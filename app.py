# app.py — ROI Workbench (Lite) with:
# - AI prefill (owner key via Streamlit secrets)
# - Sliders + Reasoning table
# - Tornado with input + BCR ranges
# - Monte Carlo: per-variable CVs, Beta/Lognormal draws (mean-preserving), "Hold cost fixed"
# - HTML one-pager

import os, json, base64, datetime, math
import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import pdfplumber, docx

# ---------------------- Page ----------------------
st.set_page_config(page_title="ROI Workbench (Lite)", layout="wide")
st.title("ROI Workbench (Lite)")

# ---------------------- Owner OpenAI key ----------------------
def _get_openai_key() -> str | None:
    try:
        return st.secrets["OPENAI_API_KEY"]
    except Exception:
        return os.getenv("OPENAI_API_KEY")

OPENAI_KEY = _get_openai_key()
if not OPENAI_KEY:
    st.error("OpenAI key missing. Add OPENAI_API_KEY to Streamlit Secrets or env vars.")
    st.stop()

# ---------------------- Sidebar -------------------
st.sidebar.header("Setup")
use_ai = st.sidebar.checkbox("Use AI to prefill from document", value=True)
model_name = st.sidebar.selectbox("Model", ["gpt-4o-mini"], index=0, help="Runs with your server-side key")

# ---------------------- Helpers -------------------
def parse_document_to_text(file):
    name = file.name.lower()
    if name.endswith(".pdf"):
        with pdfplumber.open(file) as pdf:
            return "\n\n".join(page.extract_text() or "" for page in pdf.pages)
    if name.endswith(".docx"):
        d = docx.Document(file)
        return "\n".join(p.text for p in d.paragraphs)
    return file.read().decode("utf-8", errors="ignore")

def compute_roi(params):
    n_eff = (
        params["people_reached_total"]
        * params["gitlab_share_pct"]
        * params["completion_rate"]
        * params["positive_outcome_rate"]
    )
    uplift = max(params["post_income"] - params["baseline_income"], 0.0)
    r = float(params["discount_rate"])
    T = int(params["duration_years"])
    pv_factor = (1 - (1 + r) ** (-T)) / r if r > 0 else float(T)
    pv_per_person = uplift * pv_factor
    pv_total_gain = n_eff * pv_per_person
    total_cost = params["grant_amount"] * (1 + params["overhead_rate"])
    bcr = pv_total_gain / max(total_cost, 1e-9)
    return {
        "participants_effectively_served": n_eff,
        "pv_gain_total": pv_total_gain,
        "total_cost": total_cost,
        "bcr": bcr,
    }

def extract_with_ai(doc_text: str, api_key: str) -> dict | None:
    """
    Returns {"values": {..numbers..}, "rationales": {field: {quote,page,explanation}}}
    - Page-aware prompting (adds === Page N === blocks)
    - Strict JSON schema
    - Regex fallbacks for grant_amount, total_project_cost, people_reached_total
    - Derives gitlab_share_pct if missing
    - Defaults: overhead_rate=0.097, duration_years=30, discount_rate=0.03
    - Auto-fallback to gpt-4o if required fields look thin after gpt-4o-mini
    """
    from openai import OpenAI
    import re, json, math

    client = OpenAI(api_key=api_key)

    # ---------- helpers ----------
    def _parse_money(s):
        if s is None: return None
        s = s.replace(",", "")
        m = re.search(r"(-?\d+(?:\.\d+)?)", s)
        return float(m.group(1)) if m else None

    def _money_search(text, *phrases):
        patt_money = r"\$?\s?\d{1,3}(?:[,.\s]\d{3})+(?:\.\d+)?|\$?\s?\d+(?:\.\d+)?"
        patt = r"(?i)(" + "|".join(re.escape(p) for p in phrases) + r")[^\.:\n]{0,120}?(" + patt_money + r")"
        m = re.search(patt, text)
        if not m:  # looser money sweep
            m = re.search(patt_money, text)
        return _parse_money(m.group(2) if m and m.lastindex and m.lastindex >= 2 else (m.group(0) if m else None))

    def _int_search(text, *near_words):
        patt_num = r"\b\d{1,4}\b"
        if near_words:
            patt = r"(?i)(" + "|".join(re.escape(w) for w in near_words) + r")[^\.:\n]{0,80}?(" + patt_num + r")"
            m = re.search(patt, text)
            if m: 
                try: return int(m.group(2))
                except: pass
        m = re.search(patt_num, text)
        return int(m.group(0)) if m else None

    # Wrap long text into pseudo "pages" (so model can cite a page). If you already
    # pass real pages elsewhere, this still works – it just chunks ~12–15k chars.
    def _segment_pages(s, max_chars=15000):
        s = s.replace("\r", "")
        if "\f" in s:
            pieces = [p.strip() for p in s.split("\f") if p.strip()]
        else:
            pieces = []
            cur = []
            count = 0
            for para in s.split("\n\n"):
                cur.append(para)
                count += len(para) + 2
                if count > max_chars:
                    pieces.append("\n\n".join(cur).strip()); cur=[]; count=0
            if cur: pieces.append("\n\n".join(cur).strip())
        return [p for p in pieces if p]

    pages = _segment_pages(doc_text)
    # cap to ~10–12 pages for cost; keep earliest pages which typically contain budgets/cohorts
    pages = pages[:12]

    def _build_messages():
        SYSTEM = (
            "You are building baseline inputs for a Social ROI model from an uploaded grant application.\n"
            "Return ONLY JSON with two top-level keys:\n"
            "  - values: numeric fields only (USD or fractions)\n"
            "  - rationales: for each field found, an object {quote, page, explanation}\n\n"
            "Rules (Task One):\n"
            "1) Extract the GitLab grant amount requested (USD). If multiple, take the one explicitly requested from GitLab.\n"
            "2) Overhead rate is 0.097; total cost = grant_amount * 1.097. If total_project_cost is stated, include it.\n"
            "3) Breadth: people_reached_total = total people positively impacted by the *project* (not just those served historically). "
            "   Do NOT apply completion/outcomes here; those are separate fields.\n"
            "4) completion_rate and positive_outcome_rate if clearly stated for this project/cohort. If unclear, omit (do NOT guess).\n"
            "5) Depth: baseline (counterfactual) annual income and post-program annual income in USD. Convert currencies.\n"
            "6) duration_years default 30 unless it clearly states a shorter persistent effect; discount_rate default 0.03.\n"
            "7) If total project cost or funding mix is given, also include total_project_cost and compute GitLab share as "
            "   gitlab_share_pct = grant_amount / total_project_cost.\n"
            "8) Do not compute ROI. Only provide fields and rationales.\n\n"
            "Field set for values (numbers only):\n"
            "grant_amount, total_project_cost, overhead_rate, gitlab_share_pct, "
            "people_reached_total, completion_rate, positive_outcome_rate, "
            "baseline_income, post_income, duration_years, discount_rate.\n"
        )

        # Page-labelled user content
        user_text = []
        for i, ptxt in enumerate(pages, start=1):
            user_text.append(f"=== Page {i} ===\n{ptxt}")
        USER = "Extract values with citations per the rules.\n\n" + "\n\n".join(user_text)

        # Strict schema
        schema = {
            "name": "ROI_Baseline_Extraction",
            "schema": {
                "type": "object",
                "properties": {
                    "values": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "grant_amount": {"type": "number"},
                            "total_project_cost": {"type": "number"},
                            "overhead_rate": {"type": "number"},
                            "gitlab_share_pct": {"type": "number"},
                            "people_reached_total": {"type": "number"},
                            "completion_rate": {"type": "number"},
                            "positive_outcome_rate": {"type": "number"},
                            "baseline_income": {"type": "number"},
                            "post_income": {"type": "number"},
                            "duration_years": {"type": "number"},
                            "discount_rate": {"type": "number"}
                        }
                    },
                    "rationales": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "object",
                            "properties": {
                                "quote": {"type": "string"},
                                "page":  {"type": "number"},
                                "explanation": {"type": "string"}
                            }
                        }
                    }
                },
                "required": ["values", "rationales"],
                "additionalProperties": False
            },
            "strict": True
        }
        return SYSTEM, USER, schema

    def _call(model_name: str):
        SYSTEM, USER, schema = _build_messages()
        resp = client.responses.create(
            model=model_name,               # try 'gpt-4o-mini' first; we fall back below
            input=[{"role": "system", "content": SYSTEM},
                   {"role": "user",   "content": USER}],
            response_format={"type": "json_schema", "json_schema": schema, "strict": True},
            temperature=0
        )
        raw = getattr(resp, "output_text", None)
        if not raw:
            try: raw = resp.output[0].content[0].text
            except Exception: raw = ""
        try:
            return json.loads(raw)
        except Exception:
            return None

    # 1) primary call (cheap/fast)
    data = _call("gpt-4o-mini")

    # 2) If missing criticals, escalate to 4o
    def _missing_crit(d):
        v = (d or {}).get("values", {}) if isinstance(d, dict) else {}
        needed_any = any(k in v for k in ("post_income","baseline_income","positive_outcome_rate","grant_amount","people_reached_total"))
        return (not v) or (not needed_any)
    if _missing_crit(data):
        data = _call("gpt-4o")

    # 3) If still None, bail out
    if not data or "values" not in data:
        return None

    values = {k: v for k, v in (data.get("values") or {}).items() if isinstance(v, (int, float))}
    rationales = data.get("rationales") or {}

    # ---------- regex fallbacks (only fill when missing) ----------
    joined = "\n\n".join(pages)

    # grant amount
    if "grant_amount" not in values:
        guess = _money_search(joined,
                              "request from GitLab", "requested amount", "amount requested",
                              "GitLab Foundation", "grant request", "grant amount")
        if guess:
            values["grant_amount"] = float(guess)
            rationales.setdefault("grant_amount", {
                "quote": "Regex fallback", "page": 1, "explanation": "Parsed requested amount from budget/request text."
            })

    # total project cost
    if "total_project_cost" not in values:
        tpc = _money_search(joined, "total project cost", "total project budget", "project budget", "overall budget")
        if tpc:
            values["total_project_cost"] = float(tpc)
            rationales.setdefault("total_project_cost", {
                "quote": "Regex fallback", "page": 1, "explanation": "Parsed total project cost / overall budget."
            })

    # people reached
    if "people_reached_total" not in values:
        ppl = _int_search(joined, "serve", "students", "participants", "learners", "people")
        if ppl:
            values["people_reached_total"] = int(ppl)
            rationales.setdefault("people_reached_total", {
                "quote": "Regex fallback", "page": 1, "explanation": "Parsed cohort size / people served."
            })

    # ---------- derive share if possible ----------
    if "gitlab_share_pct" not in values:
        gr = values.get("grant_amount")
        tpc = values.get("total_project_cost")
        if gr and tpc and tpc > 0:
            share = max(0.0, min(1.0, float(gr)/float(tpc)))
            values["gitlab_share_pct"] = share
            rationales.setdefault("gitlab_share_pct", {
                "quote": f"Derived from grant_amount/total_project_cost = {gr:,.0f}/{tpc:,.0f}",
                "page": 1, "explanation": "Share of total project budget covered by GitLab request."
            })

    # ---------- defaults ----------
    values.setdefault("overhead_rate", 0.097)
    values.setdefault("duration_years", 30)
    values.setdefault("discount_rate", 0.03)

    # Keep only numeric fields the app understands
    allowed = {
        "grant_amount","total_project_cost","overhead_rate","gitlab_share_pct",
        "people_reached_total","completion_rate","positive_outcome_rate",
        "baseline_income","post_income","duration_years","discount_rate"
    }
    values = {k: float(v) if k != "people_reached_total" else int(v)
              for k, v in values.items() if k in allowed and isinstance(v, (int, float))}

    return {"values": values, "rationales": rationales}

# ---------------------- Defaults (must exist before UI) ----------------------
DEFAULTS = {
    "grant_amount": 1_000_000.0,
    "overhead_rate": 0.097,
    "gitlab_share_pct": 1.0,
    "people_reached_total": 1000,
    "completion_rate": 0.8,
    "positive_outcome_rate": 1.0,
    "baseline_income": 28000.0,
    "post_income": 34000.0,
    "duration_years": 30,
    "discount_rate": 0.03,
}

# Persist defaults across Streamlit reruns
if "default_vals" not in st.session_state:
    st.session_state.default_vals = DEFAULTS.copy()

default_vals = st.session_state.default_vals  # <-- SAFE to use in the UI below


# ---------------------- Upload --------------------
uploaded = st.file_uploader("Upload grant application (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"])
if uploaded:
    text = parse_document_to_text(uploaded).strip()
    with st.expander("Extracted document text (first 5,000 chars)"):
        st.text_area("text", text[:5000], height=240)

    if text and use_ai:
        # NOTE: extract_with_ai accepts (doc_text, api_key) in your current code
        ai = extract_with_ai(text, OPENAI_KEY)
        if ai:
            vals = ai.get("values", {}) or {}
            reasons = ai.get("rationales", {}) or {}

            # Merge AI values into defaults and persist to session
            for k, v in vals.items():
                if k in default_vals and v is not None:
                    default_vals[k] = v
            st.session_state.default_vals = default_vals  # persist
            st.session_state.ai_rationales = reasons      # persist

            st.success("AI prefill applied from the uploaded document.")
            with st.expander("AI values (applied)"):
                st.json(vals)
        else:
            st.warning("AI prefill failed; using defaults.")

# ---------------------- Inputs --------------------
st.subheader("Assumptions")
col1, col2, col3 = st.columns(3)
with col1:
    grant_amount = st.number_input("GitLab grant amount (USD)", min_value=0.0, value=float(default_vals["grant_amount"]), step=1000.0)
    overhead     = st.slider("Overhead rate (rule: 9.7%)", 0.0, 0.5, float(default_vals["overhead_rate"]), 0.001)
    people       = st.number_input("People reached (total project)", min_value=0, value=int(default_vals["people_reached_total"]), step=10)
with col2:
    share        = st.slider("GitLab share of project funding", 0.0, 1.0, float(default_vals["gitlab_share_pct"]), 0.01)
    completion   = st.slider("Completion rate", 0.0, 1.0, float(default_vals["completion_rate"]), 0.01)
    positive     = st.slider("Positive outcome rate", 0.0, 1.0, float(default_vals["positive_outcome_rate"]), 0.01)
with col3:
    baseline_inc = st.number_input("Baseline income ($/yr)", min_value=0.0, value=float(default_vals["baseline_income"]), step=500.0)
    post_inc     = st.number_input("Post-program income ($/yr)", min_value=0.0, value=float(default_vals["post_income"]), step=500.0)

row = st.columns(2)
with row[0]:
    duration     = st.slider("Duration (yrs)", 1, 40, int(default_vals["duration_years"]))
with row[1]:
    discount     = st.slider("Real discount rate", 0.0, 0.2, float(default_vals["discount_rate"]), 0.005)

params = dict(
    grant_amount=grant_amount, overhead_rate=overhead,
    gitlab_share_pct=share,
    people_reached_total=people, completion_rate=completion, positive_outcome_rate=positive,
    baseline_income=baseline_inc, post_income=post_inc,
    duration_years=duration, discount_rate=discount,
)

# ---------------------- Reasoning table -------------------
reason_map = st.session_state.get("ai_rationales", {}) if "ai_rationales" in st.session_state else {}
if reason_map:
    st.markdown("### Why these values?")
    keys_in_ui = [
        "grant_amount","overhead_rate","gitlab_share_pct","people_reached_total",
        "completion_rate","positive_outcome_rate","baseline_income","post_income",
        "duration_years","discount_rate"
    ]
    current_vals = {
        "grant_amount": float(params["grant_amount"]),
        "overhead_rate": float(params["overhead_rate"]),
        "gitlab_share_pct": float(params["gitlab_share_pct"]),
        "people_reached_total": int(params["people_reached_total"]),
        "completion_rate": float(params["completion_rate"]),
        "positive_outcome_rate": float(params["positive_outcome_rate"]),
        "baseline_income": float(params["baseline_income"]),
        "post_income": float(params["post_income"]),
        "duration_years": int(params["duration_years"]),
        "discount_rate": float(params["discount_rate"]),
    }
    rows = [{"Field": k, "Value used": current_vals.get(k, ""), "Reasoning": reason_map.get(k, "—")} for k in keys_in_ui]
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ---------------------- Results -------------------
res = compute_roi(params)
k1, k2, k3, k4 = st.columns(4)
k1.metric("BCR", f"{res['bcr']:.1f}×")
k2.metric("PV Lifetime Gains (Total)", f"${res['pv_gain_total']:,.0f}")
k3.metric("Participants (effective)", f"{res['participants_effectively_served']:,.0f}")
k4.metric("Total Cost (incl. OH)", f"${res['total_cost']:,.0f}")

# ---------------------- Chart ---------------------
years = np.arange(0, int(params["duration_years"]) + 1)
r = float(params["discount_rate"])
T = int(params["duration_years"])
uplift = max(params["post_income"] - params["baseline_income"], 0.0)
pv_factor_per_year = np.array([0.0] + [1 / (1 + r) ** t for t in range(1, T + 1)])
ppl_eff = params["people_reached_total"] * params["gitlab_share_pct"] * params["completion_rate"] * params["positive_outcome_rate"]
annual_pv_total = pv_factor_per_year * uplift * ppl_eff
cum_pv = np.cumsum(annual_pv_total)
df = pd.DataFrame({"Year": years, "Cumulative PV Gains ($)": cum_pv})
chart = (
    alt.Chart(df).mark_line(point=True).encode(
        x=alt.X("Year:O"),
        y=alt.Y("Cumulative PV Gains ($):Q", title="PV ($)"),
        tooltip=["Year", alt.Tooltip("Cumulative PV Gains ($):Q", format="$,.0f")]
    ).properties(height=320)
)
st.altair_chart(chart, use_container_width=True)
st.divider()

# ---------------------- Tornado (input + BCR ranges) -------------------
st.subheader("One-way Sensitivity (Tornado)")
rng = st.slider("± Range", 0.05, 0.5, 0.2, 0.05)

def run_tornado(p, var, low, high):
    pl = p.copy(); ph = p.copy()
    pl[var] = low; ph[var] = high
    return (compute_roi(pl)["bcr"], compute_roi(ph)["bcr"])

specs = [
    ("gitlab_share_pct", 0.0, 1.0, "pct"),
    ("completion_rate", 0.0, 1.0, "pct"),
    ("positive_outcome_rate", 0.0, 1.0, "pct"),
    ("baseline_income", 0.0, None, "usd"),
    ("post_income", 0.0, None, "usd"),
    ("duration_years", 1.0, 40.0, "int"),
    ("discount_rate", 0.0, 0.2, "pct"),
    ("overhead_rate", 0.0, 0.5, "pct"),
]
rows = []
for var, lo_b, hi_b, kind in specs:
    baseline_val = params[var]
    lo_in = max(baseline_val * (1 - rng), lo_b if lo_b is not None else -1e18)
    hi_in = min(baseline_val * (1 + rng), hi_b if hi_b is not None else 1e18)
    if kind == "int":
        lo_in = int(round(lo_in)); hi_in = int(round(hi_in))
    bcr_lo, bcr_hi = run_tornado(params, var, lo_in, hi_in)
    lo_bcr, hi_bcr = (min(bcr_lo, bcr_hi), max(bcr_lo, bcr_hi))
    rows.append({"Variable": var, "Input Low": lo_in, "Input High": hi_in, "BCR Low": lo_bcr, "BCR High": hi_bcr, "Delta BCR": hi_bcr - lo_bcr})
tdf = pd.DataFrame(rows).sort_values("Delta BCR", ascending=False)

def fmt_input(val, kind):
    if kind == "usd": return f"${val:,.0f}"
    if kind == "pct": return f"{val:.0%}"
    if kind == "int": return f"{int(val)}"
    return f"{val:.2f}"

pretty_rows, kind_map = [], {v[0]: v[3] for v in specs}
for _, rrow in tdf.iterrows():
    k = rrow["Variable"]
    pretty_rows.append({"Variable": k, "Input Range": f"{fmt_input(rrow['Input Low'], kind_map[k])} → {fmt_input(rrow['Input High'], kind_map[k])}",
                        "BCR Range": f"{rrow['BCR Low']:.2f}× → {rrow['BCR High']:.2f}×"})
st.write("#### Input and BCR ranges")
st.dataframe(pd.DataFrame(pretty_rows), use_container_width=True)

chart_t = (
    alt.Chart(tdf.assign(VariableLabel=tdf["Variable"].str.replace("_", " "))).mark_bar().encode(
        x=alt.X("BCR Low:Q", title="BCR"),
        x2="BCR High:Q",
        y=alt.Y("VariableLabel:N", sort=tdf["Variable"].tolist(), title="Variable"),
        tooltip=[alt.Tooltip("Variable", title="Variable"),
                 alt.Tooltip("Input Low", format=".4f"),
                 alt.Tooltip("Input High", format=".4f"),
                 alt.Tooltip("BCR Low", format=".2f"),
                 alt.Tooltip("BCR High", format=".2f")]
    ).properties(height=min(40 * len(tdf), 520))
)
st.altair_chart(chart_t, use_container_width=True)
st.divider()

# ---------------------- Monte Carlo (per-var CVs, Beta/Lognormal, hold-cost) ---------------
st.subheader("Monte Carlo")

# ----- CV UI -----
st.markdown("**Uncertainty (Coefficient of Variation, SD as % of mean)**")
colcv0, colcv1, colcv2, colcv3 = st.columns(4)
with colcv0: sims = st.slider("Simulations", 500, 15000, 3000, 500)
with colcv1: seed = st.number_input("Random seed", value=42, step=1)
with colcv2: hold_cost_fixed = st.checkbox("Hold cost fixed in Monte Carlo", value=True)
with colcv3: base_cv = st.slider("Global starting CV", 0.02, 0.5, 0.15, 0.01, help="Seed value for the per-variable CV sliders")

st.write("Adjust CV per variable (higher = more uncertain):")
cva, cvb, cvc = st.columns(3)
with cva:
    cv_share   = st.slider("CV: GitLab share", 0.0, 0.8, base_cv, 0.01)
    cv_comp    = st.slider("CV: Completion rate", 0.0, 0.8, base_cv, 0.01)
    cv_pos     = st.slider("CV: Positive outcome rate", 0.0, 0.8, base_cv, 0.01)
with cvb:
    cv_baseinc = st.slider("CV: Baseline income", 0.0, 0.8, min(0.30, base_cv+0.05), 0.01)
    cv_postinc = st.slider("CV: Post income", 0.0, 0.8, min(0.30, base_cv+0.05), 0.01)
    cv_people  = st.slider("CV: People reached", 0.0, 0.8, min(0.20, base_cv), 0.01)
with cvc:
    cv_dur     = st.slider("CV: Duration (yrs)", 0.0, 0.5, 0.0, 0.01)
    cv_disc    = st.slider("CV: Discount rate", 0.0, 0.5, 0.05, 0.01)
    cv_grant   = st.slider("CV: Grant amount", 0.0, 0.5, 0.0 if hold_cost_fixed else 0.05, 0.01)
    cv_oh      = st.slider("CV: Overhead rate", 0.0, 0.5, 0.0 if hold_cost_fixed else 0.05, 0.01)

# ----- Beta/Lognormal helpers -----
def _beta_from_mean_cv(mean, cv, eps=1e-9):
    """Return alpha,beta for Beta(mean,var) given mean in (0,1) and CV (SD/mean).
       If infeasible (variance >= mean*(1-mean)), shrink CV to be feasible."""
    m = min(max(float(mean), eps), 1 - eps)
    if cv <= eps:
        # near-deterministic: very large concentration
        return 1e6*m, 1e6*(1-m)
    var = (cv * m) ** 2
    max_var = m * (1 - m) - 1e-9
    if var >= max_var:
        var = max_var
    k = m * (1 - m) / var - 1.0
    alpha, beta = m * k, (1 - m) * k
    return max(alpha, eps), max(beta, eps)

def draw_scaled_beta_scalar(mean, cap, cv, rng, eps=1e-9):
    """Mean-preserving scaled Beta in [0, cap]; returns a **float**."""
    if cv <= 0:
        return float(mean)
    cap = float(cap)
    if cap <= eps:
        return 0.0
    mean_frac = min(max(mean / cap, eps), 1 - eps)
    a, b = _beta_from_mean_cv(mean_frac, cv)
    y = rng.beta(a, b)          # scalar float
    return float(y * cap)

def draw_lognormal_mean_preserving_scalar(mean, cv, rng):
    """Return a **float** with E[X] = mean for lognormal (if cv > 0), else mean."""
    mean = float(mean)
    if mean <= 0 or cv <= 0:
        return mean
    sigma2 = math.log(1.0 + cv**2)
    sigma  = math.sqrt(sigma2)
    mu     = math.log(mean) - 0.5 * sigma2
    return float(rng.lognormal(mean=mu, sigma=sigma))  # scalar float

def mc(params, sims, seed,
       cv_share, cv_comp, cv_pos, cv_baseinc, cv_postinc, cv_people, cv_dur, cv_disc, cv_grant, cv_oh,
       hold_cost_fixed=True):
    rng = np.random.default_rng(seed)

    max_overhead = 0.5
    max_discount = 0.2
    dur_lo, dur_hi = 1, 40
    draws = np.empty(sims, dtype=float)

    # Baseline means
    m_share, m_comp, m_pos = params["gitlab_share_pct"], params["completion_rate"], params["positive_outcome_rate"]
    m_base, m_post = params["baseline_income"], params["post_income"]
    m_people = params["people_reached_total"]
    m_dur, m_disc = params["duration_years"], params["discount_rate"]
    m_grant, m_oh = params["grant_amount"], params["overhead_rate"]

    for i in range(sims):
        p = params.copy()

        # Rates in [0,1] via Beta (scalars)
        if cv_share > 0:
            a, b = _beta_from_mean_cv(m_share, cv_share)
            p["gitlab_share_pct"] = float(rng.beta(a, b))
        else:
            p["gitlab_share_pct"] = m_share

        if cv_comp > 0:
            a, b = _beta_from_mean_cv(m_comp, cv_comp)
            p["completion_rate"] = float(rng.beta(a, b))
        else:
            p["completion_rate"] = m_comp

        if cv_pos > 0:
            a, b = _beta_from_mean_cv(m_pos, cv_pos)
            p["positive_outcome_rate"] = float(rng.beta(a, b))
        else:
            p["positive_outcome_rate"] = m_pos

        # Discount & overhead via scaled Beta (keeps caps)
        p["discount_rate"] = draw_scaled_beta_scalar(m_disc, max_discount, cv_disc, rng) if cv_disc > 0 else m_disc

        if hold_cost_fixed:
            p["overhead_rate"] = m_oh
        else:
            p["overhead_rate"] = draw_scaled_beta_scalar(m_oh, max_overhead, cv_oh, rng) if cv_oh > 0 else m_oh

        # Positive variables via mean-preserving lognormal (scalars)
        p["baseline_income"] = draw_lognormal_mean_preserving_scalar(m_base, cv_baseinc, rng) if cv_baseinc > 0 else m_base
        p["post_income"]     = draw_lognormal_mean_preserving_scalar(m_post, cv_postinc, rng) if cv_postinc > 0 else m_post

        ppl_draw = draw_lognormal_mean_preserving_scalar(max(m_people, 1.0), cv_people, rng) if cv_people > 0 else m_people
        p["people_reached_total"] = int(round(max(1.0, ppl_draw)))

        # Duration as clipped rounded normal
        if cv_dur > 0:
            sd_dur = max(m_dur * cv_dur, 1e-9)
            p["duration_years"] = int(np.clip(round(rng.normal(m_dur, sd_dur)), dur_lo, dur_hi))
        else:
            p["duration_years"] = m_dur

        # Grant amount (cost) — hold fixed if toggled
        if hold_cost_fixed or cv_grant == 0:
            p["grant_amount"] = m_grant
        else:
            p["grant_amount"] = draw_lognormal_mean_preserving_scalar(max(m_grant, 1.0), cv_grant, rng)

        draws[i] = compute_roi(p)["bcr"]

    pct = np.percentile(draws, [5, 50, 95])
    stats = dict(
        p5=pct[0], p50=pct[1], p95=pct[2],
        mean=float(draws.mean()), std=float(draws.std()),
        frac_ge_100=float(np.mean(draws >= 100.0))
    )
    return draws, stats

if st.button("Run Monte Carlo"):
    arr, stats = mc(
        params, sims, seed,
        cv_share, cv_comp, cv_pos, cv_baseinc, cv_postinc, cv_people, cv_dur, cv_disc, cv_grant, cv_oh,
        hold_cost_fixed=hold_cost_fixed
    )
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("P5",  f"{stats['p5']:.1f}×")
    c2.metric("P50", f"{stats['p50']:.1f}×")
    c3.metric("P95", f"{stats['p95']:.1f}×")
    c4.metric("Mean",f"{stats['mean']:.1f}×")
    c5.metric("Std", f"{stats['std']:.2f}")
    c6.metric("% ≥ 100×", f"{stats['frac_ge_100']*100:.1f}%")

    hist_df = pd.DataFrame({"BCR": arr})
    chart_h = (
        alt.Chart(hist_df).mark_bar().encode(
            x=alt.X("BCR:Q", bin=alt.Bin(maxbins=40)), y="count()"
        ).properties(height=300)
    )
    st.altair_chart(chart_h, use_container_width=True)

st.divider()

# ---------------------- One-pager (HTML) ---------
st.subheader("One-pager (HTML download)")
def render_onepager_html(params, res):
    html = f"""
<!doctype html><html><head><meta charset="utf-8"><title>ROI One-Pager</title>
<style>body{{font-family:Arial;margin:24px}}
.kpi div{{border:1px solid #ddd;padding:10px;border-radius:8px;margin-right:10px;display:inline-block}}</style></head><body>
<h1>ROI Summary</h1>
<p><small>{datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}</small></p>
<div class="kpi">
  <div><b>BCR</b><br>{res['bcr']:.1f}×</div>
  <div><b>PV Lifetime Gains</b><br>${res['pv_gain_total']:,.0f}</div>
  <div><b>Total Cost</b><br>${res['total_cost']:,.0f}</div>
  <div><b>Participants</b><br>{res['participants_effectively_served']:,.0f}</div>
</div>
<h2>Assumptions</h2>
<ul>
  <li>People reached (project): {params['people_reached_total']:,}</li>
  <li>GitLab share: {params['gitlab_share_pct']:.0%}</li>
  <li>Completion: {params['completion_rate']:.0%}; Positive outcome: {params['positive_outcome_rate']:.0%}</li>
  <li>Baseline: ${params['baseline_income']:,.0f}; Post: ${params['post_income']:,.0f}</li>
  <li>Duration: {params['duration_years']} yrs; Discount: {params['discount_rate']:.1%}</li>
  <li>Grant: ${params['grant_amount']:,.0f}; Overhead: {params['overhead_rate']:.1%}</li>
</ul>
</body></html>
"""
    return html

if st.button("Download one-pager (HTML)"):
    html = render_onepager_html(params, res)
    b64 = base64.b64encode(html.encode("utf-8")).decode()
    st.markdown(f'<a download="onepager.html" href="data:text/html;base64,{b64}">Click to download</a>', unsafe_allow_html=True)
