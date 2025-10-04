# app.py — ROI Workbench (Lite, with AI rationales)
# - Uses Streamlit secret OPENAI_API_KEY (owner key)
# - AI prefill returns values + rationales; computes gitlab_share_pct if needed
# - Constant annual uplift PV (no half-life, attribution, deadweight)
# - Sliders, Tornado, Monte Carlo, HTML one-pager

import os, json, base64, datetime
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
    """
    Constant annual uplift PV with discounting; GitLab share gates breadth.
    """
    n_eff = (
        params["people_reached_total"]
        * params["gitlab_share_pct"]
        * params["completion_rate"]
        * params["positive_outcome_rate"]
    )

    uplift = max(params["post_income"] - params["baseline_income"], 0.0)  # USD per person per year
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

# ---------------------- AI extraction (values + rationales) -------------------
def extract_with_ai(doc_text: str, api_key: str, model_name: str = "gpt-4o-mini") -> dict | None:
    """
    Returns:
    {
      "values": {... numeric inputs ...},
      "rationales": {key: short explanation},
    }
    Optionally includes values.total_project_cost; we derive gitlab_share_pct when possible.
    """
    from openai import OpenAI
    import json, re

    client = OpenAI(api_key=api_key)

    SYSTEM = (
        "Analyze the uploaded grant application and return baseline JSON inputs for a Social ROI model.\n"
        "Return ONLY JSON with two top-level keys: 'values' (numeric inputs) and 'rationales' "
        "(short explanations with quotes/page cues or formulas). If a value is missing, omit it from 'values' "
        "but include a brief note in 'rationales'.\n\n"
        "Task One rules:\n"
        "1) Extract GitLab grant amount requested (USD).\n"
        "2) Overhead rate is 0.097; total cost = grant_amount * 1.097 (you may also return total_project_cost if provided).\n"
        "3) Breadth: people_reached_total = number of people positively impacted by this project overall. "
        "   The app will later apply GitLab share and completion/outcome.\n"
        "4) completion_rate and positive_outcome_rate if stated; otherwise omit rather than guess.\n"
        "5) Depth: baseline (counterfactual) annual income and post-program annual income in USD (convert currencies).\n"
        "6) duration_years default 30 unless clearly less; discount_rate default 0.03.\n"
        "7) GitLab share of project funding: If the application states a share or a total project budget, derive "
        "   gitlab_share_pct = grant_amount / total_project_cost. If neither is clear, omit it.\n"
        "8) Do not compute ROI.\n\n"
        "JSON shape:\n"
        "{\n"
        "  \"values\": {\n"
        "    \"grant_amount\": number,\n"
        "    \"overhead_rate\": number,\n"
        "    \"gitlab_share_pct\": number,\n"
        "    \"people_reached_total\": integer,\n"
        "    \"completion_rate\": number,\n"
        "    \"positive_outcome_rate\": number,\n"
        "    \"baseline_income\": number,\n"
        "    \"post_income\": number,\n"
        "    \"duration_years\": integer,\n"
        "    \"discount_rate\": number,\n"
        "    \"total_project_cost\": number\n"
        "  },\n"
        "  \"rationales\": { key: string }\n"
        "}\n"
        "Keep rationales ≤200 chars; include page cues or a formula when derived."
    )

    clipped = doc_text[:120000]

    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Grant application text (truncated):\n" + clipped}
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    raw = resp.choices[0].message.content or ""
    try:
        data = json.loads(raw)
    except Exception:
        st.warning("AI prefill returned non-JSON; showing raw output below.")
        with st.expander("Raw model output"):
            st.code(raw or "<empty>")
        m = re.search(r"\{[\s\S]*\}", raw)
        if not m:
            return None
        data = json.loads(m.group(0))

    values = data.get("values", {}) or {}
    rationales = data.get("rationales", {}) or {}

    # Fallback: derive gitlab_share_pct if grant + total_project_cost present
    if "gitlab_share_pct" not in values and "grant_amount" in values and "total_project_cost" in values:
        tpc = float(values["total_project_cost"]) or 0.0
        gr  = float(values["grant_amount"]) or 0.0
        if tpc > 0:
            share = max(0.0, min(1.0, gr / tpc))
            values["gitlab_share_pct"] = share
            rationales.setdefault(
                "gitlab_share_pct",
                f"Derived as grant_amount / total_project_cost = {gr:,.0f} / {tpc:,.0f}"
            )

    # Defaults
    values.setdefault("overhead_rate", 0.097)
    values.setdefault("duration_years", 30)
    values.setdefault("discount_rate", 0.03)

    return {"values": values, "rationales": rationales}

# ---------------------- Defaults ------------------
default_vals = {
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
    "citations": [],
}

# ---------------------- Upload --------------------
uploaded = st.file_uploader("Upload grant application (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"])

if uploaded:
    text = parse_document_to_text(uploaded).strip()

    with st.expander("Extracted document text (first 5,000 chars)"):
        st.text_area("text", text[:5000], height=240)

    if not text:
        st.warning("Couldn’t extract readable text from the file. Try a text-based PDF or DOCX.")
    elif use_ai:
        ai = extract_with_ai(text, OPENAI_KEY, model_name)
        if ai:
            vals = ai.get("values", {})
            reasons = ai.get("rationales", {})
            for k, v in vals.items():
                if k in default_vals and v is not None:
                    default_vals[k] = v
            st.session_state["ai_rationales"] = reasons
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
    rows = []
    for k in keys_in_ui:
        rows.append({"Field": k, "Value used": current_vals.get(k, ""), "Reasoning": reason_map.get(k, "—")})
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
    alt.Chart(df)
    .mark_line(point=True)
    .encode(
        x=alt.X("Year:O"),
        y=alt.Y("Cumulative PV Gains ($):Q", title="PV ($)"),
        tooltip=["Year", alt.Tooltip("Cumulative PV Gains ($):Q", format="$,.0f")]
    )
    .properties(height=320)
)
st.altair_chart(chart, use_container_width=True)

st.divider()

# ---------------------- Tornado -------------------
st.subheader("One-way Sensitivity (Tornado)")
rng = st.slider("± Range", 0.05, 0.5, 0.2, 0.05)

def run_tornado(p, var, low, high):
    pl = p.copy(); ph = p.copy()
    pl[var] = low; ph[var] = high
    return compute_roi(pl)["bcr"], compute_roi(ph)["bcr"]

rows = []
specs = [
    ("gitlab_share_pct", 0.0, 1.0),
    ("completion_rate", 0.0, 1.0),
    ("positive_outcome_rate", 0.0, 1.0),
    ("baseline_income", 0.0, None),
    ("post_income", 0.0, None),
    ("duration_years", 1.0, 40.0),
    ("discount_rate", 0.0, 0.2),
    ("overhead_rate", 0.0, 0.5),
]
for var, lo_b, hi_b in specs:
    baseline_val = params[var]
    lo = max(baseline_val * (1 - rng), lo_b if lo_b is not None else -1e18)
    hi = min(baseline_val * (1 + rng), hi_b if hi_b is not None else 1e18)
    l, h = run_tornado(params, var, lo, hi)
    rows.append({"Variable": var, "Low": min(l, h), "High": max(l, h), "Delta": abs(h - l)})
tdf = pd.DataFrame(rows).sort_values("Delta", ascending=False)
chart_t = (
    alt.Chart(tdf)
    .mark_bar()
    .encode(
        x=alt.X("Low:Q", title="BCR"),
        x2="High:Q",
        y=alt.Y("Variable:N", sort=tdf["Variable"].tolist()),
        tooltip=["Variable", alt.Tooltip("Low", format=".2f"), alt.Tooltip("High", format=".2f")]
    )
    .properties(height=min(40 * len(tdf), 520))
)
st.altair_chart(chart_t, use_container_width=True)

st.divider()

# ---------------------- Monte Carlo ---------------
st.subheader("Monte Carlo")
colmc1, colmc2, colmc3 = st.columns(3)
with colmc1: sims = st.slider("Simulations", 500, 10000, 3000, 500)
with colmc2: cv   = st.slider("Coef. of variation (SD as % of mean)", 0.02, 0.5, 0.1, 0.01)
with colmc3: seed = st.number_input("Random seed", value=42, step=1)

def mc(params, sims, cv, seed):
    rng = np.random.default_rng(seed)
    def clip(v, lo=None, hi=None):
        if lo is not None: v = np.maximum(v, lo)
        if hi is not None: v = np.minimum(v, hi)
        return v
    draws = []
    specs_local = [
        ("gitlab_share_pct", 0.0, 1.0),
        ("completion_rate", 0.0, 1.0),
        ("positive_outcome_rate", 0.0, 1.0),
        ("baseline_income", 0.0, None),
        ("post_income", 0.0, None),
        ("duration_years", 1.0, 40.0),
        ("discount_rate", 0.0, 0.2),
        ("overhead_rate", 0.0, 0.5),
        ("grant_amount", 0.0, None),
        ("people_reached_total", 0.0, None),
    ]
    for _ in range(sims):
        p = params.copy()
        for k, lo, hi in specs_local:
            m = params[k]
            sd = max(abs(m) * cv, 1e-9)
            v = rng.normal(m, sd)
            if k in ("gitlab_share_pct","completion_rate","positive_outcome_rate","discount_rate","overhead_rate"):
                v = float(clip(v, 0.0, 1.0 if k != "discount_rate" else 0.2))
            if k == "duration_years":  v = int(round(clip(v, 1.0, 40.0)))
            if k in ("baseline_income","post_income","grant_amount","people_reached_total"):
                v = float(max(v, 0.0))
            p[k] = v
        draws.append(compute_roi(p)["bcr"])
    arr = np.array(draws)
    pct = np.percentile(arr, [5, 50, 95])
    return arr, dict(p5=pct[0], p50=pct[1], p95=pct[2], mean=float(arr.mean()), std=float(arr.std()))

if st.button("Run Monte Carlo"):
    arr, stats = mc(params, sims, cv, seed)
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("P5",  f"{stats['p5']:.1f}×")
    c2.metric("P50", f"{stats['p50']:.1f}×")
    c3.metric("P95", f"{stats['p95']:.1f}×")
    c4.metric("Mean",f"{stats['mean']:.1f}×")
    c5.metric("Std", f"{stats['std']:.2f}")
    hist_df = pd.DataFrame({"BCR": arr})
    chart_h = (
        alt.Chart(hist_df)
        .mark_bar()
        .encode(x=alt.X("BCR:Q", bin=alt.Bin(maxbins=40)), y="count()")
        .properties(height=300)
    )
    st.altair_chart(chart_h, use_container_width=True)

st.divider()

# ---------------------- One-pager (HTML) ---------
st.subheader("One-pager (HTML download)")
def render_onepager_html(params, res):
    html = f"""
<!doctype html><html><head><meta charset="utf-8"><title>ROI One-Pager</title>
<style>
body{{font-family:Arial;margin:24px}}
.kpi div{{border:1px solid #ddd;padding:10px;border-radius:8px;margin-right:10px;display:inline-block}}
</style></head><body>
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
