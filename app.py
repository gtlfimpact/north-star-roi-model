import os, json, base64, datetime
import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import pdfplumber, docx

# --------- Page config ---------
st.set_page_config(page_title="ROI Workbench (Lite)", layout="wide")
st.title("ROI Workbench (Lite)")

# --------- Sidebar: optional AI prefill ---------
st.sidebar.header("Setup")
use_ai = st.sidebar.checkbox("Use AI to prefill from document", value=False)
api_key = st.sidebar.text_input("OpenAI API Key (optional)", type="password") if use_ai else None
model_name = st.sidebar.selectbox("Model", ["gpt-4o-mini", "gpt-5.0-mini"], index=0) if use_ai else None

# --------- Upload ---------
uploaded = st.file_uploader("Upload grant application (PDF, DOCX, or TXT)", type=["pdf","docx","txt"])

def parse_document_to_text(file):
    name = file.name.lower()
    if name.endswith(".pdf"):
        with pdfplumber.open(file) as pdf:
            return "\n\n".join(page.extract_text() or "" for page in pdf.pages)
    if name.endswith(".docx"):
        doc = docx.Document(file)
        return "\n".join(p.text for p in doc.paragraphs)
    return file.read().decode("utf-8", errors="ignore")

# --------- Minimal ROI engine ---------
def compute_roi(params):
    # effective participants
    n_eff = params["people_reached_total"] * params["completion_rate"] * params["positive_outcome_rate"]
    uplift = max(params["post_income"] - params["baseline_income"], 0.0)

    years = np.arange(1, params["duration_years"] + 1)
    decay = np.log(2) / params["half_life_years"]
    stream = uplift * np.exp(-decay * (years - 1))
    pv_stream = (stream / (1 + params["discount_rate"]) ** years).sum()

    adj = (1 - params["deadweight"]) * params["attribution"]
    pv_total_gain = n_eff * pv_stream * adj

    total_cost = params["grant_amount"] * (1 + params["overhead_rate"])
    bcr = pv_total_gain / max(total_cost, 1e-9)
    return {
        "participants_effectively_served": n_eff,
        "pv_gain_total": pv_total_gain,
        "total_cost": total_cost,
        "bcr": bcr
    }

# --------- Prefill with AI (optional) ---------
default_vals = {
    "grant_amount": 1_000_000.0,
    "overhead_rate": 0.097,
    "people_reached_total": 1000,
    "completion_rate": 0.8,
    "positive_outcome_rate": 1.0,
    "baseline_income": 28000.0,
    "post_income": 34000.0,
    "half_life_years": 8,
    "duration_years": 30,
    "discount_rate": 0.03,
    "attribution": 1.0,
    "deadweight": 0.0,
    "citations": [],
}

if uploaded:
    text = parse_document_to_text(uploaded)

    if use_ai and api_key:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        SYSTEM_PROMPT = (
            "Extract baseline inputs for an ROI model as strict JSON with keys: "
            "grant_amount, overhead_rate, people_reached_total, completion_rate, "
            "positive_outcome_rate, baseline_income, post_income, half_life_years, "
            "duration_years, discount_rate, attribution, deadweight, citations. "
            "Defaults: overhead_rate=0.097, duration_years=30, discount_rate=0.03, "
            "attribution=1.0, deadweight=0.0. All USD. If unknown, guess reasonable values."
        )
        try:
            resp = client.responses.create(
                model=model_name,
                input=[
                    {"role":"system","content":SYSTEM_PROMPT},
                    {"role":"user","content":text[:100000]}  # avoid very large prompts
                ]
            )
            raw = resp.output[0].content[0].text
            data = json.loads(raw)
            # merge into defaults
            for k in default_vals:
                if k in data and data[k] is not None:
                    default_vals[k] = data[k]
            if "citations" in data:
                default_vals["citations"] = data["citations"]
            st.success("AI prefill complete.")
            with st.expander("AI baseline JSON"):
                st.json(data)
        except Exception as e:
            st.warning(f"AI prefill failed; using defaults. ({e})")

# --------- Inputs (always editable) ---------
st.subheader("Assumptions")
col1, col2, col3 = st.columns(3)
with col1:
    grant_amount = st.number_input("GitLab grant amount (USD)", min_value=0.0, value=float(default_vals["grant_amount"]), step=1000.0)
    overhead     = st.slider("Overhead rate", 0.0, 0.5, float(default_vals["overhead_rate"]), 0.001)
    people       = st.number_input("People reached (total)", min_value=0, value=int(default_vals["people_reached_total"]), step=10)
with col2:
    completion   = st.slider("Completion rate", 0.0, 1.0, float(default_vals["completion_rate"]), 0.01)
    positive     = st.slider("Positive outcome rate", 0.0, 1.0, float(default_vals["positive_outcome_rate"]), 0.01)
    baseline_inc = st.number_input("Baseline income ($/yr)", min_value=0.0, value=float(default_vals["baseline_income"]), step=500.0)
with col3:
    post_inc     = st.number_input("Post-program income ($/yr)", min_value=0.0, value=float(default_vals["post_income"]), step=500.0)
    half_life    = st.slider("Uplift half-life (yrs)", 1, 40, int(default_vals["half_life_years"]))
    duration     = st.slider("Duration (yrs)", 1, 40, int(default_vals["duration_years"]))
row = st.columns(3)
with row[0]:
    discount     = st.slider("Real discount rate", 0.0, 0.2, float(default_vals["discount_rate"]), 0.005)
with row[1]:
    attribution  = st.slider("Attribution", 0.0, 1.0, float(default_vals["attribution"]), 0.05)
with row[2]:
    deadweight   = st.slider("Deadweight", 0.0, 1.0, float(default_vals["deadweight"]), 0.05)

params = dict(
    grant_amount=grant_amount, overhead_rate=overhead,
    people_reached_total=people, completion_rate=completion, positive_outcome_rate=positive,
    baseline_income=baseline_inc, post_income=post_inc, half_life_years=half_life,
    duration_years=duration, discount_rate=discount, attribution=attribution, deadweight=deadweight,
)

# --------- Results ---------
res = compute_roi(params)
k1,k2,k3,k4 = st.columns(4)
k1.metric("BCR", f"{res['bcr']:.1f}×")
k2.metric("PV Lifetime Gains (Total)", f"${res['pv_gain_total']:,.0f}")
k3.metric("Participants (effective)", f"{res['participants_effectively_served']:,.0f}")
k4.metric("Total Cost (incl. OH)", f"${res['total_cost']:,.0f}")

# --------- Chart: Cumulative PV ---------
years = list(range(0, int(params["duration_years"]) + 1))
decay = np.log(2) / params["half_life_years"]
uplift = max(params["post_income"] - params["baseline_income"], 0.0)
stream = [0.0] + [uplift * np.exp(-decay * (t-1)) for t in years[1:]]
disc   = [0.0] + [1 / (1 + params["discount_rate"]) ** t for t in years[1:]]
annual_pv_per = [s * d for s, d in zip(stream, disc)]
n_eff = params["people_reached_total"] * params["completion_rate"] * params["positive_outcome_rate"]
annual_pv_total = [x * n_eff * (1 - params["deadweight"]) * params["attribution"] for x in annual_pv_per]
cum_pv = np.cumsum([0.0] + annual_pv_total[1:])
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

# --------- Tornado (one-way) ---------
st.subheader("One-way Sensitivity (Tornado)")
rng = st.slider("± Range", 0.05, 0.5, 0.2, 0.05)
def run_tornado(p, var, low, high):
    pl = p.copy(); ph = p.copy()
    pl[var] = low; ph[var] = high
    return compute_roi(pl)["bcr"], compute_roi(ph)["bcr"]

rows = []
specs = [
    ("completion_rate", 0.0, 1.0),
    ("positive_outcome_rate", 0.0, 1.0),
    ("baseline_income", 0.0, None),
    ("post_income", 0.0, None),
    ("half_life_years", 1.0, 40.0),
    ("duration_years", 1.0, 40.0),
    ("discount_rate", 0.0, 0.2),
    ("attribution", 0.0, 1.0),
    ("deadweight", 0.0, 1.0),
    ("overhead_rate", 0.0, 0.5),
]
for var, lo_b, hi_b in specs:
    baseline_val = params[var]
    lo = max(baseline_val*(1-rng), lo_b if lo_b is not None else -1e18)
    hi = min(baseline_val*(1+rng), hi_b if hi_b is not None else 1e18)
    l, h = run_tornado(params, var, lo, hi)
    rows.append({"Variable": var, "Low": min(l,h), "High": max(l,h), "Delta": abs(h-l)})
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
    .properties(height=min(40*len(tdf), 520))
)
st.altair_chart(chart_t, use_container_width=True)

st.divider()

# --------- Monte Carlo ---------
st.subheader("Monte Carlo")
colmc1, colmc2, colmc3 = st.columns(3)
with colmc1: sims = st.slider("Simulations", 500, 10000, 3000, 500)
with colmc2: cv   = st.slider("Coef. of variation", 0.02, 0.5, 0.1, 0.01)
with colmc3: seed = st.number_input("Random seed", value=42, step=1)

def mc(params, sims, cv, seed):
    rng = np.random.default_rng(seed)
    def clip(v, lo=None, hi=None): 
        if lo is not None: v = np.maximum(v, lo)
        if hi is not None: v = np.minimum(v, hi)
        return v
    draws = []
    for _ in range(sims):
        p = params.copy()
        for k, lo, hi in specs:
            m = params[k]
            sd = max(abs(m)*cv, 1e-9)
            v = rng.normal(m, sd)
            if k in ("completion_rate","positive_outcome_rate","discount_rate","attribution","deadweight","overhead_rate"):
                v = float(clip(v, 0.0, 1.0 if k!="discount_rate" else 0.2))
            if k == "half_life_years": v = float(clip(v, 1.0, 40.0))
            if k == "duration_years":  v = int(round(clip(v, 1.0, 40.0)))
            if k in ("baseline_income","post_income","grant_amount","people_reached_total"):
                v = float(max(v, 0.0))
            p[k] = v
        draws.append(compute_roi(p)["bcr"])
    arr = np.array(draws)
    pct = np.percentile(arr, [5,50,95])
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

# --------- One-pager (HTML download) ---------
st.subheader("One-pager (HTML download)")
def render_onepager_html(params, res):
    html = f"""
<!doctype html><html><head><meta charset="utf-8"><title>ROI One-Pager</title>
<style>body{{font-family:Arial;margin:24px}}.kpi div{{border:1px solid #ddd;padding:10px;border-radius:8px;margin-right:10px;display:inline-block}}</style>
</head><body>
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
  <li>People reached: {params['people_reached_total']:,}</li>
  <li>Completion: {params['completion_rate']:.0%}; Positive outcome: {params['positive_outcome_rate']:.0%}</li>
  <li>Baseline: ${params['baseline_income']:,.0f}; Post: ${params['post_income']:,.0f}</li>
  <li>Half-life: {params['half_life_years']} yrs; Duration: {params['duration_years']} yrs</li>
  <li>Discount: {params['discount_rate']:.1%}; Attribution: {params['attribution']:.0%}; Deadweight: {params['deadweight']:.0%}</li>
  <li>Grant: ${params['grant_amount']:,.0f}; Overhead: {params['overhead_rate']:.1%}</li>
</ul>
</body></html>
"""
    return html

if st.button("Download one-pager (HTML)"):
    html = render_onepager_html(params, res)
    b64 = base64.b64encode(html.encode("utf-8")).decode()
    st.markdown(f'<a download="onepager.html" href="data:text/html;base64,{b64}">Click to download</a>', unsafe_allow_html=True)
