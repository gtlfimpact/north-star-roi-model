import os
import re
import streamlit as st
import pandas as pd
import altair as alt
from openai import OpenAI, OpenAIError
import PyPDF2
import docx
import json

# ─── Helpers ────────────────────────────────────────────────────────────────
def parse_currency(val):
    """Strip non-numeric (except dot) characters and convert to float."""
    if isinstance(val, str):
        nums = re.sub(r"[^\d.]", "", val)
        return float(nums) if nums else 0.0
    return float(val)

# ─── OpenAI Client Setup ────────────────────────────────────────────────────
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY not set in environment")
    st.stop()
client     = OpenAI(api_key=api_key)
model_name = "gpt-4.1"

# ─── Branding & Layout ──────────────────────────────────────────────────────
BRAND = {
    "red":    "#E24329",
    "orange": "#FC6D26",
    "yellow": "#FCA326",
    "black":  "#000000",
    "grey":   "#EDEDED",
    "white":  "#FFFFFF"
}

st.set_page_config(page_title="PV Benefit–Cost Ratio", layout="wide")
st.markdown(f"""
<style>
  .reportview-container, .main {{ background-color: {BRAND['white']}; }}
  .sidebar .sidebar-content {{ background-color: {BRAND['grey']}; }}
  .stMetric > div > div > div:nth-child(2) {{ color: {BRAND['red']}; }}
</style>
""", unsafe_allow_html=True)

# ─── Sidebar Inputs ─────────────────────────────────────────────────────────
sidebar = st.sidebar
sidebar.header("Inputs & Assumptions")
pre      = sidebar.number_input("Pre-income per person (USD)", value=0, format="%d")
post     = sidebar.number_input("Post-income per person (USD)", value=0, format="%d")
imp      = sidebar.number_input("People impacted", value=0, format="%d")
cost     = sidebar.number_input("Total program cost (USD)", value=0, format="%d")
yrs      = sidebar.number_input("Years of income increase", value=1, format="%d")
rate_pct = sidebar.number_input("Discount rate (%)", value=3.0, format="%.2f")
run_calc = sidebar.button("Calculate")

with sidebar.expander("How PV is calculated", expanded=False):
    st.write(
        "1. Annual gain = (post - pre) × people impacted  \n"
        "2. Discount factor = (1 - (1+r)^-yrs)/r  \n"
        "3. PV benefit = annual gain × discount factor  \n"
        "4. BCR = PV benefit / total cost"
    )

st.title("PV Benefit–Cost Ratio Calculator")

# ─── PV Calculation & Chart ──────────────────────────────────────────────────
if run_calc:
    rate     = rate_pct / 100.0
    ann      = (post - pre) * imp
    factor   = (1 - (1 + rate) ** -yrs) / rate if rate > 0 else yrs
    total_pv = ann * factor
    bcr      = total_pv / cost if cost > 0 else 0

    c1, c2 = st.columns(2)
    c1.metric("Total PV Income Increase", f"${total_pv:,.0f}")
    c2.metric("Benefit–Cost Ratio",       f"{bcr:.2f}")

    years   = list(range(0, int(yrs) + 1))
    pv_gain = [0.0] + [ann      / ((1 + rate) ** t) for t in years[1:]]
    pv_pre  = [0.0] + [(pre*imp)  / ((1 + rate) ** t) for t in years[1:]]
    pv_post = [0.0] + [(post*imp) / ((1 + rate) ** t) for t in years[1:]]

    df = pd.DataFrame({
        "Cumulative Net PV Income Gains":           pd.Series(pv_gain).cumsum(),
        "Cumulative Counterfactual PV Income Gains": pd.Series(pv_pre).cumsum(),
        "Cumulative Post PV Income Gains":           pd.Series(pv_post).cumsum(),
    }, index=years)
    df.index.name = "Year"

    st.write("### Cumulative PV Benefit Over Time")
    df_melt = df.reset_index().melt(
        id_vars="Year",
        var_name="Series",
        value_name="Value"
    )
    color_scale = alt.Scale(
        domain=list(df.columns),
        range=[BRAND['red'], BRAND['black'], BRAND['orange']]
    )
    chart = (
        alt.Chart(df_melt)
           .mark_line(point=True)
           .encode(
               x=alt.X("Year:O"),
               y=alt.Y("Value:Q", title="PV ($)"),
               color=alt.Color("Series:N", scale=color_scale),
               tooltip=["Year", "Series", alt.Tooltip("Value", format="$,.2f")],
           )
           .properties(width="container", height=400)
    )
    st.altair_chart(chart, use_container_width=True)

# ─── Extraction Prompt ───────────────────────────────────────────────────────
st.write("---")
st.markdown("**Extraction prompt:**")
prompt_template = """```text
Extract the following fields from this grant application:
- Amount requested
- Total project cost
- Estimated baseline or counterfactual annual income per person
- Post income per person
- Net change in annual income per person
- Number of people positively impacted

Return JSON with keys:
amount_requested, total_project_cost,
baseline_income_per_person, post_income_per_person,
net_income_change, people_impacted
```"""
st.code(prompt_template, language="text")

# ─── File Upload & GPT Extraction ──────────────────────────────────────────
uploaded = st.file_uploader("Upload PDF or DOCX to extract key fields", type=["pdf","docx"])
if uploaded:
    st.write(f"Uploaded file: {uploaded.name}")
    if st.button("Extract Fields"):
        with st.spinner("Extracting fields..."):
            # Read raw text
            if uploaded.type == "application/pdf":
                reader = PyPDF2.PdfReader(uploaded)
                text = "\n".join(page.extract_text() or "" for page in reader.pages)
            else:
                doc  = docx.Document(uploaded)
                text = "\n".join(p.text for p in doc.paragraphs)

            # Build and send to GPT
            full_prompt = prompt_template.strip("```text\n```") + "\n\n" + text
            try:
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "Extract fields from grant."},
                        {"role": "user",   "content": full_prompt}
                    ],
                    temperature=0
                )
                raw_fields = json.loads(resp.choices[0].message.content)
                # Convert to proper types
                fields = {
                    "amount_requested":           parse_currency(raw_fields.get("amount_requested", 0)),
                    "total_project_cost":         parse_currency(raw_fields.get("total_project_cost", 0)),
                    "baseline_income_per_person": parse_currency(raw_fields.get("baseline_income_per_person", 0)),
                    "post_income_per_person":     parse_currency(raw_fields.get("post_income_per_person", 0)),
                    "net_income_change":          parse_currency(raw_fields.get("net_income_change", 0)),
                    "people_impacted":            int(raw_fields.get("people_impacted", 0))
                }
            except (OpenAIError, json.JSONDecodeError) as e:
                st.error(f"Extraction failed: {e}")
                fields = {}

        if fields:
            st.success("Extraction complete")
            st.json(fields)

            # Build natural summary
            a   = fields["amount_requested"]
            tc  = fields["total_project_cost"]
            bi  = fields["baseline_income_per_person"]
            pi  = fields["post_income_per_person"]
            nc  = fields["net_income_change"]
            ppl = fields["people_impacted"]
            pct = (a / tc * 100) if tc else 0
            rec = int(ppl * a / tc)      if tc else 0

            summary_md = (
                f"The grant request is for **${a:,.0f}**, out of a total project cost of **${tc:,.0f}**. "
                f"Each participant’s baseline annual income is **${bi:,.0f}**, rising to **${pi:,.0f}**. "
                f"This is a net annual increase of **${nc:,.0f}** per person, impacting **{ppl:,}** individuals overall."
            )
            if a < tc:
                summary_md += (
                    f" Due to funding covering **{pct:.1f}%** of total cost, "
                    f"we recommend adjusting impacted to **{rec:,}** individuals."
                )

            st.markdown("### Summary")
            st.markdown(summary_md)
