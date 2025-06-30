import os
from openai import OpenAI
import streamlit as st
import pandas as pd
import altair as alt
import PyPDF2
import docx
import json

# Load your key into a variable
api_key = os.getenv("OPENAI_API_KEY")      
client  = OpenAI(api_key=api_key)

# …rest of your code…

if uploaded:
    # build prompt…
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role":"system","content":"…"},
            {"role":"user","content": prompt}
        ],
        temperature=0,
    )
    fields = response.choices[0].message.content

# Brand colors
BRAND = {
    "red": "#E24329",
    "orange": "#FC6D26",
    "yellow": "#FCA326",
    "black": "#000000",
    "grey": "#EDEDED",
    "white": "#FFFFFF"
}

# Page config
st.set_page_config(page_title="PV Benefit–Cost Ratio", layout="wide")

# Inject brand CSS
st.markdown(f"""
<style>
  .reportview-container, .main {{ background-color: {BRAND['white']}; }}
  .sidebar .sidebar-content {{ background-color: {BRAND['grey']}; }}
  .stMetric > div > div > div:nth-child(2) {{ color: {BRAND['red']}; }}
</style>
""", unsafe_allow_html=True)

# Sidebar inputs & help
t = st.sidebar
st.sidebar.header("Inputs & Assumptions")
pre      = t.number_input("Pre-income per person (USD)", value=0, format="%d")
post     = t.number_input("Post-income per person (USD)", value=0, format="%d")
imp      = t.number_input("People impacted", value=0, format="%d")
cost     = t.number_input("Total program cost (USD)", value=0, format="%d")
yrs      = t.number_input("Years of income increase", value=1, format="%d")
rate_pct = t.number_input("Discount rate (%)", value=3.0, format="%.2f")
run_calc = t.button("Calculate")

with st.sidebar.expander("How PV is calculated", expanded=False):
    st.write(
        "1. Annual gain = (post - pre) × people impacted  \n"
        "2. Discount factor = (1 - (1+r)^-yrs)/r  \n"
        "3. PV benefit = annual gain × discount factor  \n"
        "4. BCR = PV benefit / total cost"
    )

st.title("PV Benefit–Cost Ratio Calculator")

if run_calc:
    rate = rate_pct / 100.0
    ann = (post - pre) * imp
    factor = (1 - (1 + rate)**-yrs) / rate if rate > 0 else yrs
    total_pv = ann * factor
    bcr = total_pv / cost if cost > 0 else 0

    # KPI cards
    col1, col2 = st.columns(2)
    col1.metric("Total PV Income Increase", f"${total_pv:,.0f}")
    col2.metric("Benefit–Cost Ratio", f"{bcr:.2f}")

    # Build cumulative PV series
    cf_gain = ann
    cf_pre = pre * imp
    cf_post = post * imp
    years = list(range(0, int(yrs) + 1))
    pv_gain = [0.0] + [cf_gain / ((1 + rate) ** t) for t in years[1:]]
    pv_pre = [0.0] + [cf_pre   / ((1 + rate) ** t) for t in years[1:]]
    pv_post = [0.0] + [cf_post / ((1 + rate) ** t) for t in years[1:]]

    cum_gain = pd.Series(pv_gain).cumsum()
    cum_pre = pd.Series(pv_pre).cumsum()
    cum_post = pd.Series(pv_post).cumsum()

    df = pd.DataFrame({
        "Cumulative Net PV Income Gains": cum_gain,
        "Cumulative Counterfactual PV Income Gains": cum_pre,
        "Cumulative Post PV Income Gains": cum_post
    }, index=years)
    df.index.name = "Year"

    st.write("### Cumulative PV Benefit Over Time")
    df_reset = df.reset_index().melt(id_vars="Year", var_name="Type", value_name="Value")
    color_scale = alt.Scale(
        domain=[
            "Cumulative Net PV Income Gains",
            "Cumulative Counterfactual PV Income Gains",
            "Cumulative Post PV Income Gains"
        ],
        range=[BRAND['red'], BRAND['black'], BRAND['orange']]
    )
    chart = (
        alt.Chart(df_reset)
            .mark_line(point=True)
            .encode(
                x=alt.X("Year:O"),
                y=alt.Y("Value:Q", title="PV ($)"),
                color=alt.Color("Type:N", scale=color_scale, legend=alt.Legend(title="Series")),
                tooltip=["Year", "Type", alt.Tooltip("Value", format="$,.2f")]
            )
            .properties(width="container", height=400)
    )
    st.altair_chart(chart, use_container_width=True)

# File upload & field extraction
st.write("---")
st.write("## Upload Grant Application")
uploaded = st.file_uploader(
    "Upload a PDF or Word grant application to extract key fields", 
    type=["pdf", "docx"]
)
if uploaded:
    # Read text
    text = ""
    if uploaded.type == "application/pdf":
        reader = PyPDF2.PdfReader(uploaded)
        for p in reader.pages:
            text += (p.extract_text() or "") + "\n"
    else:
        doc = docx.Document(uploaded)
        for p in doc.paragraphs:
            text += p.text + "\n"

    # Build prompt for GPT
    prompt = (
        "Extract the following fields from this grant application text:\n"
        "- Amount requested\n"
        "- Total project cost\n"
        "- Estimated baseline or counterfactual annual income per person\n"
        "- Post income per person\n"
        "- Net change in annual income per person\n"
        "- Number of people positively impacted\n"
        "Provide output as JSON with keys: amount_requested, total_project_cost, baseline_income_per_person, post_income_per_person, net_income_change, people_impacted.\n"
        f"Application text follows:\n{text}"
    )
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You extract structured fields from a grant application."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    content = response.choices[0].message.content
    try:
        fields = json.loads(content)
    except json.JSONDecodeError:
        fields = {"error": "Unable to parse JSON", "raw": content}

    st.write("### Extracted Fields")
    st.json(fields)
