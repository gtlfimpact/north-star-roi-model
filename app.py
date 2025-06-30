import os
import streamlit as st
import pandas as pd
import altair as alt
from openai import OpenAI, OpenAIError
import PyPDF2
import docx
import json

# Load OpenAI key from environment variable
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY not found in environment")
    st.stop()
client = OpenAI(api_key=api_key)
model_name = "gpt-4.1"

# Brand colors
BRAND = {
    "red": "#E24329",
    "orange": "#FC6D26",
    "yellow": "#FCA326",
    "black": "#000000",
    "grey": "#EDEDED",
    "white": "#FFFFFF"
}

# Page configuration
st.set_page_config(page_title="PV Benefit–Cost Ratio", layout="wide")

# Inject custom CSS
st.markdown(f"""
<style>
  .reportview-container, .main {{ background-color: {BRAND['white']}; }}
  .sidebar .sidebar-content {{ background-color: {BRAND['grey']}; }}
  .stMetric > div > div > div:nth-child(2) {{ color: {BRAND['red']}; }}
</style>
""", unsafe_allow_html=True)

# Sidebar inputs
sidebar = st.sidebar
sidebar.header("Inputs & Assumptions")
pre = sidebar.number_input("Pre-income per person (USD)", value=0, format="%d")
post = sidebar.number_input("Post-income per person (USD)", value=0, format="%d")
imp = sidebar.number_input("People impacted", value=0, format="%d")
cost = sidebar.number_input("Total program cost (USD)", value=0, format="%d")
yrs = sidebar.number_input("Years of income increase", value=1, format="%d")
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

# Calculation and visualization function
def calculate_and_render():
    rate = rate_pct / 100.0
    ann = (post - pre) * imp
    factor = (1 - (1 + rate)**-yrs) / rate if rate > 0 else yrs
    total_pv = ann * factor
    bcr = total_pv / cost if cost > 0 else 0

    # Display KPIs
    col1, col2 = st.columns(2)
    col1.metric("Total PV Income Increase", f"${total_pv:,.0f}")
    col2.metric("Benefit–Cost Ratio", f"{bcr:.2f}")

    # Build cumulative PV series
    cf_gain = ann
    cf_pre = pre * imp
    cf_post = post * imp
    years = list(range(0, int(yrs) + 1))
    pv_gain = [0.0] + [cf_gain / ((1 + rate) ** t) for t in years[1:]]
    pv_pre = [0.0] + [cf_pre / ((1 + rate) ** t) for t in years[1:]]
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
    df_melt = df.reset_index().melt(id_vars="Year", var_name="Series", value_name="Value")
    color_scale = alt.Scale(
        domain=[
            "Cumulative Net PV Income Gains",
            "Cumulative Counterfactual PV Income Gains",
            "Cumulative Post PV Income Gains"
        ],
        range=[BRAND['red'], BRAND['black'], BRAND['orange']]
    )
    chart = (
        alt.Chart(df_melt)
            .mark_line(point=True)
            .encode(
                x=alt.X("Year:O"),
                y=alt.Y("Value:Q", title="PV ($)"),
                color=alt.Color("Series:N", scale=color_scale, legend=alt.Legend(title="Series")),
                tooltip=["Year", "Series", alt.Tooltip("Value", format="$,.2f")]
            )
            .properties(width="container", height=400)
    )
    st.altair_chart(chart, use_container_width=True)

if run_calc:
    calculate_and_render()

# File upload & extraction
st.write("---")
st.write("## Upload Grant Application")
uploaded = st.file_uploader("Upload PDF or DOCX to extract key fields", type=["pdf","docx"])
if uploaded:
    st.write(f"Uploaded file: {uploaded.name}")
    if st.button("Extract Fields"):
        with st.spinner("Extracting fields..."):
            # Read document text
            if uploaded.type == "application/pdf":
                reader = PyPDF2.PdfReader(uploaded)
                text = "\n".join(page.extract_text() or "" for page in reader.pages)
            else:
                doc = docx.Document(uploaded)
                text = "\n".join(p.text for p in doc.paragraphs)

            # Build prompt
            prompt = (
                "Extract the following fields from this grant application:\n"
                "- Amount requested\n"
                "- Total project cost\n"
                "- Estimated baseline or counterfactual annual income per person\n"
                "- Post income per person\n"
                "- Net change in annual income per person\n"
                "- Number of people positively impacted\n"
                "Return JSON with keys: amount_requested, total_project_cost, baseline_income_per_person, post_income_per_person, net_income_change, people_impacted.\n"
                f"Application text:\n{text}"
            )
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role":"system","content":"Extract fields from grant."},
                        {"role":"user","content":prompt}
                    ],
                    temperature=0
                )
                content = response.choices[0].message.content
                fields = json.loads(content)
            except OpenAIError as e:
                st.error(f"OpenAI API error: {e}")
                fields = {}
            except json.JSONDecodeError:
                st.error("Failed to parse JSON from response.")
                fields = {"raw_response": content}
        st.success("Extraction complete")
        st.write("### Extracted Fields")
        st.json(fields)
