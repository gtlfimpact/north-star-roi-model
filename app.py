import os
import streamlit as st
import pandas as pd
import altair as alt
from openai import OpenAI, OpenAIError
import PyPDF2
import docx
import json

# Load OpenAI key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY not set in environment")
    st.stop()
client = OpenAI(api_key=api_key)
model_name = "gpt-4.1"

# Brand colors (omitted for brevity)...

st.set_page_config(page_title="PV Benefit–Cost Ratio", layout="wide")

# Sidebar inputs
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

# Calculation
if run_calc:
    rate     = rate_pct / 100.0
    ann      = (post - pre) * imp
    factor   = (1 - (1 + rate) ** -yrs) / rate if rate > 0 else yrs
    total_pv = ann * factor
    bcr      = total_pv / cost if cost > 0 else 0

    col1, col2 = st.columns(2)
    col1.metric("Total PV Income Increase", f"${total_pv:,.0f}")
    col2.metric("Benefit–Cost Ratio", f"{bcr:.2f}")

    years   = list(range(0, int(yrs) + 1))
    pv_gain = [0.0] + [ann / ((1 + rate) ** t) for t in years[1:]]
    pv_pre  = [0.0] + [(pre * imp) / ((1 + rate) ** t) for t in years[1:]]
    pv_post = [0.0] + [(post * imp) / ((1 + rate) ** t) for t in years[1:]]

    df = pd.DataFrame({
        "Cumulative Net PV Income Gains":           pd.Series(pv_gain).cumsum(),
        "Cumulative Counterfactual PV Income Gains": pd.Series(pv_pre).cumsum(),
        "Cumulative Post PV Income Gains":          pd.Series(pv_post).cumsum(),
    }, index=years)
    df.index.name = "Year"

    st.write("### Cumulative PV Benefit Over Time")
    df_melt = df.reset_index().melt(id_vars="Year", var_name
