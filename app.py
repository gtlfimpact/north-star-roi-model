import streamlit as st
import pandas as pd

def calculate_bcr_pv(pre, post, imp, cost, yrs, r):
    ann        = (post - pre) * imp
    factor     = (1 - (1 + r)**-yrs) / r if r > 0 else yrs
    pv_benefit = ann * factor
    ratio      = pv_benefit / cost
    return ratio, pv_benefit

st.title("PV Benefit–Cost Ratio Calculator")

pre   = st.number_input("Pre-income per person",    value=0.0)
post  = st.number_input("Post-income per person",   value=0.0)
imp   = st.number_input("People impacted",          value=0,   step=1)
cost  = st.number_input("Total program cost",       value=0.0)
yrs   = st.number_input("Years of income increase", value=1,   step=1)
rate  = st.number_input("Discount rate (decimal)",  value=0.03, format="%.4f")

if st.button("Calculate"):
    if cost <= 0:
        st.error("Total cost must be > 0")
    else:
        bcr, total_pv = calculate_bcr_pv(pre, post, imp, cost, yrs, rate)
        st.write(f"**Total PV Income Increase:** ${total_pv:,.2f}")
        st.write(f"**PV Benefit–Cost Ratio:** {bcr:.2f}")

        # build year-by-year PV series
        years = list(range(1, int(yrs) + 1))
        cf_gain = (post - pre) * imp
        pv_gain = [cf_gain / ((1 + rate) ** t) for t in years]
        cf_pre   = pre * imp
        pv_pre   = [cf_pre / ((1 + rate) ** t) for t in years]

        df = pd.DataFrame({
            "Net PV Gain": pv_gain,
            "Pre-counterfactual PV": pv_pre
        }, index=years)
        df.index.name = "Year"

        st.write("### PV Benefit Over Time")
        st.line_chart(df)
