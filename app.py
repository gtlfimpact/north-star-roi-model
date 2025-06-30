import streamlit as st
import pandas as pd

def calculate_bcr_pv(pre, post, imp, cost, yrs, r):
    ann        = (post - pre) * imp
    factor     = (1 - (1 + r)**-yrs) / r if r > 0 else yrs
    pv_benefit = ann * factor
    ratio      = pv_benefit / cost
    return ratio, pv_benefit

st.title("PV Benefit–Cost Ratio Calculator")

pre      = st.number_input(
    "Pre-income per person (USD)", 
    value=0, 
    format="%d"
)
post     = st.number_input(
    "Post-income per person (USD)", 
    value=0, 
    format="%d"
)
imp      = st.number_input(
    "People impacted", 
    value=0, 
    format="%d"
)
cost     = st.number_input(
    "Total program cost (USD)", 
    value=0, 
    format="%d"
)
yrs      = st.number_input(
    "Years of income increase", 
    value=1, 
    format="%d"
)
rate_pct = st.number_input(
    "Discount rate (%)", 
    value=3.0, 
    format="%.2f"
)
rate = rate_pct / 100.0

if st.button("Calculate"):
    if cost <= 0:
        st.error("Total cost must be > 0")
    else:
        bcr, total_pv = calculate_bcr_pv(pre, post, imp, cost, yrs, rate)
        st.write(f"**Total PV Income Increase:** ${total_pv:,.0f}")
        st.write(f"**PV Benefit–Cost Ratio:** {bcr:.2f}")

        # build year-by-year series from t=0 to t=yrs
        cf_gain = (post - pre) * imp
        cf_pre  = pre * imp
        years   = list(range(0, int(yrs) + 1))

        # PV at t=0 is zero
        pv_gain = [0.0] + [cf_gain / ((1 + rate) ** t) for t in years[1:]]
        pv_pre  = [0.0] + [cf_pre  / ((1 + rate) ** t) for t in years[1:]]

        cum_gain = pd.Series(pv_gain).cumsum()
        cum_pre  = pd.Series(pv_pre).cumsum()

        df = pd.DataFrame({
            "Cumulative Net PV Gain":           cum_gain,
            "Cumulative Pre-counterfactual PV": cum_pre
        }, index=years)
        df.index.name = "Year"

        st.write("### Cumulative PV Benefit Over Time")
        st.line_chart(df)

