import streamlit as st
import pandas as pd
import altair as alt

# Page config for wide layout
st.set_page_config(page_title="PV Benefit–Cost Ratio", layout="wide")

def calculate_bcr_pv(pre, post, imp, cost, yrs, r):
    ann        = (post - pre) * imp
    factor     = (1 - (1 + r)**-yrs) / r if r > 0 else yrs
    pv_benefit = ann * factor
    ratio      = pv_benefit / cost
    return ratio, pv_benefit

# Sidebar for inputs and help
t = st.sidebar
st.sidebar.header("Inputs & Assumptions")
pre = t.number_input("Pre-income per person (USD)", value=0, format="%d")
post = t.number_input("Post-income per person (USD)", value=0, format="%d")
imp = t.number_input("People impacted", value=0, format="%d")
cost = t.number_input("Total program cost (USD)", value=0, format="%d")
yrs = t.number_input("Years of income increase", value=1, format="%d")
rate_pct = t.number_input("Discount rate (%)", value=3.0, format="%.2f")
run_calc = t.button("Calculate")

with st.sidebar.expander("How PV is calculated", expanded=False):
    st.write(
        "1. Annual gain = (post - pre) × people impacted  \
"        "2. Discount factor = (1 - (1+r)^-yrs)/r  \
"        "3. PV benefit = annual gain × discount factor  \
"        "4. BCR = PV benefit / total cost"
    )

st.title("PV Benefit–Cost Ratio Calculator")

if run_calc:
    rate = rate_pct / 100.0
    bcr, total_pv = calculate_bcr_pv(pre, post, imp, cost, yrs, rate)

    # Metrics cards
    m1, m2 = st.columns(2)
    m1.metric("Total PV Income Increase", f"${total_pv:,.0f}")
    m2.metric("Benefit–Cost Ratio", f"{bcr:.2f}")

    # Build cumulative PV series
    cf_gain = (post - pre) * imp
    cf_pre = pre * imp
    years = list(range(0, int(yrs) + 1))
    pv_gain = [0.0] + [cf_gain / ((1 + rate) ** t) for t in years[1:]]
    pv_pre = [0.0] + [cf_pre / ((1 + rate) ** t) for t in years[1:]]
    cum_gain = pd.Series(pv_gain).cumsum()
    cum_pre = pd.Series(pv_pre).cumsum()

    df = pd.DataFrame({
        "Cumulative Net PV Gain": cum_gain,
        "Cumulative Pre-counterfactual PV": cum_pre
    }, index=years)
    df.index.name = "Year"

    # Altair line chart with tooltips
    st.write("### Cumulative PV Benefit Over Time")
    df_reset = df.reset_index().melt(id_vars="Year", value_vars=[
        "Cumulative Net PV Gain", "Cumulative Pre-counterfactual PV"
    ], var_name="Type", value_name="Value")
    chart = (
        alt.Chart(df_reset)
        .mark_line(point=True)
        .encode(
            x="Year:O",
            y=alt.Y("Value:Q", title="PV ($)"),
            color="Type:N",
            tooltip=["Year", "Type", alt.Tooltip("Value", format="$,.2f")]
        )
        .properties(width="container", height=400)
    )
    st.altair_chart(chart, use_container_width=True)
