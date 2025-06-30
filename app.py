import streamlit as st
import pandas as pd
import altair as alt

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
    /* Page background */
    .reportview-container, .main {{ background-color: {BRAND['white']}; }}
    /* Sidebar styling */
    .sidebar .sidebar-content {{ background-color: {BRAND['grey']}; }}
    /* Metric card value color */
    .stMetric > div > div > div:nth-child(2) {{ color: {BRAND['red']}; }}
</style>
""" , unsafe_allow_html=True)

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
"
        "2. Discount factor = (1 - (1+r)^-yrs)/r  \
"
        "3. PV benefit = annual gain × discount factor  \
"
        "4. BCR = PV benefit / total cost"
    )

st.title("PV Benefit–Cost Ratio Calculator")

if run_calc:
    rate = rate_pct / 100.0
    # Calculate BCR & PV
    ann = (post - pre) * imp
    factor = (1 - (1 + rate)**-yrs) / rate if rate > 0 else yrs
    total_pv = ann * factor
    bcr = total_pv / cost if cost > 0 else 0

    # Display metrics
    col1, col2 = st.columns(2)
    col1.metric("Total PV Income Increase", f"${total_pv:,.0f}")
    col2.metric("Benefit–Cost Ratio", f"{bcr:.2f}")

    # Build cumulative PV series
    cf_gain = ann
    cf_pre = pre * imp
    years = list(range(0, int(yrs) + 1))
    pv_gain = [0.0] + [cf_gain / ((1 + rate) ** t) for t in years[1:]]
    pv_pre = [0.0] + [cf_pre / ((1 + rate) ** t) for t in years[1:]]
    cum_gain = pd.Series(pv_gain).cumsum()
    cum_pre = pd.Series(pv_pre).cumsum()

    df = pd.DataFrame({
        "Cumulative Net PV Gain": cum_gain,
        "Pre-counterfactual PV": cum_pre
    }, index=years)
    df.index.name = "Year"

    # Altair line chart with brand colors
    st.write("### Cumulative PV Benefit Over Time")
    df_reset = df.reset_index().melt(id_vars="Year", value_vars=[
        "Cumulative Net PV Gain", "Pre-counterfactual PV"
    ], var_name="Type", value_name="Value")
    color_scale = alt.Scale(domain=["Cumulative Net PV Gain", "Pre-counterfactual PV"],
                            range=[BRAND['red'], BRAND['black']])
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
