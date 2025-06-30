import streamlit as st

def calculate_bcr_pv(pre, post, imp, cost, yrs, r):
    ann        = (post - pre) * imp
    factor     = (1 - (1 + r) ** -yrs) / r if r > 0 else yrs
    pv_benefit = ann * factor
    ratio      = pv_benefit / cost
    return ratio, pv_benefit

st.title("PV Benefit–Cost Ratio Calculator")

# --- Inputs ---
pre = st.slider(
    "Pre-income per person", 
    min_value=0, max_value=100_000, step=1_000, 
    format="$%d"
)

post = st.slider(
    "Post-income per person", 
    min_value=0, max_value=150_000, step=1_000, 
    format="$%d"
)

imp = st.slider(
    "People impacted", 
    min_value=0, max_value=10_000_000, step=100
)

cost = st.slider(
    "Total program cost", 
    min_value=0, max_value=5_000_000, step=10_000, 
    format="$%d"
)

yrs = st.number_input(
    "Years of income increase", 
    min_value=1, step=1, value=1
)

rate_pct = st.slider(
    "Discount rate (%)", 
    min_value=0, max_value=15, step=1, value=3
)
rate = rate_pct / 100.0

# --- Calculation ---
if st.button("Calculate"):
    if cost <= 0:
        st.error("Total cost must be > 0")
    else:
        bcr, total_pv = calculate_bcr_pv(pre, post, imp, cost, yrs, rate)
        st.write(f"**Total PV Income Increase:** ${total_pv:,.0f}")
        st.write(f"**PV Benefit–Cost Ratio:** {bcr:.2f}")
