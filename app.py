import streamlit as st

def calculate_bcr_pv(pre, post, imp, cost, yrs, r):
    ann    = (post - pre) * imp
    factor = (1 - (1 + r)**-yrs) / r if r > 0 else yrs
    return ann * factor / cost

st.title("PV Benefit–Cost Ratio Calculator")

pre   = st.number_input("Pre-income per person",    value=0.0)
post  = st.number_input("Post-income per person",   value=0.0)
imp   = st.number_input("People impacted",          value=0, step=1)
cost  = st.number_input("Total program cost",       value=0.0)
yrs   = st.number_input("Years of income increase", value=1, step=1)
rate  = st.number_input("Discount rate (decimal)",  value=0.0, format="%.4f")

if st.button("Calculate"):
    if cost <= 0:
        st.error("Total cost must be > 0")
    else:
        result = calculate_bcr_pv(pre, post, imp, cost, yrs, rate)
        st.write(f"**PV Benefit–Cost Ratio:** {result:.2f}")
