import streamlit as st

# 1) install
!pip install -q ipywidgets

# 2) bring in the calculator
def calculate_bcr_pv(pre, post, imp, cost, yrs, r):
    ann = (post - pre) * imp
    factor = (1 - (1 + r)**-yrs)/r if r>0 else yrs
    return ann * factor / cost

# 3) build the form
from ipywidgets import FloatText, IntText, Button, VBox, HBox, Output

pre_w   = FloatText(value=10000, description='Pre-income')
post_w  = FloatText(value=15000, description='Post-income')
imp_w   = IntText(value=200,    description='Impacted')
cost_w  = FloatText(value=300000,description='Total cost')
yrs_w   = IntText(value=5,      description='Years')
rate_w  = FloatText(value=0.03, description='Discount rate')
out     = Output()
btn     = Button(description='Calculate BCR')

def on_click(_):
    with out:
        out.clear_output()
        c = cost_w.value
        if c<=0:
            print("Total cost must be > 0")
            return
        r = calculate_bcr_pv(
            pre_w.value,post_w.value,
            imp_w.value,c,
            yrs_w.value,rate_w.value
        )
        print(f"PV BCR: {r:.2f}")

btn.on_click(on_click)
VBox([HBox([pre_w,post_w]), HBox([imp_w,cost_w]), HBox([yrs_w,rate_w]), btn, out])
