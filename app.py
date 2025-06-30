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
    st.error("OPENAI_API_KEY not set")
    st.stop()
client = OpenAI(api_key=api_key)
model_name = "gpt-4.1"

# Branding omitted for brevity...

# Sidebar inputs (unchanged)…
sidebar = st.sidebar
# … your number_inputs …

run_calc = sidebar.button("Calculate")
if run_calc:
    # … calculation + chart …

st.write("---")
st.write("## Upload Grant Application")
uploaded = st.file_uploader("Upload PDF or DOCX…", type=["pdf","docx"])

# **Always show the prompt template** so it stays on-screen
prompt_template = """```text
Extract the following fields from this grant application:
- Amount requested
- Total project cost
- Estimated baseline or counterfactual annual income per person
- Post income per person
- Net change in annual income per person
- Number of people positively impacted

Return JSON with keys:
amount_requested, total_project_cost,
baseline_income_per_person, post_income_per_person,
net_income_change, people_impacted.
```"""
st.markdown("**Extraction prompt:**")
st.code(prompt_template, language="text")

if uploaded:
    st.write(f"Uploaded file: {uploaded.name}")
    if st.button("Extract Fields"):
        with st.spinner("Extracting…"):
            # read text …
            # build prompt (insert the actual document text after the template) …
            full_prompt = prompt_template.strip("```text\n```") + "\n\n" + document_text

            try:
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[
                      {"role":"system","content":"Extract fields from grant."},
                      {"role":"user","content": full_prompt}
                    ],
                    temperature=0
                )
                content = resp.choices[0].message.content
                fields = json.loads(content)
            except (OpenAIError, json.JSONDecodeError) as e:
                st.error(f"Extraction failed: {e}")
                fields = {}

        if fields:
            st.success("Extraction complete")
            st.json(fields)

            # Natural‐language summary with **bold** numbers
            amt = fields["amount_requested"]
            tc  = fields["total_project_cost"]
            bi  = fields["baseline_income_per_person"]
            pi  = fields["post_income_per_person"]
            nc  = fields["net_income_change"]
            ppl = fields["people_impacted"]
            pct = amt/tc*100
            rec = int(ppl * amt/tc)

            summary_md = (
                f"The grant request is for **${amt:,.0f}**, out of a total project cost of **${tc:,.0f}**. "
                f"Each participant’s baseline annual income is **${bi:,.0f}**, rising to **${pi:,.0f}**. "
                f"This is a net annual increase of **${nc:,.0f}** per person, impacting **{ppl:,}** individuals. "
            )
            if amt < tc:
                summary_md += (
                  f"Due to funding covering **{pct:.1f}%** of total cost, "
                  f"we recommend adjusting impacted to **{rec:,}** individuals."
                )

            st.markdown("### Summary")
            st.markdown(summary_md)
