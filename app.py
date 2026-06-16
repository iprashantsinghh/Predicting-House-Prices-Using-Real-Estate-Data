import streamlit as st
import pandas as pd

# PAGE CONFIG
st.set_page_config(
    page_title="Property Matching System",
    page_icon="🏠",
    layout="wide"
)

# TITLE
st.title("🏠 AI Property Matching System")

st.write("Property Recommendation Engine")

# LOAD CSV
results_df = pd.read_csv("match_results.csv")

# USER LIST
user_ids = sorted(results_df["User ID"].unique())

# SELECT USER
selected_user = st.selectbox(
    "Select User ID",
    user_ids
)

# FILTER DATA
filtered = results_df[
    results_df["User ID"] == selected_user
]

filtered = filtered.sort_values(
    by="Final Match Score",
    ascending=False
)

# BEST MATCH
best = filtered.iloc[0]

st.success(
    f"Best Match → Property {best['Property ID']} "
    f"| Score: {best['Final Match Score']}"
)

# TABLE
st.subheader("Recommended Properties")

st.dataframe(filtered)

# BAR CHART
st.subheader("Match Score Visualization")

chart_data = filtered.set_index("Property ID")

st.bar_chart(chart_data["Final Match Score"])