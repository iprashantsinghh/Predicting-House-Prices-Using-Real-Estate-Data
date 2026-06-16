import pandas as pd

file_path = "data/Case Study 2 Data (1).xlsx"

users = pd.read_excel(file_path, sheet_name="User Data")
properties = pd.read_excel(file_path, sheet_name="Property Data")



print("\nUSER DATA")
print(users.head())


#pre processing
print("\nPROPERTY DATA")
print(properties.head())
print("\nUSER COLUMNS")
print(users.columns)

print("\nPROPERTY COLUMNS")
print(properties.columns)
print("\nMISSING VALUES IN USER DATA")
print(users.isnull().sum())

print("\nMISSING VALUES IN PROPERTY DATA")
print(properties.isnull().sum())



def clean_price(value):

    value = str(value)

    value = value.replace("$", "")
    value = value.replace(",", "")
    value = value.replace("k", "000")

    return float(value)

users["Budget"] = users["Budget"].apply(clean_price)

properties["Price"] = properties["Price"].apply(clean_price)

users["Qualitative Description"] = users["Qualitative Description"].str.lower()

properties["Qualitative Description"] = properties["Qualitative Description"].str.lower()

print("\nCLEANED USER DATA")
print(users.head())

print("\nCLEANED PROPERTY DATA")
print(properties.head())
def text_match_score(user_text, property_text):

    # Convert into word sets
    user_words = set(user_text.split())

    property_words = set(property_text.split())

    # Common words
    common_words = user_words.intersection(property_words)

    # Score
    score = len(common_words) / len(user_words)

    return round(score * 100, 2)

score = text_match_score(
    users.iloc[0]["Qualitative Description"],
    properties.iloc[0]["Qualitative Description"]
)

print("\nTEXT MATCH SCORE")
print(score)

def feature_match_score(user, property_):

    score = 0

    # Budget Match
    if property_["Price"] <= user["Budget"]:
        score += 1

    # Bedroom Match
    if property_["Bedrooms"] == user["Bedrooms"]:
        score += 1

    # Bathroom Match
    if property_["Bathrooms"] == user["Bathrooms"]:
        score += 1

    # Final Feature Score
    final_score = (score / 3) * 100

    return round(final_score, 2)
feature_score = feature_match_score(
    users.iloc[0],
    properties.iloc[0]
)

print("\nFEATURE MATCH SCORE")
print(feature_score)

# FINAL MATCH SCORE
def final_match_score(text_score, feature_score):

    final_score = (text_score + feature_score) / 2

    return round(final_score, 2)
final_score = final_match_score(score, feature_score)

print("\nFINAL MATCH SCORE")
print(final_score)
results = []

for i, user in users.iterrows():

    for j, property_ in properties.iterrows():

        # Text Score
        text_score = text_match_score(
            user["Qualitative Description"],
            property_["Qualitative Description"]
        )

        # Feature Score
        feature_score = feature_match_score(
            user,
            property_
        )

        # Final Score
        final_score = final_match_score(
            text_score,
            feature_score
        )

        # Store Results
        results.append({
            "User ID": user["User ID"],
            "Property ID": property_["Property ID"],
            "Text Score": text_score,
            "Feature Score": feature_score,
            "Final Match Score": final_score
        })

        results_df = pd.DataFrame(results)

# Sort by highest score
results_df = results_df.sort_values(
    by="Final Match Score",
    ascending=False
)
print("\nTOP MATCHES")
print(results_df.head(10))

results_df.to_csv("match_results.csv", index=False)

print("\nResults saved successfully!")

import matplotlib.pyplot as plt

# Top 10 matches
top_10 = results_df.head(10)

# Labels
labels = []

for i in range(len(top_10)):
    labels.append(
        f"U{top_10.iloc[i]['User ID']} → P{top_10.iloc[i]['Property ID']}"
    )

# Colors
colors = [
    "red",
    "blue",
    "green",
    "orange",
    "purple",
    "cyan",
    "magenta",
    "gold",
    "pink",
    "brown"
]

# Figure
plt.figure(figsize=(12, 6))

# Horizontal bars
bars = plt.barh(
    labels,
    top_10["Final Match Score"],
    color=colors
)

# Labels
plt.xlabel("Match Score", fontsize=12)
plt.ylabel("Recommendations", fontsize=12)

plt.title(
    "Top Property Recommendations",
    fontsize=16,
    fontweight='bold'
)

# Score text
for i, score in enumerate(top_10["Final Match Score"]):
    plt.text(score + 1, i, f"{score}")

# Best score on top
plt.gca().invert_yaxis()

# Grid
plt.grid(axis='x', linestyle='--', alpha=0.5)

# Save chart
plt.savefig("top_property_recommendations.png")

# Show
plt.show()
import seaborn as sns

# Pivot table
heatmap_data = results_df.pivot(
    index="User ID",
    columns="Property ID",
    values="Final Match Score"
)

# Figure
plt.figure(figsize=(12, 8))

# Heatmap
sns.heatmap(
    heatmap_data,
    annot=True,
    cmap="YlGnBu",
    linewidths=0.5
)

# Title
plt.title(
    "User vs Property Match Score Heatmap",
    fontsize=16,
    fontweight='bold'
)

# Save
plt.savefig("match_heatmap.png")

# Show
plt.show()
import streamlit as st
import pandas as pd

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Property Matching System",
    layout="wide"
)

# =========================
# LOAD DATA
# =========================

results_df = pd.read_csv("match_results.csv")

# =========================
# TITLE
# =========================

st.title("🏠 Property Matching System")

st.markdown("### AI-Based Property Recommendation Engine")

# =========================
# USER IDs
# =========================

user_ids = results_df["User ID"].unique()

selected_user = st.selectbox(
    "Select User ID",
    user_ids
)

# =========================
# FILTER DATA
# =========================

filtered = results_df[
    results_df["User ID"] == selected_user
]

filtered = filtered.sort_values(
    by="Final Match Score",
    ascending=False
)

# =========================
# SHOW TABLE
# =========================

st.subheader("📋 Recommended Properties")

st.dataframe(
    filtered,
    use_container_width=True
)

# =========================
# BEST MATCH
# =========================

best_property = filtered.iloc[0]

st.success(
    f"🏆 Best Match → Property {best_property['Property ID']} "
    f"with score {best_property['Final Match Score']}"
)

# =========================
# BAR CHART
# =========================

st.subheader("📊 Match Score Chart")

chart_data = filtered.set_index("Property ID")

st.bar_chart(chart_data["Final Match Score"])