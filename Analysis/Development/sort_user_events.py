# ============================================
# STEP 4.1 — SORT USER EVENTS
# ============================================

import pandas as pd

# Load dataset
raw_df = pd.read_csv("zerve_hackathon_for_reviewc8fa7c7.csv")

df = raw_df.copy()

# Convert timestamp
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

# Create user-level table
user_level = df.groupby("person_id").agg(
    first_event=("timestamp","min"),
    last_event=("timestamp","max")
).reset_index()

user_level["lifetime_days"] = (
    user_level["last_event"] - user_level["first_event"]
).dt.days

user_level["success"] = (user_level["lifetime_days"] >= 30).astype(int)

# Merge success label
df = df.merge(
    user_level[["person_id","first_event","success"]],
    on="person_id",
    how="left"
)

# Calculate days since first event
df["days_since_first_event"] = (
    df["timestamp"] - df["first_event"]
).dt.days

# Keep first 7 days
first7 = df[df["days_since_first_event"] <= 7].copy()

# Sort events chronologically
first7 = first7.sort_values(
    ["person_id","timestamp"]
)

print("First 7 day dataset ready")
print("Shape:", first7.shape)
print(first7[["person_id","event","timestamp"]].head())