# =========================================
# SAFE STEP 2.1 — CREATE + MERGE USER DATA
# =========================================

import pandas as pd
import numpy as np

# Reload dataset if df not found
if "df" not in globals():
    raw_df = pd.read_csv("zerve_hackathon_for_reviewc8fa7c7.csv")
    df = raw_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

# Create user_level_dataset_step1 if missing
if "user_level_dataset_step1" not in globals():

    user_level_dataset_step1 = df.groupby("person_id").agg(
        first_event_timestamp=("timestamp", "min"),
        last_event_timestamp=("timestamp", "max"),
        total_events=("event", "count"),
        total_credits_used=("prop_credits_used", "sum")
    ).reset_index()

    # Success logic
    user_level_dataset_step1["success"] = np.where(
        user_level_dataset_step1["total_credits_used"] > 0,
        1,
        0
    )

    print("User-level dataset created.")

# Merge
df = df.merge(
    user_level_dataset_step1[
        ["person_id", "first_event_timestamp", "success"]
    ],
    on="person_id",
    how="left"
)

print("Merge successful.")
print("Final shape:", df.shape)
print(df.head())