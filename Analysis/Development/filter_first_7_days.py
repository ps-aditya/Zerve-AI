# ============================================
# STEP 2.1 — FILTER FIRST 7 DAYS PER USER
# ============================================

import pandas as pd
import numpy as np

# Safety: Reload dataset if needed
if "df" not in globals():
    raw_df = pd.read_csv("zerve_hackathon_for_reviewc8fa7c7.csv")
    df = raw_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

# Recreate user-level dataset if missing
if "user_level_dataset_step1" not in globals():
    user_level_dataset_step1 = df.groupby("person_id").agg(
        first_event_timestamp=("timestamp", "min"),
        last_event_timestamp=("timestamp", "max")
    ).reset_index()

    user_level_dataset_step1["lifetime_days"] = (
        user_level_dataset_step1["last_event_timestamp"] -
        user_level_dataset_step1["first_event_timestamp"]
    ).dt.days

    user_level_dataset_step1["success"] = np.where(
        user_level_dataset_step1["lifetime_days"] >= 30,
        1, 0
    )

# Merge first_event_timestamp + success
df = df.merge(
    user_level_dataset_step1[
        ["person_id", "first_event_timestamp", "success"]
    ],
    on="person_id",
    how="left"
)

# Calculate days since first event
df["days_since_first_event"] = (
    df["timestamp"] - df["first_event_timestamp"]
).dt.days

# Keep only first 7 days
first_7_days_df = df[
    df["days_since_first_event"] <= 7
].copy()

print("First 7 days dataset created.")
print("Shape:", first_7_days_df.shape)
print(first_7_days_df.head())