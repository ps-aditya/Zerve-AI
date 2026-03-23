# ================================
# STEP 1 OUTPUT — USER LEVEL DATA
# ================================
import pandas as pd
import numpy as np

if "df" not in globals():
    raw_df = pd.read_csv("zerve_hackathon_for_reviewc8fa7c7.csv")
    df = raw_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

user_level_dataset_step1 = df.groupby("person_id").agg(
    first_event_timestamp=("timestamp", "min"),
    last_event_timestamp=("timestamp", "max"),
    total_events=("event", "count"),
    total_credits_used=("prop_credits_used", "sum")
).reset_index()

# Define success logic (customize if needed)
user_level_dataset_step1["success"] = np.where(
    user_level_dataset_step1["total_credits_used"] > 0,
    1,
    0
)

print("User-level dataset created.")
print(user_level_dataset_step1.head())
print("Shape:", user_level_dataset_step1.shape)