# =========================================
# FINAL USER-LEVEL DATASET (ONE ROW PER USER)
# =========================================

import pandas as pd
import numpy as np

# Reload df safely if needed
if "df" not in globals():
    raw_df = pd.read_csv("zerve_hackathon_for_reviewc8fa7c7.csv")
    df = raw_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

# Group to one row per user
user_level_final = df.groupby("person_id").agg(
    first_event_timestamp=("timestamp", "min"),
    last_event_timestamp=("timestamp", "max"),
    total_events=("event", "count"),
    total_credits_used=("prop_credits_used", "sum")
).reset_index()

# Days active
user_level_final["days_active"] = (
    user_level_final["last_event_timestamp"] -
    user_level_final["first_event_timestamp"]
).dt.days

# Retained 30 days definition
user_level_final["retained_30d"] = np.where(
    user_level_final["days_active"] >= 30,
    1,
    0
)

# Keep success column (based on credit usage)
user_level_final["success"] = np.where(
    user_level_final["total_credits_used"] > 0,
    1,
    0
)

print("Final user-level dataset created.")
print("Shape:", user_level_final.shape)
print(user_level_final.head())