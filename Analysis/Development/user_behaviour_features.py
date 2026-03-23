# ============================================
# STEP 2.2 — USER BEHAVIOUR FEATURES
# ============================================

import pandas as pd
import numpy as np

# Total events in first 7 days
total_events_7d = first_7_days_df.groupby("person_id").agg(
    total_events_7d=("event", "count")
).reset_index()

# Unique event types used
unique_event_types_7d = first_7_days_df.groupby("person_id").agg(
    unique_event_types_7d=("event", "nunique")
).reset_index()

# Total credits used in first 7 days
if "prop_credits_used" in first_7_days_df.columns:
    total_credits_7d = first_7_days_df.groupby("person_id").agg(
        total_credits_7d=("prop_credits_used", "sum")
    ).reset_index()
else:
    total_credits_7d = pd.DataFrame(
        columns=["person_id", "total_credits_7d"]
    )

# Merge all
features_step2 = total_events_7d.merge(
    unique_event_types_7d,
    on="person_id",
    how="left"
)

if not total_credits_7d.empty:
    features_step2 = features_step2.merge(
        total_credits_7d,
        on="person_id",
        how="left"
    )

# Merge success label
features_step2 = features_step2.merge(
    user_level_dataset_step1[["person_id", "success"]],
    on="person_id",
    how="left"
)

features_step2.fillna(0, inplace=True)

print("Behaviour feature dataset created.")
print("Shape:", features_step2.shape)
print(features_step2.head())