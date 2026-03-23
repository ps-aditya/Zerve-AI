# ================================
# BLOCK 2 — STANDARDIZE DATAFRAME
# ================================

import pandas as pd
import numpy as np

# If raw_df doesn't exist, reload it
try:
    raw_df
except NameError:
    raw_df = pd.read_csv("zerve_hackathon_for_reviewc8fa7c7.csv")

# Create working dataframe
df = raw_df.copy()

# Convert timestamp to datetime
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

print("df successfully created.")
print("Shape:", df.shape)
print("Timestamp dtype:", df["timestamp"].dtype)