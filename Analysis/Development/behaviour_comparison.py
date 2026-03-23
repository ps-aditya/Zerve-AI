# ============================================
# FIXED STEP 2.3 — NUMERIC ONLY COMPARISON
# ============================================

import pandas as pd
import numpy as np

# Select only numeric columns
numeric_cols = features_step2.select_dtypes(include=["number"]).columns

# Remove success from numeric list to avoid averaging it
numeric_cols = [col for col in numeric_cols if col != "success"]

# Compute grouped means safely
comparison = features_step2.groupby("success")[numeric_cols].mean()

print("Average early behaviour comparison:")
print(comparison)