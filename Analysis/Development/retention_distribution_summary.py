import pandas as pd

# --- Retention Distribution (Step 5.1) ---
# Aggregate at user level — user_level_final is already one row per user
_dist_df = user_level_final.copy()

# Core counts
_total_users     = len(_dist_df)
_retained_users  = int(_dist_df["success"].sum())
_churned_users   = _total_users - _retained_users
_retention_rate  = round(_retained_users / _total_users * 100, 2)

# Build the summary table
retention_distribution_summary = pd.DataFrame({
    "metric": [
        "total_users",
        "retained_users",
        "churned_users",
        "retention_rate (%)"
    ],
    "value": [
        _total_users,
        _retained_users,
        _churned_users,
        f"{_retention_rate}%"
    ]
})

print("=" * 40)
print("  STEP 5.1 — RETENTION DISTRIBUTION")
print("=" * 40)
print(retention_distribution_summary.to_string(index=False))
print("=" * 40)