# ── Step 5.1: Retention Distribution ─────────────────────────────────────────
# Aggregate to user level and compute retention stats from user_level_final

# Core counts
retention_total_users   = len(user_level_final)
retention_retained      = int(user_level_final["success"].sum())
retention_churned       = retention_total_users - retention_retained
retention_rate_pct      = round((retention_retained / retention_total_users) * 100, 2)

# Build summary table
import pandas as pd

retention_summary = pd.DataFrame([
    {"metric": "total_users",    "value": retention_total_users},
    {"metric": "retained_users", "value": retention_retained},
    {"metric": "churned_users",  "value": retention_churned},
    {"metric": "retention_rate", "value": f"{retention_rate_pct}%"},
])

# Display
print("=" * 40)
print("  STEP 5.1 — RETENTION DISTRIBUTION")
print("=" * 40)
print(retention_summary.to_string(index=False))
print("=" * 40)