import pandas as pd

# ── STEP 4.5 — RETENTION CORRELATED EVENTS ───────────────────────────────────
# Uses: first_7_days_df (events in first 7 days with person_id + event col)
#       retained_user_ids / churned_user_ids (sets from churn_sequence_analysis)
#       user_level_final (for total user counts per group)

_n_retained = len(retained_user_ids)
_n_churned  = len(churned_user_ids)
_n_total    = _n_retained + _n_churned

print(f"Retained users : {_n_retained:,}")
print(f"Churned users  : {_n_churned:,}")
print(f"Total users    : {_n_total:,}\n")

# ── Step 1 & 2: Count how many USERS triggered each event (not raw occurrences)
# Working at user level avoids power-user inflation
_events_df = first_7_days_df[['person_id', 'event']].copy()

# Tag each row with retention label
_events_df['is_retained'] = _events_df['person_id'].isin(retained_user_ids).astype(int)

# Deduplicate: one row per (user, event) — we care about reach, not volume
_user_event = _events_df.drop_duplicates(subset=['person_id', 'event'])

# ── Step 3: Aggregate per event ───────────────────────────────────────────────
_agg = (
    _user_event
    .groupby('event')
    .agg(
        retained_frequency=('is_retained', 'sum'),
        total_frequency=('person_id', 'count')
    )
    .reset_index()
)

# ── Step 4: Compute retention_ratio = retained_frequency / total_frequency ────
_agg['retention_ratio'] = _agg['retained_frequency'] / _agg['total_frequency']

# Also compute normalised lift vs baseline retention rate (optional insight)
_baseline_retention = _n_retained / _n_total
_agg['retention_lift'] = _agg['retention_ratio'] / _baseline_retention

# ── Step 5: Rank by retention_ratio descending ───────────────────────────────
event_retention_corr = (
    _agg
    .sort_values('retention_ratio', ascending=False)
    .reset_index(drop=True)
)

# Rename for clean output
event_retention_corr.rename(columns={'event': 'event_name'}, inplace=True)

# ── Step 6: Print ranked table ────────────────────────────────────────────────
_top_n = 30

print("=" * 90)
print("STEP 4.5 — RETENTION CORRELATED EVENTS (ranked by retention_ratio)")
print(f"Baseline retention rate: {_baseline_retention:.4f} ({_baseline_retention*100:.2f}%)")
print("=" * 90)
print(f"\n{'#':<4} {'event_name':<45} {'ret_freq':>8} {'total_freq':>11} {'ret_ratio':>10} {'lift':>7}")
print("-" * 90)

for _rank, _row in enumerate(event_retention_corr.head(_top_n).itertuples(), start=1):
    print(
        f"{_rank:<4} {_row.event_name:<45} "
        f"{int(_row.retained_frequency):>8} "
        f"{int(_row.total_frequency):>11} "
        f"{_row.retention_ratio:>10.4f} "
        f"{_row.retention_lift:>7.2f}x"
    )

print("=" * 90)
print(f"\nTotal unique events analysed : {len(event_retention_corr):,}")
print(f"Events above baseline retention ({_baseline_retention*100:.2f}%) : "
      f"{(event_retention_corr['retention_ratio'] > _baseline_retention).sum():,}")

# ── Summary: top 10 high-signal events (min 5% user reach for statistical weight)
_min_reach = max(10, int(_n_total * 0.05))
_high_signal = event_retention_corr[
    event_retention_corr['total_frequency'] >= _min_reach
].head(10)

print(f"\n── Top 10 high-signal events (min {_min_reach} users reach) ──")
print(_high_signal[['event_name', 'retained_frequency', 'total_frequency', 'retention_ratio', 'retention_lift']].to_string(index=False))