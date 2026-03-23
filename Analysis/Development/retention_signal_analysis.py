import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np

# ── STEP 5.5 — RETENTION SIGNAL EVENTS ───────────────────────────────────────
# Uses: event_retention_corr from retention_event_analysis (Step 4.5)
#       columns: event_name | retained_frequency | total_frequency
#                retention_ratio | retention_lift
#
# Baseline retention already embedded in lift = retention_ratio / baseline_retention

_baseline_retention = len(retained_user_ids) / (len(retained_user_ids) + len(churned_user_ids))

# ── Step 1: Apply minimum reach filter (5% of total users) for statistical weight
_n_total = len(retained_user_ids) + len(churned_user_ids)
_min_reach = max(10, int(_n_total * 0.05))

_signal_pool = event_retention_corr[
    event_retention_corr['total_frequency'] >= _min_reach
].copy()

# ── Step 2: Rank by retention_lift (already computed in Step 4.5) and take Top 10
top10_retention_signals = (
    _signal_pool
    .sort_values('retention_lift', ascending=False)
    .head(10)
    .reset_index(drop=True)
)
top10_retention_signals.index = top10_retention_signals.index + 1
top10_retention_signals.index.name = 'rank'

# ── Step 3: Print ranked table ────────────────────────────────────────────────
print("=" * 95)
print("STEP 5.5 — TOP 10 RETENTION SIGNAL EVENTS  (ranked by retention_lift)")
print(f"Baseline retention rate : {_baseline_retention:.4f}  ({_baseline_retention * 100:.2f}%)")
print(f"Minimum reach threshold : {_min_reach:,} users  ({_min_reach / _n_total * 100:.1f}% of cohort)")
print("=" * 95)
print(f"\n{'#':<4} {'event_name':<45} {'ret_freq':>8} {'total_freq':>11} {'lift':>7}")
print("-" * 95)

for _rank, _row in top10_retention_signals.iterrows():
    print(
        f"{_rank:<4} {_row.event_name:<45} "
        f"{int(_row.retained_frequency):>8,} "
        f"{int(_row.total_frequency):>11,} "
        f"{_row.retention_lift:>6.2f}x"
    )

print("=" * 95)
print(f"\n  Events above baseline in full pool : {(event_retention_corr['retention_lift'] > 1.0).sum():,} / {len(event_retention_corr):,}")
print(f"  Max lift observed                  : {event_retention_corr['retention_lift'].max():.2f}x")
print(f"  Median lift (all events)           : {event_retention_corr['retention_lift'].median():.2f}x")

# ── Step 4: Horizontal bar chart — Retention Lift ────────────────────────────
_fig, _ax = plt.subplots(figsize=(13, 7))

_colors = [
    '#1a7f5e' if v >= 3.0 else ('#2ecc71' if v >= 1.5 else '#a8d5c2')
    for v in top10_retention_signals['retention_lift']
]

_bars = _ax.barh(
    range(len(top10_retention_signals)),
    top10_retention_signals['retention_lift'],
    color=_colors,
    edgecolor='white',
    linewidth=0.6
)

# Baseline line at lift = 1.0
_ax.axvline(1.0, color='#e74c3c', linewidth=1.5, linestyle='--', label=f'Baseline (1.0x = {_baseline_retention*100:.2f}%)')

# Axis labels
_short_labels = [
    name[:42] + ('…' if len(name) > 42 else '')
    for name in top10_retention_signals['event_name']
]
_ax.set_yticks(range(len(top10_retention_signals)))
_ax.set_yticklabels(_short_labels, fontsize=10)
_ax.invert_yaxis()

# Annotate bars with lift value + reach
for _i, (_bar, (_idx, _row)) in enumerate(zip(_bars, top10_retention_signals.iterrows())):
    _w = _bar.get_width()
    _reach_pct = _row.total_frequency / _n_total * 100
    _ax.text(
        _w + 0.05, _bar.get_y() + _bar.get_height() / 2,
        f"{_w:.2f}x  (n={int(_row.total_frequency):,}, {_reach_pct:.1f}% reach)",
        va='center', ha='left', fontsize=8.5, color='#2c3e50'
    )

_ax.set_xlabel('Retention Lift  (vs baseline)', fontsize=11)
_ax.set_title(
    'Step 5.5 — Top 10 Retention Signal Events\nHighest lift = strongest indicator of long-term engagement',
    fontsize=13, fontweight='bold'
)

_high_patch  = mpatches.Patch(color='#1a7f5e', label='Very High Lift  (≥ 3.0x)')
_med_patch   = mpatches.Patch(color='#2ecc71', label='High Lift  (1.5x – 3.0x)')
_low_patch   = mpatches.Patch(color='#a8d5c2', label='Above Baseline  (< 1.5x)')
_ax.legend(handles=[_high_patch, _med_patch, _low_patch,
                     mpatches.Patch(color='#e74c3c', label=f'Baseline ({_baseline_retention*100:.2f}%)')],
           fontsize=9, loc='lower right')

_ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('top10_retention_signal_events.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✓ Saved: top10_retention_signal_events.png")

# ── Step 5: Bubble chart — Lift vs Reach (size = retained_frequency) ─────────
_fig2, _ax2 = plt.subplots(figsize=(13, 7))

_reach_pct = top10_retention_signals['total_frequency'] / _n_total * 100
_bubble_sizes = (top10_retention_signals['retained_frequency'] / top10_retention_signals['retained_frequency'].max() * 1200) + 100

_scatter = _ax2.scatter(
    _reach_pct,
    top10_retention_signals['retention_lift'],
    s=_bubble_sizes,
    c=top10_retention_signals['retention_lift'],
    cmap='YlGn',
    alpha=0.85,
    edgecolors='#2c3e50',
    linewidths=0.8,
    zorder=3
)

# Label each bubble
for _i, (_idx, _row) in enumerate(top10_retention_signals.iterrows()):
    _name = _row.event_name[:30] + ('…' if len(_row.event_name) > 30 else '')
    _ax2.annotate(
        _name,
        xy=(_reach_pct.iloc[_i], _row.retention_lift),
        xytext=(6, 4), textcoords='offset points',
        fontsize=8, color='#2c3e50'
    )

_ax2.axhline(1.0, color='#e74c3c', linewidth=1.4, linestyle='--', label='Baseline lift = 1.0x')
_ax2.set_xlabel('Event Reach  (% of all users)', fontsize=11)
_ax2.set_ylabel('Retention Lift', fontsize=11)
_ax2.set_title(
    'Step 5.5 — Lift vs Reach  (bubble size ∝ retained_frequency)\nHigh-lift + high-reach = ideal product engagement signal',
    fontsize=12, fontweight='bold'
)
_ax2.legend(fontsize=9)
_ax2.grid(alpha=0.3)

_cbar = plt.colorbar(_scatter, ax=_ax2)
_cbar.set_label('Retention Lift', fontsize=9)

plt.tight_layout()
plt.savefig('retention_signals_lift_vs_reach.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Saved: retention_signals_lift_vs_reach.png")

# ── Final export ──────────────────────────────────────────────────────────────
print(f"\n✓ top10_retention_signals — {len(top10_retention_signals)} rows × {len(top10_retention_signals.columns)} cols")
print(top10_retention_signals[['event_name', 'retained_frequency', 'total_frequency', 'retention_lift']].to_string())