import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

# ── Step 5.4 — Retained vs Churned Workflow Comparison ───────────────────────
# workflow_comparison columns: sequence | retained_count | churned_count | difference
# Already sorted by difference descending from churn_sequence_analysis

# ── Top N sequences most favouring retained users ────────────────────────────
_top_n = 20
_top_retained = workflow_comparison.head(_top_n).copy()
_top_retained['rank'] = range(1, len(_top_retained) + 1)

# ── Top N sequences most favouring churned users ─────────────────────────────
_top_churned = workflow_comparison.tail(_top_n).sort_values('difference').copy()
_top_churned['rank'] = range(1, len(_top_churned) + 1)

# ── Pretty-print the full ranked table ───────────────────────────────────────
_divider = "=" * 100
print(_divider)
print("STEP 5.4 — RETAINED vs CHURNED WORKFLOW COMPARISON  (ranked by difference)")
print(_divider)
print(f"\n{'RANK':<5} {'SEQUENCE':<60} {'RETAINED':>9} {'CHURNED':>8} {'DIFF':>7}")
print("-" * 100)
for _r in _top_retained.itertuples():
    _seq = str(_r.sequence)[:58]
    print(f"{_r.rank:<5} {_seq:<60} {_r.retained_count:>9,} {_r.churned_count:>8,} {_r.difference:>+7,}")

print(f"\n  ... (total unique sequences: {len(workflow_comparison):,})")
print(f"\n  Sequences where retained > churned : {(workflow_comparison['difference'] > 0).sum():,}")
print(f"  Sequences where churned  > retained: {(workflow_comparison['difference'] < 0).sum():,}")
print(_divider)

# ── Fig 1: Diverging bar chart — top 20 retention-driving vs churn-driving ───
_fig, _axes = plt.subplots(1, 2, figsize=(20, 8))
_fig.suptitle(
    "Step 5.4 — Retained vs Churned: Workflow Sequence Comparison",
    fontsize=15, fontweight='bold', y=1.01
)

# -- LEFT: Top 20 retention-driving sequences ---------------------------------
_ax_ret = _axes[0]
_labels_ret = [str(s)[:55] + ('…' if len(str(s)) > 55 else '') for s in _top_retained['sequence']]
_bars_ret = _ax_ret.barh(
    range(len(_top_retained)), _top_retained['difference'],
    color='#2ecc71', edgecolor='white', linewidth=0.6
)
_ax_ret.set_yticks(range(len(_top_retained)))
_ax_ret.set_yticklabels(_labels_ret, fontsize=8)
_ax_ret.invert_yaxis()
_ax_ret.set_xlabel('Difference (retained − churned)', fontsize=10)
_ax_ret.set_title('Top 20 Retention-Driving Sequences\n(largest positive difference)', fontsize=11, fontweight='bold')
_ax_ret.axvline(0, color='black', linewidth=0.8, linestyle='--')
_ax_ret.grid(axis='x', alpha=0.3)

# Annotate bar values
for _bar in _bars_ret:
    _w = _bar.get_width()
    _ax_ret.text(_w + 1, _bar.get_y() + _bar.get_height() / 2,
                 f'+{int(_w):,}', va='center', ha='left', fontsize=7.5, color='#27ae60')

# -- RIGHT: Top 20 churn-driving sequences ------------------------------------
_ax_churn = _axes[1]
_labels_churn = [str(s)[:55] + ('…' if len(str(s)) > 55 else '') for s in _top_churned['sequence']]
_bars_churn = _ax_churn.barh(
    range(len(_top_churned)), _top_churned['difference'],
    color='#e74c3c', edgecolor='white', linewidth=0.6
)
_ax_churn.set_yticks(range(len(_top_churned)))
_ax_churn.set_yticklabels(_labels_churn, fontsize=8)
_ax_churn.invert_yaxis()
_ax_churn.set_xlabel('Difference (retained − churned)', fontsize=10)
_ax_churn.set_title('Top 20 Churn-Associated Sequences\n(largest negative difference)', fontsize=11, fontweight='bold')
_ax_churn.axvline(0, color='black', linewidth=0.8, linestyle='--')
_ax_churn.grid(axis='x', alpha=0.3)

# Annotate bar values
for _bar in _bars_churn:
    _w = _bar.get_width()
    _ax_churn.text(_w - 1, _bar.get_y() + _bar.get_height() / 2,
                   f'{int(_w):,}', va='center', ha='right', fontsize=7.5, color='#c0392b')

plt.tight_layout()
plt.savefig('workflow_comparison_diverging.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Saved: workflow_comparison_diverging.png")

# ── Fig 2: Grouped bar chart — retained vs churned counts side-by-side ───────
_top15 = workflow_comparison.head(15).copy()
_short_labels = [str(s)[:45] + ('…' if len(str(s)) > 45 else '') for s in _top15['sequence']]
_x = range(len(_top15))
_width = 0.38

_fig2, _ax2 = plt.subplots(figsize=(18, 7))
_b1 = _ax2.bar([_i - _width / 2 for _i in _x], _top15['retained_count'], _width,
               label='Retained', color='#2ecc71', edgecolor='white')
_b2 = _ax2.bar([_i + _width / 2 for _i in _x], _top15['churned_count'], _width,
               label='Churned', color='#e74c3c', edgecolor='white')

_ax2.set_xticks(list(_x))
_ax2.set_xticklabels(_short_labels, rotation=35, ha='right', fontsize=8.5)
_ax2.set_ylabel('Sequence Count', fontsize=11)
_ax2.set_title(
    'Top 15 Retention-Driving Sequences — Retained vs Churned Counts',
    fontsize=13, fontweight='bold'
)
_ax2.legend(fontsize=11)
_ax2.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('workflow_comparison_grouped.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Saved: workflow_comparison_grouped.png")

# ── Export ranked table ───────────────────────────────────────────────────────
retention_workflow_table = workflow_comparison.copy()
retention_workflow_table.index = retention_workflow_table.index + 1
retention_workflow_table.index.name = 'rank'
print(f"\n✓ retention_workflow_table ready — {len(retention_workflow_table):,} rows × {len(retention_workflow_table.columns)} cols")
print(retention_workflow_table.head(10).to_string())