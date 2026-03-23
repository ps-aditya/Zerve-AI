import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Step 6.4: Product Feature Impact Simulation ──────────────────────────────
# We retrain the same Logistic Regression on features_step2 and then simulate
# behaviour increases to estimate retention lift via counterfactual scoring.

# ── 1. Map requested behaviours to our model features ────────────────────────
# The 4 target behaviours (canvas_create, run_block, agent_open,
# agent_accept_suggestion) all drive engagement volume & diversity:
#   - run_block       → total_events_7d   (+20%)     Scenario A
#   - canvas_create   → total_events_7d   (+30%)     Scenario B (also bumps event types)
#   - agent interaction → unique_event_types_7d (+25%)  Scenario C
#
# Scenario B (canvas_create) is modelled as:
#   • total_events_7d    × 1.30  (more raw activity)
#   • unique_event_types_7d × 1.15 (creates a new canvas = new event type engaged)

_FEATURE_COLS = ['total_events_7d', 'unique_event_types_7d', 'total_credits_7d']

# ── 2. Retrain model (balanced, scaled) ──────────────────────────────────────
_X_raw = features_step2[_FEATURE_COLS].values
_y     = features_step2['success'].values

_scaler = StandardScaler()
_X_scaled = _scaler.fit_transform(_X_raw)

_model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    solver='lbfgs',
    random_state=42
)
_model.fit(_X_scaled, _y)

# ── 3. Baseline predicted retention (mean probability across all users) ───────
_baseline_probs   = _model.predict_proba(_X_scaled)[:, 1]
_baseline_ret_pct = _baseline_probs.mean() * 100  # as %

print(f"Baseline mean predicted retention probability: {_baseline_ret_pct:.4f}%")

# ── 4. Define simulation scenarios ───────────────────────────────────────────
_scenarios = [
    {
        'scenario'        : 'A',
        'behavior_change' : 'Increase run_block (+20%)',
        'multipliers'     : {'total_events_7d': 1.20, 'unique_event_types_7d': 1.00, 'total_credits_7d': 1.00},
    },
    {
        'scenario'        : 'B',
        'behavior_change' : 'Increase canvas_create (+30%)',
        'multipliers'     : {'total_events_7d': 1.30, 'unique_event_types_7d': 1.15, 'total_credits_7d': 1.00},
    },
    {
        'scenario'        : 'C',
        'behavior_change' : 'Increase agent interaction (+25%)',
        'multipliers'     : {'total_events_7d': 1.00, 'unique_event_types_7d': 1.25, 'total_credits_7d': 1.00},
    },
]

# ── 5. Score each scenario ────────────────────────────────────────────────────
_sim_rows = []

for _s in _scenarios:
    # Apply multipliers to raw feature values
    _sim_df = features_step2[_FEATURE_COLS].copy().astype(float)
    for _feat, _mult in _s['multipliers'].items():
        _sim_df[_feat] = _sim_df[_feat] * _mult

    # Scale using the same fitted scaler
    _sim_scaled = _scaler.transform(_sim_df.values)

    # Predict
    _sim_probs  = _model.predict_proba(_sim_scaled)[:, 1]
    _sim_ret    = _sim_probs.mean() * 100

    _lift = _sim_ret - _baseline_ret_pct

    _sim_rows.append({
        'scenario'            : _s['scenario'],
        'behavior_change'     : _s['behavior_change'],
        'predicted_retention' : round(_sim_ret, 2),
        'retention_lift'      : round(_lift, 2),
    })

# ── 6. Build output DataFrame ─────────────────────────────────────────────────
feature_impact_simulation = pd.DataFrame(_sim_rows)
feature_impact_simulation.insert(0, 'baseline_retention', round(_baseline_ret_pct, 2))

# ── 7. Print results table ────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("  STEP 6.4 — PRODUCT FEATURE IMPACT SIMULATION")
print("=" * 80)
print(f"  Baseline predicted retention: {_baseline_ret_pct:.2f}%")
print("-" * 80)
print(f"  {'Scenario':<6} {'Behaviour Change':<38} {'Pred. Retention':>16} {'Lift':>10}")
print("-" * 80)
for _, _r in feature_impact_simulation.iterrows():
    _lift_str = f"+{_r['retention_lift']:.2f}%" if _r['retention_lift'] >= 0 else f"{_r['retention_lift']:.2f}%"
    print(
        f"  {_r['scenario']:<6} "
        f"{_r['behavior_change']:<38} "
        f"{_r['predicted_retention']:>14.2f}%"
        f"{_lift_str:>12}"
    )
print("=" * 80)

# ── 8. Visualisation ─────────────────────────────────────────────────────────
_fig, (_ax_ret, _ax_lift) = plt.subplots(1, 2, figsize=(14, 5))
_fig.suptitle('Step 6.4 — Product Feature Impact Simulation', fontsize=14, fontweight='bold', y=1.01)

_scenario_labels = [
    f"Scenario {r['scenario']}\n{r['behavior_change'].split('(')[1].rstrip(')')}"
    for _, r in feature_impact_simulation.iterrows()
]
_bar_colors   = ['#4C72B0', '#DD8452', '#55A868']
_lift_colors  = ['#27AE60' if l >= 0 else '#E74C3C' for l in feature_impact_simulation['retention_lift']]

# --- Left: predicted retention vs baseline ---
_baseline_line = _baseline_ret_pct
_bars_ret = _ax_ret.bar(_scenario_labels, feature_impact_simulation['predicted_retention'], color=_bar_colors, width=0.5, zorder=3)
_ax_ret.axhline(_baseline_line, color='red', linestyle='--', linewidth=1.5, label=f'Baseline: {_baseline_line:.2f}%')
_ax_ret.set_title('Predicted Retention by Scenario', fontweight='bold')
_ax_ret.set_ylabel('Mean Predicted Retention (%)')
_ax_ret.set_ylim(0, max(feature_impact_simulation['predicted_retention'].max() * 1.3, _baseline_line * 1.5))
_ax_ret.legend()
_ax_ret.grid(axis='y', alpha=0.4)
for _b, _v in zip(_bars_ret, feature_impact_simulation['predicted_retention']):
    _ax_ret.text(_b.get_x() + _b.get_width() / 2, _b.get_height() + 0.01,
                 f'{_v:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

# --- Right: retention lift ---
_bars_lift = _ax_lift.bar(_scenario_labels, feature_impact_simulation['retention_lift'], color=_lift_colors, width=0.5, zorder=3)
_ax_lift.axhline(0, color='black', linewidth=1.0)
_ax_lift.set_title('Retention Lift vs Baseline', fontweight='bold')
_ax_lift.set_ylabel('Retention Lift (pp)')
_ax_lift.grid(axis='y', alpha=0.4)
for _b, _v in zip(_bars_lift, feature_impact_simulation['retention_lift']):
    _sign = '+' if _v >= 0 else ''
    _ax_lift.text(_b.get_x() + _b.get_width() / 2,
                  _b.get_height() + (0.005 if _v >= 0 else -0.01),
                  f'{_sign}{_v:.2f}pp', ha='center', va='bottom', fontsize=9, fontweight='bold')

_fig.tight_layout()
plt.savefig('feature_impact_simulation.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nChart saved → feature_impact_simulation.png")