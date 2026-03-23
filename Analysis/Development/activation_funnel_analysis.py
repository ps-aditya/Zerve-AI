import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ================================================
# STEP 5.6 — ACTIVATION FUNNEL ANALYSIS
# ================================================
# Map funnel stages to exact event names in the dataset
funnel_stages = [
    ("sign_in",                 "sign_in"),
    ("canvas_create",           "canvas_create"),
    ("run_block",               "run_block"),
    ("agent_accept_suggestion", "agent_accept_suggestion"),
]

# Use person_id as the canonical user identifier (consistent across sessions)
_user_col = "person_id"

# Count unique users who triggered each funnel event (at least once)
_stage_names   = []
_stage_labels  = []
_users_reached = []

for label, event_name in funnel_stages:
    _mask = df["event"] == event_name
    _count = df.loc[_mask, _user_col].nunique()
    _stage_labels.append(label)
    _stage_names.append(event_name)
    _users_reached.append(_count)

# Build funnel DataFrame
activation_funnel = pd.DataFrame({
    "stage":         _stage_labels,
    "event_name":    _stage_names,
    "users_reached": _users_reached,
})

# Stage-to-stage conversion rate (each stage vs. the previous one)
activation_funnel["conversion_rate"] = None
for _i in range(len(activation_funnel)):
    if _i == 0:
        # Top of funnel = 100% baseline
        activation_funnel.at[_i, "conversion_rate"] = 100.0
    else:
        _prev = activation_funnel.at[_i - 1, "users_reached"]
        _curr = activation_funnel.at[_i,     "users_reached"]
        activation_funnel.at[_i, "conversion_rate"] = round((_curr / _prev * 100) if _prev > 0 else 0.0, 1)

activation_funnel["conversion_rate"] = activation_funnel["conversion_rate"].astype(float)

# Absolute drop-off at each stage
activation_funnel["drop_off_users"] = activation_funnel["users_reached"].shift(1).fillna(activation_funnel["users_reached"].iloc[0]).astype(int) - activation_funnel["users_reached"]

# Identify largest drop-off stage
_max_drop_idx = activation_funnel["drop_off_users"].iloc[1:].idxmax()
_bottleneck_stage = activation_funnel.at[_max_drop_idx, "stage"]
_bottleneck_drop  = activation_funnel.at[_max_drop_idx, "drop_off_users"]

# ── Print table ──────────────────────────────────────────────────────────────
print("=" * 70)
print("ACTIVATION FUNNEL — Stage-by-Stage Breakdown")
print("=" * 70)
print(f"{'Stage':<28} {'Users Reached':>14} {'Conv. Rate':>12} {'Drop-off':>10}")
print("-" * 70)
for _, _row in activation_funnel.iterrows():
    _conv = f"{_row['conversion_rate']:.1f}%" if _row["conversion_rate"] is not None else "—"
    _drop = f"-{int(_row['drop_off_users'])}" if _row["drop_off_users"] > 0 else "—"
    print(f"{_row['stage']:<28} {int(_row['users_reached']):>14,} {_conv:>12} {_drop:>10}")
print("=" * 70)
print(f"\n🔴 Largest drop-off:  '{_bottleneck_stage}'  ({int(_bottleneck_drop):,} users lost)")
print(f"   This is the primary activation bottleneck in the onboarding flow.")

# ── Funnel chart ─────────────────────────────────────────────────────────────
_fig, _ax = plt.subplots(figsize=(10, 6))

_colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
_bar_heights = activation_funnel["users_reached"].tolist()
_x = range(len(_stage_labels))

_bars = _ax.bar(_x, _bar_heights, color=_colors, width=0.55, edgecolor="white", linewidth=1.2)

# Annotate bars with user counts + conversion rates
for _idx, (_bar, _row) in enumerate(zip(_bars, activation_funnel.itertuples())):
    _ax.text(
        _bar.get_x() + _bar.get_width() / 2,
        _bar.get_height() + max(_bar_heights) * 0.01,
        f"{int(_row.users_reached):,}",
        ha="center", va="bottom", fontsize=11, fontweight="bold"
    )
    if _idx > 0:
        _ax.text(
            _bar.get_x() + _bar.get_width() / 2,
            _bar.get_height() / 2,
            f"{_row.conversion_rate:.1f}%",
            ha="center", va="center", fontsize=10, color="white", fontweight="bold"
        )

# Highlight bottleneck with a red border
_bars[_max_drop_idx].set_edgecolor("#FF4444")
_bars[_max_drop_idx].set_linewidth(3)

_ax.set_xticks(list(_x))
_ax.set_xticklabels(_stage_labels, fontsize=11)
_ax.set_ylabel("Unique Users", fontsize=12)
_ax.set_title("Activation Funnel — User Drop-off by Stage", fontsize=14, fontweight="bold", pad=15)
_ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
_ax.spines[["top", "right"]].set_visible(False)
_ax.set_facecolor("#f9f9f9")
_fig.patch.set_facecolor("#f9f9f9")

plt.tight_layout()
plt.show()