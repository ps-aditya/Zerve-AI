import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# ── 1. Merge behaviour features with retention labels ───────────────────────
activation_df = (
    user_level_final[["person_id", "retained_30d"]]
    .merge(total_events_7d,      on="person_id", how="inner")
    .merge(unique_event_types_7d, on="person_id", how="inner")
)

# ── 2. Define behavioural buckets ────────────────────────────────────────────
event_bins   = [0, 10, 30, 60, 120, np.inf]
event_labels = ["0–10", "10–30", "30–60", "60–120", "120+"]

etype_bins   = [0, 3, 6, 10, np.inf]
etype_labels = ["1–3", "4–6", "7–10", "10+"]

activation_df["event_bucket"]      = pd.cut(activation_df["total_events_7d"],
                                              bins=event_bins, labels=event_labels,
                                              right=True, include_lowest=True)
activation_df["event_type_bucket"] = pd.cut(activation_df["unique_event_types_7d"],
                                              bins=etype_bins, labels=etype_labels,
                                              right=True, include_lowest=True)

# ── 3. Bucket-level aggregations ─────────────────────────────────────────────
def bucket_summary(df, bucket_col):
    grp = (df.groupby(bucket_col, observed=True)
             .agg(users=("person_id", "count"),
                  retained_users=("retained_30d", "sum"))
             .reset_index()
             .rename(columns={bucket_col: "bucket"}))
    grp["retention_rate"] = (grp["retained_users"] / grp["users"] * 100).round(2)
    return grp

event_bucket_df = bucket_summary(activation_df, "event_bucket")
etype_bucket_df = bucket_summary(activation_df, "event_type_bucket")

# ── 4. Detect activation threshold (largest single-step jump in retention) ──
def find_threshold(bucket_df):
    rates = bucket_df["retention_rate"].values
    deltas = np.diff(rates)
    jump_idx = int(np.argmax(deltas))  # bucket AFTER which the jump occurs
    return bucket_df.iloc[jump_idx + 1]["bucket"]

activation_threshold_events      = find_threshold(event_bucket_df)
activation_threshold_event_types = find_threshold(etype_bucket_df)

# ── 5. Pretty-print tables ───────────────────────────────────────────────────
print("=" * 62)
print("  EVENT BUCKET  —  Activation Threshold Analysis")
print("=" * 62)
print(event_bucket_df.to_string(index=False))
print()
print("=" * 62)
print("  EVENT-TYPE BUCKET  —  Activation Threshold Analysis")
print("=" * 62)
print(etype_bucket_df.to_string(index=False))
print()
print("━" * 62)
print(f"  🎯 Activation Threshold (Events)       : {activation_threshold_events}")
print(f"  🎯 Activation Threshold (Event Types)  : {activation_threshold_event_types}")
print("━" * 62)

# ── 6. Visualisation ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("User Activation Threshold Discovery", fontsize=16, fontweight="bold", y=1.02)

palette_bar  = "#4C72B0"
palette_line = "#DD8452"

for ax, bdf, threshold, title in [
    (axes[0], event_bucket_df,      activation_threshold_events,      "Total Events in First 7 Days"),
    (axes[1], etype_bucket_df,      activation_threshold_event_types, "Unique Event Types in First 7 Days"),
]:
    x = np.arange(len(bdf))

    ax2 = ax.twinx()

    bars = ax.bar(x, bdf["users"], color=palette_bar, alpha=0.65, label="Total Users", zorder=2)
    line, = ax2.plot(x, bdf["retention_rate"], color=palette_line,
                     marker="o", linewidth=2.5, markersize=8, label="Retention Rate %", zorder=3)

    # Highlight threshold bucket
    thresh_idx = list(bdf["bucket"].astype(str)).index(str(threshold))
    ax.bar(thresh_idx, bdf.iloc[thresh_idx]["users"], color="#2ca02c",
           alpha=0.85, zorder=3, label=f"Threshold: {threshold}")

    ax.set_xticks(x)
    ax.set_xticklabels(bdf["bucket"].astype(str), rotation=25, ha="right")
    ax.set_xlabel("Bucket", fontsize=11)
    ax.set_ylabel("Number of Users", fontsize=11)
    ax2.set_ylabel("Retention Rate (%)", fontsize=11)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_title(title, fontsize=13, fontweight="bold")

    # Annotate retention rate on each point
    for xi, rate in zip(x, bdf["retention_rate"]):
        ax2.annotate(f"{rate:.1f}%", xy=(xi, rate),
                     xytext=(0, 9), textcoords="offset points",
                     ha="center", fontsize=9, color=palette_line)

    # Legend
    lines  = [bars, line]
    labels = ["Total Users", "Retention Rate %", f"Threshold: {threshold}"]
    ax.legend(handles=[bars,
                        ax.patches[-1] if thresh_idx < len(ax.patches) else bars,
                        line],
              labels=["Total Users", f"Threshold: {threshold}", "Retention Rate %"],
              loc="upper right", fontsize=9)

plt.tight_layout()
plt.savefig("activation_threshold_discovery.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nChart saved → activation_threshold_discovery.png")