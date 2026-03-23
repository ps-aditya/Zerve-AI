import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Step 5.3: Workflow Frequency Visualization ──────────────────────────────
# Count frequency of each unique sequence across all users
workflow_freq = (
    sequence_df
    .groupby("sequence", as_index=False)
    .agg(count=("person_id", "count"))
    .sort_values("count", ascending=False)
    .reset_index(drop=True)
)

# Rank sequences 1-based
workflow_freq.insert(0, "rank", range(1, len(workflow_freq) + 1))

# Top 10 dominant workflows
top10_workflows = workflow_freq.head(10).copy()

# ── Print ranked table ───────────────────────────────────────────────────────
print("=" * 90)
print(f"{'STEP 5.3 — TOP 10 MOST FREQUENT USER WORKFLOWS (First 7 Days)':^90}")
print("=" * 90)
print(f"{'Rank':<6} {'Count':>7}   {'Sequence'}")
print("-" * 90)
for _, row in top10_workflows.iterrows():
    # Wrap long sequences for readability
    seq_display = row["sequence"]
    print(f"  {int(row['rank']):<4} {int(row['count']):>7}   {seq_display}")
print("-" * 90)
print(f"\nTotal unique sequences observed: {len(workflow_freq):,}")
print(f"Total sequence instances:        {workflow_freq['count'].sum():,}")
print(f"Top-10 coverage:                 {top10_workflows['count'].sum() / workflow_freq['count'].sum() * 100:.1f}% of all instances\n")

# ── Visualization ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 7))

# Shorten long sequence labels for the chart
def shorten_seq(seq, max_len=60):
    return seq if len(seq) <= max_len else seq[:max_len] + "…"

labels = [f"#{int(r['rank'])}  {shorten_seq(r['sequence'])}" for _, r in top10_workflows.iterrows()]
counts = top10_workflows["count"].values

colors = plt.cm.Blues_r([i / (len(counts) + 2) for i in range(len(counts))])

bars = ax.barh(labels[::-1], counts[::-1], color=colors[::-1], edgecolor="white", linewidth=0.5)

# Annotate bars with count values
for bar, cnt in zip(bars, counts[::-1]):
    ax.text(bar.get_width() + max(counts) * 0.005, bar.get_y() + bar.get_height() / 2,
            f"{cnt:,}", va="center", ha="left", fontsize=9, color="#333333")

ax.set_xlabel("Number of Occurrences", fontsize=11)
ax.set_title("Top 10 Most Frequent User Workflows — First 7 Days", fontsize=14, fontweight="bold", pad=15)
ax.set_xlim(0, max(counts) * 1.18)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(axis="y", labelsize=8.5)
ax.tick_params(axis="x", labelsize=9)
plt.tight_layout()
plt.show()

# ── Interpretation note ───────────────────────────────────────────────────────
print("INTERPRETATION")
print("-" * 90)
print("The sequences above represent the most common 3-event interaction chains")
print("observed during a user's first 7 days. High-frequency sequences indicate")
print("dominant onboarding paths and habitual usage patterns.")
print()
for _, row in top10_workflows.iterrows():
    pct = row["count"] / workflow_freq["count"].sum() * 100
    print(f"  #{int(row['rank']):02d} ({pct:.2f}%)  {row['sequence']}")