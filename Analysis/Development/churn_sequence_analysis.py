from collections import Counter
import pandas as pd

# ── Step 1: Split users into retained vs churned ──────────────────────────────
# user_level_final has columns: person_id, retained_30d (1=retained, 0=churned)
retained_user_ids  = set(user_level_final.loc[user_level_final['retained_30d'] == 1, 'person_id'])
churned_user_ids   = set(user_level_final.loc[user_level_final['retained_30d'] == 0, 'person_id'])

print(f"Retained users : {len(retained_user_ids):,}")
print(f"Churned users  : {len(churned_user_ids):,}")

# ── Step 2: Pull sequences per group from sequence_df ─────────────────────────
# sequence_df has columns: person_id, sequence, success  (built in build_event_sequences)
retained_seqs = sequence_df[sequence_df['person_id'].isin(retained_user_ids)]['sequence']
churned_seqs  = sequence_df[sequence_df['person_id'].isin(churned_user_ids)]['sequence']

# ── Step 3: Count frequency per group ─────────────────────────────────────────
retained_counts = Counter(retained_seqs)
churned_counts  = Counter(churned_seqs)

# ── Step 4: Build comparison table ────────────────────────────────────────────
all_sequences = set(retained_counts.keys()) | set(churned_counts.keys())

workflow_comparison = pd.DataFrame([
    {
        'sequence'       : seq,
        'retained_count' : retained_counts.get(seq, 0),
        'churned_count'  : churned_counts.get(seq, 0),
    }
    for seq in all_sequences
])

workflow_comparison['difference'] = (
    workflow_comparison['retained_count'] - workflow_comparison['churned_count']
)

# Sort by difference descending → sequences most over-represented in retained users first
workflow_comparison = (
    workflow_comparison
    .sort_values('difference', ascending=False)
    .reset_index(drop=True)
)

# ── Step 5: Display results ───────────────────────────────────────────────────
_top_n = 30   # sequences most favoured by retained users

print("\n" + "=" * 90)
print("STEP 4.4 — RETAINED vs CHURNED: TOP WORKFLOW SEQUENCES (by difference)")
print("=" * 90)
print(f"\n{'#':<4} {'sequence':<58} {'ret':>5} {'churn':>6} {'diff':>6}")
print("-" * 90)

for _rank, _row in enumerate(workflow_comparison.head(_top_n).itertuples(), start=1):
    print(f"{_rank:<4} {_row.sequence:<58} {_row.retained_count:>5} {_row.churned_count:>6} {_row.difference:>+6}")

print("=" * 90)
print(f"\nTotal unique sequences : {len(workflow_comparison):,}")
print(f"Sequences where retained > churned : "
      f"{(workflow_comparison['difference'] > 0).sum():,}")
print(f"\nBottom {_top_n} — sequences most favoured by churned users:")
print("-" * 90)

_bottom = workflow_comparison.tail(_top_n).sort_values('difference')
for _rank, _row in enumerate(_bottom.itertuples(), start=1):
    print(f"{_rank:<4} {_row.sequence:<58} {_row.retained_count:>5} {_row.churned_count:>6} {_row.difference:>+6}")

print("=" * 90)