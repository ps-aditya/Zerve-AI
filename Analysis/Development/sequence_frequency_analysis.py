from collections import Counter
import pandas as pd
# Count frequency of each 3-event sequence
sequence_counts = Counter(sequence_df['sequence'])

# Build Top 20 sequences DataFrame
top_20_sequences = (
    pd.DataFrame(sequence_counts.most_common(20), columns=['sequence', 'count'])
)

# Display results
print("=" * 70)
print("TOP 20 MOST COMMON 3-EVENT SEQUENCES (First 7 Days)")
print("=" * 70)
print(f"{'#':<4} {'sequence':<55} {'count':>6}")
print("-" * 70)
for rank, row in enumerate(top_20_sequences.itertuples(), start=1):
    print(f"{rank:<4} {row.sequence:<55} {row.count:>6}")
print("=" * 70)
print(f"\nTotal unique sequences: {len(sequence_counts):,}")
print(f"Total sequence instances: {len(sequence_df):,}")