# ============================================
# STEP 4.2 — BUILD EVENT SEQUENCES
# ============================================

import pandas as pd

# Load dataset
raw_df = pd.read_csv("zerve_hackathon_for_reviewc8fa7c7.csv")

df = raw_df.copy()
sequences = []

for user, group in first7.groupby("person_id"):
    
    events = group["event"].tolist()
    success = group["success"].iloc[0]
    
    if len(events) < 3:
        continue
    
    for i in range(len(events)-2):
        seq = (
            events[i],
            events[i+1],
            events[i+2]
        )
        
        sequences.append({
            "person_id": user,
            "sequence": " → ".join(seq),
            "success": success
        })

sequence_df = pd.DataFrame(sequences)

print("Sequences created")
print("Total sequences:", len(sequence_df))
print(sequence_df.head())