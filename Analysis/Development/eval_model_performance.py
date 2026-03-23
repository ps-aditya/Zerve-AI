# ============================================
# MODEL EVALUATION ON ORIGINAL DATA
# ============================================
import pandas as pd
import numpy as np

# --------------------------------------------
# 1. LOAD DATA
# --------------------------------------------

raw_df = pd.read_csv("zerve_hackathon_for_reviewc8fa7c7.csv")
df = raw_df.copy()
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

# Recreate original feature matrix
X = features[["total_events_7d", "unique_event_types_7d"]].values
y = features["success"].values.reshape(-1,1)

# Add intercept column
X = np.hstack([np.ones((X.shape[0], 1)), X])

# Predict probabilities
z = np.dot(X, weights)
probs = 1 / (1 + np.exp(-z))

# Convert to predictions
predictions = (probs >= 0.5).astype(int)

# Metrics
accuracy = np.mean(predictions == y)

tp = np.sum((predictions == 1) & (y == 1))
fp = np.sum((predictions == 1) & (y == 0))
fn = np.sum((predictions == 0) & (y == 1))

precision = tp / (tp + fp + 1e-9)
recall = tp / (tp + fn + 1e-9)
f1 = 2 * (precision * recall) / (precision + recall + 1e-9)

print("\nFINAL MODEL PERFORMANCE")
print("-------------------------")
print("Accuracy:", round(float(accuracy), 4))
print("Precision:", round(float(precision), 4))
print("Recall:", round(float(recall), 4))
print("F1 Score:", round(float(f1), 4))

print("\nConfusion counts:")
print("True Positives:", int(tp))
print("False Positives:", int(fp))
print("False Negatives:", int(fn))