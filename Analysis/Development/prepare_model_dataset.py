# ============================================
# STEP 3 — PURE NUMPY LOGISTIC REGRESSION
# ============================================

import pandas as pd
import numpy as np

# --------------------------------------------
# 1. LOAD DATA
# --------------------------------------------

raw_df = pd.read_csv("zerve_hackathon_for_reviewc8fa7c7.csv")
df = raw_df.copy()
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

# --------------------------------------------
# 2. CREATE SUCCESS LABEL
# --------------------------------------------

user_level = df.groupby("person_id").agg(
    first_event=("timestamp", "min"),
    last_event=("timestamp", "max")
).reset_index()

user_level["lifetime_days"] = (
    user_level["last_event"] - user_level["first_event"]
).dt.days

user_level["success"] = np.where(
    user_level["lifetime_days"] >= 30,
    1, 0
)

# --------------------------------------------
# 3. MERGE FIRST EVENT
# --------------------------------------------

df = df.merge(
    user_level[["person_id", "first_event", "success"]],
    on="person_id",
    how="left"
)

df["days_since_first_event"] = (
    df["timestamp"] - df["first_event"]
).dt.days

# --------------------------------------------
# 4. FIRST 7 DAYS
# --------------------------------------------

first_7 = df[df["days_since_first_event"] <= 7].copy()

features = first_7.groupby("person_id").agg(
    total_events_7d=("event", "count"),
    unique_event_types_7d=("event", "nunique")
).reset_index()

features = features.merge(
    user_level[["person_id", "success"]],
    on="person_id",
    how="left"
)

features.fillna(0, inplace=True)

print("Dataset ready:", features.shape)

# --------------------------------------------
# 5. PREPARE MATRIX
# --------------------------------------------

X = features[["total_events_7d", "unique_event_types_7d"]].values
y = features["success"].values.reshape(-1, 1)

# Add bias term
X = np.hstack([np.ones((X.shape[0], 1)), X])

# --------------------------------------------
# 6. LOGISTIC REGRESSION (GRADIENT DESCENT)
# --------------------------------------------

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Initialize weights
weights = np.zeros((X.shape[1], 1))

learning_rate = 0.0000001
epochs = 500

for i in range(epochs):
    z = np.dot(X, weights)
    predictions = sigmoid(z)
    gradient = np.dot(X.T, (predictions - y)) / len(y)
    weights -= learning_rate * gradient

print("Model trained.")

# --------------------------------------------
# 7. PREDICTIONS
# --------------------------------------------

z = np.dot(X, weights)
probs = sigmoid(z)
predictions = (probs >= 0.5).astype(int)

# --------------------------------------------
# 8. EVALUATION
# --------------------------------------------

accuracy = np.mean(predictions == y)

tp = np.sum((predictions == 1) & (y == 1))
fp = np.sum((predictions == 1) & (y == 0))
fn = np.sum((predictions == 0) & (y == 1))

precision = tp / (tp + fp + 1e-9)
recall = tp / (tp + fn + 1e-9)
f1 = 2 * (precision * recall) / (precision + recall + 1e-9)

print("\nModel Performance:")
print("Accuracy:", round(float(accuracy), 4))
print("Precision:", round(float(precision), 4))
print("Recall:", round(float(recall), 4))
print("F1 Score:", round(float(f1), 4))

print("\nWeights:")
print("Intercept:", weights[0][0])
print("total_events_7d weight:", weights[1][0])
print("unique_event_types_7d weight:", weights[2][0])