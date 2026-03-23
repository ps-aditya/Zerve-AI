# --------------------------------------------
# BALANCE DATA (OVERSAMPLING)
# --------------------------------------------
import pandas as pd
import numpy as np

# --------------------------------------------
# 1. LOAD DATA
# --------------------------------------------

raw_df = pd.read_csv("zerve_hackathon_for_reviewc8fa7c7.csv")
df = raw_df.copy()
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

# Separate classes
retained = features[features["success"] == 1]
not_retained = features[features["success"] == 0]

# Oversample retained users
retained_upsampled = retained.sample(
    n=len(not_retained),
    replace=True,
    random_state=42
)

balanced = pd.concat([not_retained, retained_upsampled])

# Shuffle
balanced = balanced.sample(frac=1, random_state=42)

print("Balanced dataset shape:", balanced.shape)
print(balanced["success"].value_counts())

# --------------------------------------------
# TRAIN AGAIN
# --------------------------------------------

X_bal = balanced[["total_events_7d", "unique_event_types_7d"]].values
y_bal = balanced["success"].values.reshape(-1,1)

X_bal = np.hstack([np.ones((X_bal.shape[0], 1)), X_bal])

weights = np.zeros((X_bal.shape[1], 1))

learning_rate = 0.00000001
epochs = 800

for i in range(epochs):
    z = np.dot(X_bal, weights)
    predictions = 1 / (1 + np.exp(-z))
    gradient = np.dot(X_bal.T, (predictions - y_bal)) / len(y_bal)
    weights -= learning_rate * gradient

print("Balanced model trained.")