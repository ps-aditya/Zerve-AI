import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ── Feature columns for Step 2 ──────────────────────────────────────────────
feature_cols = ['total_events_7d', 'unique_event_types_7d', 'total_credits_7d']
target_col   = 'success'

# ── Prepare X / y ───────────────────────────────────────────────────────────
_X_raw = features_step2[feature_cols].values
_y     = features_step2[target_col].values

# Scale features so coefficients are directly comparable
_scaler = StandardScaler()
_X_scaled = _scaler.fit_transform(_X_raw)

# ── Train Logistic Regression ────────────────────────────────────────────────
# class_weight='balanced' handles the heavy class imbalance in this dataset
_lr = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    solver='lbfgs',
    random_state=42
)
_lr.fit(_X_scaled, _y)

# ── Extract & rank coefficients ──────────────────────────────────────────────
_coefs = _lr.coef_[0]  # shape: (n_features,)

retention_driver_importance = pd.DataFrame({
    'feature_name'    : feature_cols,
    'model_weight'    : _coefs,
    'importance_score': np.abs(_coefs),
})

retention_driver_importance['rank'] = (
    retention_driver_importance['importance_score']
    .rank(ascending=False)
    .astype(int)
)

retention_driver_importance = (
    retention_driver_importance
    .sort_values('rank')
    .reset_index(drop=True)
)

# ── Display results ──────────────────────────────────────────────────────────
print("=" * 65)
print("  RETENTION DRIVER IMPORTANCE  (Logistic Regression Coefficients)")
print("=" * 65)
print(f"{'feature_name':<28} {'model_weight':>13} {'importance_score':>17} {'rank':>5}")
print("-" * 65)
for _, row in retention_driver_importance.iterrows():
    print(
        f"{row['feature_name']:<28} "
        f"{row['model_weight']:>13.4f} "
        f"{row['importance_score']:>17.4f} "
        f"{int(row['rank']):>5}"
    )
print("=" * 65)
print(f"\nModel trained on {len(_y):,} users  |  "
      f"Retained: {_y.sum():,}  |  Churned: {(1-_y).sum():,}")
print(f"Intercept: {_lr.intercept_[0]:.4f}")