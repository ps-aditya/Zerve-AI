import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# --- Feature columns to analyse ---
feature_cols = ['total_events_7d', 'unique_event_types_7d', 'total_credits_7d']

# --- Prepare X and y ---
_X_raw = features_step2[feature_cols].values
_y     = features_step2['success'].values

# --- Scale features so coefficients are comparable ---
_scaler = StandardScaler()
_X_scaled = _scaler.fit_transform(_X_raw)

# --- Train Logistic Regression (balanced class weights to handle imbalance) ---
_lr = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    solver='lbfgs',
    random_state=42
)
_lr.fit(_X_scaled, _y)

# --- Extract coefficients ---
_coefs = _lr.coef_[0]

# --- Build importance table ---
retention_driver_importance = pd.DataFrame({
    'feature_name':     feature_cols,
    'model_weight':     _coefs,
    'importance_score': np.abs(_coefs),
})

# Rank by importance score (highest = rank 1)
retention_driver_importance = (
    retention_driver_importance
    .sort_values('importance_score', ascending=False)
    .reset_index(drop=True)
)
retention_driver_importance['rank'] = retention_driver_importance.index + 1

# Round for readability
retention_driver_importance['model_weight']     = retention_driver_importance['model_weight'].round(4)
retention_driver_importance['importance_score'] = retention_driver_importance['importance_score'].round(4)

print("=" * 60)
print("  RETENTION DRIVER IMPORTANCE  (Logistic Regression)")
print("=" * 60)
print(retention_driver_importance.to_string(index=False))
print()
print("Note: Features scaled with StandardScaler before fitting,")
print("so coefficients reflect relative importance on a common scale.")
print("Model trained with class_weight='balanced' to handle class imbalance.")