"""Export an enriched customer-level CSV for Tableau Public.

Adds segmentation fields and out-of-fold XGBoost churn probabilities
so Tableau can slice by any dimension without needing to run Python.

Output: data/processed/tableau_churn.csv
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from xgboost import XGBClassifier

from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features

RANDOM_STATE = 42

print("Loading raw data...")
df = load_data("data/raw/Customer-Churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

# --- Segmentation fields ---
tenure_bins = [0, 6, 12, 24, 48, int(df["tenure"].max()) + 1]
tenure_labels = ["0-6 mo", "7-12 mo", "13-24 mo", "25-48 mo", "49+ mo"]
df["tenure_bucket"] = pd.cut(
    df["tenure"], bins=tenure_bins, labels=tenure_labels, include_lowest=True
).astype(str)

addon_cols = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]
df["addon_count"] = (df[addon_cols] == "Yes").sum(axis=1)

df["is_churned"] = (df["Churn"] == "Yes").astype(int)
df["annual_revenue"] = df["MonthlyCharges"] * 12

# --- Out-of-fold churn probabilities ---
print("Building features for OOF scoring...")
extra_cols = ["tenure_bucket", "addon_count", "is_churned", "annual_revenue"]
df_model = preprocess_data(
    df.drop(columns=extra_cols).copy(), target_col="Churn"
)
df_enc = build_features(df_model, target_col="Churn")
X = df_enc.drop(columns=["Churn"])
y = df_enc["Churn"].astype(int)

scale_pos_weight = (y == 0).sum() / (y == 1).sum()
model = XGBClassifier(
    n_estimators=301, learning_rate=0.034, max_depth=7,
    subsample=0.95, colsample_bytree=0.98,
    scale_pos_weight=scale_pos_weight, eval_metric="logloss",
    n_jobs=-1, random_state=RANDOM_STATE,
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

print("Running 5-fold OOF predictions (this takes ~1 min)...")
proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]

df["churn_probability"] = proba
df["annual_revenue_at_risk"] = df["annual_revenue"] * df["churn_probability"]

# --- Export ---
out = "data/processed/tableau_churn.csv"
df.to_csv(out, index=False)
print(f"\nWrote {len(df):,} rows x {df.shape[1]} columns to {out}")
print(f"Columns: {', '.join(df.columns)}")
