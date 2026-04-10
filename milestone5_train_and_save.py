"""
CreditPath AI – Milestone 5
Train & Serialize Model
──────────────────────────────────────────────────────────────────────────────
PURPOSE
  This script trains the best model from Milestone 4 (XGBoost with tuned
  hyperparameters) and serializes three artefacts to disk:

    model.pkl          – fitted XGBClassifier
    scaler.pkl         – fitted StandardScaler (MUST match API transform)
    feature_names.pkl  – ordered list of feature columns

  The FastAPI app (main_api.py) loads these at startup so predictions are
  instant and consistent with the training pipeline.

WHY JOBLIB OVER PICKLE?
  joblib is optimised for large numpy arrays (it memory-maps them), making
  save/load 3-5× faster than raw pickle for sklearn/XGBoost objects.

RUN ONCE before starting the API:
    python milestone5_train_and_save.py
──────────────────────────────────────────────────────────────────────────────
"""

import sys, io
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

# ── re-use the feature engineering already built in Milestone 3 ──────────────
from feature_engineering import load_data, engineer_features, split_X_y, get_train_test


def _banner(text: str):
    line = "=" * (len(text) + 4)
    print(f"\n+{line}+")
    print(f"|  {text}  |")
    print(f"+{line}+\n")


def _section(text: str):
    print(f"\n{'-' * 60}")
    print(f"  {text}")
    print(f"{'-' * 60}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MAIN                                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def train_and_save(filepath: str = "clean_loans.csv"):
    _banner("CreditPath AI – Milestone 5: Train & Serialize")

    # ── Step 1 : Load & engineer features (identical to Milestone 3/4) ──────
    _section("STEP 1: Load Data & Engineer Features")
    df = load_data(filepath)
    df = engineer_features(df)
    X, y = split_X_y(df)

    # Save feature column order – the API MUST use this exact order
    feature_names = X.columns.tolist()
    print(f"  [INFO] Feature count  : {len(feature_names)}")
    print(f"  [INFO] Features       : {feature_names}")

    # ── Step 2 : Train / test split ──────────────────────────────────────────
    _section("STEP 2: Train / Test Split (80 / 20, stratified)")
    X_train, X_test, y_train, y_test = get_train_test(X, y)

    # ── Step 3 : Scale – fit ONLY on train ───────────────────────────────────
    _section("STEP 3: Fit StandardScaler on Training Data")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    print("  [OK] Scaler fitted.")

    # ── Step 4 : Train best XGBoost model ────────────────────────────────────
    _section("STEP 4: Train XGBoost (best params from Milestone 4 tuning)")
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())

    model = XGBClassifier(
        n_estimators      = 200,
        max_depth         = 3,         # best_params from GridSearch: max_depth=3
        learning_rate     = 0.1,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        scale_pos_weight  = neg / pos, # handles class imbalance
        eval_metric       = "auc",
        use_label_encoder = False,
        random_state      = 42,
        n_jobs            = -1,
    )
    model.fit(X_train_scaled, y_train)
    print("  [OK] XGBoost training complete.")

    # ── Step 5 : Quick evaluation ─────────────────────────────────────────────
    _section("STEP 5: Evaluate on Test Set")
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    auc    = roc_auc_score(y_test, y_prob)
    print(f"  [RESULT] Test AUC-ROC : {auc:.4f}")
    if auc >= 0.75:
        print("  [OK] Model quality is VERY GOOD (>= 0.75).")
    elif auc >= 0.70:
        print("  [OK] Model quality is GOOD (>= 0.70).")
    else:
        print("  [!] Model quality is below 0.70 – revisit features.")

    # ── Step 6 : Serialize artefacts ─────────────────────────────────────────
    _section("STEP 6: Serialize Model, Scaler & Feature Names")
    joblib.dump(model,         "model.pkl")
    joblib.dump(scaler,        "scaler.pkl")
    joblib.dump(feature_names, "feature_names.pkl")
    print("  [SAVED] model.pkl")
    print("  [SAVED] scaler.pkl")
    print("  [SAVED] feature_names.pkl")

    # ── Final summary ─────────────────────────────────────────────────────────
    _section("DONE – Serialization Summary")
    print(f"""
  ┌─────────────────────────────────────────────────────┐
  │  Artefact          Size / Info                      │
  ├─────────────────────────────────────────────────────┤
  │  model.pkl         XGBClassifier (200 trees, d=3)   │
  │  scaler.pkl        StandardScaler ({len(feature_names)} features)         │
  │  feature_names.pkl {len(feature_names)} column names                      │
  │  Test AUC-ROC      {auc:.4f}                              │
  └─────────────────────────────────────────────────────┘

  Next step:  uvicorn main_api:app --reload
              Then open  http://127.0.0.1:8000/docs
    """)

    return model, scaler, feature_names, auc


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_and_save()
