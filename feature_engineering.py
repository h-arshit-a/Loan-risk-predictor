"""
CreditPath AI – Milestone 3
Feature Engineering Pipeline
------------------------------------------------------
Loads clean_loans.csv, engineers new features, and
returns X (features) and y (target) ready for modelling.
No pickling – all transformations live here in memory.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# ── 1. Load cleaned dataset ──────────────────────────────────────────────────
def load_data(filepath: str = "clean_loans.csv") -> pd.DataFrame:
    """Read the cleaned loans CSV produced in Milestone 2."""
    df = pd.read_csv(filepath)
    print(f"[LOAD] Dataset shape: {df.shape}")
    return df


# ── 2. Engineer new features ──────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add domain-relevant derived features on top of the cleaned columns.

    New features:
      - debt_to_income_score : dtiratio × loanamount  (combined risk signal)
      - loan_per_month       : loanamount / (loanterm + 1)  (monthly burden)
      - credit_income_ratio  : creditscore / (income + 1)   (credit quality per income unit)
      - high_risk_flag       : 1 if dtiratio > 0.4 AND creditscore < 0  (z-score threshold)
    """
    df = df.copy()

    # Combined debt stress indicator
    df["debt_to_income_score"] = df["dtiratio"] * df["loanamount"]

    # Monthly repayment burden proxy
    df["loan_per_month"] = df["loanamount"] / (df["loanterm"] + 1)

    # Credit quality relative to income
    df["credit_income_ratio"] = df["creditscore"] / (df["income"].abs() + 1)

    # Binary high-risk flag: high DTI AND low credit score (both already z-scored)
    df["high_risk_flag"] = (
        (df["dtiratio"] > 0.4) & (df["creditscore"] < 0)
    ).astype(int)

    print(f"[ENGINEER] Shape after feature engineering: {df.shape}")
    print(f"[ENGINEER] New features: debt_to_income_score, loan_per_month, "
          f"credit_income_ratio, high_risk_flag")
    return df


# ── 3. Split features / target ────────────────────────────────────────────────
def split_X_y(df: pd.DataFrame, target: str = "defaultstatus"):
    """Separate feature matrix X from target vector y."""
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset.")

    X = df.drop(columns=[target])
    y = df[target]
    print(f"[SPLIT] Features: {X.shape[1]}  |  Target: '{target}'  |  Rows: {len(y)}")
    return X, y


# ── 4. Train / test split ─────────────────────────────────────────────────────
def get_train_test(X, y, test_size: float = 0.2, random_state: int = 42):
    """Stratified train-test split (preserves class ratio)."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    print(f"[SPLIT] Train size: {len(X_train)}  |  Test size: {len(X_test)}")
    print(f"[SPLIT] Default rate – Train: {y_train.mean():.3f}  |  Test: {y_test.mean():.3f}")
    return X_train, X_test, y_train, y_test


# ── 5. Scale numerical features ───────────────────────────────────────────────
def scale_features(X_train, X_test):
    """
    Fit StandardScaler on training data only, then transform both splits.
    Returns scaled numpy arrays and the fitted scaler object.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    print(f"[SCALE] Scaling complete – {X_train.shape[1]} features scaled.")
    return X_train_scaled, X_test_scaled, scaler


# ── 6. Full pipeline (convenience wrapper) ────────────────────────────────────
def build_features(filepath: str = "clean_loans.csv"):
    """
    End-to-end feature pipeline.
    Returns: X_train, X_test, y_train, y_test, feature_names, scaler
    """
    df = load_data(filepath)
    df = engineer_features(df)
    X, y = split_X_y(df)
    feature_names = X.columns.tolist()
    X_train, X_test, y_train, y_test = get_train_test(X, y)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names, scaler


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te, names, sc = build_features()
    print("\n[OK] Feature pipeline ran successfully.")
    print(f"     X_train shape : {X_tr.shape}")
    print(f"     X_test  shape : {X_te.shape}")
    print(f"     Features      : {names}")
