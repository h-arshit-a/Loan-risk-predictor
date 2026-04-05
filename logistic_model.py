"""
CreditPath AI – Milestone 3
Logistic Regression Baseline Model
------------------------------------------------------
Trains a Logistic Regression classifier on engineered
features, evaluates with AUC-ROC, and prints coefficient
interpretation.  No model files are saved (in-memory only).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

from feature_engineering import build_features


# ── 1. Train the model ────────────────────────────────────────────────────────
def train_model(X_train, y_train) -> LogisticRegression:
    """
    Fit Logistic Regression with balanced class weights
    to handle the class imbalance common in credit datasets.
    """
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    )
    model.fit(X_train, y_train)
    print("[TRAIN] Logistic Regression training complete.")
    return model


# ── 2. Evaluate the model ─────────────────────────────────────────────────────
def evaluate_model(model, X_test, y_test, feature_names):
    """
    Print full evaluation metrics and save diagnostic plots.
    Returns the AUC-ROC score.
    """
    y_pred  = model.predict(X_test)
    y_prob  = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)

    print("\n" + "=" * 55)
    print("  CreditPath AI – Baseline Model Evaluation Results")
    print("=" * 55)
    print(f"\n  AUC-ROC Score : {auc:.4f}", end="  ")

    # Grade the result
    if auc >= 0.80:
        print("★ Excellent (≥ 0.80)")
    elif auc >= 0.75:
        print("✓ Very Good (≥ 0.75)")
    elif auc >= 0.70:
        print("~ Good (≥ 0.70)")
    elif auc >= 0.65:
        print("△ Acceptable – consider more feature engineering")
    else:
        print("✗ Below threshold – revisit feature pipeline!")

    print("\n  Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=["No Default", "Default"]))

    # ROC Curve plot
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color="#4A90D9", lw=2,
             label=f"Logistic Regression (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], color="#AAAAAA", linestyle="--", lw=1, label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve – CreditPath AI Baseline Model")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("roc_curve.png", dpi=120)
    plt.close()
    print("  [PLOT] ROC curve saved → roc_curve.png")

    # Confusion Matrix plot
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["No Default", "Default"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix – CreditPath AI Baseline")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=120)
    plt.close()
    print("  [PLOT] Confusion matrix saved → confusion_matrix.png")

    return auc


# ── 3. Interpret coefficients ─────────────────────────────────────────────────
def interpret_coefficients(model, feature_names):
    """
    Print the model's coefficients ranked from highest to lowest.
    Positive coefficient → feature increases default risk.
    Negative coefficient → feature decreases default risk.
    """
    coeff_df = pd.DataFrame({
        "Feature"    : feature_names,
        "Coefficient": model.coef_[0],
    }).sort_values(by="Coefficient", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 55)
    print("  Feature Coefficients (Impact on Default Risk)")
    print("=" * 55)

    print("\n  ▲ TOP 5 features INCREASING default risk:")
    print(coeff_df.head(5).to_string(index=False))

    print("\n  ▼ TOP 5 features DECREASING default risk:")
    print(coeff_df.tail(5).to_string(index=False))

    print("\n  Full coefficient table:")
    print(coeff_df.to_string(index=False))

    return coeff_df


# ── 4. Main orchestrator ──────────────────────────────────────────────────────
def run_baseline():
    """
    Full Milestone 3 pipeline:
      1. Build engineered features
      2. Train Logistic Regression
      3. Evaluate (AUC-ROC + plots)
      4. Interpret coefficients
    """
    print("\n" + "=" * 55)
    print("  CreditPath AI – Milestone 3: Baseline Model")
    print("=" * 55 + "\n")

    # Step 1 – Feature pipeline
    print("--- STEP 1: Feature Engineering Pipeline ---")
    X_train, X_test, y_train, y_test, feature_names, scaler = build_features()

    # Step 2 – Train
    print("\n--- STEP 2: Model Training ---")
    model = train_model(X_train, y_train)

    # Step 3 – Evaluate
    print("\n--- STEP 3: Evaluation ---")
    auc = evaluate_model(model, X_test, y_test, feature_names)

    # Step 4 – Interpret
    print("\n--- STEP 4: Coefficient Interpretation ---")
    coeff_df = interpret_coefficients(model, feature_names)

    print("\n" + "=" * 55)
    print(f"  Milestone 3 Complete  |  Final AUC-ROC: {auc:.4f}")
    print("=" * 55)

    return model, auc, coeff_df


if __name__ == "__main__":
    run_baseline()
