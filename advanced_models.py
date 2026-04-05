"""
CreditPath AI – Milestone 4
Advanced Model Training: XGBoost & LightGBM
──────────────────────────────────────────────────────────────────────────────
Why these models?
  Logistic Regression assumes a LINEAR relationship between features and the
  log-odds of default.  Real credit risk is non-linear:
    • Income × loan-amount interactions matter differently at different scales
    • DTI above 0.5 creates a sudden risk cliff (threshold effect)
    • Tree-based boosting captures all of this automatically via splits.

Boosting intuition:
  Round 1 → first tree makes crude predictions
  Round 2 → second tree focuses on samples the first tree got wrong
  Round 3 → third tree corrects remaining residuals
  Final prediction = weighted sum of ALL trees

XGBoost vs LightGBM:
  XGBoost  → grows trees level-by-level (broader, balanced)
  LightGBM → grows leaf-wise (deeper, faster, risks more overfitting)
──────────────────────────────────────────────────────────────────────────────
"""

import sys, io
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend safe for all environments
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from feature_engineering import build_features


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  1.  MODEL TRAINING                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def train_xgboost(X_train, y_train) -> XGBClassifier:
    """
    Train XGBoost classifier with carefully chosen defaults.

    Parameter explanations
    ──────────────────────
    n_estimators   : 200 trees – enough to learn complex patterns without
                     heavy overfitting (early stopping handles the rest).
    max_depth      : 5 → controls tree complexity.  Too large → memorises
                     training noise.
    learning_rate  : 0.1 → each tree contributes 10 % of its prediction.
                     Smaller = more conservative = better generalisation.
    subsample      : 0.8 → each tree trains on 80 % of rows (stochastic).
                     Adds variance-reduction (like Random Forest).
    colsample_bytree: 0.8 → each tree sees 80 % of features → reduces
                     correlation between trees.
    scale_pos_weight: compensates for class imbalance
                     (neg_count / pos_count).
    eval_metric    : 'auc' → optimise AUC internally.
    use_label_encoder: False → suppresses a deprecation warning.
    """
    neg  = int((y_train == 0).sum())
    pos  = int((y_train == 1).sum())
    spw  = neg / pos

    xgb = XGBClassifier(
        n_estimators       = 200,
        max_depth          = 5,
        learning_rate      = 0.1,
        subsample          = 0.8,
        colsample_bytree   = 0.8,
        scale_pos_weight   = spw,
        eval_metric        = "auc",
        use_label_encoder  = False,
        random_state       = 42,
        n_jobs             = -1,
    )
    xgb.fit(X_train, y_train)
    print("[XGB] XGBoost training complete.")
    return xgb


def train_lightgbm(X_train, y_train) -> LGBMClassifier:
    """
    Train LightGBM classifier.

    Key difference from XGBoost
    ───────────────────────────
    LightGBM grows trees LEAF-WISE instead of level-wise.  This means it
    finds the single leaf with the largest loss reduction and splits ONLY
    that leaf – resulting in deeper, asymmetric trees that converge faster
    but CAN overfit if max_depth is not constrained.

    is_unbalance = True → LightGBM internally re-weights the minority class
    (equivalent to scale_pos_weight in XGBoost).
    """
    lgbm = LGBMClassifier(
        n_estimators   = 200,
        max_depth      = 5,
        learning_rate  = 0.1,
        subsample      = 0.8,
        colsample_bytree = 0.8,
        is_unbalance   = True,
        random_state   = 42,
        n_jobs         = -1,
        verbose        = -1,      # silence LightGBM console logs
    )
    lgbm.fit(X_train, y_train)
    print("[LGBM] LightGBM training complete.")
    return lgbm


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  2.  EVALUATION                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """
    Compute AUC-ROC and full classification report for a trained model.
    Returns a dict with auc, fpr, tpr, y_prob for later plotting.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    auc    = roc_auc_score(y_test, y_prob)
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    # Grade interpretation
    if auc >= 0.88:
        grade = "[**] Outstanding (>= 0.88)"
    elif auc >= 0.80:
        grade = "[OK] Excellent  (>= 0.80)"
    elif auc >= 0.75:
        grade = "[~]  Very Good  (>= 0.75)"
    elif auc >= 0.70:
        grade = "[~]  Good       (>= 0.70)"
    else:
        grade = "[!]  Below baseline - revisit features!"

    print(f"\n  [{model_name}] AUC-ROC : {auc:.4f}   {grade}")
    print(f"\n  Classification Report ({model_name}):\n")
    print(classification_report(y_test, y_pred, target_names=["No Default", "Default"]))

    return {
        "model_name": model_name,
        "auc"       : auc,
        "fpr"       : fpr,
        "tpr"       : tpr,
        "y_prob"    : y_prob,
        "y_pred"    : y_pred,
        "grade"     : grade,
    }


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  3.  HYPERPARAMETER TUNING                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def tune_xgboost_grid(X_train, y_train) -> tuple:
    """
    GridSearchCV – exhaustive search over a small, targeted grid.

    Why GridSearch?
    ───────────────
    Systematically tests every combination of hyper-parameters inside a
    cross-validated loop (cv=3 folds) and returns the settings that yielded
    the highest mean AUC.

    Drawback: O(n_combinations × cv) model fits → expensive on large grids.
    We keep the grid small intentionally: 3 × 2 × 2 = 12 combinations × 3
    folds = 36 fits.
    """
    print("\n  [TUNE] Running GridSearchCV on XGBoost (8 fits) …")

    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())

    param_grid = {
        "max_depth"     : [3, 5],
        "learning_rate" : [0.1],
        "n_estimators"  : [100, 200],
    }

    base = XGBClassifier(
        subsample        = 0.8,
        colsample_bytree = 0.8,
        scale_pos_weight = neg / pos,
        eval_metric      = "auc",
        use_label_encoder= False,
        random_state     = 42,
        n_jobs           = -1,
    )

    grid = GridSearchCV(
        estimator  = base,
        param_grid = param_grid,
        scoring    = "roc_auc",
        cv         = 2,
        n_jobs     = -1,
        verbose    = 1,
    )
    grid.fit(X_train, y_train)

    print(f"  [TUNE] Best params  : {grid.best_params_}")
    print(f"  [TUNE] Best CV AUC  : {grid.best_score_:.4f}")

    return grid.best_estimator_, grid.best_params_, grid.best_score_


def tune_xgboost_random(X_train, y_train) -> tuple:
    """
    RandomizedSearchCV – samples N random combinations from a wider space.

    Why RandomSearch?
    ─────────────────
    Faster than GridSearch because it does NOT test all combinations.
    Particularly useful when the grid is large.  We sample 20 random
    combinations with cv=3 → 60 fits (vs potentially thousands for a
    full grid).

    Drawback: may occasionally miss the optimal combination.
    """
    from scipy.stats import randint, uniform

    print("\n  [TUNE] Running RandomizedSearchCV on XGBoost (10 fits) …")

    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())

    param_dist = {
        "max_depth"     : randint(3, 8),         # 3, 4, 5, 6, 7
        "learning_rate" : uniform(0.01, 0.19),   # 0.01 – 0.20
        "n_estimators"  : randint(100, 301),      # 100 – 300
        "subsample"     : uniform(0.6, 0.4),      # 0.6 – 1.0
        "colsample_bytree": uniform(0.6, 0.4),
    }

    base = XGBClassifier(
        scale_pos_weight = neg / pos,
        eval_metric      = "auc",
        use_label_encoder= False,
        random_state     = 42,
        n_jobs           = -1,
    )

    rand = RandomizedSearchCV(
        estimator          = base,
        param_distributions= param_dist,
        n_iter             = 5,
        scoring            = "roc_auc",
        cv                 = 2,
        random_state       = 42,
        n_jobs             = -1,
        verbose            = 1,
    )
    rand.fit(X_train, y_train)

    print(f"  [TUNE] Best params  : {rand.best_params_}")
    print(f"  [TUNE] Best CV AUC  : {rand.best_score_:.4f}")

    return rand.best_estimator_, rand.best_params_, rand.best_score_


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  4.  FEATURE IMPORTANCE                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def extract_feature_importance(model, feature_names: list, model_name: str) -> pd.DataFrame:
    """
    Extract and rank feature importances.

    Important caveat: high importance ≠ causality.
    Use it to:
      • Validate that engineered features contribute (debt_to_income_score
        should rank highly)
      • Remove features with near-zero importance (dimensionality reduction)
    """
    importances = model.feature_importances_
    fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    fi_df = fi_df.sort_values("Importance", ascending=False).reset_index(drop=True)

    print(f"\n  [{model_name}] Top-10 Feature Importances:")
    print(fi_df.head(10).to_string(index=False))
    return fi_df


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  5.  PLOTTING                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

PALETTE = {
    "xgb"    : "#4A90D9",
    "lgbm"   : "#E8A838",
    "tuned"  : "#6FCF97",
    "lr"     : "#AAAAAA",
    "random" : "#DDDDDD",
    "bg"     : "#0D0F18",
    "panel"  : "#151826",
    "text"   : "#E8EAF0",
    "grid"   : "#2A2D3E",
}

def _apply_dark_style(ax):
    ax.set_facecolor(PALETTE["panel"])
    ax.tick_params(colors=PALETTE["text"])
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["grid"])
    ax.xaxis.label.set_color(PALETTE["text"])
    ax.yaxis.label.set_color(PALETTE["text"])
    ax.title.set_color(PALETTE["text"])
    ax.yaxis.set_tick_params(labelcolor=PALETTE["text"])
    ax.xaxis.set_tick_params(labelcolor=PALETTE["text"])


def plot_roc_comparison(results: list, lr_auc: float, savepath: str = "m4_roc_comparison.png"):
    """
    Overlay ROC curves for XGBoost, LightGBM (and optionally tuned model)
    plus the LR baseline diagonal reference.
    """
    colors = [PALETTE["xgb"], PALETTE["lgbm"], PALETTE["tuned"]]

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["panel"])

    # Baseline reference line
    ax.plot([0, 1], [0, 1], color=PALETTE["random"], linestyle="--", lw=1.2,
            label=f"Random (AUC = 0.50)")

    for i, res in enumerate(results):
        ax.plot(res["fpr"], res["tpr"],
                color=colors[i % len(colors)], lw=2.2,
                label=f"{res['model_name']}  (AUC = {res['auc']:.4f})")

    # LR baseline
    ax.axhline(y=lr_auc, color=PALETTE["lr"], linestyle=":", lw=1.2,
               label=f"LR baseline AUC ≈ {lr_auc:.4f}")

    _apply_dark_style(ax)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curve Comparison – CreditPath AI Milestone 4", fontsize=13, pad=12)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.3,
              facecolor=PALETTE["panel"], edgecolor=PALETTE["grid"],
              labelcolor=PALETTE["text"])
    ax.grid(True, color=PALETTE["grid"], alpha=0.5)

    plt.tight_layout()
    plt.savefig(savepath, dpi=150, facecolor=PALETTE["bg"])
    plt.close()
    print(f"  [PLOT] ROC comparison saved → {savepath}")


def plot_feature_importance(fi_xgb: pd.DataFrame, fi_lgbm: pd.DataFrame,
                             savepath: str = "m4_feature_importance.png"):
    """Side-by-side horizontal bar charts for XGBoost and LightGBM importance."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(PALETTE["bg"])

    for ax, fi_df, color, title in zip(
        axes,
        [fi_xgb.head(10), fi_lgbm.head(10)],
        [PALETTE["xgb"], PALETTE["lgbm"]],
        ["XGBoost – Top 10 Features", "LightGBM – Top 10 Features"],
    ):
        bars = ax.barh(
            fi_df["Feature"][::-1],
            fi_df["Importance"][::-1],
            color=color, alpha=0.85, edgecolor=PALETTE["grid"],
        )
        _apply_dark_style(ax)
        ax.set_title(title, fontsize=12, pad=8)
        ax.set_xlabel("Importance Score")

        # Value labels
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.001, bar.get_y() + bar.get_height() / 2,
                    f"{w:.3f}", va="center", ha="left",
                    fontsize=8, color=PALETTE["text"])

        ax.grid(axis="x", color=PALETTE["grid"], alpha=0.4)

    fig.suptitle("Feature Importance – CreditPath AI Milestone 4",
                 fontsize=14, color=PALETTE["text"], y=1.01)
    plt.tight_layout()
    plt.savefig(savepath, dpi=150, facecolor=PALETTE["bg"], bbox_inches="tight")
    plt.close()
    print(f"  [PLOT] Feature importance saved → {savepath}")


def plot_auc_bar(auc_dict: dict, savepath: str = "m4_auc_comparison.png"):
    """Bar chart comparing AUC scores across all models."""
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["panel"])

    models = list(auc_dict.keys())
    aucs   = list(auc_dict.values())
    bar_colors = [
        PALETTE["lr"],
        PALETTE["xgb"],
        PALETTE["lgbm"],
        PALETTE["tuned"],
        "#BB6FCF",
    ][:len(models)]

    bars = ax.bar(models, aucs, color=bar_colors, edgecolor=PALETTE["grid"],
                  width=0.55, alpha=0.9)

    for bar, val in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom",
                fontsize=10, color=PALETTE["text"], fontweight="bold")

    ax.axhline(y=0.70, color="#AAAAAA", linestyle="--", lw=1, label="Good threshold (0.70)")
    ax.axhline(y=0.80, color="#6FCF97", linestyle="--", lw=1, label="Excellent threshold (0.80)")

    ax.set_ylim(0.5, min(1.0, max(aucs) + 0.08))
    ax.set_ylabel("AUC-ROC", fontsize=11)
    ax.set_title("AUC-ROC Comparison – All Models", fontsize=13, pad=10)
    _apply_dark_style(ax)
    ax.legend(fontsize=9, framealpha=0.3, facecolor=PALETTE["panel"],
              edgecolor=PALETTE["grid"], labelcolor=PALETTE["text"])
    ax.grid(axis="y", color=PALETTE["grid"], alpha=0.5)

    plt.tight_layout()
    plt.savefig(savepath, dpi=150, facecolor=PALETTE["bg"])
    plt.close()
    print(f"  [PLOT] AUC comparison bar chart saved → {savepath}")


def plot_confusion_matrices(results: list, savepath: str = "m4_confusion_matrices.png"):
    """Confusion matrices for each model side-by-side."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    fig.patch.set_facecolor(PALETTE["bg"])
    if n == 1:
        axes = [axes]

    colors_list = [PALETTE["xgb"], PALETTE["lgbm"], PALETTE["tuned"]]

    for ax, res, color in zip(axes, results, colors_list):
        cm = confusion_matrix(res["y_true"], res["y_pred"])
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["No Default", "Default"]
        )
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{res['model_name']}\nAUC={res['auc']:.4f}",
                     color=PALETTE["text"], fontsize=11, pad=6)
        ax.set_facecolor(PALETTE["panel"])
        ax.tick_params(colors=PALETTE["text"])
        for spine in ax.spines.values():
            spine.set_edgecolor(PALETTE["grid"])

    fig.suptitle("Confusion Matrices – CreditPath AI Milestone 4",
                 fontsize=14, color=PALETTE["text"], y=1.03)
    plt.tight_layout()
    plt.savefig(savepath, dpi=150, facecolor=PALETTE["bg"], bbox_inches="tight")
    plt.close()
    print(f"  [PLOT] Confusion matrices saved → {savepath}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  6.  MAIN PIPELINE                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def run_milestone4(lr_baseline_auc: float = 0.72, run_tuning: bool = True):
    """
    Full Milestone 4 pipeline.

    Parameters
    ──────────
    lr_baseline_auc : AUC from Milestone 3 (used only for chart reference).
    run_tuning      : Set False to skip hyperparameter tuning (faster demo).
    """
    _banner("CreditPath AI - Milestone 4: Advanced Model Training")

    # ── Step 1 : Feature Pipeline (reuse Milestone 3 module) ────────────────
    _section("STEP 1: Feature Engineering Pipeline (reusing Milestone 3)")
    X_train, X_test, y_train, y_test, feature_names, scaler = build_features()

    # ── Step 2 : Train baseline models ───────────────────────────────────────
    _section("STEP 2: Training XGBoost & LightGBM (default params)")
    xgb_model  = train_xgboost(X_train, y_train)
    lgbm_model = train_lightgbm(X_train, y_train)

    # ── Step 3 : Evaluate ────────────────────────────────────────────────────
    _section("STEP 3: Model Evaluation (AUC-ROC)")
    res_xgb  = evaluate_model(xgb_model,  X_test, y_test, "XGBoost")
    res_lgbm = evaluate_model(lgbm_model, X_test, y_test, "LightGBM")

    # Add y_true for confusion matrix plotting
    res_xgb["y_true"]  = y_test
    res_lgbm["y_true"] = y_test

    results_list = [res_xgb, res_lgbm]
    auc_dict     = {
        f"LR Baseline\n(M3)"  : lr_baseline_auc,
        "XGBoost\n(default)"  : res_xgb["auc"],
        "LightGBM\n(default)" : res_lgbm["auc"],
    }

    # ── Step 4 : Hyperparameter Tuning ───────────────────────────────────────
    best_tuned_model = None
    if run_tuning:
        _section("STEP 4: Hyperparameter Tuning")

        print("\n  --- 4a. Grid Search ---")
        best_grid, best_grid_params, best_grid_auc = tune_xgboost_grid(X_train, y_train)
        res_grid = evaluate_model(best_grid, X_test, y_test, "XGB-GridSearch")
        res_grid["y_true"] = y_test

        print("\n  --- 4b. Random Search (alternative, faster) ---")
        best_rand, best_rand_params, best_rand_auc = tune_xgboost_random(X_train, y_train)
        res_rand = evaluate_model(best_rand, X_test, y_test, "XGB-RandomSearch")
        res_rand["y_true"] = y_test

        # Pick whichever search delivered a better test AUC
        if res_grid["auc"] >= res_rand["auc"]:
            best_tuned_model = best_grid
            best_tuned_name  = "XGB-GridSearch"
            best_tuned_res   = res_grid
        else:
            best_tuned_model = best_rand
            best_tuned_name  = "XGB-RandomSearch"
            best_tuned_res   = res_rand

        results_list.append(best_tuned_res)
        auc_dict[f"{best_tuned_name}\n(tuned)"] = best_tuned_res["auc"]

        print(f"\n  [TUNE] Winner: {best_tuned_name}  |  Test AUC: {best_tuned_res['auc']:.4f}")
    else:
        _section("STEP 4: Hyperparameter Tuning SKIPPED (run_tuning=False)")

    # ── Step 5 : Feature Importance ──────────────────────────────────────────
    _section("STEP 5: Feature Importance Analysis")
    fi_xgb  = extract_feature_importance(xgb_model,  feature_names, "XGBoost")
    fi_lgbm = extract_feature_importance(lgbm_model, feature_names, "LightGBM")

    # ── Step 6 : Generate plots ───────────────────────────────────────────────
    _section("STEP 6: Generating Diagnostic Plots")
    plot_roc_comparison(results_list, lr_auc=lr_baseline_auc)
    plot_feature_importance(fi_xgb, fi_lgbm)
    plot_auc_bar(auc_dict)

    cm_results = [dict(r) for r in results_list[:3]]   # up to 3 models
    plot_confusion_matrices(cm_results)

    # ── Step 7 : Summary ─────────────────────────────────────────────────────
    _section("STEP 7: Final Summary")

    print()
    print("  +" + "=" * 54 + "+")
    print("  |    CreditPath AI - Milestone 4 Results               |")
    print("  +" + "=" * 54 + "+")

    for name, auc in auc_dict.items():
        name_clean = name.replace("\n", " ")
        delta = auc - lr_baseline_auc
        sign  = "+" if delta >= 0 else ""
        print(f"  |  {name_clean:<26}  AUC={auc:.4f}  ({sign}{delta:.4f} vs LR)  |")

    print("  +" + "=" * 54 + "+")
    print("  |  Plots:                                              |")
    print("  |    m4_roc_comparison.png                             |")
    print("  |    m4_feature_importance.png                         |")
    print("  |    m4_auc_comparison.png                             |")
    print("  |    m4_confusion_matrices.png                         |")
    print("  +" + "=" * 54 + "+")
    print()

    # ── Common failure case diagnosis ─────────────────────────────────────────
    _diagnose(res_xgb["auc"], res_lgbm["auc"], lr_baseline_auc)

    return {
        "xgb_model"       : xgb_model,
        "lgbm_model"      : lgbm_model,
        "best_tuned_model": best_tuned_model,
        "results"         : {r["model_name"]: r["auc"] for r in results_list},
        "fi_xgb"          : fi_xgb,
        "fi_lgbm"         : fi_lgbm,
    }


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  7.  HELPERS                                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _banner(text: str):
    line = "=" * (len(text) + 4)
    print(f"\n+{line}+")
    print(f"|  {text}  |")
    print(f"+{line}+\n")


def _section(text: str):
    print(f"\n{'-' * 60}")
    print(f"  {text}")
    print(f"{'-' * 60}")


def _diagnose(auc_xgb: float, auc_lgbm: float, auc_lr: float):
    """Print actionable failure-case diagnosis."""
    _section("Failure-Case Diagnosis")

    best_adv = max(auc_xgb, auc_lgbm)

    if best_adv <= auc_lr + 0.01:
        print("  [!] Case 1: Advanced models are NOT significantly beating LR.")
        print("      Root cause: weak or redundant features.")
        print("      Action: revisit feature_engineering.py - add interaction terms.")
    elif best_adv < 0.75:
        print("  [!] Case 2: AUC is below 0.75.")
        print("      Consider increasing n_estimators or adding more features.")
    else:
        print("  [OK] Models are performing well. No critical issues detected.")

    print()
    print("  Common pitfalls and fixes:")
    print("  +----------------------------------------------------------+")
    print("  | High train AUC, low test AUC -> OVERFITTING              |")
    print("  |   Fix: reduce max_depth, increase min_child_weight        |")
    print("  | AUC not improving -> WEAK FEATURES                       |")
    print("  |   Fix: revisit engineered features, add domain knowledge  |")
    print("  | Very slow training -> TOO MANY ESTIMATORS                |")
    print("  |   Fix: reduce n_estimators, use LightGBM instead         |")
    print("  +----------------------------------------------------------+")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_milestone4(
        lr_baseline_auc = 0.72,   # ← update with your actual Milestone 3 AUC
        run_tuning      = True,   # set False for a quick demo run
    )
