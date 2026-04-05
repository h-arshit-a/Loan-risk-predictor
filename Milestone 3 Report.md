# CreditPath AI – Milestone 3 Report
## Baseline Model Development (Logistic Regression)

---

## 1. Objective

Train a Logistic Regression baseline model using engineered features derived from the cleaned loans dataset, and evaluate it using the AUC-ROC metric.

---

## 2. Files Delivered

| File | Role |
|---|---|
| `feature_engineering.py` | Feature engineering pipeline (load → engineer → split → scale) |
| `logistic_model.py` | Model training, evaluation, and coefficient interpretation |
| `milestone3_runner.py` | Single entry-point to run the full pipeline |

**How to run:**
```
python milestone3_runner.py
```

---

## 3. Dataset Overview

- **Source:** `clean_loans.csv` (produced in Milestone 2)
- **Rows:** 255,347
- **Original features:** 24 (9 numeric + 15 boolean encoded)
- **Target column:** `defaultstatus` (0 = No Default, 1 = Default)

---

## 4. Feature Engineering Pipeline

### Step 1 – Load Data
Read `clean_loans.csv` into a DataFrame; verify shape and column types.

### Step 2 – Engineer New Features

Four domain-derived features were added to improve signal quality:

| New Feature | Formula / Logic | Why It Helps |
|---|---|---|
| `debt_to_income_score` | `dtiratio × loanamount` | Combined stress indicator |
| `loan_per_month` | `loanamount / (loanterm + 1)` | Monthly repayment burden proxy |
| `credit_income_ratio` | `creditscore / (|income| + 1)` | Credit quality per unit of income |
| `high_risk_flag` | 1 if `dtiratio > 0.4` AND `creditscore < 0` | Binary risk alert |

### Step 3 – Train/Test Split
- 80% train / 20% test
- **Stratified split** to preserve class ratio

### Step 4 – Scaling (Mandatory)
`StandardScaler` fitted **only on training data**, then applied to test set — no data leakage.

---

## 5. Model

```
LogisticRegression(
    max_iter=1000,
    class_weight='balanced',   # handles class imbalance
    solver='lbfgs',
    random_state=42
)
```

`class_weight='balanced'` ensures the model doesn't ignore the minority (default) class.

---

## 6. Evaluation Results

### AUC-ROC Score

| Metric | Value |
|---|---|
| **AUC-ROC** | **0.7532** |
| Grade | **[OK] Very Good (>= 0.75)** |

### Classification Report

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| No Default (0) | 0.93 | 0.56 | 0.70 |
| Default (1) | 0.38 | 0.87 | 0.53 |
| **Weighted Avg** | **0.86** | **0.68** | **0.73** |

> **Note:** Accuracy (68%) is intentionally deprioritized. AUC-ROC is the correct metric for imbalanced credit risk classification.

### Plots Generated
- `roc_curve.png` – ROC Curve with AUC annotation
- `confusion_matrix.png` – Predicted vs Actual classes

---

## 7. Feature Coefficient Interpretation

Full table (ranked highest → lowest impact on default risk):

| Rank | Feature | Coefficient | Direction |
|---|---|---|---|
| 1 | `interestrate` | +0.4587 | Increases default risk |
| 2 | `loanamount` | +0.2960 | Increases default risk |
| 3 | `employmenttype_Unemployed` | +0.1899 | Increases default risk |
| 4 | `employmenttype_Part-time` | +0.1168 | Increases default risk |
| 5 | `employmenttype_Self-employed` | +0.1073 | Increases default risk |
| ... | ... | ... | ... |
| 24 | `monthsemployed` | -0.3373 | Decreases default risk |
| 25 | `income` | -0.3139 | Decreases default risk |
| 26 | `creditscore` | -0.1313 | Decreases default risk |
| 27 | `hascosigner_Yes` | -0.1292 | Decreases default risk |
| 28 | `age` | -0.5832 | Decreases default risk |

### Interpretation

**Top features INCREASING default risk:**
- **`interestrate` (+0.459):** Higher interest rates are assigned to riskier borrowers — a strong positive predictor of default.
- **`loanamount` (+0.296):** Larger loans increase default probability, likely due to higher repayment burden.
- **`employmenttype_Unemployed` (+0.190):** Unemployed borrowers have no stable income, significantly elevating default risk.
- **`employmenttype_Part-time` (+0.117):** Partial employment provides less financial stability than full-time.
- **`numcreditlines` (+0.102):** More open credit lines may indicate over-leveraging.

**Top features DECREASING default risk:**
- **`age` (-0.583):** Older borrowers tend to be more financially experienced and stable.
- **`monthsemployed` (-0.337):** Longer employment history signals stable income and lowers default risk.
- **`income` (-0.314):** Higher income directly reduces default likelihood.
- **`creditscore` (-0.131):** Higher credit scores reflect responsible past credit behavior.
- **`hascosigner_Yes` (-0.129):** Having a cosigner provides a financial safety net, reducing risk.

---

## 8. AUC-ROC Grading

| Threshold | Grade | Our Score |
|---|---|---|
| AUC >= 0.80 | Excellent | — |
| AUC >= 0.75 | **Very Good** | **0.7532 ✓** |
| AUC >= 0.70 | Good | — |
| AUC >= 0.65 | Acceptable | — |
| AUC < 0.65 | Needs improvement | — |

**Conclusion:** The baseline Logistic Regression model achieves an AUC-ROC of **0.7532**, meeting the "Very Good" threshold. The feature engineering pipeline (especially `interestrate`, `loanamount`, and employment type features) successfully captures meaningful credit risk signals.

---

## 9. Rules Compliance

| Rule | Status |
|---|---|
| No raw data used | PASS – uses `clean_loans.csv` from Milestone 2 |
| Scaling applied | PASS – `StandardScaler` applied before training |
| Accuracy not relied upon | PASS – AUC-ROC used as primary metric |
| No model files saved (pkl/joblib) | PASS – fully in-memory pipeline |

---

*Milestone 3 completed – 2026-03-23*
