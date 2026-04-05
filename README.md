# AI CreditPath – Data Cleaning & EDA Pipeline

This repository contains the **Milestone 2** work for the AI CreditPath project: a modular Python pipeline that connects to a PostgreSQL database, cleans the loan default dataset, performs exploratory data analysis, and exports a clean CSV ready for machine learning.

---

## Project Structure

| File | Description |
|------|-------------|
| `main.py` | Entry point – runs all steps in sequence |
| `data_cleaning.py` | Self-contained monolithic version of the full pipeline |
| `cleaner.py` | Functions: handle missing values, remove duplicates, drop non-predictive columns |
| `eda.py` | Functions: inspect dataset structure, visualise outliers (boxplots) |
| `encoder.py` | Function: one-hot encode categorical variables |
| `scaler.py` | Function: standardise numerical features with `StandardScaler` |
| `EDA Milestone 2 report.md` | Full written EDA report documenting each step |

---

## Requirements

```bash
pip install pandas psycopg2-binary sqlalchemy matplotlib seaborn scikit-learn
```

---

## How to Run

1. Make sure PostgreSQL is running and the `CreditpathAI` database is accessible.
2. Update the connection string in `main.py` if your credentials differ.
3. Run:

```bash
python main.py
```

The cleaned dataset will be saved as `clean_loans.csv` in the same directory.

---

## EDA Steps Performed (Milestone 2)

### Step 1 – Connect Python to PostgreSQL
Connected to the local PostgreSQL instance (`CreditpathAI`) using `psycopg2` and `SQLAlchemy`. The raw dataset `loan_default` (255,347 records, 18 columns) was loaded into a Pandas DataFrame.

### Step 2 – Inspect Dataset Structure
Used `df.info()` to examine column names, datatypes, and non-null counts. All 18 columns imported correctly with no structural issues.

**Dataset overview:**
- 255,347 rows, 18 columns
- 6 integer columns, 4 float columns, 8 object (string) columns

### Step 3 – Identify Missing Values
Checked with `df.isnull().sum()`. **No missing values** were found across any column. The `income` fillna logic remains in place as a safeguard for future data.

### Step 4 – Detect Duplicate Records
Used `df.duplicated().sum()`. **Zero duplicate records** were detected. No rows were removed.

### Step 5 – Verify Column Datatypes
Confirmed correct datatypes:
| Column | Type |
|--------|------|
| `age`, `creditscore`, `monthsemployed`, `numcreditlines`, `loanterm`, `defaultstatus` | int64 |
| `income`, `loanamount`, `interestrate`, `dtiratio` | float64 |
| `education`, `employmenttype`, `maritalstatus`, `hasmortgage`, `hasdependents`, `loanpurpose`, `hascosigner` | object |

### Step 6 – Remove Non-Predictive Columns
Dropped `loanid` (unique per-row identifier). Since every value is distinct, it cannot help a model learn patterns and only adds noise.

### Step 7 – Detect Outliers in Numerical Data
Ran `df.describe()` and generated a boxplot for `income`. Key statistics:
- **Income**: min 15,000 – max 149,999 (mean ≈ 82,499) — well distributed, no severe outliers
- **DTI Ratio**: 0.1 – 0.9, mean ≈ 0.50 — evenly spread
- **Default Status**: 11.6% positive class — mild class imbalance noted

Boxplot saved as `income_boxplot.png`.

### Step 8 – Encode Categorical Variables
Applied `pd.get_dummies(..., drop_first=True)` to convert all 7 object columns into numerical binary indicators. This avoids the dummy variable trap by dropping the first category.

Encoded columns: `education`, `employmenttype`, `maritalstatus`, `hasmortgage`, `hasdependents`, `loanpurpose`, `hascosigner`.

### Step 9 – Scale Numerical Features
Applied `sklearn.preprocessing.StandardScaler` to `income`, `loanamount`, and `creditscore`. Each feature is now zero-mean and unit-variance, preventing high-magnitude features from dominating gradient-based models.

### Step 10 – Save the Clean Dataset
Exported the final cleaned and encoded DataFrame to `clean_loans.csv`. The raw PostgreSQL data remains untouched and serves as the source of truth.

---

## Output

- **`clean_loans.csv`** – Cleaned, encoded, and scaled dataset ready for ML
- **`income_boxplot.png`** – Boxplot of income distribution

---

*Milestone 2 – AI CreditPath Project | Data Cleaning & EDA*
