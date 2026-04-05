# Data Cleaning and Exploratory Data Analysis (EDA)

This document summarizes the steps performed to establish a clean and robust dataset from an initial raw state stored in a PostgreSQL database.

## Step 1 – Connect Python to PostgreSQL
We connected to a local PostgreSQL instance (`creditpath` database) using `psycopg2` and `SQLAlchemy`. The raw dataset `loan_default` was then loaded into a Pandas DataFrame for downstream operations.

## Step 2 – Inspect Dataset Structure
Printed the structure (`df.info()`) to examine column names, datatypes, and the number of non-null values. This confirmed correct extraction and helped identify columns with missing data.

## Step 3 – Identify Missing Values
Detected columns containing missing data (`df.isnull().sum()`). For example, missing variations in the `income` column were imputed using the median of that column (`df['income'].fillna(df['income'].median())`) to ensure a complete dataset without biasing future predictive models.

## Step 4 – Detect Duplicate Records
Counted all identical duplicate rows using `df.duplicated().sum()`. Identified duplicates were subsequently deleted (`df.drop_duplicates()`) because they can skew statistical distributions and add unintended weights during training.

## Step 5 – Verify Column Datatypes
Displayed column datatypes (`df.dtypes`). We confirmed that variables match their logical types (e.g., `loan_amount` and `income` as floats, `age` and `credit_score` as integers, and demographics as objects). 

## Step 6 – Remove Non-Predictive Columns
Identified unneeded analytical features like `loan_id`. Using `df.drop()`, these columns were removed since unique constraints/identifiers add distinct noise that ML models cannot learn useful historical patterns from.

## Step 7 – Detect Outliers in Numerical Data
Exposed raw numeric summaries utilizing `df.describe()`. Furthermore, generated a graphical Boxplot (`income_boxplot.png` via Seaborn) over the `income` variable to visualize right/left tail variance and assess extreme bounds (outliers).

## Step 8 – Encode Categorical Variables
Extracted features storing contextual string logic (e.g., `education`, `employment_type`). Used Pandas `get_dummies(..., drop_first=True)` to convert these textual labels into numerical indicator variables (One-Hot Encoding). This makes categorical concepts readable to machine learning paradigms without risking collinearity among dummy variables.

## Step 9 – Scale Numerical Features
Applied Scikit-Learn's `StandardScaler` to align numerical magnitude. Predictors like `income`, `loan_amount`, and `credit_score` were scaled so they maintain uniformly stable, standardized variance, preventing arbitrarily massive scaled factors from manipulating model weight priority.

## Step 10 – Save the Clean Dataset
After sequentially applying the operations above, we serialized the completely cleansed data frame to an independent local format: `clean_loans.csv`. This separates our production-ready analytical input from the unadulterated raw PostgreSQL database.
