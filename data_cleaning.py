import pandas as pd
from sqlalchemy import create_engine
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def clean_data_and_eda():
    # Step 1 – Connect Python to PostgreSQL
    print("Step 1: Connecting to PostgreSQL...")
    engine = create_engine("postgresql://postgres:Harshu%4012@localhost:5432/CreditpathAI")
    # Note: Ensure that PostgreSQL is running and the database details are correct
    df = pd.read_sql("SELECT * FROM loan_default", engine)

    # Step 2 – Inspect Dataset Structure
    print("\n--- Step 2: Inspect Dataset Structure ---")
    df.info()

    # Step 3 – Identify Missing Values
    print("\n--- Step 3: Missing Values ---")
    print(df.isnull().sum())
    if 'income' in df.columns:
        df['income'] = df['income'].fillna(df['income'].median())
        print("Filled missing values in 'income' with median.")

    # Step 4 – Detect Duplicate Records
    print("\n--- Step 4: Duplicate Records ---")
    duplicates = df.duplicated().sum()
    print(f"Found {duplicates} duplicate records. Removing them...")
    df = df.drop_duplicates()

    # Step 5 – Verify Column Datatypes
    print("\n--- Step 5: Column Datatypes ---")
    print(df.dtypes)

    # Step 6 – Remove Non-Predictive Columns
    print("\n--- Step 6: Removing Non-Predictive Columns ---")
    if 'loan_id' in df.columns:
        df = df.drop(columns=['loan_id'])
        print("Dropped 'loan_id'.")
    if 'loanid' in df.columns:
        df = df.drop(columns=['loanid'])
        print("Dropped 'loanid'.")

    # Step 7 – Detect Outliers in Numerical Data
    print("\n--- Step 7: Detecting Outliers ---")
    print(df.describe())
    if 'income' in df.columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=df['income'])
        plt.title('Income Distribution')
        plt.savefig('income_boxplot.png')
        print("Saved boxplot of 'income' to 'income_boxplot.png'.")
        plt.close()

    # Step 8 – Encode Categorical Variables
    print("\n--- Step 8: Encoding Categorical Variables ---")
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        print(f"Encoded variables: {list(categorical_cols)}")

    # Step 9 – Scale Numerical Features
    print("\n--- Step 9: Scaling Numerical Features ---")
    scaler = StandardScaler()
    cols_to_scale = ['income', 'loanamount', 'creditscore', 'loan_amount', 'credit_score']
    # Only scale columns that actually exist in the dataframe
    existing_cols_to_scale = [col for col in cols_to_scale if col in df.columns]
    
    if existing_cols_to_scale:
        df[existing_cols_to_scale] = scaler.fit_transform(df[existing_cols_to_scale])
        print(f"Scaled features: {existing_cols_to_scale}")

    # Step 10 – Save the Clean Dataset
    print("\n--- Step 10: Saving Clean Dataset ---")
    output_filename = "clean_loans.csv"
    df.to_csv(output_filename, index=False)
    print(f"Cleaned dataset saved successfully to {output_filename}!")

if __name__ == "__main__":
    clean_data_and_eda()