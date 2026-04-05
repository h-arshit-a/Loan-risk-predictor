import pandas as pd
from sqlalchemy import create_engine

# Import modularized functions
from eda import inspect_structure, visualize_outliers
from cleaner import handle_missing_values, remove_duplicates, remove_non_predictive
from encoder import encode_categorical
from scaler import scale_numerical

def main():
    # Step 1: Connect Python to PostgreSQL
    print("Step 1: Connecting to PostgreSQL...")
    engine = create_engine("postgresql://postgres:Harshu%4012@localhost:5432/CreditpathAI")
    df = pd.read_sql("SELECT * FROM loan_default", engine)

    # Step 2: Inspect Dataset Structure
    print("\n--- Step 2: Inspect Dataset Structure ---")
    inspect_structure(df)

    # Step 3: Identify Missing Values
    print("\n--- Step 3: Missing Values ---")
    df = handle_missing_values(df)

    # Step 4: Detect Duplicate Records
    print("\n--- Step 4: Duplicate Records ---")
    df = remove_duplicates(df)

    # Step 5: Verify Column Datatypes
    print("\n--- Step 5: Column Datatypes ---")
    print(df.dtypes)

    # Step 6: Remove Non-Predictive Columns
    print("\n--- Step 6: Removing Non-Predictive Columns ---")
    df = remove_non_predictive(df, ['loanid', 'loan_id'])

    # Step 7: Detect Outliers in Numerical Data
    print("\n--- Step 7: Detecting Outliers ---")
    visualize_outliers(df, 'income')

    # Step 8: Encode Categorical Variables
    print("\n--- Step 8: Encoding Categorical Variables ---")
    df = encode_categorical(df)

    # Step 9: Scale Numerical Features
    print("\n--- Step 9: Scaling Numerical Features ---")
    df = scale_numerical(df, ['income', 'loanamount', 'creditscore', 'loan_amount', 'credit_score'])

    # Step 10: Save the Clean Dataset
    print("\n--- Step 10: Saving Clean Dataset ---")
    output_filename = "clean_loans.csv"
    df.to_csv(output_filename, index=False)
    print(f"Cleaned dataset saved successfully to {output_filename}!")

if __name__ == "__main__":
    main()
