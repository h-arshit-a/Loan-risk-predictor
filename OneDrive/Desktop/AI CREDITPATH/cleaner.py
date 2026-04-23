import pandas as pd

def handle_missing_values(df):
    """
    Step 3: Identify Missing Values
    """
    print("Missing values before:\n", df.isnull().sum())
    if 'income' in df.columns:
        df['income'] = df['income'].fillna(df['income'].median())
        print("Filled missing values in 'income' with median.")
    return df

def remove_duplicates(df):
    """
    Step 4: Detect Duplicate Records
    """
    duplicates = df.duplicated().sum()
    print(f"Found {duplicates} duplicate records. Removing them...")
    df = df.drop_duplicates()
    return df

def remove_non_predictive(df, columns_to_remove):
    """
    Step 6: Remove Non-Predictive Columns
    """
    for col in columns_to_remove:
        if col in df.columns:
            df = df.drop(columns=[col])
            print(f"Dropped '{col}'.")
    return df
