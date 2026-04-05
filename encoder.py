import pandas as pd

def encode_categorical(df):
    """
    Step 8: Encode Categorical Variables
    """
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        print(f"Encoded variables: {list(categorical_cols)}")
    else:
        print("No categorical columns to encode.")
    return df
