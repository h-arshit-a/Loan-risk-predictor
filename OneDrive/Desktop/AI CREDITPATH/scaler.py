from sklearn.preprocessing import StandardScaler

def scale_numerical(df, cols_to_scale):
    """
    Step 9: Scale Numerical Features
    """
    scaler = StandardScaler()
    existing_cols_to_scale = [col for col in cols_to_scale if col in df.columns]
    
    if existing_cols_to_scale:
        df[existing_cols_to_scale] = scaler.fit_transform(df[existing_cols_to_scale])
        print(f"Scaled features: {existing_cols_to_scale}")
    else:
        print("No columns to scale.")
    return df
