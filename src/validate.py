EXPECTED_COLUMNS = ["feature1", "feature2", "feature3", "target"]

def validate_data(df):
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    if df[EXPECTED_COLUMNS].isnull().mean().max() > 0.2:
        raise ValueError("Too many missing values")

    return True
