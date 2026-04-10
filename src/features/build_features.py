import pandas as pd


def _map_binary_series(s: pd.Series) -> pd.Series:
    vals = list(pd.Series(s.dropna().unique()).astype(str))
    valset = set(vals)

    if valset == {"Yes", "No"}:
        return s.map({"No": 0, "Yes": 1}).astype("Int64")

    if valset == {"Male", "Female"}:
        return s.map({"Female": 0, "Male": 1}).astype("Int64")

    if len(vals) == 2:
        sorted_vals = sorted(vals)
        mapping = {sorted_vals[0]: 0, sorted_vals[1]: 1}
        return s.astype(str).map(mapping).astype("Int64")

    return s


def build_features(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    df = df.copy()
    print(f"Starting feature engineering on {df.shape[1]} columns...")

    obj_cols = [c for c in df.select_dtypes(include=["object"]).columns if c != target_col]
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    print(f"  Found {len(obj_cols)} categorical and {len(numeric_cols)} numeric columns")

    binary_cols = [c for c in obj_cols if df[c].dropna().nunique() == 2]
    multi_cols = [c for c in obj_cols if df[c].dropna().nunique() > 2]

    for c in binary_cols:
        df[c] = _map_binary_series(df[c].astype(str))

    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype(int)

    if multi_cols:
        df = pd.get_dummies(df, columns=multi_cols, drop_first=True)

    for c in binary_cols:
        if pd.api.types.is_integer_dtype(df[c]):
            df[c] = df[c].fillna(0).astype(int)

    print(f"Feature engineering complete: {df.shape[1]} final features")
    return df
