import numpy as np
import pandas as pd


def load_and_clean(filepath):
    df = pd.read_csv(filepath, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"[1] Loaded {len(df):,} rows")

    df = enforce_5min_grid(df)
    print(f"[2] After resampling to 5-min grid: {len(df):,} rows")

    missing_before = df["glucose_mg_dl"].isna().sum()
    df = impute_missing(df)
    print(f"[3] Imputed {missing_before} missing values")

    outliers = detect_outliers(df)
    df.loc[outliers, "glucose_mg_dl"] = np.nan
    df = impute_missing(df)
    print(f"[4] Removed {outliers.sum()} outliers")

    df = add_derived_columns(df)
    print(f"[5] Added time columns")

    return df


def enforce_5min_grid(df):
    df = df.set_index("timestamp")
    df_resampled = df["glucose_mg_dl"].resample("5min").mean()
    df_out = df_resampled.reset_index()
    df_out.columns = ["timestamp", "glucose_mg_dl"]
    df_out["patient_id"] = df["patient_id"].iloc[0]
    df_out["device"] = df["device"].iloc[0]
    return df_out


def impute_missing(df, max_gap_minutes=30):
    df = df.copy()
    glucose = df["glucose_mg_dl"].copy()

    is_nan = glucose.isna()
    nan_groups = (is_nan != is_nan.shift()).cumsum()
    gap_sizes = is_nan.groupby(nan_groups).transform("sum")
    max_gap_steps = max_gap_minutes // 5

    interpolated = glucose.interpolate(method="linear", limit_direction="both")
    glucose_clean = interpolated.copy()
    glucose_clean[is_nan & (gap_sizes > max_gap_steps)] = np.nan

    df["glucose_mg_dl"] = glucose_clean
    return df


def detect_outliers(df):
    g = df["glucose_mg_dl"]
    out_of_range = (g < 39) | (g > 401)
    impossible_delta = g.diff().abs() > 30
    return out_of_range | impossible_delta


def add_derived_columns(df):
    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["minute"] = df["timestamp"].dt.minute
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["time_of_day_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["time_of_day_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    return df


if __name__ == "__main__":
    df = load_and_clean("data/raw/cgm_raw.csv")
    df.to_csv("data/processed/cgm_clean.csv", index=False)
    print(f"\nCleaned data saved — shape: {df.shape}")
    print(f"Remaining NaNs: {df['glucose_mg_dl'].isna().sum()}")