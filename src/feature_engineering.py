import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib


PREDICTION_HORIZON = 6   # 6 x 5min = 30 minutes ahead
LAG_STEPS = 24           # 24 x 5min = 2 hours of history


def build_features(df):
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    g = df["glucose_mg_dl"]

    # 1. Lag features: glucose at t-5, t-10, ... t-120 min
    for lag in range(1, LAG_STEPS + 1):
        df[f"lag_{lag}"] = g.shift(lag)

    # 2. Rolling statistics
    for w in [6, 12, 24]:
        df[f"roll_mean_{w}"] = g.shift(1).rolling(w).mean()
        df[f"roll_std_{w}"]  = g.shift(1).rolling(w).std()
        df[f"roll_min_{w}"]  = g.shift(1).rolling(w).min()
        df[f"roll_max_{w}"]  = g.shift(1).rolling(w).max()

    # 3. Rate of change
    df["velocity_1"]   = g.diff(1)
    df["velocity_2"]   = g.diff(2) / 2
    df["velocity_6"]   = g.diff(6) / 6
    df["acceleration"] = df["velocity_1"].diff(1)

    # 4. Trend slopes
    df["slope_6"]  = rolling_slope(g, window=6)
    df["slope_12"] = rolling_slope(g, window=12)

    # 5. Target: glucose 30 minutes from now
    df["target_glucose"] = g.shift(-PREDICTION_HORIZON)

    # Drop rows with any NaN
    df = df.dropna().reset_index(drop=True)

    return df


def rolling_slope(series, window):
    x = np.arange(window)
    slopes = series.rolling(window).apply(
        lambda y: np.polyfit(x, y, 1)[0], raw=True
    )
    return slopes


def get_feature_columns(df):
    exclude = {"timestamp", "patient_id", "device", "glucose_mg_dl", "target_glucose"}
    return [c for c in df.columns if c not in exclude]


def split_train_test(df, test_days=14):
    split_time = df["timestamp"].max() - pd.Timedelta(days=test_days)
    train = df[df["timestamp"] <= split_time].copy()
    test  = df[df["timestamp"] >  split_time].copy()
    print(f"Train: {len(train):,} samples")
    print(f"Test:  {len(test):,} samples")
    print(f"Train ends:   {train['timestamp'].max()}")
    print(f"Test starts:  {test['timestamp'].min()}")
    return train, test


def scale_features(train, test, feature_cols):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[feature_cols])
    X_test  = scaler.transform(test[feature_cols])
    y_train = train["target_glucose"].values
    y_test  = test["target_glucose"].values

    joblib.dump(scaler, "models/scaler.pkl")
    print("Scaler saved to models/scaler.pkl")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    df_clean = pd.read_csv("data/processed/cgm_clean.csv", parse_dates=["timestamp"])
    df_feat  = build_features(df_clean)
    df_feat.to_csv("data/processed/cgm_features.csv", index=False)

    feature_cols = get_feature_columns(df_feat)
    print(f"Total features: {len(feature_cols)}")
    print(f"Feature matrix shape: {df_feat.shape}")
    print(f"First 5 features: {feature_cols[:5]}")