import numpy as np
import pandas as pd
import joblib
import json
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


def evaluate(y_true, y_pred, label="Model"):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100

    print(f"\n{'─'*40}")
    print(f"  {label}")
    print(f"  MAE:  {mae:.2f} mg/dL")
    print(f"  RMSE: {rmse:.2f} mg/dL")
    print(f"  R²:   {r2:.4f}")
    print(f"  MAPE: {mape:.2f}%")

    return {
        "model": label,
        "MAE":   round(mae, 3),
        "RMSE":  round(rmse, 3),
        "R2":    round(r2, 4),
        "MAPE":  round(mape, 3),
    }


def get_models():
    return {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=42,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        ),
    }


def train_all(X_train, y_train, X_test, y_test):
    models = get_models()
    all_metrics = []
    trained_models = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = evaluate(y_test, y_pred, label=name)
        all_metrics.append(metrics)
        trained_models[name] = (model, y_pred)
        joblib.dump(model, f"models/{name}.pkl")
        print(f"  Saved to models/{name}.pkl")

    with open("reports/model_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print("\nAll metrics saved to reports/model_metrics.json")

    best = min(all_metrics, key=lambda m: m["MAE"])
    print(f"\n{'='*40}")
    print(f"  BEST MODEL: {best['model']}")
    print(f"  MAE = {best['MAE']} mg/dL  |  R² = {best['R2']}")

    return all_metrics, trained_models


def get_feature_importance(model, feature_cols):
    if hasattr(model, "feature_importances_"):
        return pd.DataFrame({
            "feature":    feature_cols,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)
    return pd.DataFrame()


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")
    from feature_engineering import split_train_test, scale_features, get_feature_columns

    df = pd.read_csv("data/processed/cgm_features.csv", parse_dates=["timestamp"])
    feature_cols = get_feature_columns(df)

    train, test = split_train_test(df, test_days=14)
    X_train, X_test, y_train, y_test = scale_features(train, test, feature_cols)

    all_metrics, trained_models = train_all(X_train, y_train, X_test, y_test)

    # Save best model predictions for later use
    best_name = min(all_metrics, key=lambda m: m["MAE"])["model"]
    _, y_pred_best = trained_models[best_name]
    np.save("reports/y_test.npy", y_test)
    np.save("reports/y_pred_best.npy", y_pred_best)
    pd.Series(test["timestamp"].values).to_csv("reports/test_timestamps.csv", index=False)

    # Save feature importance
    best_model, _ = trained_models[best_name]
    imp = get_feature_importance(best_model, feature_cols)
    if not imp.empty:
        imp.to_csv("reports/feature_importance.csv", index=False)
        print(f"Top 5 features: {imp.head(5)['feature'].tolist()}")