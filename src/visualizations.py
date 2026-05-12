import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import json

plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#1a1d27",
    "axes.edgecolor":   "#3a3d4d",
    "axes.labelcolor":  "#c8cad4",
    "text.color":       "#c8cad4",
    "xtick.color":      "#8a8d9a",
    "ytick.color":      "#8a8d9a",
    "grid.color":       "#2a2d3a",
    "grid.linewidth":   0.6,
    "axes.spines.top":  False,
    "axes.spines.right":False,
})


def shade_zones(ax):
    ax.axhspan(40,  70,  alpha=0.08, color="#ef4444")
    ax.axhspan(70,  180, alpha=0.06, color="#22c55e")
    ax.axhspan(180, 400, alpha=0.08, color="#f59e0b")
    ax.axhline(70,  color="#ef4444", lw=0.8, ls="--", alpha=0.5)
    ax.axhline(180, color="#f59e0b", lw=0.8, ls="--", alpha=0.5)


def plot_raw_overview(df, save=True):
    fig, axes = plt.subplots(3, 1, figsize=(16, 10),
                             gridspec_kw={"height_ratios": [3, 1, 1]})
    fig.suptitle("CGM Data Overview — 90-Day Raw Readings",
                 fontsize=16, fontweight="bold", color="white")

    # Panel 1: time series
    ax = axes[0]
    subset = df.iloc[:1000]
    ax.plot(subset["timestamp"], subset["glucose_mg_dl"],
            color="#60a5fa", lw=1.0, alpha=0.9)
    shade_zones(ax)
    ax.set_ylabel("Glucose (mg/dL)")
    ax.set_title("First 3.5 Days of Readings")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d %H:%M"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)

    # Panel 2: distribution
    ax2 = axes[1]
    vals = df["glucose_mg_dl"].dropna()
    ax2.hist(vals, bins=80, color="#60a5fa", alpha=0.8, edgecolor="none")
    ax2.axvline(70,  color="#ef4444", lw=1.5, ls="--", label="Low <70")
    ax2.axvline(180, color="#f59e0b", lw=1.5, ls="--", label="High >180")
    ax2.set_xlabel("Glucose (mg/dL)")
    ax2.set_ylabel("Count")
    ax2.set_title("Distribution of All Readings")
    ax2.legend()

    # Panel 3: time-in-range
    ax3 = axes[2]
    low    = (vals < 70).mean() * 100
    target = ((vals >= 70) & (vals <= 180)).mean() * 100
    high   = (vals > 180).mean() * 100
    ax3.barh([""], [low, target, high],
             color=["#ef4444", "#22c55e", "#f59e0b"], height=0.5)
    ax3.set_xlim(0, 100)
    ax3.set_xlabel("Time (%)")
    ax3.set_title(f"Time-in-Range  |  Low: {low:.1f}%  Target: {target:.1f}%  High: {high:.1f}%")
    ax3.set_yticks([])

    plt.tight_layout()
    if save:
        plt.savefig("visualizations/01_raw_overview.png", dpi=150, bbox_inches="tight")
        print("Saved: visualizations/01_raw_overview.png")
    plt.close()


def plot_model_comparison(save=True):
    with open("reports/model_metrics.json") as f:
        metrics = json.load(f)

    df = pd.DataFrame(metrics)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Model Comparison", fontsize=14, fontweight="bold", color="white")

    palette = ["#60a5fa", "#f472b6", "#34d399", "#fbbf24"]

    for ax, col, title in zip(
        axes,
        ["MAE", "RMSE", "R2"],
        ["MAE (mg/dL) — lower is better",
         "RMSE (mg/dL) — lower is better",
         "R² Score — higher is better"]
    ):
        bars = ax.bar(df["model"], df[col], color=palette,
                      edgecolor="none", alpha=0.85)
        best_idx = df[col].idxmin() if col != "R2" else df[col].idxmax()
        bars[best_idx].set_edgecolor("white")
        bars[best_idx].set_linewidth(2)
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", rotation=20, labelsize=9)
        ax.grid(axis="y", alpha=0.4)

    plt.tight_layout()
    if save:
        plt.savefig("visualizations/02_model_comparison.png", dpi=150, bbox_inches="tight")
        print("Saved: visualizations/02_model_comparison.png")
    plt.close()


def plot_actual_vs_predicted(y_true, y_pred, model_name="RandomForest",
                              timestamps=None, save=True):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"{model_name} — Actual vs Predicted (30-min ahead)",
                 fontsize=14, fontweight="bold", color="white")

    # Left: time series
    ax1 = axes[0]
    n = min(500, len(y_true))
    x = pd.to_datetime(timestamps[:n]) if timestamps is not None else range(n)
    ax1.plot(x, y_true[:n], color="#60a5fa", lw=1.2, label="Actual")
    ax1.plot(x, y_pred[:n], color="#f472b6", lw=1.2, ls="--", label="Predicted")
    shade_zones(ax1)
    ax1.set_ylabel("Glucose (mg/dL)")
    ax1.set_title("Time Series — First 500 Test Points")
    ax1.legend()
    if timestamps is not None:
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30)

    # Right: scatter
    ax2 = axes[1]
    errors = np.abs(y_pred - y_true)
    sc = ax2.scatter(y_true, y_pred, c=errors, cmap="plasma",
                     alpha=0.3, s=8, edgecolors="none")
    lim = [min(y_true.min(), y_pred.min()) - 5,
           max(y_true.max(), y_pred.max()) + 5]
    ax2.plot(lim, lim, color="white", lw=1, ls="--", alpha=0.5, label="Perfect")
    ax2.set_xlim(lim); ax2.set_ylim(lim)
    ax2.set_xlabel("Actual (mg/dL)")
    ax2.set_ylabel("Predicted (mg/dL)")
    ax2.set_title("Scatter: Actual vs Predicted")
    plt.colorbar(sc, ax=ax2, label="|Error| (mg/dL)")
    ax2.legend()

    plt.tight_layout()
    if save:
        plt.savefig("visualizations/03_actual_vs_predicted.png", dpi=150, bbox_inches="tight")
        print("Saved: visualizations/03_actual_vs_predicted.png")
    plt.close()


def plot_feature_importance(importance_df, model_name="RandomForest", save=True):
    df = importance_df.head(20)
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle(f"{model_name} — Top 20 Feature Importances",
                 fontsize=13, fontweight="bold", color="white")

    colors = plt.cm.plasma(np.linspace(0.3, 0.9, len(df)))[::-1]
    ax.barh(df["feature"][::-1], df["importance"][::-1],
            color=colors, edgecolor="none", alpha=0.9)
    ax.set_xlabel("Importance Score")
    ax.grid(axis="x", alpha=0.4)

    plt.tight_layout()
    if save:
        plt.savefig("visualizations/04_feature_importance.png", dpi=150, bbox_inches="tight")
        print("Saved: visualizations/04_feature_importance.png")
    plt.close()


def plot_error_distribution(y_true, y_pred, model_name="RandomForest", save=True):
    errors = y_pred - y_true
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{model_name} — Prediction Error Analysis",
                 fontsize=13, fontweight="bold", color="white")

    ax1 = axes[0]
    ax1.hist(errors, bins=80, color="#60a5fa", alpha=0.8, edgecolor="none")
    ax1.axvline(0, color="white", lw=1.5, ls="--")
    ax1.axvline(errors.mean(), color="#f472b6", lw=1.5,
                label=f"Mean = {errors.mean():.2f}")
    ax1.set_xlabel("Prediction Error (mg/dL)")
    ax1.set_ylabel("Count")
    ax1.set_title("Error Distribution")
    ax1.legend()

    ax2 = axes[1]
    bins = np.arange(40, 401, 20)
    bin_idx = np.digitize(y_true, bins)
    bin_mae = [np.mean(np.abs(errors[bin_idx == i]))
               if (bin_idx == i).sum() > 0 else 0
               for i in range(1, len(bins))]
    ax2.bar(bins[:-1], bin_mae, width=18, color="#34d399",
            edgecolor="none", alpha=0.85)
    ax2.axvline(70,  color="#ef4444", lw=1, ls="--")
    ax2.axvline(180, color="#f59e0b", lw=1, ls="--")
    ax2.set_xlabel("True Glucose (mg/dL)")
    ax2.set_ylabel("Mean Absolute Error (mg/dL)")
    ax2.set_title("MAE by Glucose Range")
    ax2.grid(axis="y", alpha=0.4)

    plt.tight_layout()
    if save:
        plt.savefig("visualizations/05_error_distribution.png", dpi=150, bbox_inches="tight")
        print("Saved: visualizations/05_error_distribution.png")
    plt.close()


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")
    from feature_engineering import get_feature_columns

    df_raw  = pd.read_csv("data/raw/cgm_raw.csv", parse_dates=["timestamp"])
    y_test  = np.load("reports/y_test.npy")
    y_pred  = np.load("reports/y_pred_best.npy")
    ts      = pd.read_csv("reports/test_timestamps.csv").iloc[:, 0].values
    imp_df  = pd.read_csv("reports/feature_importance.csv")

    plot_raw_overview(df_raw)
    plot_model_comparison()
    plot_actual_vs_predicted(y_test, y_pred, timestamps=ts)
    plot_feature_importance(imp_df)
    plot_error_distribution(y_test, y_pred)
    print("\nAll plots generated!")