"""
visualize.py
------------
Generates all charts used to interpret the data and evaluate model
performance. All plots are saved to outputs/plots/.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

sns.set_theme(style="darkgrid")
PLOT_DIR = Path("outputs/plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def plot_historical_trend(df: pd.DataFrame, target_col: str = "sales", save_name: str = "01_historical_trend.png"):
    plt.figure(figsize=(14, 5))
    plt.plot(df.index, df[target_col], color="#2563eb", linewidth=0.9)
    plt.title("Historical Sales Trend", fontsize=14, fontweight="bold")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / save_name, dpi=150)
    plt.close()


def plot_seasonal_decomposition(df: pd.DataFrame, target_col: str = "sales", period: int = 365, save_name: str = "02_seasonal_decomposition.png"):
    result = seasonal_decompose(df[target_col], model="additive", period=period, extrapolate_trend="freq")
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    result.observed.plot(ax=axes[0], color="#2563eb"); axes[0].set_ylabel("Observed")
    result.trend.plot(ax=axes[1], color="#16a34a"); axes[1].set_ylabel("Trend")
    result.seasonal.plot(ax=axes[2], color="#ea580c"); axes[2].set_ylabel("Seasonal")
    result.resid.plot(ax=axes[3], color="#dc2626"); axes[3].set_ylabel("Residual")
    fig.suptitle("Time Series Decomposition", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / save_name, dpi=150)
    plt.close()


def plot_predictions_vs_actual(
    test_index,
    y_true,
    predictions_dict: dict,
    save_name: str = "03_predictions_vs_actual.png",
):
    """predictions_dict: {'Model Name': predicted_values_array}"""
    plt.figure(figsize=(14, 6))
    plt.plot(test_index, y_true, label="Actual", color="black", linewidth=2)
    colors = ["#2563eb", "#ea580c", "#16a34a", "#9333ea"]
    for (name, preds), color in zip(predictions_dict.items(), colors):
        plt.plot(test_index, preds, label=name, linestyle="--", linewidth=1.5, color=color)
    plt.title("Forecast vs Actual Sales (Test Period)", fontsize=14, fontweight="bold")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / save_name, dpi=150)
    plt.close()


def plot_metric_comparison(comparison_df: pd.DataFrame, save_name: str = "04_metric_comparison.png"):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    metrics = ["MAE", "RMSE", "R2 Score"]
    palette = ["#2563eb", "#ea580c", "#16a34a"]
    for ax, metric, color in zip(axes, metrics, palette):
        comparison_df[metric].plot(kind="bar", ax=ax, color=color)
        ax.set_title(metric, fontweight="bold")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / save_name, dpi=150)
    plt.close()


def plot_feature_importance(importance: pd.Series, save_name: str = "05_feature_importance.png"):
    plt.figure(figsize=(10, 6))
    importance.sort_values().plot(kind="barh", color="#2563eb")
    plt.title("Random Forest — Feature Importance", fontsize=14, fontweight="bold")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / save_name, dpi=150)
    plt.close()


def plot_residuals(y_true, y_pred, model_name: str, save_name: str):
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].scatter(y_pred, residuals, alpha=0.5, color="#2563eb", s=15)
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Residual")
    axes[0].set_title(f"{model_name} — Residuals vs Predicted")

    sns.histplot(residuals, kde=True, ax=axes[1], color="#ea580c")
    axes[1].set_title(f"{model_name} — Residual Distribution")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / save_name, dpi=150)
    plt.close()
