"""
test_pipeline.py
-----------------
Basic sanity tests for the predictive analytics pipeline.
Run with: pytest tests/
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.data_generator import generate_sales_data
from src.preprocessing import (
    handle_missing_values,
    handle_outliers,
    add_calendar_features,
    add_lag_and_rolling_features,
)
from src.evaluate import evaluate_predictions


def test_generate_sales_data_shape():
    df = generate_sales_data(start_date="2023-01-01", end_date="2023-12-31")
    assert len(df) == 365
    assert set(df.columns) == {"date", "sales"}


def test_generate_sales_data_has_missing_and_outliers():
    df = generate_sales_data(start_date="2020-01-01", end_date="2023-12-31")
    assert df["sales"].isna().sum() > 0


def test_handle_missing_values_fills_all_nans():
    dates = pd.date_range("2023-01-01", periods=30, freq="D")
    df = pd.DataFrame({"sales": np.random.rand(30)}, index=dates)
    df.loc[dates[5], "sales"] = np.nan
    df.loc[dates[15], "sales"] = np.nan
    result = handle_missing_values(df, "sales")
    assert result["sales"].isna().sum() == 0


def test_handle_outliers_caps_extremes():
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    values = np.random.normal(100, 10, 100)
    values[0] = 10000  # extreme outlier
    df = pd.DataFrame({"sales": values}, index=dates)
    result = handle_outliers(df, "sales")
    assert result["sales"].max() < 10000


def test_calendar_features_added():
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    df = pd.DataFrame({"sales": np.random.rand(10)}, index=dates)
    result = add_calendar_features(df)
    for col in ["year", "month", "day", "day_of_week", "is_weekend"]:
        assert col in result.columns


def test_lag_features_added():
    dates = pd.date_range("2023-01-01", periods=50, freq="D")
    df = pd.DataFrame({"sales": np.random.rand(50)}, index=dates)
    result = add_lag_and_rolling_features(df, "sales", lags=(1, 7), windows=(7,))
    assert "sales_lag_1" in result.columns
    assert "sales_lag_7" in result.columns
    assert "sales_roll_mean_7" in result.columns


def test_evaluate_predictions_perfect_score():
    y_true = np.array([10, 20, 30, 40, 50])
    y_pred = np.array([10, 20, 30, 40, 50])
    metrics = evaluate_predictions(y_true, y_pred, "Perfect Model")
    assert metrics["MAE"] == 0
    assert metrics["RMSE"] == 0
    assert metrics["R2 Score"] == 1.0


def test_evaluate_predictions_returns_expected_keys():
    y_true = np.array([10, 20, 30])
    y_pred = np.array([12, 18, 33])
    metrics = evaluate_predictions(y_true, y_pred, "Test Model")
    assert set(metrics.keys()) == {"Model", "MAE", "RMSE", "MAPE (%)", "R2 Score"}
