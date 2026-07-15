"""
evaluate.py
-----------
Computes standard regression/forecasting accuracy metrics:
    - MAE  (Mean Absolute Error)
    - RMSE (Root Mean Squared Error)
    - MAPE (Mean Absolute Percentage Error)
    - R2   (Coefficient of Determination)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray, model_name: str = "Model") -> dict:
    """Compute and return a dictionary of accuracy metrics."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100
    r2 = r2_score(y_true, y_pred)

    metrics = {
        "Model": model_name,
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "MAPE (%)": round(mape, 3),
        "R2 Score": round(r2, 4),
    }
    return metrics


def compare_models(results: list) -> pd.DataFrame:
    """Build a comparison table from a list of metric dictionaries."""
    df = pd.DataFrame(results).set_index("Model")
    return df.sort_values("RMSE")


def print_report(results: list):
    df = compare_models(results)
    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE COMPARISON")
    print("=" * 60)
    print(df.to_string())
    print("=" * 60)
    best_model = df["RMSE"].idxmin()
    print(f"\nBest performing model (lowest RMSE): {best_model}")
    return df
