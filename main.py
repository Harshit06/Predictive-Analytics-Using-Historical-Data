"""
main.py
-------
End-to-end predictive analytics pipeline:

    1. Generate / load historical dataset
    2. Clean & preprocess data
    3. Engineer features
    4. Train multiple models (Linear Regression, Random Forest, ARIMA)
    5. Evaluate and compare accuracy
    6. Generate visualizations
    7. Save best model to disk

Run with:  python main.py
"""

import json
from pathlib import Path
import joblib

from src.data_generator import save_sample_data
from src.preprocessing import build_feature_dataset, time_based_split, load_data, handle_missing_values, handle_outliers
from src.models import train_linear_regression, train_random_forest, train_arima, get_feature_importance
from src.evaluate import evaluate_predictions, print_report
from src.visualize import (
    plot_historical_trend,
    plot_seasonal_decomposition,
    plot_predictions_vs_actual,
    plot_metric_comparison,
    plot_feature_importance,
    plot_residuals,
)

DATA_PATH = "data/sample_sales_data.csv"
TARGET_COL = "sales"


def main():
    print("\n STEP 1: Preparing dataset")
    print("-" * 60)
    if not Path(DATA_PATH).exists():
        save_sample_data(DATA_PATH)
    else:
        print(f"Using existing dataset at {DATA_PATH}")

    print("\n STEP 2: Cleaning & preprocessing")
    print("-" * 60)
    features_df = build_feature_dataset(DATA_PATH, target_col=TARGET_COL)

    print("\n STEP 3: Train/test split (chronological)")
    print("-" * 60)
    train, test = time_based_split(features_df, test_size=0.15)

    print("\n STEP 4: Visualizing raw historical data")
    print("-" * 60)
    raw_df = load_data(DATA_PATH)
    raw_df = handle_missing_values(raw_df, TARGET_COL)
    raw_df = handle_outliers(raw_df, TARGET_COL)
    plot_historical_trend(raw_df, TARGET_COL)
    plot_seasonal_decomposition(raw_df, TARGET_COL)
    print("Saved: outputs/plots/01_historical_trend.png, 02_seasonal_decomposition.png")

    print("\n STEP 5: Training models")
    print("-" * 60)

    print("Training Linear Regression...")
    lr_model, lr_preds, scaler = train_linear_regression(train, test, TARGET_COL)

    print("Training Random Forest Regressor...")
    rf_model, rf_preds = train_random_forest(train, test, TARGET_COL)

    print("Training ARIMA...")
    arima_model, arima_preds = train_arima(train, test, TARGET_COL)

    print("\n STEP 6: Evaluating models")
    print("-" * 60)
    y_test = test[TARGET_COL]
    results = [
        evaluate_predictions(y_test, lr_preds, "Linear Regression"),
        evaluate_predictions(y_test, rf_preds, "Random Forest"),
        evaluate_predictions(y_test, arima_preds, "ARIMA"),
    ]
    comparison_df = print_report(results)
    comparison_df.to_csv("outputs/model_comparison.csv")

    print("\n STEP 7: Generating visualizations")
    print("-" * 60)
    predictions_dict = {
        "Linear Regression": lr_preds,
        "Random Forest": rf_preds,
        "ARIMA": arima_preds,
    }
    plot_predictions_vs_actual(test.index, y_test, predictions_dict)
    plot_metric_comparison(comparison_df)

    importance = get_feature_importance(rf_model)
    plot_feature_importance(importance)

    plot_residuals(y_test, rf_preds, "Random Forest", "06_rf_residuals.png")
    print("Saved all plots to outputs/plots/")

    print("\n STEP 8: Saving best model")
    print("-" * 60)
    Path("outputs/models").mkdir(parents=True, exist_ok=True)
    best_model_name = comparison_df["RMSE"].idxmin()
    model_map = {"Linear Regression": lr_model, "Random Forest": rf_model, "ARIMA": arima_model}
    joblib.dump(model_map[best_model_name], f"outputs/models/best_model_{best_model_name.replace(' ', '_').lower()}.pkl")
    print(f"Best model ('{best_model_name}') saved to outputs/models/")

    with open("outputs/metrics_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Metrics summary saved to outputs/metrics_summary.json")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
