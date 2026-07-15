# 📈 Predictive Analytics Using Historical Data

A complete, end-to-end machine learning project that forecasts future trends from historical time-series data. It combines **regression models** (Linear Regression, Random Forest) and a **classical time-series model** (ARIMA) so you can compare approaches side by side, with a full pipeline for cleaning, feature engineering, evaluation, and visualization.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)

---

## 🎯 Project Overview

This project demonstrates a realistic, production-style predictive analytics workflow:

- **Data preprocessing** — handling missing values, outliers, and irregular time gaps
- **Feature engineering** — calendar features, lag features, rolling statistics
- **Model training** — Linear Regression, Random Forest Regressor, ARIMA
- **Evaluation** — MAE, RMSE, MAPE, R² with a clear model comparison table
- **Visualization** — trend charts, seasonal decomposition, forecast vs actual, residual analysis, feature importance

It ships with a **synthetic historical sales dataset generator** so the project runs out-of-the-box — but it's built to be pointed at *any* historical dataset (sales, stock prices, website traffic, energy demand, weather, etc.) with minimal changes.

---

## 🗂️ Project Structure

```
predictive-analytics-project/
├── data/
│   └── sample_sales_data.csv       # Auto-generated synthetic dataset
├── src/
│   ├── data_generator.py           # Synthetic dataset generator
│   ├── preprocessing.py            # Cleaning + feature engineering
│   ├── models.py                   # Linear Regression / Random Forest / ARIMA
│   ├── evaluate.py                 # Accuracy metrics + comparison report
│   └── visualize.py                # All chart generation
├── notebooks/
│   └── predictive_analysis.ipynb   # Interactive walkthrough notebook
├── tests/
│   └── test_pipeline.py            # Unit tests for core pipeline logic
├── outputs/
│   ├── plots/                      # Generated charts (created on run)
│   └── models/                     # Saved best model (created on run)
├── main.py                         # Runs the full end-to-end pipeline
├── requirements.txt
├── LICENSE
└── README.md
```

---

## ⚙️ Installation

```bash
# Clone the repo
git clone https://github.com/<your-username>/predictive-analytics-project.git
cd predictive-analytics-project

# (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### Run the full pipeline
```bash
python main.py
```

This will:
1. Generate a synthetic historical sales dataset (if one doesn't already exist in `data/`)
2. Clean and preprocess it
3. Engineer calendar, lag, and rolling-window features
4. Train Linear Regression, Random Forest, and ARIMA models
5. Evaluate all models and print a comparison table
6. Save charts to `outputs/plots/`
7. Save the best-performing model to `outputs/models/`

### Explore interactively
```bash
jupyter notebook notebooks/predictive_analysis.ipynb
```

### Run tests
```bash
pytest tests/ -v
```

---

## 🔧 Using Your Own Dataset

Replace `data/sample_sales_data.csv` with your own CSV — it just needs:
- A **date column** (any parseable date format)
- A **numeric target column** you want to forecast

Then update the column names in `main.py`:

```python
DATA_PATH = "data/your_dataset.csv"
TARGET_COL = "your_target_column"
```

And in `src/preprocessing.py`, adjust `load_data(path, date_col="your_date_column", ...)` if your date column isn't named `date`.

---

## 🧠 Models Used

| Model | Type | Why it's included |
|---|---|---|
| **Linear Regression** | Regression | Fast, interpretable baseline |
| **Random Forest Regressor** | Regression (ensemble) | Captures non-linear patterns, gives feature importance |
| **ARIMA** | Classical time-series | Models autocorrelation & trend directly from the series |

The pipeline evaluates all three on a **chronological train/test split** (no shuffling — critical for time series) and reports which one generalizes best.

---

## 📊 Evaluation Metrics

- **MAE** (Mean Absolute Error) — average magnitude of errors
- **RMSE** (Root Mean Squared Error) — penalizes larger errors more heavily
- **MAPE** (Mean Absolute Percentage Error) — error as a percentage, easy to communicate
- **R² Score** — proportion of variance explained by the model

---

## 📈 Sample Outputs

Running `main.py` generates the following in `outputs/plots/`:

1. `01_historical_trend.png` — raw historical trend
2. `02_seasonal_decomposition.png` — trend / seasonality / residual breakdown
3. `03_predictions_vs_actual.png` — all models' forecasts vs actual values
4. `04_metric_comparison.png` — bar charts comparing MAE / RMSE / R²
5. `05_feature_importance.png` — which features drive the Random Forest's predictions
6. `06_rf_residuals.png` — residual diagnostics

---

## 🛠️ Tech Stack

- **Python 3.9+**
- **pandas / numpy** — data manipulation
- **scikit-learn** — Linear Regression, Random Forest, metrics
- **statsmodels** — ARIMA, seasonal decomposition
- **matplotlib / seaborn** — visualization
- **joblib** — model persistence

---

## 🎓 What You'll Learn

- How to clean and prepare messy, real-world time-series data
- How to engineer date/lag/rolling features for regression on time series
- The difference between regression-based and classical statistical forecasting
- How to fairly evaluate forecasting models (chronological splits, multiple metrics)
- How to visualize trends, seasonality, and forecast accuracy

---

## 🔮 Future Improvements

- Add Facebook Prophet / LSTM / XGBoost as additional model options
- Hyperparameter tuning with `GridSearchCV` / `Optuna`
- Add a Streamlit dashboard for interactive forecasting
- Support multivariate forecasting (multiple correlated time series)
- Add automated hyperparameter search for ARIMA `(p, d, q)` order

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## 🙌 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.
