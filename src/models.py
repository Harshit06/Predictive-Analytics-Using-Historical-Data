"""
models.py
---------
Trains and returns predictions from multiple model families so results can
be compared side-by-side:

    1. Linear Regression        -> baseline regression model
    2. Random Forest Regressor  -> non-linear regression model
    3. ARIMA                    -> classical statistical time-series model

Each function follows the same contract: fit on train, predict on test,
return the fitted model + predictions.
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

FEATURE_COLUMNS = [
    "year", "month", "day", "day_of_week", "day_of_year",
    "is_weekend", "week_of_year",
    "sales_lag_1", "sales_lag_7", "sales_lag_14", "sales_lag_30",
    "sales_roll_mean_7", "sales_roll_std_7",
    "sales_roll_mean_30", "sales_roll_std_30",
]


def train_linear_regression(train: pd.DataFrame, test: pd.DataFrame, target_col: str = "sales"):
    """Train a Linear Regression model on engineered features."""
    X_train, y_train = train[FEATURE_COLUMNS], train[target_col]
    X_test = test[FEATURE_COLUMNS]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    return model, preds, scaler


def train_random_forest(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str = "sales",
    n_estimators: int = 300,
    max_depth: int = 10,
    random_state: int = 42,
):
    """Train a Random Forest Regressor on engineered features."""
    X_train, y_train = train[FEATURE_COLUMNS], train[target_col]
    X_test = test[FEATURE_COLUMNS]

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return model, preds


def train_arima(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str = "sales",
    order=(5, 1, 2),
):
    """Train a classical ARIMA model directly on the target time series."""
    model = ARIMA(train[target_col], order=order)
    fitted = model.fit()
    preds = fitted.forecast(steps=len(test))
    return fitted, preds.values


def get_feature_importance(model: RandomForestRegressor) -> pd.Series:
    """Return sorted feature importances from a fitted Random Forest."""
    importance = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS)
    return importance.sort_values(ascending=False)
