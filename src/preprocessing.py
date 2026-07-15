"""
preprocessing.py
-----------------
Cleans and prepares historical time-series data for modeling:
    1. Load & parse dates
    2. Handle missing values (interpolation)
    3. Handle outliers (IQR capping)
    4. Feature engineering:
        - Calendar features (year, month, day of week, is_weekend)
        - Lag features (sales_lag_1, sales_lag_7, sales_lag_30)
        - Rolling statistics (rolling mean / std)
    5. Train/test split that respects time order (no shuffling!)
"""

import numpy as np
import pandas as pd


def load_data(path: str, date_col: str = "date", target_col: str = "sales") -> pd.DataFrame:
    """Load CSV and ensure proper datetime index, sorted chronologically."""
    df = pd.read_csv(path, parse_dates=[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    df = df.set_index(date_col)
    df = df.asfreq("D")  # ensure continuous daily frequency, introduces NaN for gaps
    return df


def handle_missing_values(df: pd.DataFrame, target_col: str = "sales") -> pd.DataFrame:
    """Fill missing values via time-based linear interpolation."""
    df = df.copy()
    n_missing = df[target_col].isna().sum()
    df[target_col] = df[target_col].interpolate(method="time")
    # Fill any remaining edge NaNs (start/end of series)
    df[target_col] = df[target_col].bfill().ffill()
    print(f"Filled {n_missing} missing values via time-interpolation.")
    return df


def handle_outliers(df: pd.DataFrame, target_col: str = "sales", factor: float = 1.5) -> pd.DataFrame:
    """Cap outliers using the IQR method."""
    df = df.copy()
    q1 = df[target_col].quantile(0.25)
    q3 = df[target_col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr

    n_outliers = ((df[target_col] < lower) | (df[target_col] > upper)).sum()
    df[target_col] = df[target_col].clip(lower=lower, upper=upper)
    print(f"Capped {n_outliers} outliers outside [{lower:.2f}, {upper:.2f}].")
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add date-derived features useful for regression models."""
    df = df.copy()
    df["year"] = df.index.year
    df["month"] = df.index.month
    df["day"] = df.index.day
    df["day_of_week"] = df.index.dayofweek
    df["day_of_year"] = df.index.dayofyear
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["week_of_year"] = df.index.isocalendar().week.astype(int)
    return df


def add_lag_and_rolling_features(
    df: pd.DataFrame,
    target_col: str = "sales",
    lags=(1, 7, 14, 30),
    windows=(7, 30),
) -> pd.DataFrame:
    """Add lag features and rolling window statistics."""
    df = df.copy()
    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)
    for window in windows:
        df[f"{target_col}_roll_mean_{window}"] = df[target_col].shift(1).rolling(window).mean()
        df[f"{target_col}_roll_std_{window}"] = df[target_col].shift(1).rolling(window).std()
    return df


def build_feature_dataset(
    path: str,
    date_col: str = "date",
    target_col: str = "sales",
) -> pd.DataFrame:
    """Full preprocessing pipeline: load -> clean -> engineer features."""
    df = load_data(path, date_col, target_col)
    df = handle_missing_values(df, target_col)
    df = handle_outliers(df, target_col)
    df = add_calendar_features(df)
    df = add_lag_and_rolling_features(df, target_col)
    df = df.dropna()  # drop rows with NaN lag/rolling values (start of series)
    return df


def time_based_split(df: pd.DataFrame, test_size: float = 0.2):
    """Split chronologically: earliest data = train, most recent = test."""
    split_idx = int(len(df) * (1 - test_size))
    train, test = df.iloc[:split_idx], df.iloc[split_idx:]
    print(f"Train period: {train.index.min().date()} to {train.index.max().date()} ({len(train)} rows)")
    print(f"Test period:  {test.index.min().date()} to {test.index.max().date()} ({len(test)} rows)")
    return train, test


if __name__ == "__main__":
    features_df = build_feature_dataset("data/sample_sales_data.csv")
    print(features_df.head())
    print(f"\nFinal feature dataset shape: {features_df.shape}")
