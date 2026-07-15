"""
data_generator.py
------------------
Generates a realistic synthetic historical sales dataset for demo/testing
purposes. In a real project, replace this with your own historical CSV
(e.g., sales records, stock prices, website traffic, energy demand, etc.)
loaded via pandas.read_csv().

The synthetic data includes:
    - A long-term upward trend
    - Yearly seasonality (higher sales in Nov-Dec, lower in Feb)
    - Weekly seasonality (higher on weekends)
    - Random noise
    - A handful of missing values and outliers (to make preprocessing
      meaningful and realistic)
"""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_sales_data(
    start_date: str = "2019-01-01",
    end_date: str = "2024-12-31",
    base_sales: float = 500.0,
    trend_slope: float = 0.35,
    yearly_amplitude: float = 150.0,
    weekly_amplitude: float = 60.0,
    noise_std: float = 40.0,
    missing_fraction: float = 0.01,
    outlier_fraction: float = 0.005,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic daily sales data with trend + seasonality + noise."""
    rng = np.random.default_rng(random_seed)

    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    n = len(dates)
    t = np.arange(n)

    # Long-term trend
    trend = base_sales + trend_slope * t

    # Yearly seasonality (sine wave, period = 365.25 days)
    yearly_seasonality = yearly_amplitude * np.sin(2 * np.pi * t / 365.25 - np.pi / 2)

    # Weekly seasonality (higher sales on weekends)
    day_of_week = dates.dayofweek  # 0=Mon ... 6=Sun
    weekly_seasonality = np.where(day_of_week >= 5, weekly_amplitude, -weekly_amplitude / 3)

    # Random noise
    noise = rng.normal(0, noise_std, n)

    # A few special "promotion spikes" to mimic real-world events
    promo_spikes = np.zeros(n)
    promo_days = rng.choice(n, size=int(n * 0.02), replace=False)
    promo_spikes[promo_days] = rng.uniform(100, 300, size=len(promo_days))

    sales = trend + yearly_seasonality + weekly_seasonality + noise + promo_spikes
    sales = np.maximum(sales, 0)  # sales can't be negative

    df = pd.DataFrame({"date": dates, "sales": sales.round(2)})

    # Inject missing values (realistic data quality issue)
    missing_idx = rng.choice(n, size=int(n * missing_fraction), replace=False)
    df.loc[missing_idx, "sales"] = np.nan

    # Inject outliers (realistic data quality issue)
    outlier_idx = rng.choice(n, size=int(n * outlier_fraction), replace=False)
    df.loc[outlier_idx, "sales"] = df.loc[outlier_idx, "sales"] * rng.uniform(3, 5)

    return df


def save_sample_data(output_path: str = "data/sample_sales_data.csv") -> pd.DataFrame:
    """Generate and persist the sample dataset to disk."""
    df = generate_sales_data()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Sample dataset saved to: {output_path}")
    print(f"Shape: {df.shape}")
    print(df.head())
    return df


if __name__ == "__main__":
    save_sample_data()
