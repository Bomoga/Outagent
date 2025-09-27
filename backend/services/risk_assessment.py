"""Risk assessment utilities for energy outage forecasting."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train_hourly_consumption_model(
    dataset: pd.DataFrame,
    *,
    target_column: str = "value",
    model_path: Optional[Path] = None,
) -> Dict[str, float]:
    """Train a simple regression model for hourly energy consumption.

    The dataset is expected to have a DateTimeIndex and contain the target column
    alongside engineered features (weather, calendar features, etc.).
    """

    if dataset.empty:
        raise ValueError("Training dataset is empty.")

    if target_column not in dataset.columns:
        raise ValueError(f"Target column '{target_column}' not present in dataset.")

    df = dataset.copy()
    df = df.dropna(subset=[target_column])
    if df.empty:
        raise ValueError("No rows remaining after dropping NaNs from target column.")

    features = df.drop(columns=[target_column])
    features = features.replace({np.inf: np.nan, -np.inf: np.nan}).ffill().bfill()
    features = features.select_dtypes(include=["number"])
    feature_columns = features.columns.tolist()

    X = features.to_numpy(dtype=float)
    y = df[target_column].to_numpy(dtype=float)

    if len(df) < 24:
        raise ValueError("Not enough observations to train the model (need >= 24).")

    split_index = int(len(df) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = r2_score(y_test, y_pred)

    if model_path is not None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": model, "features": feature_columns}, model_path)

    return {
        "mae": float(mae),
        "rmse": rmse,
        "r2": float(r2),
    }

