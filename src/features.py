"""Feature engineering para el pronostico de demanda.

Todas las features usadas son calculables en el momento de prediccion (no usan
informacion futura), condicion necesaria para que el backtesting temporal sea
valido.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PERIODS_PER_DAY = 96  # 24h * 4 (15 min)
LAGS = (1, 4, 96, 96 * 7)  # 15min, 1h, 1 dia, 1 semana atras
ROLLING_WINDOWS = (4, 96)  # 1h, 1 dia


def add_calendar_features(df: pd.DataFrame, time_col: str = "observed_at") -> pd.DataFrame:
    df = df.copy()
    ts = df[time_col]
    df["hour"] = ts.dt.hour
    df["minute_of_day"] = ts.dt.hour * 60 + ts.dt.minute
    df["dow"] = ts.dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["minute_of_day"] / (24 * 60))
    df["hour_cos"] = np.cos(2 * np.pi * df["minute_of_day"] / (24 * 60))
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Requiere df ordenado por (station_id, observed_at) y sin huecos."""
    df = df.copy()
    grouped = df.groupby("station_id")["demand"]
    for lag in LAGS:
        df[f"lag_{lag}"] = grouped.shift(lag)
    for window in ROLLING_WINDOWS:
        df[f"roll_mean_{window}"] = grouped.shift(1).rolling(window).mean().reset_index(level=0, drop=True)
    return df


def build_feature_frame(observations: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    """Une observaciones + contexto y agrega todas las features."""
    df = observations.merge(context, on="observed_at", how="left")
    df = add_calendar_features(df)
    df = add_lag_features(df)
    return df


FEATURE_COLUMNS = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend",
    "rain_mm", "temperature_c", "event_intensity",
    "lag_1", "lag_4", "lag_96", "lag_672",
    "roll_mean_4", "roll_mean_96",
]
