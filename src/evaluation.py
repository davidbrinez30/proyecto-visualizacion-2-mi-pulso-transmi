"""Metrica oficial del reto: WAPE / Accuracy, calculada por estacion.

    WAPE = sum(abs(real - prediccion)) / sum(real)
    Accuracy = 100 * max(0, 1 - WAPE)

La metrica se calcula por estacion y luego se promedia (ver README.md del
SDK oficial).
"""
from __future__ import annotations

import pandas as pd


def wape_by_station(y_true: pd.Series, y_pred: pd.Series, station_id: pd.Series) -> pd.Series:
    error = (y_true - y_pred).abs()
    return error.groupby(station_id).sum() / y_true.groupby(station_id).sum()


def accuracy_by_station(y_true: pd.Series, y_pred: pd.Series, station_id: pd.Series) -> pd.Series:
    wape = wape_by_station(y_true, y_pred, station_id)
    return (100 * (1 - wape)).clip(lower=0)


def summarize(y_true: pd.Series, y_pred: pd.Series, station_id: pd.Series) -> dict:
    wape = wape_by_station(y_true, y_pred, station_id)
    acc = (100 * (1 - wape)).clip(lower=0)
    return {
        "wape_mean": float(wape.mean()),
        "accuracy_mean": float(acc.mean()),
        "wape_by_station": wape.round(4).to_dict(),
        "accuracy_by_station": acc.round(2).to_dict(),
    }
