import pandas as pd

from src.evaluation import accuracy_by_station, wape_by_station


def test_wape_perfect_prediction_is_zero():
    y_true = pd.Series([10, 20, 30])
    y_pred = pd.Series([10, 20, 30])
    station = pd.Series(["A", "A", "A"])
    assert wape_by_station(y_true, y_pred, station)["A"] == 0


def test_accuracy_is_100_for_perfect_prediction():
    y_true = pd.Series([10, 20, 30])
    y_pred = pd.Series([10, 20, 30])
    station = pd.Series(["A", "A", "A"])
    assert accuracy_by_station(y_true, y_pred, station)["A"] == 100


def test_accuracy_clips_at_zero_for_very_bad_predictions():
    y_true = pd.Series([10, 10])
    y_pred = pd.Series([1000, 1000])
    station = pd.Series(["A", "A"])
    assert accuracy_by_station(y_true, y_pred, station)["A"] == 0


def test_metric_is_computed_per_station_independently():
    y_true = pd.Series([10, 10, 100, 100])
    y_pred = pd.Series([10, 10, 50, 50])
    station = pd.Series(["A", "A", "B", "B"])
    wape = wape_by_station(y_true, y_pred, station)
    assert wape["A"] == 0
    assert round(wape["B"], 2) == 0.5
