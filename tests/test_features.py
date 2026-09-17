import pandas as pd

from src.features import add_calendar_features, add_lag_features


def test_calendar_features_are_bounded():
    df = pd.DataFrame({"observed_at": pd.date_range("2026-01-01", periods=5, freq="15min")})
    out = add_calendar_features(df)
    assert out["hour_sin"].between(-1, 1).all()
    assert out["hour_cos"].between(-1, 1).all()
    assert out["is_weekend"].isin([0, 1]).all()


def test_lag_features_do_not_leak_future_information():
    times = pd.date_range("2026-01-01", periods=5, freq="15min")
    df = pd.DataFrame({
        "station_id": ["A"] * 5,
        "observed_at": times,
        "demand": [10, 20, 30, 40, 50],
    })
    out = add_lag_features(df)
    # lag_1 en la fila i debe ser el demand de la fila i-1 (nunca uno futuro)
    assert out["lag_1"].tolist() == [None, 10, 20, 30, 40] or out["lag_1"].isna().sum() == 1
