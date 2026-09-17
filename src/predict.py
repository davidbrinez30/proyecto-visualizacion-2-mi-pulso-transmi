"""Genera pronosticos para los 4 horizontes solicitados (15/30/45/60 min
adelante) a partir del ultimo modelo entrenado (artifacts/model_gbr.joblib).

Reconstruye features de forma recursiva: cada paso usa, cuando hace falta,
las predicciones de los pasos anteriores como lag_1 / roll_mean_4, y datos
reales de historia para el resto (lag_4/lag_96/lag_672 siempre caen dentro
de historia real para horizontes <= 4).

Limitacion conocida: el reto no libera clima/eventos futuros
(`future_included: false` en /v1/meta), asi que para las features de
contexto de los periodos futuros se usa el ultimo valor de contexto
conocido (forward-fill) - hay que revisar este supuesto cuando el reto
libere una fuente de pronostico de clima.

Uso:
    python -m src.predict
Genera:
    artifacts/predictions_latest.csv
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import joblib
import pandas as pd

from src.features import LAGS, ROLLING_WINDOWS, add_calendar_features
from src.ingest import load_all

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"
HORIZONS = (1, 2, 3, 4)  # pasos de 15 min: 15/30/45/60 min adelante


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent.parent
                                        ).decode().strip()
    except Exception:
        return "unknown"


def predict_station(demand_hist: pd.Series, context_row: dict, last_time: pd.Timestamp,
                     model, feature_cols: list[str]) -> list[dict]:
    """demand_hist: serie de demanda historica (indexada por observed_at) de
    UNA estacion, ordenada ascendente. Se va extendiendo con cada prediccion."""
    series = demand_hist.copy()
    rows = []
    for h in HORIZONS:
        target_at = last_time + pd.Timedelta(minutes=15 * h)
        feat = {"observed_at": target_at, **context_row}
        feat_df = add_calendar_features(pd.DataFrame([feat]))
        for lag in LAGS:
            idx = target_at - pd.Timedelta(minutes=15 * lag)
            feat_df[f"lag_{lag}"] = series.get(idx, series.iloc[-1])
        for window in ROLLING_WINDOWS:
            feat_df[f"roll_mean_{window}"] = series.iloc[-window:].mean()
        pred = float(model.predict(feat_df[feature_cols])[0])
        pred = max(0.0, pred)
        series.loc[target_at] = pred
        rows.append({"target_at": target_at, "horizon_steps": h, "predicted_demand": round(pred, 1)})
    return rows


def main():
    bundle = joblib.load(ARTIFACTS / "model_gbr.joblib")
    model, feature_cols = bundle["model"], bundle["features"]

    stations, observations, context = load_all()
    last_context = context.sort_values("observed_at").iloc[-1]
    context_row = {c: last_context[c] for c in ("rain_mm", "temperature_c", "event_intensity")}

    commit = git_commit()
    all_rows = []
    for station_id, grp in observations.groupby("station_id"):
        grp = grp.sort_values("observed_at")
        demand_hist = pd.Series(grp["demand"].values, index=grp["observed_at"].values)
        last_time = grp["observed_at"].max()
        preds = predict_station(demand_hist, context_row, last_time, model, feature_cols)
        for row in preds:
            row.update({"station_id": station_id, "model_version": f"gbr@{commit}"})
            all_rows.append(row)

    out = pd.DataFrame(all_rows)[
        ["station_id", "target_at", "horizon_steps", "predicted_demand", "model_version"]
    ]
    out.to_csv(ARTIFACTS / "predictions_latest.csv", index=False)
    print(out.to_string(index=False))
    print(f"\n{len(out)} predicciones guardadas en artifacts/predictions_latest.csv")


if __name__ == "__main__":
    main()
