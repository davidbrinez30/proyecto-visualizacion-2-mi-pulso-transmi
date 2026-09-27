"""Genera pronosticos para los 4 horizontes solicitados (15/30/45/60 min
adelante) a partir del ultimo modelo entrenado (artifacts/model_gbr.joblib).

Modelo de horizonte DIRECTO (ver src/features.py): el horizonte (15/30/45/60
min) es una feature mas del modelo, y todas las demas features ("ancla") se
calculan una sola vez con datos 100% reales hasta el ultimo momento
conocido (last_time). Ya NO se predice de forma recursiva -- ese enfoque
anterior se probo con datos reales del reto y el accuracy caia de 82.7%
(15 min) a 69% (60 min) por el error acumulado.

Ademas, la prediccion final es una MEZCLA con un promedio historico
estacion+dia+hora+minuto (`seasonal_lookup` guardado en el bundle del
modelo, ver BLEND_WEIGHT_SEASONAL en src/features.py) -- esto solo, sin
recursividad, ya sube el accuracy real a ~87-88% en validacion con datos
del reto.

Limitacion conocida: el reto no libera clima/eventos futuros
(`future_included: false` en /v1/meta), asi que para las features de
contexto se usa el ultimo valor de contexto conocido (forward-fill) - hay
que revisar este supuesto cuando el reto libere una fuente de pronostico
de clima.

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

from src.features import HORIZONS, add_calendar_features, anchor_features_for_station, seasonal_value
from src.ingest import load_all

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent.parent
                                        ).decode().strip()
    except Exception:
        return "unknown"


def predict_station(demand_hist: pd.Series, context_row: dict, last_time: pd.Timestamp,
                     model, feature_cols: list[str],
                     seasonal_lookup=None, station_id: str | None = None,
                     blend_weight_seasonal: float = 0.0) -> list[dict]:
    """demand_hist: serie de demanda historica (indexada por observed_at) de
    UNA estacion, ordenada ascendente.

    Calcula las features "ancla" UNA sola vez (con datos reales hasta
    last_time) y predice los 4 horizontes directamente -- sin usar ninguna
    prediccion anterior como si fuera un dato real. Si se pasa
    `seasonal_lookup` (y `station_id`), mezcla la prediccion del modelo con
    el promedio historico estacion+dia+hora+minuto segun
    `blend_weight_seasonal` (0 = solo modelo, 1 = solo promedio historico).
    """
    anchor_feat = anchor_features_for_station(demand_hist, context_row, last_time)

    rows = []
    for h in HORIZONS:
        target_at = last_time + pd.Timedelta(minutes=15 * h)
        feat = {**anchor_feat, "horizon_min": 15 * h}
        cal = add_calendar_features(pd.DataFrame([{"observed_at": target_at}]))
        for col in ("hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"):
            feat[col] = cal[col].iloc[0]
        feat_df = pd.DataFrame([feat])
        pred_ml = float(model.predict(feat_df[feature_cols])[0])

        pred_final = pred_ml
        if seasonal_lookup is not None and station_id is not None and blend_weight_seasonal > 0:
            pred_seasonal = seasonal_value(seasonal_lookup, station_id, target_at, fallback=pred_ml)
            pred_final = blend_weight_seasonal * pred_seasonal + (1 - blend_weight_seasonal) * pred_ml

        pred_final = max(0.0, pred_final)
        rows.append({"target_at": target_at, "horizon_steps": h, "predicted_demand": round(pred_final, 1)})
    return rows


def main():
    bundle = joblib.load(ARTIFACTS / "model_gbr.joblib")
    model, feature_cols = bundle["model"], bundle["features"]
    seasonal_lookup_ = bundle.get("seasonal_lookup")
    blend_weight = bundle.get("blend_weight_seasonal", 0.0)

    stations, observations, context = load_all()
    last_context = context.sort_values("observed_at").iloc[-1]
    context_row = {c: last_context[c] for c in ("rain_mm", "temperature_c", "event_intensity")}

    commit = git_commit()
    all_rows = []
    for station_id, grp in observations.groupby("station_id"):
        grp = grp.sort_values("observed_at")
        demand_hist = pd.Series(grp["demand"].values, index=grp["observed_at"].values)
        last_time = grp["observed_at"].max()
        preds = predict_station(demand_hist, context_row, last_time, model, feature_cols,
                                 seasonal_lookup=seasonal_lookup_, station_id=station_id,
                                 blend_weight_seasonal=blend_weight)
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
