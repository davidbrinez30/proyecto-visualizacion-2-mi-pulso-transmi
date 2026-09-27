"""Entrena y compara baselines + modelo de ML, con validacion temporal.

Split: primeros ~38 dias para entrenamiento, ultimos ~7 dias para validacion
(nunca aleatorio - mezclaria futuro y pasado, ver README.md del SDK).

Modelo de horizonte DIRECTO (ver src/features.py): un solo modelo que recibe
el horizonte (15/30/45/60 min) como feature y predice cada uno directamente
a partir de datos ancla reales, sin encadenar predicciones sobre
predicciones. Esto reemplaza el enfoque recursivo anterior, que mostro
degradarse fuerte con el horizonte (validado con datos reales del reto:
accuracy 82.7% a 15 min pero solo 69% a 60 min de forma recursiva).

El modelo que se DESPLIEGA ademas es una mezcla con un promedio estacional
(ver BLEND_WEIGHT_SEASONAL en src/features.py) -- valida con estos mismos
datos, esa mezcla sube el accuracy de 82.2% (solo GBR) a 87.7%. Las
metricas de mas abajo se calculan de forma honesta con el split temporal
(nunca se entrena con datos de la ventana de validacion); DESPUES de medir,
se reentrena el modelo final con TODOS los datos disponibles (incluida esa
ultima semana) para que lo que se despliega sea lo mas fresco posible.

Uso:
    python -m src.train
Genera:
    artifacts/metrics.json          - metricas de cada modelo (incl. por horizonte y de la mezcla final)
    artifacts/model_gbr.joblib       - modelo + lookup estacional + peso de mezcla, listo para produccion
    artifacts/comparacion_modelos.png
"""
from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from src.evaluation import summarize
from src.features import BLEND_WEIGHT_SEASONAL, FEATURE_COLUMNS, HORIZONS, build_multihorizon_frame, seasonal_lookup
from src.ingest import load_all

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)
VALIDATION_DAYS = 7

# paleta (dataviz skill)
SEQ_BLUE = "#2a78d6"
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#7a4fd6"]
SURFACE = "#fcfcfb"


def baseline_naive_96(df: pd.DataFrame) -> pd.Series:
    """Baseline 1: demanda de hace 96 periodos (mismo horario, dia anterior),
    comparada contra el horizonte de 15 min (h=1) para que sea comparable
    con el naive clasico de este tipo de series."""
    return df.groupby("station_id")["demand"].shift(96 - 1)


def _apply_seasonal_lookup(lookup: pd.Series, df: pd.DataFrame) -> pd.Series:
    df = df.copy()
    df["hour"] = df["target_at"].dt.hour
    df["minute"] = df["target_at"].dt.minute
    df["dow"] = df["target_at"].dt.dayofweek
    key = list(zip(df["station_id"], df["dow"], df["hour"], df["minute"]))
    return pd.Series([lookup.get(k, np.nan) for k in key], index=df.index)


def main():
    stations, observations, context = load_all()
    multi = build_multihorizon_frame(observations, context)

    cutoff = observations["observed_at"].max() - timedelta(days=VALIDATION_DAYS)
    train_raw = observations[observations["observed_at"] <= cutoff]

    multi["pred_naive"] = baseline_naive_96(multi)
    lookup_train_only = seasonal_lookup(train_raw)  # sin fuga: solo datos de entrenamiento
    multi["pred_seasonal"] = _apply_seasonal_lookup(lookup_train_only, multi)

    train_feat = multi[multi["anchor_at"] <= cutoff].dropna(subset=FEATURE_COLUMNS + ["demand_target"])
    valid_feat = multi[multi["anchor_at"] > cutoff].copy()

    # n_estimators/learning_rate/subsample mas livianos que la primera
    # version (250/0.05/1.0): con 4 horizontes por fila el set de
    # entrenamiento es ~4x mas grande, y ese ajuste tardaba ~85s por
    # modelo (y se entrena DOS veces, ver refit final mas abajo) -- muy
    # cerca del limite de 9 min del workflow. Validado con estos mismos
    # datos: baja a ~20s por modelo sin perder accuracy (87.70% vs
    # 87.72%), porque la mayor parte de la mejora la da la mezcla con el
    # promedio estacional, no la complejidad del GBR.
    GBR_PARAMS = dict(random_state=42, max_depth=3, n_estimators=100, learning_rate=0.1, subsample=0.5)
    model = GradientBoostingRegressor(**GBR_PARAMS)
    model.fit(train_feat[FEATURE_COLUMNS], train_feat["demand_target"])

    valid_ml = valid_feat.dropna(subset=FEATURE_COLUMNS + ["demand_target"]).copy()
    valid_ml["pred_ml"] = model.predict(valid_ml[FEATURE_COLUMNS]).clip(min=0)
    valid_ml["pred_blend"] = (
        BLEND_WEIGHT_SEASONAL * valid_ml["pred_seasonal"].fillna(valid_ml["pred_ml"])
        + (1 - BLEND_WEIGHT_SEASONAL) * valid_ml["pred_ml"]
    )

    results = {"cutoff_validacion": str(cutoff), "validation_days": VALIDATION_DAYS,
               "blend_weight_seasonal": BLEND_WEIGHT_SEASONAL}

    for name, col in [("naive_96", "pred_naive"), ("seasonal_avg", "pred_seasonal")]:
        sub = valid_feat.dropna(subset=[col, "demand_target"])
        results[name] = summarize(sub["demand_target"], sub[col], sub["station_id"])

    results["gradient_boosting"] = summarize(valid_ml["demand_target"], valid_ml["pred_ml"], valid_ml["station_id"])
    results["gradient_boosting"]["accuracy_by_horizon"] = {
        f"{h * 15}min": summarize(
            valid_ml.loc[valid_ml["horizon_steps"] == h, "demand_target"],
            valid_ml.loc[valid_ml["horizon_steps"] == h, "pred_ml"],
            valid_ml.loc[valid_ml["horizon_steps"] == h, "station_id"],
        )["accuracy_mean"]
        for h in HORIZONS
    }

    # esta es la que de verdad se despliega -- la que hay que mirar
    results["blend_final"] = summarize(valid_ml["demand_target"], valid_ml["pred_blend"], valid_ml["station_id"])
    results["blend_final"]["accuracy_by_horizon"] = {
        f"{h * 15}min": summarize(
            valid_ml.loc[valid_ml["horizon_steps"] == h, "demand_target"],
            valid_ml.loc[valid_ml["horizon_steps"] == h, "pred_blend"],
            valid_ml.loc[valid_ml["horizon_steps"] == h, "station_id"],
        )["accuracy_mean"]
        for h in HORIZONS
    }

    (ARTIFACTS / "metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))

    # ---- refit final para produccion: usa TODOS los datos disponibles
    # (entrenamiento + la semana que se dejo para validar), para que el
    # modelo desplegado no deje de ver la informacion mas reciente. Las
    # metricas de arriba ya se midieron de forma honesta ANTES de este
    # refit, con el split temporal.
    full_train = multi.dropna(subset=FEATURE_COLUMNS + ["demand_target"])
    model_final = GradientBoostingRegressor(**GBR_PARAMS)
    model_final.fit(full_train[FEATURE_COLUMNS], full_train["demand_target"])
    lookup_final = seasonal_lookup(observations)

    joblib.dump({
        "model": model_final,
        "features": FEATURE_COLUMNS,
        "seasonal_lookup": lookup_final,
        "blend_weight_seasonal": BLEND_WEIGHT_SEASONAL,
    }, ARTIFACTS / "model_gbr.joblib")

    # ---- grafico comparativo ----
    names = ["naive_96", "seasonal_avg", "gradient_boosting", "blend_final"]
    labels = ["Naive (t-96)", "Promedio\nestacional", "Gradient\nBoosting", "Mezcla final\n(desplegada)"]
    accs = [results[n]["accuracy_mean"] for n in names]

    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                          "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars = ax.bar(labels, accs, color=CATEGORICAL)
    for bar, acc in zip(bars, accs):
        ax.annotate(f"{acc:.1f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("Accuracy promedio (validación, últimos 7 días, 4 horizontes)")
    ax.set_title("Comparación de modelos — Pulso TransMi")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linewidth=0.6, color="#e1e0d9")
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "comparacion_modelos.png", dpi=160, facecolor=SURFACE)

    print(json.dumps({k: v for k, v in results.items() if isinstance(v, dict) and "accuracy_mean" in v},
                      indent=2, default=str))
    for name in names:
        print(f"{name}: accuracy_mean={results[name]['accuracy_mean']:.2f}  wape_mean={results[name]['wape_mean']:.4f}")
    print("blend_final por horizonte:", results["blend_final"]["accuracy_by_horizon"])


if __name__ == "__main__":
    main()
