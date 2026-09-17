"""Entrena y compara baselines + modelo de ML, con validacion temporal.

Split: primeros ~38 dias para entrenamiento, ultimos ~7 dias para validacion
(nunca aleatorio - mezclaria futuro y pasado, ver README.md del SDK).

Uso:
    python -m src.train
Genera:
    artifacts/metrics.json          - metricas de cada modelo
    artifacts/model_gbr.joblib       - modelo de ML entrenado
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
from src.features import FEATURE_COLUMNS, build_feature_frame
from src.ingest import load_all

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)
VALIDATION_DAYS = 7

# paleta (dataviz skill)
SEQ_BLUE = "#2a78d6"
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE = "#fcfcfb"


def temporal_split(df: pd.DataFrame, validation_days: int = VALIDATION_DAYS):
    cutoff = df["observed_at"].max() - timedelta(days=validation_days)
    train = df.loc[df["observed_at"] <= cutoff].copy()
    valid = df.loc[df["observed_at"] > cutoff].copy()
    return train, valid, cutoff


def baseline_naive_96(df: pd.DataFrame) -> pd.Series:
    """Baseline 1: demanda de hace 96 periodos (mismo horario, dia anterior)."""
    return df.groupby("station_id")["demand"].shift(96)


def baseline_seasonal_avg(train: pd.DataFrame, full: pd.DataFrame) -> pd.Series:
    """Baseline 2: promedio historico por estacion + hora + dia de la semana,
    calculado SOLO con datos de entrenamiento (sin fuga de informacion)."""
    train = train.copy()
    train["hour"] = train["observed_at"].dt.hour
    train["minute"] = train["observed_at"].dt.minute
    train["dow"] = train["observed_at"].dt.dayofweek
    lookup = train.groupby(["station_id", "dow", "hour", "minute"])["demand"].mean()

    full = full.copy()
    full["hour"] = full["observed_at"].dt.hour
    full["minute"] = full["observed_at"].dt.minute
    full["dow"] = full["observed_at"].dt.dayofweek
    key = list(zip(full["station_id"], full["dow"], full["hour"], full["minute"]))
    return pd.Series([lookup.get(k, np.nan) for k in key], index=full.index)


def main():
    stations, observations, context = load_all()
    full = build_feature_frame(observations, context)

    # baselines calculados sobre la serie completa (usan solo pasado por construccion)
    full["pred_naive"] = baseline_naive_96(full)
    train_raw, valid_raw, cutoff = temporal_split(observations)
    full["pred_seasonal"] = baseline_seasonal_avg(train_raw, full)

    train_feat, valid_feat, _ = temporal_split(full)

    model = GradientBoostingRegressor(random_state=42, max_depth=3, n_estimators=250, learning_rate=0.05)
    train_ml = train_feat.dropna(subset=FEATURE_COLUMNS + ["demand"])
    model.fit(train_ml[FEATURE_COLUMNS], train_ml["demand"])

    valid_ml = valid_feat.dropna(subset=FEATURE_COLUMNS + ["demand"]).copy()
    valid_ml["pred_ml"] = model.predict(valid_ml[FEATURE_COLUMNS])

    results = {"cutoff_validacion": str(cutoff), "validation_days": VALIDATION_DAYS}

    for name, col in [("naive_96", "pred_naive"), ("seasonal_avg", "pred_seasonal")]:
        sub = valid_feat.dropna(subset=[col, "demand"])
        results[name] = summarize(sub["demand"], sub[col], sub["station_id"])

    results["gradient_boosting"] = summarize(valid_ml["demand"], valid_ml["pred_ml"], valid_ml["station_id"])

    (ARTIFACTS / "metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    joblib.dump({"model": model, "features": FEATURE_COLUMNS}, ARTIFACTS / "model_gbr.joblib")

    # ---- grafico comparativo ----
    names = ["naive_96", "seasonal_avg", "gradient_boosting"]
    labels = ["Naive (t-96)", "Promedio estacional", "Gradient Boosting"]
    accs = [results[n]["accuracy_mean"] for n in names]

    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                          "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, accs, color=CATEGORICAL)
    for bar, acc in zip(bars, accs):
        ax.annotate(f"{acc:.1f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("Accuracy promedio (validación, últimos 7 días)")
    ax.set_title("Comparación de modelos — Pulso TransMi")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linewidth=0.6, color="#e1e0d9")
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "comparacion_modelos.png", dpi=160, facecolor=SURFACE)

    print(json.dumps({k: v for k, v in results.items() if isinstance(v, dict) and "accuracy_mean" in v},
                      indent=2, default=str))
    for name in names:
        print(f"{name}: accuracy_mean={results[name]['accuracy_mean']:.2f}  wape_mean={results[name]['wape_mean']:.4f}")


if __name__ == "__main__":
    main()
