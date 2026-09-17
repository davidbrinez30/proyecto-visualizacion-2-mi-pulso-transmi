"""Monitoreo de datos/desempeno, decision de reentrenamiento, y persistencia
en Supabase (tablas pipeline_runs / predictions / metrics / drift_signals -
ver docs/er-diagram.md).

Se ejecuta despues de src/train.py y src/predict.py dentro del pipeline de
GitHub Actions. Si SUPABASE_URL / SUPABASE_KEY no estan definidas (por
ejemplo corriendo local), guarda todo en artifacts/ y no intenta conectarse.

Estrategia de reentrenamiento (simple e interpretable a proposito):
  - reentrena si accuracy_mean de la corrida actual cae mas de
    RETRAIN_ACCURACY_DROP puntos por debajo del accuracy_mean historico
    reciente, o si no hay modelo previo.
  - los DRIFT_SIGNALS son informativos (se registran siempre) y ademas
    alimentan esa decision cuando superan su umbral.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pandas as pd
import requests

from src.ingest import load_observations

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"
RETRAIN_ACCURACY_DROP = 5.0  # puntos porcentuales
DRIFT_RELATIVE_CHANGE = 0.25  # 25% de cambio en demanda media estacion-hora


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent.parent
        ).decode().strip()
    except Exception:
        return "unknown"


def detect_data_drift(observations: pd.DataFrame, window_days: int = 7) -> list[dict]:
    """Compara la demanda media por estacion de la ultima ventana contra la
    ventana inmediatamente anterior. Senal simple de "el patron cambio"."""
    obs = observations.copy()
    obs["date"] = obs["observed_at"].dt.floor("D")
    cutoff = obs["observed_at"].max() - pd.Timedelta(days=window_days)
    prev_cutoff = cutoff - pd.Timedelta(days=window_days)

    recent = obs.loc[obs["observed_at"] > cutoff].groupby("station_id")["demand"].mean()
    previous = obs.loc[(obs["observed_at"] > prev_cutoff) & (obs["observed_at"] <= cutoff)].groupby("station_id")["demand"].mean()

    signals = []
    for station_id in recent.index:
        if station_id not in previous.index or previous[station_id] == 0:
            continue
        rel_change = (recent[station_id] - previous[station_id]) / previous[station_id]
        triggered = abs(rel_change) > DRIFT_RELATIVE_CHANGE
        signals.append({
            "station_id": station_id,
            "signal_type": "demand_mean_shift_7d",
            "signal_value": round(float(rel_change), 4),
            "threshold": DRIFT_RELATIVE_CHANGE,
            "triggered": bool(triggered),
        })
    return signals


def decide_retrain(current_accuracy: float, history_accuracy: float | None) -> tuple[bool, str]:
    if history_accuracy is None:
        return True, "sin corrida previa registrada"
    drop = history_accuracy - current_accuracy
    if drop > RETRAIN_ACCURACY_DROP:
        return True, f"accuracy cayo {drop:.1f} pts vs corrida anterior ({history_accuracy:.1f} -> {current_accuracy:.1f})"
    return False, f"accuracy estable ({current_accuracy:.1f}, vs {history_accuracy:.1f} anterior)"


def _supabase_headers(key: str) -> dict:
    return {"apikey": key, "Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def persist_to_supabase(run_row: dict, predictions_df: pd.DataFrame, metrics_rows: list[dict],
                         drift_rows: list[dict]) -> dict:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        return {"skipped": True, "reason": "SUPABASE_URL/SUPABASE_KEY no configuradas"}

    headers = _supabase_headers(key)
    rest = f"{url}/rest/v1"

    run_resp = requests.post(f"{rest}/pipeline_runs", headers={**headers, "Prefer": "return=representation"},
                              json=run_row, timeout=30)
    run_resp.raise_for_status()
    run_id = run_resp.json()[0]["id"]

    if not predictions_df.empty:
        preds = predictions_df.copy()
        preds["run_id"] = run_id
        preds["target_at"] = preds["target_at"].astype(str)
        requests.post(f"{rest}/predictions", headers=headers, json=preds.to_dict("records"), timeout=30).raise_for_status()

    for m in metrics_rows:
        m["run_id"] = run_id
    if metrics_rows:
        requests.post(f"{rest}/metrics", headers=headers, json=metrics_rows, timeout=30).raise_for_status()

    for d in drift_rows:
        d["run_id"] = run_id
    if drift_rows:
        requests.post(f"{rest}/drift_signals", headers=headers, json=drift_rows, timeout=30).raise_for_status()

    return {"skipped": False, "run_id": run_id}


def main():
    metrics_path = ARTIFACTS / "metrics.json"
    predictions_path = ARTIFACTS / "predictions_latest.csv"
    if not metrics_path.exists():
        raise SystemExit("Corre primero src/train.py (falta artifacts/metrics.json)")

    metrics = json.loads(metrics_path.read_text())
    current_accuracy = metrics["gradient_boosting"]["accuracy_mean"]

    history_path = ARTIFACTS / "accuracy_history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    last_accuracy = history[-1]["accuracy_mean"] if history else None

    retrain, reason = decide_retrain(current_accuracy, last_accuracy)

    observations = load_observations()
    drift_signals = detect_data_drift(observations)

    run_row = {
        "git_commit": git_commit(),
        "model_version": f"gbr@{git_commit()}",
        "data_cutoff": str(observations["observed_at"].max()),
        "status": "success",
        "retrained": retrain,
        "retrain_reason": reason,
    }
    metrics_rows = [
        {"station_id": None, "wape": metrics[m]["wape_mean"], "accuracy": metrics[m]["accuracy_mean"],
         "window_start": None, "window_end": None}
        for m in ("naive_96", "seasonal_avg", "gradient_boosting")
    ]

    predictions_df = pd.read_csv(predictions_path) if predictions_path.exists() else pd.DataFrame()

    result = persist_to_supabase(run_row, predictions_df, metrics_rows, drift_signals)

    history.append({"git_commit": run_row["git_commit"], "accuracy_mean": current_accuracy})
    history_path.write_text(json.dumps(history[-30:], indent=2))

    report = {"run": run_row, "drift_signals": drift_signals, "supabase": result}
    (ARTIFACTS / "monitor_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
