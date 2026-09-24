"""Genera y envia la primera submission de predicciones al reto Pulso TransMi.

Flujo:
  1. Consulta GET /v1/forecast-cycles/current para saber el ciclo activo:
     que estaciones, que target_at exacto y con que data_cutoff hay que
     pronosticar (esto lo define el servidor del reto, no nosotros).
  2. Filtra las observaciones historicas a solo lo disponible HASTA ese
     data_cutoff (nunca se usa nada posterior, para no "ver el futuro").
  3. Reusa exactamente la misma logica de prediccion recursiva de
     src/predict.py (predict_station) con el modelo ya entrenado
     (artifacts/model_gbr.joblib). predict_station ya calcula los 4
     horizontes (15/30/45/60 min) de una sola vez por estacion; aqui se
     llama UNA vez por estacion (no una vez por target) y luego se toma,
     para cada target que pida el ciclo, el horizonte que le corresponda
     (el ciclo real del profesor pide los 4 horizontes por estacion, no
     solo 15 min).
  4. Arma el payload segun el schema real de POST /v1/submissions
     (descubierto via /openapi.json) y lo envia con un Idempotency-Key
     unico, para poder reintentar sin duplicar si algo falla a mitad de
     camino.

Requiere PULSO_API_URL y PULSO_API_KEY en el entorno.
"""
from __future__ import annotations

import json
import os
import subprocess
import uuid
from pathlib import Path

import joblib
import pandas as pd
import requests

from src.ingest import load_all
from src.predict import predict_station

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"
BASE_URL = os.environ.get("PULSO_API_URL", "https://pulso-transmi.72-60-245-2.sslip.io").rstrip("/")
API_KEY = os.environ["PULSO_API_KEY"]


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent.parent
        ).decode().strip()
    except Exception:
        return "unknown"


def main() -> None:
    headers = {"Authorization": f"Bearer {API_KEY}"}

    cycle_resp = requests.get(f"{BASE_URL}/v1/forecast-cycles/current", headers=headers, timeout=20)
    if cycle_resp.status_code == 404:
        print("No hay ningun ciclo activo en este momento (404 en /v1/forecast-cycles/current); no se envia nada.")
        return
    cycle_resp.raise_for_status()
    cycle = cycle_resp.json()
    print("Ciclo activo:")
    print(json.dumps(cycle, indent=2, ensure_ascii=False))

    if cycle.get("state") != "open":
        raise SystemExit(f"El ciclo {cycle.get('cycle_id')} no esta abierto (state={cycle.get('state')}); no se envia nada.")

    cutoff = pd.Timestamp(cycle["data_cutoff"])
    targets = cycle["targets"]

    bundle = joblib.load(ARTIFACTS / "model_gbr.joblib")
    model, feature_cols = bundle["model"], bundle["features"]

    stations, observations, context = load_all()
    obs = observations[observations["observed_at"] <= cutoff]
    ctx = context[context["observed_at"] <= cutoff].sort_values("observed_at")
    if ctx.empty:
        raise SystemExit("No hay contexto (clima/eventos) disponible hasta el data_cutoff del ciclo.")
    last_context = ctx.iloc[-1]
    context_row = {c: last_context[c] for c in ("rain_mm", "temperature_c", "event_intensity")}

    # Agrupa los targets por estacion para llamar predict_station una sola
    # vez por estacion (calcula los 4 horizontes de una) en vez de una vez
    # por target, que era el bug: antes se llamaba por cada target y se
    # exigia que el horizonte fuera siempre 15 min.
    targets_by_station: dict[str, list[dict]] = {}
    for t in targets:
        targets_by_station.setdefault(t["station_id"], []).append(t)

    predictions = []
    skipped = []
    for station_id, station_targets in targets_by_station.items():
        grp = obs[obs["station_id"] == station_id].sort_values("observed_at")
        if grp.empty:
            raise SystemExit(f"Sin observaciones historicas para la estacion {station_id} antes del data_cutoff.")
        demand_hist = pd.Series(grp["demand"].values, index=grp["observed_at"].values)
        last_time = grp["observed_at"].max()

        # Los 4 horizontes (15/30/45/60 min) de esta estacion, calculados una
        # sola vez. horizon_steps 1..4 corresponde a 15/30/45/60 min.
        results_by_step = {
            r["horizon_steps"]: r
            for r in predict_station(demand_hist, context_row, last_time, model, feature_cols)
        }

        for t in station_targets:
            expected_target_at = pd.Timestamp(t["target_at"])
            horizon_min = t["horizon_minutes"]

            if horizon_min % 15 != 0:
                skipped.append((station_id, horizon_min, "horizonte no es multiplo de 15 min"))
                continue
            step = horizon_min // 15
            result = results_by_step.get(step)
            if result is None:
                skipped.append((station_id, horizon_min, f"predict_station no calculo el paso {step} (solo calcula hasta 60 min)"))
                continue

            computed_target_at = pd.Timestamp(result["target_at"])
            if computed_target_at.tz_convert("UTC") != expected_target_at.tz_convert("UTC"):
                skipped.append((
                    station_id, horizon_min,
                    f"target_at calculado ({computed_target_at}) no coincide con el esperado ({expected_target_at})",
                ))
                continue

            predictions.append({
                "station_id": station_id,
                "target_at": t["target_at"],
                "value": result["predicted_demand"],
            })

    if skipped:
        print(f"\nAVISO: {len(skipped)} targets omitidos (no se pudieron predecir):")
        for station_id, horizon_min, reason in skipped:
            print(f"  - {station_id} ({horizon_min} min): {reason}")

    if not predictions:
        raise SystemExit("Ningun target del ciclo se pudo predecir; no se envia nada.")

    commit = git_commit()
    client_run_id = f"pulso-transmi-{commit}-{uuid.uuid4().hex[:8]}"
    payload = {
        "schema_version": "1.0",
        "cycle_id": cycle["cycle_id"],
        "client_run_id": client_run_id,
        "data_cutoff": cycle["data_cutoff"],
        "model": {
            "version": f"gbr-{commit}",
            "trained_at": None,
            "training_data_end": cycle["data_cutoff"],
            "git_commit": commit if commit != "unknown" and len(commit) >= 7 else None,
        },
        "predictions": predictions,
    }

    print(f"\n{len(predictions)} predicciones a enviar (de {len(targets)} targets pedidos por el ciclo).")
    print("\nPayload a enviar:")
    print(json.dumps(payload, indent=2, ensure_ascii=False))

    resp = requests.post(
        f"{BASE_URL}/v1/submissions",
        headers={**headers, "Idempotency-Key": client_run_id, "Content-Type": "application/json"},
        json=payload,
        timeout=30,
    )
    print(f"\nPOST /v1/submissions -> {resp.status_code}")
    print(resp.text)
    resp.raise_for_status()
    print("\nSubmission enviada correctamente.")


if __name__ == "__main__":
    main()

