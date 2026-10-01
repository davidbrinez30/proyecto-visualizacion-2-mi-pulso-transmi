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
from src.ensemble import live_forecast

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
    stations, observations, context = load_all()
    obs = observations[observations["observed_at"] <= cutoff]
    if obs.empty:
        raise SystemExit("Sin observaciones historicas antes del data_cutoff del ciclo.")

    # Ensamble adaptativo (src/ensemble.py): usa SOLO observaciones <= data_cutoff
    # del ciclo, tanto para los pronosticos como para los pesos por estacion.
    # Si el GBR vigente se entreno con datos POSTERIORES al corte del ciclo,
    # no se usa en esta entrega (contrato: solo informacion hasta el corte);
    # el ensamble sigue con los candidatos estadisticos.
    use_bundle = bundle
    train_cutoff = bundle.get("train_cutoff") if isinstance(bundle, dict) else None
    if train_cutoff is not None and pd.Timestamp(train_cutoff) > cutoff:
        print(f"AVISO: el GBR se entreno hasta {train_cutoff} (> corte {cutoff}); se excluye de esta entrega.")
        use_bundle = {**bundle, "model": None}
    preds_df, _weights = live_forecast(obs, context, use_bundle, cutoff=cutoff)
    by_key = {(r.station_id, pd.Timestamp(r.target_at).tz_convert("UTC")): r.predicted_demand
              for r in preds_df.itertuples()}

    predictions = []
    skipped = []
    for t in targets:
        station_id = str(t["station_id"])
        horizon_min = t["horizon_minutes"]
        expected_target_at = pd.Timestamp(t["target_at"]).tz_convert("UTC")
        value = by_key.get((station_id, expected_target_at))
        if value is None:
            skipped.append((station_id, horizon_min,
                            f"no hay prediccion para target_at {expected_target_at} (ultimo dato: {obs['observed_at'].max()})"))
            continue
        predictions.append({"station_id": station_id, "target_at": t["target_at"], "value": float(value)})

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
            "version": f"ens-{bundle.get('model_id', 'gbr')}",
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
