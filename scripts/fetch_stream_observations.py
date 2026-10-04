"""Trae las observaciones EN VIVO del reto desde /v1/stream/observations
(distinto del dataset estatico que descarga fetch_data.py) y las agrega a
data/observations.csv.

Segun /v1/meta y /v1/clock, el reto corre sobre un "reloj virtual"
(virtual_now, tick_number) que avanza durante la competencia: el dataset
estatico (pulso-transmi-starter-v1) llega hasta 2026-09-08T23:45, y de ahi
en adelante los datos nuevos se liberan por este endpoint de stream, con
paginacion por cursor.

Guarda el ultimo cursor usado en data/stream_cursor.txt para no tener que
re-descargar todo el stream en cada corrida (el pipeline hace commit de
data/, asi que el cursor persiste entre corridas).
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import requests

DATA = Path(__file__).resolve().parent.parent / "data"
CURSOR_FILE = DATA / "stream_cursor.txt"
OBSERVATIONS_CSV = DATA / "observations.csv"
BASE_URL = os.environ.get("PULSO_API_URL", "https://pulso-transmi.72-60-245-2.sslip.io").rstrip("/")
API_KEY = os.environ["PULSO_API_KEY"]
HEADERS = {"Authorization": f"Bearer {API_KEY}"}
PAGE_LIMIT = 5000


def _normalize(r: dict) -> dict:
    """Contrato de observacion v2 (fase final, docs/fase-final.md del reto):
    desde 2026-09-20T12:00Z (virtual) los registros traen
    measurement.value (texto decimal o null) y measurement.quality
    (observed|missing) en vez del campo plano `demand`. Una pagina puede
    mezclar v1 y v2. Un faltante NO es cero: queda como NaN y se descarta."""
    if "measurement" in r:
        m = r.get("measurement") or {}
        val = m.get("value")
        demand = int(round(float(val))) if (val is not None and m.get("quality", "observed") == "observed") else None  # Supabase guarda demand como integer
    else:
        demand = r.get("demand")
    return {"observed_at": r["observed_at"], "station_id": str(r["station_id"]), "demand": demand}


def main() -> None:
    cursor = CURSOR_FILE.read_text().strip() if CURSOR_FILE.exists() else None
    if cursor == "":
        cursor = None

    new_rows: list[dict] = []
    pages = 0
    while True:
        params = {"limit": PAGE_LIMIT}
        if cursor:
            params["cursor"] = cursor
        resp = requests.get(f"{BASE_URL}/v1/stream/observations", headers=HEADERS, params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        data = payload.get("data", [])
        pages += 1
        print(f"Pagina {pages}: {len(data)} filas nuevas (cursor usado: {cursor})")
        if data:
            new_rows.extend(data)
        next_cursor = payload.get("next_cursor")
        if not next_cursor or next_cursor == cursor or not data:
            break
        cursor = next_cursor

    print(f"Total filas nuevas del stream: {len(new_rows)}")

    if new_rows:
        new_df = pd.DataFrame([_normalize(r) for r in new_rows])
        n_missing = int(new_df["demand"].isna().sum())
        n_v2 = sum(1 for r in new_rows if r.get("schema_version") == 2)
        print(f"Registros v2: {n_v2} | faltantes (quality=missing, se descartan, NO se toman como cero): {n_missing}")
        new_df = new_df.dropna(subset=["demand"])
        new_df["station_id"] = new_df["station_id"].astype(str)
        new_df["observed_at"] = pd.to_datetime(new_df["observed_at"], utc=True, errors="coerce")

        if OBSERVATIONS_CSV.exists():
            existing = pd.read_csv(OBSERVATIONS_CSV, dtype={"station_id": "string"})
            # El dataset estatico y el stream en vivo pueden venir con
            # distintos formatos de zona horaria (-05:00 vs UTC); se
            # normaliza todo a UTC antes de combinar para que el CSV
            # resultante quede con un unico formato consistente (si no,
            # cualquier lectura posterior con parse_dates falla en
            # silencio y deja la columna como texto).
            existing["observed_at"] = pd.to_datetime(existing["observed_at"], utc=True, errors="coerce")
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["station_id", "observed_at"], keep="last")
        else:
            combined = new_df
        combined = combined.dropna(subset=["observed_at", "demand"])
        combined["demand"] = combined["demand"].round().astype(int)
        combined = combined.sort_values(["station_id", "observed_at"])
        combined["observed_at"] = combined["observed_at"].apply(lambda ts: ts.isoformat())
        combined.to_csv(OBSERVATIONS_CSV, index=False)
        print(f"data/observations.csv actualizado: {len(combined)} filas totales.")

    if cursor:
        CURSOR_FILE.write_text(cursor)
        print(f"Cursor guardado para la proxima corrida: {cursor}")


if __name__ == "__main__":
    main()
