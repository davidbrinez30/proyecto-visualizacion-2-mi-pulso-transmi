"""Mantiene Supabase (observations/context) al dia con lo ultimo que el
profesor publica en el reto.

fetch_data.py ya trae el dataset completo mas reciente a data/*.csv en cada
corrida. Este script sube a Supabase solo las filas mas nuevas que lo que ya
esta guardado alla (consulta el maximo observed_at existente por tabla y
sube el resto), para no reenviar decenas de miles de filas sin cambios en
cada corrida del pipeline automatico.

Si SUPABASE_URL / SUPABASE_KEY no estan definidas, no hace nada (permite
correr local sin Supabase).
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import requests

DATA = Path(__file__).resolve().parent.parent / "data"
BATCH_SIZE = 2000


def _headers(key: str) -> dict:
    return {"apikey": key, "Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _max_observed_at(rest: str, headers: dict, table: str) -> pd.Timestamp | None:
    resp = requests.get(
        f"{rest}/{table}?select=observed_at&order=observed_at.desc&limit=1",
        headers=headers, timeout=30,
    )
    resp.raise_for_status()
    rows = resp.json()
    if not rows:
        return None
    return pd.Timestamp(rows[0]["observed_at"]).tz_convert("UTC")


def _upsert(rest: str, headers: dict, table: str, df: pd.DataFrame, on_conflict: str) -> int:
    if df.empty:
        return 0
    records = df.copy()
    records["observed_at"] = records["observed_at"].apply(lambda ts: ts.isoformat())
    total = 0
    for start in range(0, len(records), BATCH_SIZE):
        chunk = records.iloc[start:start + BATCH_SIZE].to_dict("records")
        resp = requests.post(
            f"{rest}/{table}?on_conflict={on_conflict}",
            headers={**headers, "Prefer": "resolution=merge-duplicates,return=minimal"},
            json=chunk, timeout=60,
        )
        if not resp.ok:
            print(f"SUPABASE ERROR {table} {resp.status_code}: {resp.text[:500]}")
        resp.raise_for_status()
        total += len(chunk)
    return total


def sync_table(rest: str, headers: dict, table: str, csv_name: str, dtype: dict) -> None:
    csv_path = DATA / csv_name
    if not csv_path.exists():
        print(f"{csv_name} no existe, se omite {table}")
        return
    df = pd.read_csv(csv_path, dtype=dtype)
    # observed_at puede venir con distintos formatos de zona horaria mezclados
    # (el dataset estatico usa -05:00, el stream en vivo llega en UTC), lo que
    # hace que parse_dates de read_csv falle en silencio y deje la columna
    # como texto. Se normaliza explicitamente a UTC aqui.
    df["observed_at"] = pd.to_datetime(df["observed_at"], utc=True, errors="coerce")
    if df["observed_at"].isna().any():
        bad = df["observed_at"].isna().sum()
        print(f"AVISO: {bad} filas de {csv_name} tienen observed_at invalido y se descartan.")
        df = df.dropna(subset=["observed_at"])
    cutoff = _max_observed_at(rest, headers, table)
    new_rows = df if cutoff is None else df[df["observed_at"] > cutoff]
    n = _upsert(rest, headers, table, new_rows, on_conflict="station_id,observed_at" if "station_id" in df.columns else "observed_at")
    print(f"{table}: {n} filas nuevas subidas (cutoff previo: {cutoff})")


def main() -> None:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        print("SUPABASE_URL/SUPABASE_KEY no configuradas; se omite la sincronizacion.")
        return
    headers = _headers(key)
    rest = f"{url}/rest/v1"

    sync_table(rest, headers, "observations", "observations.csv", dtype={"station_id": "string"})
    sync_table(rest, headers, "context", "context.csv", dtype={})


if __name__ == "__main__":
    main()
