"""Reconstruye data/observations.csv y data/context.csv COMPLETOS desde
Supabase, para que el entrenamiento nunca dependa de que
fetch_data.py + fetch_stream_observations.py hayan mantenido el CSV local
consistente entre corridas.

Por que existe: el 28/09 se encontro un hueco de 3 dias (10-12 sep) en
data/observations.csv que Supabase nunca tuvo (Supabase se llena via
upsert idempotente en sync_supabase.py, asi que nunca pierde filas ya
subidas). La causa exacta en fetch_data.py/fetch_stream_observations.py no
se pudo confirmar sin logs de esa corrida, asi que en vez de perseguir ese
bug puntual, este script hace que Supabase sea la UNICA fuente de verdad
para lo que entrena el modelo: se corre despues de sync_supabase.py (que
ya subio lo nuevo que trajeron fetch_data.py/fetch_stream_observations.py)
y antes de src/train.py.

Si SUPABASE_URL/SUPABASE_KEY no estan configuradas (por ejemplo corriendo
local sin Supabase), no hace nada y deja los CSV como estaban -- igual que
el resto de scripts/*.py de este pipeline.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import requests

DATA = Path(__file__).resolve().parent.parent / "data"
PAGE_SIZE = 5000


def _headers(key: str) -> dict:
    return {"apikey": key, "Authorization": f"Bearer {key}"}


def _get_all(rest: str, headers: dict, table: str, select: str, order: str) -> list[dict]:
    # OJO (28/09): Supabase/PostgREST limita cada respuesta a un maximo fijo
    # de filas (por defecto 1000) sin importar el `limit` que pidamos -- una
    # corrida real devolvio solo 1000 filas cuando la tabla tenia 59,868 y
    # el pipeline entreno con casi nada. La condicion de parada NO puede ser
    # "recibi menos de lo que pedi" (PAGE_SIZE), porque el servidor puede
    # recortar cada pagina a su propio tope aunque todavia queden mas datos.
    # La unica senal confiable de "se acabo" es una pagina vacia.
    rows: list[dict] = []
    offset = 0
    while True:
        resp = requests.get(
            f"{rest}/{table}?select={select}&order={order}&limit={PAGE_SIZE}&offset={offset}",
            headers=headers, timeout=60,
        )
        resp.raise_for_status()
        page = resp.json()
        if not page:
            break
        rows.extend(page)
        offset += len(page)
    return rows


def _write_csv(rows: list[dict], columns: list[str], sort_cols: list[str], path: Path) -> int:
    if not rows:
        print(f"{path.name}: Supabase no devolvio filas, se deja el CSV existente sin tocar.")
        return 0
    df = pd.DataFrame(rows)[columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)
    df.to_csv(path, index=False)
    return len(df)


def main() -> None:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        print("SUPABASE_URL/SUPABASE_KEY no configuradas; se omite la reconstruccion desde Supabase.")
        return

    headers = _headers(key)
    rest = f"{url}/rest/v1"

    obs_rows = _get_all(rest, headers, "observations", "observed_at,station_id,demand", "station_id.asc,observed_at.asc")
    n_obs = _write_csv(obs_rows, ["observed_at", "station_id", "demand"], ["station_id", "observed_at"], DATA / "observations.csv")
    print(f"data/observations.csv reconstruido desde Supabase: {n_obs} filas.")

    ctx_rows = _get_all(rest, headers, "context", "observed_at,rain_mm,rain_forecast,temperature_c,temperature_forecast,event_intensity", "observed_at.asc")
    n_ctx = _write_csv(ctx_rows, ["observed_at", "rain_mm", "rain_forecast", "temperature_c", "temperature_forecast", "event_intensity"], ["observed_at"], DATA / "context.csv")
    print(f"data/context.csv reconstruido desde Supabase: {n_ctx} filas.")


if __name__ == "__main__":
    main()
