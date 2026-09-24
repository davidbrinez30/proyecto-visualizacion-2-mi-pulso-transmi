"""Descubre como obtener las observaciones EN VIVO que libera el profesor
durante la evaluacion real, distintas del dataset estatico que baja
fetch_data.py (siempre las mismas 51.840 filas, corte fijo).

El README del SDK dice que los datos nuevos llegan "via consultas
recurrentes a la API" (/v1/observations), no por descarga de archivo. Este
script prueba ese y otros paths candidatos, con varios parametros de
consulta, para encontrar el formato real.
"""
from __future__ import annotations

import json
import os

import requests

BASE_URL = os.environ.get("PULSO_API_URL", "https://pulso-transmi.72-60-245-2.sslip.io").rstrip("/")
API_KEY = os.environ.get("PULSO_API_KEY", "")
HEADERS = {"Authorization": f"Bearer {API_KEY}"}


def show(label: str, resp: requests.Response) -> None:
    body = resp.text
    if len(body) > 2000:
        body = body[:2000] + f"... (truncado, {len(resp.text)} caracteres totales)"
    print(f"--- {label} -> HTTP {resp.status_code} ---")
    print(body)
    print()


def main() -> None:
    print("=== openapi.json: lista completa de paths ===")
    spec_resp = requests.get(f"{BASE_URL}/openapi.json", headers=HEADERS, timeout=20)
    spec_resp.raise_for_status()
    spec = spec_resp.json()
    paths = list(spec.get("paths", {}).keys())
    print(json.dumps(paths, indent=2, ensure_ascii=False))

    print("\n=== Detalle de todo path que contenga 'observ', 'data', 'stream', 'feed', 'context' ===")
    for path, item in spec.get("paths", {}).items():
        if any(k in path.lower() for k in ("observ", "data", "stream", "feed", "context")):
            print(f"--- {path} ---")
            print(json.dumps(item, indent=2, ensure_ascii=False))
            print()

    print("\n=== Pruebas directas a /v1/observations con distintos parametros ===")
    candidate_paths = [
        "/v1/observations",
        "/v1/observations/latest",
        "/v1/observations/live",
        "/v1/data/observations",
    ]
    query_variants = [
        "",
        "?limit=5",
        "?since=2026-09-08T00:00:00Z",
        "?after=2026-09-08T00:00:00Z",
        "?station_id=02300&limit=5",
    ]
    for path in candidate_paths:
        for q in query_variants:
            try:
                r = requests.get(f"{BASE_URL}{path}{q}", headers=HEADERS, timeout=20)
                print(f"GET {path}{q} -> {r.status_code}")
                if r.status_code == 200:
                    show(f"{path}{q}", r)
            except Exception as exc:  # noqa: BLE001
                print(f"GET {path}{q} -> ERROR {exc}")

    print("\n=== /v1/meta completo (puede tener info de version/release del dataset) ===")
    r = requests.get(f"{BASE_URL}/v1/meta", headers=HEADERS, timeout=20)
    print(f"GET /v1/meta -> {r.status_code}")
    print(json.dumps(r.json(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
