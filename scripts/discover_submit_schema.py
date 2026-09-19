"""Segunda fase de descubrimiento: formato exacto de POST /v1/submissions.

Ya sabemos (via /openapi.json) que la API real del reto tiene, entre otros,
estos endpoints nuevos que el SDK oficial todavia no envuelve:
  POST /v1/submissions              <- aqui se mandan las predicciones
  GET  /v1/submissions/{id}
  GET  /v1/forecast-cycles/current  <- ventana de pronostico activa
  GET  /v1/me
  GET  /v1/clock
  GET  /v1/leaderboard, /v1/portal/leaderboard

Este script:
  1. Trae el openapi.json completo y muestra el schema detallado (parametros,
     requestBody, responses) de esos paths, resolviendo referencias ($ref)
     a los esquemas en components/schemas.
  2. Llama (GET, sin efectos secundarios) a /v1/me, /v1/forecast-cycles/current
     y /v1/clock para ver los datos reales que devuelve el servidor.

No hace ningun POST todavia: el objetivo es solo reunir la informacion
necesaria para construir el envio real en un siguiente paso.
"""
from __future__ import annotations

import json
import os

import requests

BASE_URL = os.environ.get("PULSO_API_URL", "https://pulso-transmi.72-60-245-2.sslip.io").rstrip("/")
API_KEY = os.environ.get("PULSO_API_KEY", "")
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

PATHS_OF_INTEREST = [
    "/v1/submissions",
    "/v1/submissions/{submission_id}",
    "/v1/forecast-cycles/current",
    "/v1/me",
    "/v1/clock",
    "/v1/leaderboard",
    "/v1/portal/leaderboard",
]


def resolve_refs(node, components, seen=None):
    """Resuelve recursivamente los $ref de un fragmento de OpenAPI contra components."""
    if seen is None:
        seen = set()
    if isinstance(node, dict):
        if "$ref" in node and node["$ref"].startswith("#/components/"):
            ref = node["$ref"]
            if ref in seen:
                return {"$ref": ref, "_circular": True}
            seen = seen | {ref}
            parts = ref.split("/")[2:]  # ["schemas", "NombreDelSchema"]
            target = components
            for p in parts:
                target = target.get(p, {})
            return resolve_refs(target, components, seen)
        return {k: resolve_refs(v, components, seen) for k, v in node.items()}
    if isinstance(node, list):
        return [resolve_refs(v, components, seen) for v in node]
    return node


def main() -> None:
    spec_resp = requests.get(f"{BASE_URL}/openapi.json", headers=HEADERS, timeout=20)
    spec_resp.raise_for_status()
    spec = spec_resp.json()
    components = spec.get("components", {})
    paths = spec.get("paths", {})

    print("=== 1) Schema detallado de los endpoints de interes ===\n")
    for path in PATHS_OF_INTEREST:
        item = paths.get(path)
        if item is None:
            print(f"--- {path}: NO aparece en openapi.json ---\n")
            continue
        resolved = resolve_refs(item, components)
        print(f"--- {path} ---")
        print(json.dumps(resolved, indent=2, ensure_ascii=False))
        print()

    print("\n=== 2) Datos reales (GET, sin efectos secundarios) ===\n")
    for path in ["/v1/me", "/v1/forecast-cycles/current", "/v1/clock"]:
        try:
            r = requests.get(f"{BASE_URL}{path}", headers=HEADERS, timeout=20)
            print(f"GET {path} -> {r.status_code}")
            try:
                print(json.dumps(r.json(), indent=2, ensure_ascii=False))
            except Exception:
                print(r.text[:2000])
            print()
        except Exception as exc:  # noqa: BLE001
            print(f"GET {path} -> ERROR {exc}\n")


if __name__ == "__main__":
    main()
