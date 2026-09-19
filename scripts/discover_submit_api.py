"""Descubre el endpoint real de envio de predicciones del API de Pulso TransMi.

El SDK oficial (pulso_transmi.client) todavia solo trae metodos de lectura,
pero el profesor entrego una API key nueva (PULSO_API_KEY) para un mecanismo
de submissions que aun no esta documentado. Este script prueba, contra la
misma API base (PULSO_API_URL), varias rutas y formas de autenticacion
comunes, e imprime que responde cada una para poder identificar el endpoint
y formato reales antes de integrarlo en el pipeline.

No falla si algo da error: el objetivo es solo recolectar informacion en el
log de GitHub Actions.
"""
from __future__ import annotations

import json
import os

import requests

BASE_URL = os.environ.get("PULSO_API_URL", "https://pulso-transmi.72-60-245-2.sslip.io").rstrip("/")
API_KEY = os.environ.get("PULSO_API_KEY", "")

AUTH_VARIANTS = {
    "Authorization Bearer": {"Authorization": f"Bearer {API_KEY}"},
    "apikey header": {"apikey": API_KEY},
    "X-API-Key header": {"X-API-Key": API_KEY},
}

# Rutas candidatas para el spec de la API y para un posible endpoint de submissions.
SPEC_PATHS = ["/openapi.json", "/docs", "/redoc"]
CANDIDATE_SUBMIT_PATHS = [
    "/v1/predictions",
    "/v1/submissions",
    "/v1/submit",
    "/v1/forecast",
    "/v1/forecasts",
]


def show(label: str, resp: requests.Response) -> None:
    body = resp.text
    if len(body) > 1500:
        body = body[:1500] + f"... (truncado, {len(resp.text)} caracteres totales)"
    print(f"--- {label} -> HTTP {resp.status_code} ---")
    print(body)
    print()


def main() -> None:
    if not API_KEY:
        print("PULSO_API_KEY no esta configurada en el entorno; solo se probaran rutas sin auth.")

    print("=== 1) Spec de la API (openapi.json / docs) ===")
    for path in SPEC_PATHS:
        for auth_name, headers in AUTH_VARIANTS.items():
            try:
                r = requests.get(f"{BASE_URL}{path}", headers=headers, timeout=20)
                print(f"GET {path} [{auth_name}] -> {r.status_code} ({len(r.text)} bytes)")
            except Exception as exc:  # noqa: BLE001
                print(f"GET {path} [{auth_name}] -> ERROR {exc}")

    # Si algun openapi.json respondio 200, lo mostramos completo (para ver todos los paths reales).
    print("\n=== 2) Contenido completo de openapi.json (si alguno funciono) ===")
    for auth_name, headers in AUTH_VARIANTS.items():
        try:
            r = requests.get(f"{BASE_URL}/openapi.json", headers=headers, timeout=20)
            if r.status_code == 200:
                try:
                    spec = r.json()
                    paths = list(spec.get("paths", {}).keys())
                    print(f"[{auth_name}] paths encontrados en openapi.json: {json.dumps(paths, indent=2, ensure_ascii=False)}")
                except Exception:
                    show(f"openapi.json crudo [{auth_name}]", r)
                break
        except Exception as exc:  # noqa: BLE001
            print(f"openapi.json [{auth_name}] -> ERROR {exc}")

    print("\n=== 3) Rutas candidatas de submission (GET, solo para ver si existen) ===")
    for path in CANDIDATE_SUBMIT_PATHS:
        for auth_name, headers in AUTH_VARIANTS.items():
            try:
                r = requests.get(f"{BASE_URL}{path}", headers=headers, timeout=20)
                print(f"GET {path} [{auth_name}] -> {r.status_code}")
                if r.status_code not in (404, 405):
                    show(f"{path} [{auth_name}]", r)
            except Exception as exc:  # noqa: BLE001
                print(f"GET {path} [{auth_name}] -> ERROR {exc}")

    print("\n=== 4) /v1/meta con cada variante de auth (endpoint que sabemos que existe) ===")
    for auth_name, headers in AUTH_VARIANTS.items():
        try:
            r = requests.get(f"{BASE_URL}/v1/meta", headers=headers, timeout=20)
            print(f"GET /v1/meta [{auth_name}] -> {r.status_code}")
        except Exception as exc:  # noqa: BLE001
            print(f"GET /v1/meta [{auth_name}] -> ERROR {exc}")


if __name__ == "__main__":
    main()
