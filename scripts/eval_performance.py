"""Mide el desempeno REAL de nuestras predicciones contra lo que el profesor
termina publicando como dato real (observations), y lo guarda en la tabla
submission_performance de Supabase.

Como el profesor libera el dato real varias horas despues del target_at que
pronosticamos, este script se corre en cada pasada del pipeline automatico
y solo evalua las predicciones que:
  1. ya tienen target_at <= ahora (ya "deberia" existir el dato real), y
  2. ya aparecen en la tabla observations con ese mismo station_id/observed_at
     (es decir, el profesor ya lo publico), y
  3. todavia no se evaluaron (no estan en submission_performance).

Requiere que exista la tabla submission_performance (ver
migrations/0002_tabla_desempeno_real.sql) y que SUPABASE_URL/SUPABASE_KEY
esten configuradas; si no, no hace nada.
"""
from __future__ import annotations

import os
from urllib.parse import quote

import pandas as pd
import requests

PAGE_SIZE = 1000


def _headers(key: str) -> dict:
    return {"apikey": key, "Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _get_all(rest: str, headers: dict, path: str) -> list[dict]:
    rows: list[dict] = []
    offset = 0
    while True:
        sep = "&" if "?" in path else "?"
        resp = requests.get(
            f"{rest}/{path}{sep}limit={PAGE_SIZE}&offset={offset}",
            headers=headers, timeout=30,
        )
        resp.raise_for_status()
        page = resp.json()
        rows.extend(page)
        if len(page) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    return rows


def main() -> None:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        print("SUPABASE_URL/SUPABASE_KEY no configuradas; se omite la evaluacion de desempeno.")
        return

    headers = _headers(key)
    rest = f"{url}/rest/v1"
    now = pd.Timestamp.now(tz="UTC")

    # Predicciones ya vencidas (target_at <= ahora) que aun no se evaluaron.
    already = {(r["station_id"], r["target_at"]) for r in _get_all(rest, headers, "submission_performance?select=station_id,target_at")}
    # El "+00:00" del ISO 8601 debe ir URL-encodeado (%2B); sin encodear,
    # PostgREST/el servidor lo interpreta como espacio y responde 400.
    preds = _get_all(
        rest, headers,
        f"predictions?select=station_id,target_at,predicted_demand&target_at=lte.{quote(now.isoformat())}",
    )
    pending = [p for p in preds if (p["station_id"], p["target_at"]) not in already]
    if not pending:
        print("No hay predicciones vencidas pendientes de evaluar.")
        return

    obs_by_key: dict[tuple[str, str], float] = {}
    for p in pending:
        obs_by_key.setdefault(p["station_id"], None)
    for station_id in list(obs_by_key.keys()):
        rows = _get_all(
            rest, headers,
            f"observations?select=observed_at,demand&station_id=eq.{station_id}",
        )
        for r in rows:
            obs_by_key[(station_id, r["observed_at"])] = r["demand"]
        del obs_by_key[station_id]

    rows_to_insert = []
    for p in pending:
        key_ = (p["station_id"], p["target_at"])
        actual = obs_by_key.get(key_)
        if actual is None:
            continue  # el profesor todavia no publica el dato real para ese target_at
        predicted = p["predicted_demand"]
        abs_error = abs(actual - predicted)
        wape = abs_error / actual if actual else None
        accuracy = max(0.0, 100 * (1 - wape)) if wape is not None else None
        rows_to_insert.append({
            "station_id": p["station_id"],
            "target_at": p["target_at"],
            "predicted_demand": predicted,
            "actual_demand": actual,
            "abs_error": abs_error,
            "wape": wape,
            "accuracy": accuracy,
            "evaluated_at": now.isoformat(),
        })

    if not rows_to_insert:
        print(f"{len(pending)} predicciones vencidas, pero el profesor aun no publica el dato real de ninguna.")
        return

    # La tabla predictions puede tener mas de una fila para el mismo
    # (station_id, target_at) -- por ejemplo si el pipeline recalculo y
    # reinserto la prediccion en corridas sucesivas sin que esa tabla haga
    # upsert por esa misma llave. Si se manda mas de una fila con la misma
    # llave en el mismo INSERT ... ON CONFLICT, Postgres rechaza todo el
    # lote ("cannot affect row a second time"), asi que aqui se deja solo
    # la ultima por llave antes de subir.
    dedup: dict[tuple[str, str], dict] = {}
    for row in rows_to_insert:
        dedup[(row["station_id"], row["target_at"])] = row
    if len(dedup) != len(rows_to_insert):
        print(f"AVISO: {len(rows_to_insert) - len(dedup)} filas duplicadas (misma station_id+target_at) descartadas antes de insertar.")
    rows_to_insert = list(dedup.values())

    resp = requests.post(
        f"{rest}/submission_performance?on_conflict=station_id,target_at",
        headers={**headers, "Prefer": "resolution=merge-duplicates,return=minimal"},
        json=rows_to_insert, timeout=30,
    )
    if not resp.ok:
        print(f"SUPABASE ERROR submission_performance {resp.status_code}: {resp.text[:500]}")
    resp.raise_for_status()

    accuracies = [r["accuracy"] for r in rows_to_insert if r["accuracy"] is not None]
    avg = sum(accuracies) / len(accuracies) if accuracies else None
    print(f"Evaluadas {len(rows_to_insert)} predicciones. Accuracy real promedio de esta tanda: {avg}")


if __name__ == "__main__":
    main()
