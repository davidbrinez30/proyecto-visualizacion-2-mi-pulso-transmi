"""Carga de datos crudos del reto Pulso TransMi.

En la primera etapa lee el corte estatico en data/*.csv (bajado por
scripts/fetch_data.py). En la etapa incremental, la version que corra dentro
de GitHub Actions debe cambiar a consultar solo observaciones nuevas via
`pulso_transmi.PulsoTransmiClient(...).observations_dataframe(start=...)`,
usando el cursor/ultimo timestamp persistido (ver src/monitor.py).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_stations(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    return pd.read_csv(data_dir / "stations.csv", dtype={"station_id": "string"})


def load_observations(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    obs = pd.read_csv(data_dir / "observations.csv", dtype={"station_id": "string"})
    # parse_dates de read_csv falla en silencio (deja texto sin avisar) si el
    # CSV tiene observed_at con formatos de zona horaria mezclados (p.ej. el
    # dataset estatico en -05:00 combinado con datos del stream en UTC), asi
    # que se normaliza explicitamente aqui en vez de confiar en parse_dates.
    obs["observed_at"] = pd.to_datetime(obs["observed_at"], utc=True, errors="coerce")
    obs = obs.dropna(subset=["observed_at"])
    return obs.sort_values(["station_id", "observed_at"]).reset_index(drop=True)


def load_context(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    ctx = pd.read_csv(data_dir / "context.csv")
    ctx["observed_at"] = pd.to_datetime(ctx["observed_at"], utc=True, errors="coerce")
    ctx = ctx.dropna(subset=["observed_at"])
    return ctx.sort_values("observed_at").reset_index(drop=True)


def load_all(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return load_stations(data_dir), load_observations(data_dir), load_context(data_dir)
