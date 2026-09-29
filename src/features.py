"""Feature engineering para el pronostico de demanda.

Enfoque de HORIZONTE DIRECTO (sin recursividad): en vez de predecir 15 min
adelante y despues usar esa prediccion como si fuera un dato real para
predecir 30/45/60 min (lo que acumulaba error -- confirmado empiricamente:
82.7% de accuracy a 15 min caia a 69% a 60 min), todas las features de
"ancla" (anchor_*) se calculan SIEMPRE con datos 100% reales hasta el
ultimo momento conocido, y el horizonte (15/30/45/60 min) se le pasa al
modelo como una feature mas. Asi el modelo predice cada horizonte
directamente, sin encadenar predicciones sobre predicciones.

Todas las features usadas son calculables en el momento de prediccion (no
usan informacion futura), condicion necesaria para que el backtesting
temporal sea valido.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PERIODS_PER_DAY = 96  # 24h * 4 (15 min)
HORIZONS = (1, 2, 3, 4)  # pasos de 15 min: 15/30/45/60 min adelante
ANCHOR_LAGS = (4, 96, 96 * 7)  # 1h, 1 dia, 1 semana ANTES del ancla
ROLLING_WINDOWS = (4, 96)  # 1h, 1 dia

# El modelo final que se despliega es una MEZCLA de Gradient Boosting (que
# reacciona rapido a lo que acaba de pasar, via anchor_value/anchor_lag_*) y
# un promedio historico estacion+dia+hora+minuto (que captura el patron
# periodico fuerte de la demanda, sin acumular error entre horizontes).
# Validado con datos reales del reto: GBR solo = 82.2% de accuracy, mezcla
# con peso 0.85 hacia el promedio estacional = 87.7%, y ademas la caida por
# horizonte (15->60 min) es mucho mas plana (87.8% -> 87.6%, contra 84.5%
# -> 78.1% de GBR solo). El promedio estacional se recalcula con TODOS los
# datos disponibles en cada corrida (igual que el modelo), asi que tambien
# se actualiza solo si el patron cambia (drift).
BLEND_WEIGHT_SEASONAL = 0.85


def seasonal_lookup(observations: pd.DataFrame) -> pd.Series:
    """Promedio historico de demanda por (station_id, dow, hour, minute),
    calculado sobre las observaciones que se le pasen (para entrenamiento:
    solo datos antes del cutoff de validacion; para el modelo final que se
    despliega: TODOS los datos disponibles)."""
    obs = observations.copy()
    obs["hour"] = obs["observed_at"].dt.hour
    obs["minute"] = obs["observed_at"].dt.minute
    obs["dow"] = obs["observed_at"].dt.dayofweek
    return obs.groupby(["station_id", "dow", "hour", "minute"])["demand"].mean()


def seasonal_value(lookup: pd.Series, station_id: str, target_at: pd.Timestamp, fallback: float) -> float:
    """Busca el promedio estacional para (station_id, target_at); si esa
    combinacion exacta nunca se vio en el historico, usa `fallback` (por
    ejemplo, el valor ancla mas reciente) en vez de fallar."""
    key = (station_id, target_at.dayofweek, target_at.hour, target_at.minute)
    value = lookup.get(key)
    return float(value) if value is not None and not pd.isna(value) else float(fallback)


def add_calendar_features(df: pd.DataFrame, time_col: str = "observed_at") -> pd.DataFrame:
    df = df.copy()
    ts = df[time_col]
    df["hour"] = ts.dt.hour
    df["minute_of_day"] = ts.dt.hour * 60 + ts.dt.minute
    df["dow"] = ts.dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["minute_of_day"] / (24 * 60))
    df["hour_cos"] = np.cos(2 * np.pi * df["minute_of_day"] / (24 * 60))
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
    return df


def add_anchor_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features calculadas en el momento ANCLA (el ultimo dato real conocido
    en ese punto de la serie) -- nunca en el momento objetivo (target_at).
    Se usan igual para entrenar (anclas historicas con su target real ya
    conocido) y para predecir en produccion (last_time = el dato mas
    reciente que ya llego). Por eso predict_station ya NO necesita
    recursividad: cada horizonte usa exactamente las mismas features
    ancla, calculadas una sola vez con datos reales.

    Requiere df ordenado por (station_id, observed_at) y sin huecos.
    """
    df = df.copy()
    grouped = df.groupby("station_id")["demand"]
    df["anchor_value"] = df["demand"]  # el dato mas reciente conocido (lag 0)
    for lag in ANCHOR_LAGS:
        df[f"anchor_lag_{lag}"] = grouped.shift(lag)
    for window in ROLLING_WINDOWS:
        df[f"anchor_roll_mean_{window}"] = grouped.rolling(window).mean().reset_index(level=0, drop=True)
    return df


def build_multihorizon_frame(observations: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    """Arma el set de entrenamiento/evaluacion de horizonte directo: cada
    fila es (ancla t, horizonte h) -> target = demanda real en t + 15*h
    min. Las features de calendario (hora/dia) se calculan sobre el
    momento OBJETIVO (target_at), porque el patron de demanda depende de
    la hora que se esta prediciendo; las features de ancla (anchor_*) se
    calculan sobre el momento t, siempre con datos reales.
    """
    # OJO (28/09): un merge EXACTO por observed_at deja NaN en las columnas
    # de contexto para cualquier fila mas nueva que la ultima fecha que
    # tenga context.csv. El 28/09 context.csv se quedo fijo en el 9 de
    # sept (el dataset de clima/eventos del reto no se sigue actualizando)
    # mientras observations.csv (ya arreglado, ver sync_from_supabase.py)
    # sigue creciendo -- la ventana de validacion (VALIDATION_DAYS=7 en
    # src/train.py) cayo COMPLETA despues del 9 de sept, asi que el dropna
    # de mas abajo se comia TODAS las filas de validacion (0 filas) y
    # GradientBoostingRegressor tronaba al predecir con un array vacio.
    # merge_asof con direction="backward" arrastra el ultimo contexto
    # conocido hacia adelante en vez de dejar NaN -- sigue sin usar
    # informacion futura (nunca mira un context_row posterior al ancla),
    # asi que el backtesting temporal sigue siendo valido.
    merged = pd.merge_asof(
        observations.sort_values("observed_at"), context, on="observed_at", direction="backward"
    )
    merged = merged.sort_values(["station_id", "observed_at"]).reset_index(drop=True)
    merged = add_anchor_features(merged)

    frames = []
    for h in HORIZONS:
        f = merged.copy()
        f["anchor_at"] = f["observed_at"]
        f["target_at"] = f["anchor_at"] + pd.Timedelta(minutes=15 * h)
        f["horizon_steps"] = h
        f["horizon_min"] = 15 * h
        f["demand_target"] = f.groupby("station_id")["demand"].shift(-h)
        cal = add_calendar_features(f[["target_at"]].rename(columns={"target_at": "observed_at"}))
        for col in ("hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"):
            f[col] = cal[col].values
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def anchor_features_for_station(demand_hist: pd.Series, context_row: dict, last_time: pd.Timestamp) -> dict:
    """Version de add_anchor_features() para UNA sola estacion en produccion
    (usada por src/predict.py), operando sobre la serie historica real de
    esa estacion (demand_hist, indexada por observed_at) hasta last_time.
    """
    feat = dict(context_row)
    feat["anchor_value"] = float(demand_hist.iloc[-1])
    for lag in ANCHOR_LAGS:
        idx = last_time - pd.Timedelta(minutes=15 * lag)
        feat[f"anchor_lag_{lag}"] = float(demand_hist.get(idx, demand_hist.iloc[-1]))
    for window in ROLLING_WINDOWS:
        feat[f"anchor_roll_mean_{window}"] = float(demand_hist.iloc[-window:].mean())
    return feat


# Features del modelo de horizonte directo (el que usan src/train.py,
# src/predict.py y scripts/submit_first_prediction.py).
FEATURE_COLUMNS = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend", "horizon_min",
    "rain_mm", "temperature_c", "event_intensity",
    "anchor_value", "anchor_lag_4", "anchor_lag_96", "anchor_lag_672",
    "anchor_roll_mean_4", "anchor_roll_mean_96",
]
