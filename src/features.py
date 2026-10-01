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

# --- Ajuste de NIVEL reciente (fase de drift, 30/09) ---
# El promedio estacional por si solo es un perfil historico FIJO: cuando la
# demanda de una estacion cambia de nivel (la fase de drift que anuncio el
# profesor el 28/09), el perfil tarda semanas en enterarse. Medido con
# submission_performance real: desde el 16/sep 12:00 (tiempo del reto) el
# accuracy cayo a ~60%, con estaciones sub-predichas a la mitad (05000,
# 02300) y otras sobre-predichas 2-4x (03000, 05100).
# Solucion: se conserva la FORMA del perfil estacional (hora/dia), pero se
# re-escala por estacion con el nivel real de los ultimos
# LEVEL_WINDOW_POINTS periodos:  pred = perfil(target) * real_reciente / perfil_reciente.
# Backtest temporal con datos reales (solo datos disponibles al corte):
#   - antes del drift (12-16 sep): 76.7% (mezcla anterior) -> 82.9%
#   - durante el drift (16-18 sep): 62.1% (mezcla anterior) -> 84.7%
# LEVEL_WINDOW_POINTS=3 (45 min) salio entre los mejores del barrido
# (0.25h..24h); ventanas largas (6-24h) reaccionan tarde al drift.
LEVEL_WINDOW_POINTS = 3
LEVEL_RATIO_CLIP = (0.05, 20.0)  # amplio a proposito: 05100 cayo a ~1/4 de su nivel historico


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


def _as_utc_series(demand_hist: pd.Series) -> pd.Series:
    """Garantiza indice DatetimeIndex tz-aware en UTC. src/predict.py y
    submit_first_prediction.py construyen demand_hist con `.values`, lo que
    deja el indice SIN zona horaria; buscar ahi un Timestamp con zona (UTC)
    fallaba en silencio y devolvia el valor por defecto -- bug encontrado el
    30/09: en produccion anchor_lag_4/96/672 eran SIEMPRE iguales a
    anchor_value (entrenamiento y produccion veian features distintas)."""
    idx = pd.DatetimeIndex(demand_hist.index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    return pd.Series(demand_hist.values, index=idx)


def _as_utc_ts(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def level_ratio_for_station(demand_hist: pd.Series, lookup: pd.Series, station_id: str,
                            last_time, n: int = LEVEL_WINDOW_POINTS) -> float:
    """Nivel real reciente / nivel esperado por el perfil estacional, sobre
    los ultimos n periodos hasta last_time (inclusive). Solo usa datos ya
    observados al momento de predecir. Si no hay perfil para esos periodos,
    devuelve 1.0 (sin ajuste)."""
    hist = _as_utc_series(demand_hist)
    last_time = _as_utc_ts(last_time)
    act_sum = exp_sum = 0.0
    for k in range(n):
        t = last_time - pd.Timedelta(minutes=15 * k)
        act = hist.get(t)
        exp = lookup.get((station_id, t.dayofweek, t.hour, t.minute))
        if act is None or exp is None or pd.isna(act) or pd.isna(exp):
            continue
        act_sum += float(act)
        exp_sum += float(exp)
    if exp_sum <= 0:
        return 1.0
    return float(np.clip(act_sum / exp_sum, *LEVEL_RATIO_CLIP))


def level_ratio_frame(observations: pd.DataFrame, lookup: pd.Series, n: int = LEVEL_WINDOW_POINTS) -> pd.DataFrame:
    """Version vectorizada de level_ratio_for_station para TODAS las filas
    (station_id, observed_at) -- la usa src/train.py para validar la mezcla
    con ajuste de nivel exactamente igual a como se predice en produccion."""
    obs = observations[["station_id", "observed_at", "demand"]].sort_values(["station_id", "observed_at"]).copy()
    ts = obs["observed_at"]
    keys = pd.MultiIndex.from_arrays([obs["station_id"], ts.dt.dayofweek, ts.dt.hour, ts.dt.minute])
    obs["expected"] = lookup.reindex(keys).values
    valid = obs["expected"].notna() & obs["demand"].notna()
    obs["act_v"] = obs["demand"].where(valid, 0.0)
    obs["exp_v"] = obs["expected"].where(valid, 0.0)
    g = obs.groupby("station_id")
    act_sum = g["act_v"].rolling(n, min_periods=1).sum().reset_index(level=0, drop=True)
    exp_sum = g["exp_v"].rolling(n, min_periods=1).sum().reset_index(level=0, drop=True)
    ratio = (act_sum / exp_sum.where(exp_sum > 0)).clip(*LEVEL_RATIO_CLIP).fillna(1.0)
    obs["level_ratio"] = ratio
    return obs[["station_id", "observed_at", "level_ratio"]]


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
    demand_hist = _as_utc_series(demand_hist)
    last_time = _as_utc_ts(last_time)
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
