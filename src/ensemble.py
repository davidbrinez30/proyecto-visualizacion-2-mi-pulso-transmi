"""Pronostico ADAPTATIVO para la fase de drift (30/09).

Idea central (alineada con la guia del profesor, docs/fase-drift.md y
docs/drift-control.md del repo del reto): la demanda puede cambiar de NIVEL,
de DISTRIBUCION HORARIA o de RELACION ENTRE ESTACIONES cada ~4-6 h. Un unico
modelo fijo no sigue esos cambios, y reentrenar cada 15 min gasta computo sin
garantizar mejora. Por eso se separan dos cosas:

1. Adaptacion ONLINE barata (cada corrida, sin reentrenar): varios
   pronosticadores simples que usan los datos mas frescos, combinados con
   pesos POR ESTACION segun su error real de las ultimas WEIGHT_WINDOW_HOURS.
   Si una estacion cambia de regimen, en pocas horas gana peso el candidato
   que mejor lo sigue.
2. Reentrenamiento del GBR (costoso) solo cuando lo dispara la politica de
   src/train.py (datos nuevos suficientes o degradacion sostenida), con
   comparacion campeon vs retador antes de promover.

Candidatos (todos usan SOLO datos observados hasta el ancla t):
  - nivel_30m:          perfil historico (estacion, dia, hora, min) x nivel real de los ultimos 30 min
  - nivel_45m_perfil7d: perfil de los ultimos 7 dias (hora, min) x nivel real de los ultimos 45 min
  - persistencia:       ultimo valor observado
  - tendencia:          ultimo valor + pendiente de los ultimos 15 min, amortiguada (cambios rapidos de forma)
  - ayer_escalado:      mismo horario de ayer x (ultimos 30 min / mismos 30 min de ayer)
  - ciclo_detectado:    detecta el periodo dominante de las ultimas 12 h (2-12 h, por autocorrelacion)
                        y repite ese ciclo escalado al nivel actual. Existe porque en la fase de drift
                        del 18/sep la demanda paso a un ciclo de 4 h que ningun perfil diario captura.
  - gbr:                Gradient Boosting de horizonte directo (src/features.py)

Simulacion paso a paso cada 30 min con datos reales (solo datos observados
hasta cada instante, politica de reentrenamiento de src/train.py incluida),
contra el accuracy real del sistema anterior (submission_performance):
  16 sep 12-18 h (inicio del drift)   65.7%  ->  86.8%
  17 sep 12-18 h                      55.8%  ->  88.0%
  18 sep 06-10 h (cambio fuerte)      33.9%  ->  53.4%
Regimen de ciclo de 4 h (18 sep 12-24 h): ensamble sin ciclo_detectado 55.1%
(igual a lo observado en produccion) -> con ciclo_detectado 80.7%.
Detalle en docs/monitoreo-y-reentrenamiento.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import FEATURE_COLUMNS, HORIZONS, build_multihorizon_frame

CANDIDATES = ("nivel_30m", "nivel_45m_perfil7d", "persistencia", "tendencia", "ayer_escalado", "ciclo_detectado", "gbr")
WEIGHT_WINDOW_HOURS = 2     # ventana de error reciente para los pesos: corta para cambiar rapido de candidato
WEIGHT_POWER = 3            # peso ~ (1 / WAPE)^3: premia fuerte al que va mejor
CYCLE_LAGS = range(8, 49)    # periodos candidatos: 2 h a 12 h (en pasos de 15 min)
CYCLE_WINDOW = 48            # autocorrelacion medida sobre las ultimas 12 h
TREND_DAMPING = 0.8         # tendencia amortiguada: P(t) + pendiente * (0.8 + 0.8^2 + ...)
MIN_POINTS_FOR_WEIGHT = 8   # minimo de errores observados para confiar en el WAPE de un candidato
RATIO_CLIP = (0.05, 20.0)
PERIOD = pd.Timedelta(minutes=15)
GBR_HISTORY_DAYS = 9        # historia necesaria para las features del GBR (lag de 1 semana + margen)


def to_pivot(observations: pd.DataFrame) -> pd.DataFrame:
    """Matriz tiempo x estacion, con grilla regular de 15 min (huecos = NaN)."""
    P = observations.pivot_table(index="observed_at", columns="station_id", values="demand", aggfunc="last")
    P = P.sort_index().astype(float)
    if len(P):
        P = P.asfreq("15min")
    P.columns = P.columns.astype(str)
    return P


def build_lookups(P_hist: pd.DataFrame) -> dict:
    """Perfiles estacionales calculados SOLO con P_hist (datos ya observados)."""
    ix = P_hist.index
    full = P_hist.groupby([ix.dayofweek, ix.hour, ix.minute]).mean()
    recent = P_hist[ix > ix.max() - pd.Timedelta(days=7)]
    rix = recent.index
    last7 = recent.groupby([rix.hour, rix.minute]).mean()
    return {"full": full, "last7": last7}


def _expected(lookups: dict, kind: str, ts: pd.DatetimeIndex, columns) -> pd.DataFrame:
    if kind == "full":
        keys = pd.MultiIndex.from_arrays([ts.dayofweek, ts.hour, ts.minute])
    else:
        keys = pd.MultiIndex.from_arrays([ts.hour, ts.minute])
    E = lookups[kind].reindex(keys)
    E.index = ts
    return E.reindex(columns=columns)


def _ratio(num: pd.DataFrame, den: pd.DataFrame) -> pd.DataFrame:
    r = num / den.where(den > 0)
    return r.clip(*RATIO_CLIP).fillna(1.0)


def gbr_forecasts(observations: pd.DataFrame, context: pd.DataFrame, model, feature_cols,
                  anchors: pd.DatetimeIndex) -> dict:
    """Predicciones del GBR para cada ancla (features calculadas con datos <= ancla)."""
    out = {h: pd.DataFrame(np.nan, index=anchors, columns=sorted(observations["station_id"].astype(str).unique()))
           for h in HORIZONS}
    if model is None or len(anchors) == 0:
        return out
    start = anchors.min() - pd.Timedelta(days=GBR_HISTORY_DAYS)
    obs = observations[observations["observed_at"] >= start]
    multi = build_multihorizon_frame(obs, context)
    multi = multi[multi["anchor_at"].isin(anchors)].dropna(subset=feature_cols)
    if multi.empty:
        return out
    multi = multi.assign(pred=model.predict(multi[feature_cols]).clip(min=0))
    for h in HORIZONS:
        sub = multi[multi["horizon_steps"] == h].pivot_table(index="anchor_at", columns="station_id", values="pred")
        sub.columns = sub.columns.astype(str)
        out[h].loc[sub.index, sub.columns] = sub.values
    return out


def candidate_forecasts(P: pd.DataFrame, lookups: dict, anchors: pd.DatetimeIndex, gbr: dict | None) -> dict:
    """{candidato: {h: DataFrame(anclas x estaciones)}} usando solo datos <= cada ancla."""
    cols = P.columns
    # grilla extendida para poder mirar targets (t+60 min) y pasado (t-1 dia)
    full_ix = pd.date_range(anchors.min() - pd.Timedelta(days=1, hours=1),
                            anchors.max() + pd.Timedelta(hours=1), freq="15min")
    A = P.reindex(full_ix)
    Ef = _expected(lookups, "full", full_ix, cols)
    E7 = _expected(lookups, "last7", full_ix, cols)

    r2_full = _ratio(A.rolling(2, min_periods=1).sum(), Ef.rolling(2, min_periods=1).sum())
    r3_7d = _ratio(A.rolling(3, min_periods=1).sum(), E7.rolling(3, min_periods=1).sum())
    r_ayer = _ratio(A.rolling(2, min_periods=1).sum(), A.shift(96).rolling(2, min_periods=1).sum())

    # periodo dominante en cada ancla, usando solo datos <= ancla
    corr = {}
    for L in CYCLE_LAGS:
        corr[L] = A.rolling(CYCLE_WINDOW, min_periods=CYCLE_WINDOW // 2).corr(A.shift(L)).mean(axis=1)
    corr = pd.DataFrame(corr)
    valid_rows = corr.notna().any(axis=1)
    best_lag = corr.fillna(-np.inf).idxmax(axis=1).where(valid_rows).reindex(anchors)
    pos = {ts: i for i, ts in enumerate(full_ix)}
    Av = A.values

    out = {c: {} for c in CANDIDATES}
    for h in HORIZONS:
        cyc = np.full((len(anchors), len(cols)), np.nan)
        for k, t in enumerate(anchors):
            L = best_lag.iloc[k]
            if pd.isna(L):
                continue
            L = int(L); i = pos[t]
            if i - L - 1 < 0:
                continue
            base = Av[i + h - L]
            num = np.nansum(Av[i - 1:i + 1], axis=0)
            den = np.nansum(Av[i - 1 - L:i + 1 - L], axis=0)
            r = np.where(den > 0, num / np.where(den > 0, den, 1), 1.0)
            cyc[k] = base * np.clip(r, *RATIO_CLIP)
        out["ciclo_detectado"][h] = pd.DataFrame(cyc, index=anchors, columns=cols)
        tgt = anchors + h * PERIOD
        out["nivel_30m"][h] = pd.DataFrame(Ef.reindex(tgt).values * r2_full.reindex(anchors).values,
                                           index=anchors, columns=cols)
        out["nivel_45m_perfil7d"][h] = pd.DataFrame(E7.reindex(tgt).values * r3_7d.reindex(anchors).values,
                                                    index=anchors, columns=cols)
        out["persistencia"][h] = A.reindex(anchors)
        damp = sum(TREND_DAMPING ** k for k in range(1, h + 1))
        out["tendencia"][h] = (A + (A - A.shift(1)) * damp).clip(lower=0).reindex(anchors)
        out["ayer_escalado"][h] = pd.DataFrame(A.reindex(tgt - pd.Timedelta(days=1)).values * r_ayer.reindex(anchors).values,
                                               index=anchors, columns=cols)
        if gbr is not None:
            out["gbr"][h] = gbr[h].reindex(index=anchors, columns=cols)
        else:
            out["gbr"][h] = pd.DataFrame(np.nan, index=anchors, columns=cols)
    return out


def recent_wape(cands: dict, P: pd.DataFrame, at: pd.Timestamp, window_hours: float = WEIGHT_WINDOW_HOURS,
                gbr_train_cutoff: pd.Timestamp | None = None) -> pd.DataFrame:
    """WAPE por (candidato, estacion) de los pronosticos cuyo TARGET ya se
    observo en (at - ventana, at]. Para el GBR solo cuentan targets
    posteriores a su corte de entrenamiento (fuera de muestra)."""
    lo = at - pd.Timedelta(hours=window_hours)
    rows = {}
    for c in CANDIDATES:
        err = pd.Series(0.0, index=P.columns)
        act = pd.Series(0.0, index=P.columns)
        npts = pd.Series(0, index=P.columns)
        for h in HORIZONS:
            pred = cands[c][h]
            tgt = pred.index + h * PERIOD
            mask = (tgt > lo) & (tgt <= at)
            if c == "gbr" and gbr_train_cutoff is not None:
                mask &= tgt > gbr_train_cutoff
            if not mask.any():
                continue
            p = pred[mask]
            real = P.reindex(tgt[mask])
            real.index = p.index
            ok = p.notna() & real.notna()
            err += (p - real).abs().where(ok).sum()
            act += real.where(ok).sum()
            npts += ok.sum()
        w = err / act.where(act > 0)
        w[npts < MIN_POINTS_FOR_WEIGHT] = np.nan
        rows[c] = w
    return pd.DataFrame(rows)  # estaciones x candidatos


def weights_from_wape(wape: pd.DataFrame, gbr_fallback_wape: dict | None = None) -> pd.DataFrame:
    wape = wape.copy()
    if gbr_fallback_wape and "gbr" in wape:
        fb = pd.Series(gbr_fallback_wape, dtype=float).reindex(wape.index)
        wape["gbr"] = wape["gbr"].fillna(fb)
    w = (1.0 / wape.clip(lower=0.01)) ** WEIGHT_POWER
    w = w.fillna(0.0)
    tot = w.sum(axis=1)
    # sin informacion suficiente: reparte entre los candidatos estadisticos
    no_info = tot <= 0
    if no_info.any():
        w.loc[no_info, ["nivel_30m", "nivel_45m_perfil7d", "ayer_escalado"]] = 1.0
        tot = w.sum(axis=1)
    return w.div(tot, axis=0)


def combine(cands: dict, weights: pd.DataFrame, h: int, rows: pd.DatetimeIndex) -> pd.DataFrame:
    """Combinacion ponderada; si un candidato no tiene valor, se renormaliza
    entre los que si tienen."""
    num = 0.0
    den = 0.0
    for c in CANDIDATES:
        p = cands[c][h].reindex(rows)
        w = pd.DataFrame(np.broadcast_to(weights[c].reindex(p.columns).fillna(0).values, p.shape),
                         index=p.index, columns=p.columns)
        w = w.where(p.notna(), 0.0)
        num = num + (p.fillna(0) * w)
        den = den + w
    res = num / den.where(den > 0)
    # respaldo final: nivel_30m y luego persistencia
    res = res.fillna(cands["nivel_30m"][h].reindex(rows)).fillna(cands["persistencia"][h].reindex(rows))
    return res.clip(lower=0)


def live_forecast(observations: pd.DataFrame, context: pd.DataFrame, bundle: dict | None,
                  cutoff: pd.Timestamp | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pronostico de los 4 horizontes desde el ultimo dato <= cutoff.
    Devuelve (predicciones, pesos_por_estacion)."""
    obs = observations if cutoff is None else observations[observations["observed_at"] <= cutoff]
    P = to_pivot(obs)
    t = P.index.max()
    win_start = t - pd.Timedelta(hours=WEIGHT_WINDOW_HOURS) - pd.Timedelta(hours=1)
    lookups = build_lookups(P[P.index <= win_start])
    anchors = pd.date_range(win_start, t, freq="15min")
    model = bundle.get("model") if bundle else None
    feats = bundle.get("features", FEATURE_COLUMNS) if bundle else FEATURE_COLUMNS
    gbr = gbr_forecasts(obs, context, model, feats, anchors) if model is not None else None
    cands = candidate_forecasts(P, lookups, anchors, gbr)
    wape = recent_wape(cands, P, t, gbr_train_cutoff=bundle.get("train_cutoff") if bundle else None)
    weights = weights_from_wape(wape, bundle.get("holdout_wape_by_station") if bundle else None)
    rows = []
    last = pd.DatetimeIndex([t])
    last_known = P.ffill().iloc[-1]  # respaldo final: nunca enviar NaN (la API exige valores finitos)
    for h in HORIZONS:
        pred = combine(cands, weights, h, last).iloc[0]
        pred = pred.fillna(last_known).fillna(0.0)
        for s, v in pred.items():
            rows.append({"station_id": s, "target_at": t + h * PERIOD, "horizon_steps": h,
                         "predicted_demand": round(float(v), 1)})
    return pd.DataFrame(rows), weights


def backtest(observations: pd.DataFrame, context: pd.DataFrame, bundle: dict | None,
             start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Simula lo que el sistema habria predicho en cada ancla de [start, end]
    usando solo datos <= ancla (pesos recalculados en cada ancla). El GBR solo
    participa en anclas cuyos targets son posteriores a su corte de
    entrenamiento, para que la metrica sea honesta (fuera de muestra).
    Devuelve filas (station_id, anchor_at, horizon_steps, target_at, real, cada candidato, ensamble)."""
    P = to_pivot(observations)
    anchors_all = pd.date_range(start - pd.Timedelta(hours=WEIGHT_WINDOW_HOURS + 1), end, freq="15min")
    lookups = build_lookups(P[P.index < anchors_all.min()])
    model = bundle.get("model") if bundle else None
    feats = bundle.get("features", FEATURE_COLUMNS) if bundle else FEATURE_COLUMNS
    train_cutoff = bundle.get("train_cutoff") if bundle else None
    gbr = gbr_forecasts(observations, context, model, feats, anchors_all) if model is not None else None
    if gbr is not None and train_cutoff is not None:
        for h in HORIZONS:
            g = gbr[h].copy()
            g.loc[(g.index + h * PERIOD) <= train_cutoff] = np.nan
            gbr[h] = g
    cands = candidate_forecasts(P, lookups, anchors_all, gbr)
    eval_anchors = anchors_all[(anchors_all >= start) & (anchors_all <= end)]
    fb = bundle.get("holdout_wape_by_station") if bundle else None
    out = []
    for t in eval_anchors:
        w = weights_from_wape(recent_wape(cands, P, t, gbr_train_cutoff=train_cutoff), fb)
        for h in HORIZONS:
            tgt = t + h * PERIOD
            if tgt not in P.index:
                continue
            ens = combine(cands, w, h, pd.DatetimeIndex([t])).iloc[0]
            real = P.loc[tgt]
            for s in P.columns:
                row = {"station_id": s, "anchor_at": t, "horizon_steps": h, "target_at": tgt,
                       "real": real[s], "ensamble": ens[s]}
                for c in CANDIDATES:
                    row[c] = cands[c][h].at[t, s]
                out.append(row)
    return pd.DataFrame(out)
