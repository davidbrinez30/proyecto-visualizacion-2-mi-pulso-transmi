"""Entrenamiento con POLITICA de reentrenamiento + monitoreo temporal (fase de drift).

Que hace en cada corrida del pipeline (cada ~15 min):

1. Carga el modelo vigente (artifacts/model_gbr.joblib, el "campeon").
2. Decide si hay que reentrenar el GBR (lo unico costoso) con una politica
   explicita -- NO se reentrena en cada corrida:
     a) no hay modelo compatible                               -> reentrenar
     b) hay >= RETRAIN_MIN_NEW_HOURS de datos nuevos desde el ultimo
        entrenamiento (las fases de drift duran >= 6 h segun
        docs/drift-control.md del reto)                         -> reentrenar
     c) degradacion SOSTENIDA: el accuracy real del GBR (fuera de muestra)
        en las dos ultimas ventanas de 3 h esta >= DEGRADATION_POINTS por
        debajo de su accuracy de validacion, y ya pasaron al menos
        RETRAIN_COOLDOWN_HOURS desde el ultimo entrenamiento
        (histeresis: no reaccionar a un solo resultado malo)    -> reentrenar
     d) en otro caso                                            -> se conserva el campeon
3. Si se reentrena: el RETADOR se entrena con datos hasta (ultimo dato -
   HOLDOUT_HOURS) y se compara contra el campeon en esas ultimas
   HOLDOUT_HOURS (orden temporal, sin informacion futura). Compiten dos
   retadores: historia completa y solo los ultimos 3 dias (TRAIN_WINDOWS_DAYS);
   gana el de mejor accuracy en ese holdout. Solo se promueve
   si iguala o mejora al campeon; si se promueve, se reajusta con todos los
   datos. La decision queda en artifacts/model_registry.json.
4. Monitoreo: simula lo que el sistema desplegado (ensamble adaptativo de
   src/ensemble.py + GBR vigente) habria predicho en las ultimas
   MONITOR_HOURS, usando solo datos disponibles en cada ancla, y guarda las
   metricas en artifacts/metrics.json (src/monitor.py las sube a Supabase).

La adaptacion rapida a los cambios de demanda NO depende de reentrenar: la
hace el ensamble en cada prediccion con los datos mas frescos (ver
src/ensemble.py). El reentrenamiento corrige lo que el ensamble no alcanza.

Uso:
    python -m src.train
Genera:
    artifacts/metrics.json, artifacts/model_gbr.joblib,
    artifacts/model_registry.json, artifacts/comparacion_modelos.png
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from src import ensemble as ens
from src.evaluation import summarize
from src.features import FEATURE_COLUMNS, HORIZONS, build_multihorizon_frame
from src.ingest import load_all

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)
BUNDLE_PATH = ARTIFACTS / "model_gbr.joblib"
REGISTRY_PATH = ARTIFACTS / "model_registry.json"
BUNDLE_VERSION = 2

RETRAIN_MIN_NEW_HOURS = 6.0
RETRAIN_COOLDOWN_HOURS = 2.0
DEGRADATION_POINTS = 5.0
HOLDOUT_HOURS = 6.0
MONITOR_HOURS = 24.0

GBR_PARAMS = dict(random_state=42, max_depth=3, n_estimators=100, learning_rate=0.1, subsample=0.5)

SURFACE = "#fcfcfb"
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#7a4fd6"]


def load_bundle() -> dict | None:
    if not BUNDLE_PATH.exists():
        return None
    try:
        b = joblib.load(BUNDLE_PATH)
    except Exception:
        return None
    if not isinstance(b, dict) or b.get("version") != BUNDLE_VERSION or b.get("train_cutoff") is None:
        return None
    return b


# Ventanas de entrenamiento que compiten como retadores (dias de historia;
# None = toda la historia). En drift, entrenar solo con lo reciente puede
# servir mas que arrastrar semanas de un regimen que ya no existe.
TRAIN_WINDOWS_DAYS = (None, 3)


def fit_gbr(multi: pd.DataFrame, cutoff: pd.Timestamp, window_days=None) -> GradientBoostingRegressor:
    """Entrena con anclas cuyo target es <= cutoff (nada posterior al corte);
    con window_days, solo con los ultimos window_days dias antes del corte."""
    rows = multi[multi["target_at"] <= cutoff]
    if window_days is not None:
        rows = rows[rows["target_at"] > cutoff - pd.Timedelta(days=window_days)]
    rows = rows.dropna(subset=FEATURE_COLUMNS + ["demand_target"])
    model = GradientBoostingRegressor(**GBR_PARAMS)
    model.fit(rows[FEATURE_COLUMNS], rows["demand_target"])
    return model


def gbr_eval(model, observations, context, lo: pd.Timestamp, hi: pd.Timestamp) -> dict | None:
    """Accuracy del GBR solo, para targets en (lo, hi], con features calculadas
    con datos <= cada ancla."""
    anchors = pd.date_range(lo - pd.Timedelta(hours=1), hi - pd.Timedelta(minutes=15), freq="15min")
    preds = ens.gbr_forecasts(observations, context, model, FEATURE_COLUMNS, anchors)
    P = ens.to_pivot(observations)
    real_l, pred_l, st_l = [], [], []
    for h in HORIZONS:
        p = preds[h]
        tgt = p.index + h * ens.PERIOD
        mask = (tgt > lo) & (tgt <= hi)
        p = p[mask]
        r = P.reindex(tgt[mask])
        r.index = p.index
        for s in p.columns:
            if s not in r.columns:
                continue
            ok = p[s].notna() & r[s].notna()
            real_l.append(r[s][ok]); pred_l.append(p[s][ok]); st_l.append(pd.Series(s, index=r[s][ok].index))
    if not real_l:
        return None
    real = pd.concat(real_l, ignore_index=True)
    if len(real) < 48:
        return None
    return summarize(real, pd.concat(pred_l, ignore_index=True), pd.concat(st_l, ignore_index=True))


def decide_retrain(bundle, observations, context, data_max) -> tuple[bool, str, dict]:
    info: dict = {}
    if bundle is None:
        return True, "no hay modelo compatible con el esquema actual (primera corrida v2)", info
    train_cutoff = bundle["train_cutoff"]
    hours_new = (data_max - train_cutoff) / pd.Timedelta(hours=1)
    info["horas_datos_nuevos"] = round(hours_new, 2)
    info["accuracy_validacion_campeon"] = bundle.get("holdout_accuracy")
    if hours_new >= RETRAIN_MIN_NEW_HOURS:
        return True, f"{hours_new:.1f} h de datos nuevos desde el ultimo entrenamiento (umbral {RETRAIN_MIN_NEW_HOURS:.0f} h)", info
    if hours_new < RETRAIN_COOLDOWN_HOURS:
        return False, f"solo {hours_new:.1f} h de datos nuevos (< {RETRAIN_COOLDOWN_HOURS:.0f} h de enfriamiento); se conserva el campeon", info
    a = gbr_eval(bundle["model"], observations, context, max(data_max - pd.Timedelta(hours=3), train_cutoff), data_max)
    b = gbr_eval(bundle["model"], observations, context, max(data_max - pd.Timedelta(hours=6), train_cutoff),
                 data_max - pd.Timedelta(hours=3))
    acc_a = a["accuracy_mean"] if a else None
    acc_b = b["accuracy_mean"] if b else None
    info["accuracy_gbr_ultimas_3h"] = acc_a
    info["accuracy_gbr_3h_anteriores"] = acc_b
    ref = bundle.get("holdout_accuracy")
    if ref is not None and acc_a is not None and acc_b is not None:
        if acc_a < ref - DEGRADATION_POINTS and acc_b < ref - DEGRADATION_POINTS:
            return True, (f"degradacion sostenida del GBR: {acc_b:.1f}% y {acc_a:.1f}% en las dos ultimas ventanas de 3 h "
                          f"vs {ref:.1f}% de validacion (umbral -{DEGRADATION_POINTS:.0f} pts)"), info
    return False, (f"sin degradacion sostenida y {hours_new:.1f} h de datos nuevos (< {RETRAIN_MIN_NEW_HOURS:.0f} h); "
                   "se conserva el campeon"), info


def append_registry(entry: dict) -> None:
    reg = []
    if REGISTRY_PATH.exists():
        try:
            reg = json.loads(REGISTRY_PATH.read_text())
        except Exception:
            reg = []
    reg.append(entry)
    REGISTRY_PATH.write_text(json.dumps(reg[-300:], indent=2, ensure_ascii=False, default=str))


def monitor_metrics(observations, context, bundle, data_max) -> dict:
    start = data_max - pd.Timedelta(hours=MONITOR_HOURS)
    end = data_max - pd.Timedelta(minutes=15)
    bt = ens.backtest(observations, context, bundle, start, end).dropna(subset=["real"])
    P = ens.to_pivot(observations)
    lk = ens.build_lookups(P[P.index < start - pd.Timedelta(hours=ens.WEIGHT_WINDOW_HOURS + 1)])
    tg = pd.DatetimeIndex(bt["target_at"])
    stacked = P.stack()
    bt["naive_96"] = stacked.reindex(pd.MultiIndex.from_arrays([tg - pd.Timedelta(days=1), bt["station_id"]])).values
    Ef = lk["full"].reindex(pd.MultiIndex.from_arrays([tg.dayofweek, tg.hour, tg.minute]))
    col_pos = {s: i for i, s in enumerate(Ef.columns)}
    vals = Ef.values
    bt["seasonal_avg"] = [vals[i, col_pos[s]] if s in col_pos else np.nan for i, s in enumerate(bt["station_id"])]

    def summ(col):
        sub = bt.dropna(subset=[col])
        if sub.empty:
            return {"accuracy_mean": None, "wape_mean": None}
        return summarize(sub["real"], sub[col], sub["station_id"])

    res = {
        "naive_96": summ("naive_96"),
        "seasonal_avg": summ("seasonal_avg"),
        "gradient_boosting": summ("gbr"),
        "blend_final": summ("ensamble"),
        "candidatos": {c: summ(c)["accuracy_mean"] for c in ens.CANDIDATES},
    }
    res["blend_final"]["accuracy_by_horizon"] = {
        f"{h * 15}min": summarize(bt.loc[bt.horizon_steps == h, "real"], bt.loc[bt.horizon_steps == h, "ensamble"],
                                  bt.loc[bt.horizon_steps == h, "station_id"])["accuracy_mean"]
        for h in HORIZONS
    }
    last6 = bt[bt["target_at"] > data_max - pd.Timedelta(hours=6)]
    res["blend_final"]["accuracy_ultimas_6h"] = (
        summarize(last6["real"], last6["ensamble"], last6["station_id"])["accuracy_mean"] if len(last6) else None)
    return res


def main():
    stations, observations, context = load_all()
    data_max = observations["observed_at"].max()
    now = datetime.now(timezone.utc).isoformat()

    champion = load_bundle()
    retrain, reason, policy_info = decide_retrain(champion, observations, context, data_max)
    deployed = champion
    decision = {"reentrenar": retrain, "razon": reason, "politica": {
        "RETRAIN_MIN_NEW_HOURS": RETRAIN_MIN_NEW_HOURS, "RETRAIN_COOLDOWN_HOURS": RETRAIN_COOLDOWN_HOURS,
        "DEGRADATION_POINTS": DEGRADATION_POINTS, "HOLDOUT_HOURS": HOLDOUT_HOURS}, **policy_info}

    if retrain:
        multi = build_multihorizon_frame(observations, context)
        holdout_cut = data_max - pd.Timedelta(hours=HOLDOUT_HOURS)
        # Retadores: misma receta con distintas ventanas de historia, todos
        # entrenados SOLO con targets <= holdout_cut y evaluados en las
        # ultimas HOLDOUT_HOURS (que ninguno vio). El campeon se evalua en la
        # misma ventana.
        retadores = {}
        for w in TRAIN_WINDOWS_DAYS:
            name = "historia_completa" if w is None else f"ultimos_{w}_dias"
            m = fit_gbr(multi, holdout_cut, w)
            ev = gbr_eval(m, observations, context, holdout_cut, data_max)
            retadores[name] = {"window_days": w, "accuracy_holdout": ev["accuracy_mean"] if ev else None, "eval": ev}
        best_name = max(retadores, key=lambda k: retadores[k]["accuracy_holdout"] if retadores[k]["accuracy_holdout"] is not None else -1)
        best = retadores[best_name]
        ch_eval = best["eval"]
        cp_eval = gbr_eval(champion["model"], observations, context, holdout_cut, data_max) if champion else None
        acc_ch = best["accuracy_holdout"]
        acc_cp = cp_eval["accuracy_mean"] if cp_eval else None
        promote = champion is None or acc_cp is None or (acc_ch is not None and acc_ch >= acc_cp)
        decision.update({
            "retadores_holdout": {k: v["accuracy_holdout"] for k, v in retadores.items()},
            "mejor_retador": best_name,
            "accuracy_retador_holdout": acc_ch, "accuracy_campeon_holdout": acc_cp, "promovido": promote,
        })
        if promote:
            final = fit_gbr(multi, data_max, best["window_days"])
            deployed = {
                "version": BUNDLE_VERSION,
                "model": final,
                "features": FEATURE_COLUMNS,
                "train_cutoff": data_max,
                "trained_at": now,
                "model_id": f"gbr-{data_max:%Y%m%dT%H%M}" + ("" if best["window_days"] is None else f"-w{best['window_days']}d"),
                "train_window_days": best["window_days"],
                "holdout_accuracy": acc_ch,
                "holdout_wape_by_station": (ch_eval or {}).get("wape_by_station", {}),
            }
            joblib.dump(deployed, BUNDLE_PATH)
        else:
            decision["razon"] += f"; retador NO promovido ({acc_ch} < {acc_cp} en las ultimas {HOLDOUT_HOURS:.0f} h)"
        decision["modelo_desplegado"] = deployed["model_id"]
        append_registry({"evaluado_en": now, "corte_datos": data_max, **decision})
    else:
        decision["modelo_desplegado"] = champion["model_id"]

    results = monitor_metrics(observations, context, deployed, data_max)
    results.update({
        "data_cutoff": str(data_max),
        "monitor_hours": MONITOR_HOURS,
        "modelo_desplegado": deployed["model_id"],
        "gbr_train_cutoff": str(deployed["train_cutoff"]),
        "decision_modelo": decision,
        "refit_final": {"hecho": bool(retrain and decision.get("promovido")),
                        "razon": f"[{deployed['model_id']}] {decision['razon']}"},
    })
    (ARTIFACTS / "metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str))

    names = ["naive_96", "seasonal_avg", "gradient_boosting", "blend_final"]
    labels = ["Mismo horario\nde ayer", "Promedio\nestacional fijo", "Gradient\nBoosting", "Ensamble\nadaptativo"]
    accs = [results[n]["accuracy_mean"] or 0 for n in names]
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars = ax.bar(labels, accs, color=CATEGORICAL)
    for bar, a in zip(bars, accs):
        ax.annotate(f"{a:.1f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel(f"Accuracy (ultimas {MONITOR_HOURS:.0f} h, 4 horizontes)")
    ax.set_title("Comparacion de modelos — Pulso TransMi")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linewidth=0.6, color="#e1e0d9")
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "comparacion_modelos.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)

    for n in names:
        print(f"{n}: accuracy_mean={results[n]['accuracy_mean']}")
    print("candidatos:", results["candidatos"])
    print("decision_modelo:", json.dumps(decision, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
