"""Analisis exploratorio de datos (EDA) - Pulso TransMi.

Genera las graficas mas relevantes en eda/graficos/ y un resumen en
eda/hallazgos.md a partir del corte oficial descargado en data/.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "graficos"
OUT.mkdir(exist_ok=True)

# ---- paleta (dataviz skill: reference palette, light surface) ----
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SEQ_BLUE = "#2a78d6"
SEQ_BLUE_DARK = "#184f95"
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
DIVERGING = ("#2a78d6", "#e34948")  # blue <-> red

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_SECONDARY,
    "text.color": INK_PRIMARY,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "grid.color": GRID,
    "font.family": "sans-serif",
    "axes.titleweight": "bold",
    "axes.titlecolor": INK_PRIMARY,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def load_data():
    stations = pd.read_csv("data/stations.csv", dtype={"station_id": "string"})
    observations = pd.read_csv(
        "data/observations.csv", dtype={"station_id": "string"}, parse_dates=["observed_at"]
    )
    context = pd.read_csv("data/context.csv", parse_dates=["observed_at"])
    return stations, observations, context


def fig_save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def chart_distribucion_demanda(observations):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(observations["demand"], bins=60, color=SEQ_BLUE, edgecolor=SURFACE, linewidth=0.3)
    ax.axvline(observations["demand"].median(), color="#e34948", linewidth=2, linestyle="--",
               label=f"mediana = {observations['demand'].median():.0f}")
    ax.set_title("Distribución de la variable objetivo (demanda / 15 min)")
    ax.set_xlabel("Demanda (pasajeros por periodo de 15 min)")
    ax.set_ylabel("Frecuencia")
    ax.legend(frameon=False)
    ax.grid(axis="y", linewidth=0.6)
    fig_save(fig, "01_distribucion_demanda.png")


def chart_serie_total(observations):
    total = observations.groupby("observed_at")["demand"].sum().sort_index()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(total.index, total.values, color=SEQ_BLUE, linewidth=1.1)
    ax.set_title("Demanda agregada (12 estaciones) cada 15 minutos - 45 días")
    ax.set_ylabel("Demanda total")
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
    ax.grid(axis="y", linewidth=0.6)
    fig.autofmt_xdate()
    fig_save(fig, "02_serie_temporal_total.png")


def chart_perfil_horario(observations):
    obs = observations.copy()
    obs["hour"] = obs["observed_at"].dt.hour + obs["observed_at"].dt.minute / 60
    perfil = obs.groupby("hour")["demand"].mean()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(perfil.index, perfil.values, color=SEQ_BLUE, linewidth=2)
    ax.fill_between(perfil.index, perfil.values, color=SEQ_BLUE, alpha=0.12)
    ax.set_title("Perfil horario promedio de demanda (todas las estaciones)")
    ax.set_xlabel("Hora del día")
    ax.set_ylabel("Demanda promedio")
    ax.set_xticks(range(0, 25, 3))
    ax.grid(axis="y", linewidth=0.6)
    fig_save(fig, "03_perfil_horario.png")


def chart_perfil_dia_semana(observations):
    obs = observations.copy()
    dias = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
    obs["dow"] = obs["observed_at"].dt.dayofweek
    perfil = obs.groupby("dow")["demand"].mean().reindex(range(7))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = [SEQ_BLUE if d < 5 else "#eb6834" for d in range(7)]
    ax.bar(dias, perfil.values, color=colors)
    ax.set_title("Demanda promedio por día de la semana")
    ax.set_ylabel("Demanda promedio (15 min)")
    ax.grid(axis="y", linewidth=0.6)
    fig_save(fig, "04_perfil_dia_semana.png")


def chart_demanda_por_estacion(observations, stations):
    total = observations.groupby("station_id")["demand"].mean().sort_values(ascending=True)
    names = stations.set_index("station_id")["station_name"]
    labels = [f"{names.get(sid, sid)} ({sid})" for sid in total.index]
    norm = (total.values - total.values.min()) / (total.values.max() - total.values.min())
    colors = plt.cm.Blues(0.35 + 0.55 * norm)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(labels, total.values, color=colors)
    ax.set_title("Demanda promedio por estación (15 min)")
    ax.set_xlabel("Demanda promedio")
    ax.grid(axis="x", linewidth=0.6)
    fig_save(fig, "05_demanda_por_estacion.png")


def chart_boxplot_estacion(observations, stations):
    names = stations.set_index("station_id")["station_name"]
    order = observations.groupby("station_id")["demand"].median().sort_values().index
    data = [observations.loc[observations["station_id"] == sid, "demand"].values for sid in order]
    labels = [f"{names.get(sid, sid)}" for sid in order]
    fig, ax = plt.subplots(figsize=(9, 6))
    bp = ax.boxplot(data, vert=False, labels=labels, patch_artist=True, showfliers=True,
                     flierprops=dict(marker="o", markersize=2, alpha=0.3, markerfacecolor=SEQ_BLUE, markeredgecolor="none"))
    for box in bp["boxes"]:
        box.set(facecolor=SEQ_BLUE, alpha=0.55, edgecolor=SEQ_BLUE_DARK)
    for med in bp["medians"]:
        med.set(color="#e34948", linewidth=1.6)
    ax.set_title("Dispersión de la demanda por estación (outliers y variabilidad)")
    ax.set_xlabel("Demanda (15 min)")
    ax.grid(axis="x", linewidth=0.6)
    fig_save(fig, "06_boxplot_por_estacion.png")


def chart_correlacion(observations, context):
    total = observations.groupby("observed_at")["demand"].sum().rename("demand_total")
    merged = context.set_index("observed_at").join(total, how="inner")
    merged["hour"] = merged.index.hour + merged.index.minute / 60
    cols = ["demand_total", "rain_mm", "temperature_c", "event_intensity", "hour"]
    corr = merged[cols].corr()

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    labels = ["demanda total", "lluvia (mm)", "temperatura (°C)", "intensidad evento", "hora del día"]
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_yticklabels(labels)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                     color="white" if abs(corr.values[i, j]) > 0.5 else INK_PRIMARY, fontsize=9)
    ax.set_title("Correlación: demanda vs. variables de contexto")
    fig.colorbar(im, ax=ax, shrink=0.8, label="correlación de Pearson")
    fig_save(fig, "07_correlacion_contexto.png")
    return corr


def chart_mapa_estaciones(observations, stations):
    total = observations.groupby("station_id")["demand"].mean()
    stations = stations.copy()
    stations["demanda_media"] = stations["station_id"].map(total)
    norm = (stations["demanda_media"] - stations["demanda_media"].min()) / (
        stations["demanda_media"].max() - stations["demanda_media"].min()
    )
    fig, ax = plt.subplots(figsize=(6.5, 7))
    sizes = 200 + 1400 * norm
    sc = ax.scatter(stations["longitude"], stations["latitude"], s=sizes,
                     c=stations["demanda_media"], cmap="Blues", edgecolor=SEQ_BLUE_DARK, linewidth=0.8, alpha=0.85)
    for _, row in stations.iterrows():
        ax.annotate(row["station_name"], (row["longitude"], row["latitude"]),
                    fontsize=7, color=INK_SECONDARY, xytext=(4, 4), textcoords="offset points")
    ax.set_title("Estaciones: ubicación y demanda promedio")
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    fig.colorbar(sc, ax=ax, shrink=0.7, label="demanda promedio")
    fig_save(fig, "08_mapa_estaciones.png")


def main():
    stations, observations, context = load_data()

    chart_distribucion_demanda(observations)
    chart_serie_total(observations)
    chart_perfil_horario(observations)
    chart_perfil_dia_semana(observations)
    chart_demanda_por_estacion(observations, stations)
    chart_boxplot_estacion(observations, stations)
    corr = chart_correlacion(observations, context)
    chart_mapa_estaciones(observations, stations)

    print("Gráficos generados en", OUT)
    print(corr["demand_total"])


if __name__ == "__main__":
    main()
