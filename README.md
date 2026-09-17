# Proyecto Visualización 2 — Pulso TransMi

Proyecto del reto MLOps **Pulso TransMi** (Universidad Externado de Colombia):
pronóstico de demanda de pasajeros cada 15 minutos para 12 estaciones de
TransMilenio, con un pipeline que mide, decide y reentrena en el tiempo.

Autor: David — Marketing, Universidad Externado de Colombia.

## Arquitectura

```text
proyecto-visualizacion-2-mi-pulso-transmi/
├── data/                    # corte de datos oficial (stations/observations/context/metadata)
├── eda/                     # análisis exploratorio (etapa 1)
│   ├── eda.py
│   ├── graficos/            # 8 gráficos más relevantes del EDA
│   └── hallazgos.md
├── src/                     # pipeline de modelado (etapa 1 y 2)
│   ├── ingest.py            # carga de datos crudos
│   ├── features.py          # feature engineering (calendario + lags, sin fuga de futuro)
│   ├── evaluation.py        # métrica oficial: WAPE / Accuracy por estación
│   ├── train.py             # baselines + modelo ML, validación temporal
│   ├── predict.py           # pronóstico recursivo a 4 horizontes (15/30/45/60 min)
│   └── monitor.py           # drift, decisión de reentrenamiento, persistencia en Supabase
├── tests/                   # pruebas unitarias (pytest)
├── artifacts/                # evidencia de cada experimento (métricas, modelo, gráfico, predicciones)
├── docs/
│   └── er-diagram.md        # diagrama entidad-relación del esquema en Supabase
├── scripts/
│   └── fetch_data.py        # descarga el corte oficial vía el SDK del reto
├── .github/workflows/
│   ├── fetch-data.yml       # workflow manual: descarga y commitea data/
│   └── pipeline.yml         # workflow del pipeline completo (train + predict + monitor)
└── requirements.txt
```

## Decisiones de diseño

**Validación temporal, nunca aleatoria.** El split de entrenamiento/validación
usa los primeros ~38 días para entrenar y los últimos 7 para validar
(`src/train.py:temporal_split`). Una partición aleatoria mezclaría futuro y
pasado y daría métricas optimistas engañosas — el README del SDK oficial lo
advierte explícitamente.

**Dos baselines, calculados sin fuga de información:**
1. *Naive (t-96)* — la demanda de hace 96 periodos (mismo horario, día
   anterior). Es el que trae de ejemplo el SDK.
2. *Promedio estacional* — promedio histórico por `(estación, día de semana,
   hora, minuto)`, calculado **solo** con la partición de entrenamiento.

**Modelo de ML:** `GradientBoostingRegressor` (scikit-learn) sobre features de
calendario (hora/día en seno-coseno para capturar la ciclicidad, fin de
semana), clima/eventos del contexto, y lags/rolling de la propia demanda
(15min, 1h, 1 día, 1 semana atrás). Todas las features son calculables en el
momento de la predicción — condición necesaria para que el backtesting sea
válido y para poder correr el mismo código en producción.

**Resultado de la comparación** (ver `artifacts/comparacion_modelos.png` y
`artifacts/metrics.json`, validación = últimos 7 días del corte inicial):

| Modelo | Accuracy promedio |
|---|---:|
| Naive (t-96) | 77.9 |
| Promedio estacional | **87.3** |
| Gradient Boosting | 86.6 |

El promedio estacional resultó marginalmente mejor que el modelo de ML en
este corte — es un resultado honesto y esperable con solo 45 días de datos y
un patrón fuertemente periódico: el baseline estacional ya captura casi toda
la señal explicable por calendario. El modelo de ML debería sacar más
ventaja conforme se acumulen más semanas de historia (más señal para separar
efectos de clima/eventos de la estacionalidad pura). Ambos superan claramente
al naive simple.

**Pronóstico a 4 horizontes** (`src/predict.py`): predicción recursiva a
15/30/45/60 minutos por estación. Como el reto no libera clima/eventos
futuros, los periodos futuros reutilizan el último contexto conocido
(forward-fill) — una limitación explícita a revisar si el reto libera un
pronóstico de clima más adelante.

**Monitoreo y reentrenamiento** (`src/monitor.py`): cada corrida calcula
señales de drift (cambio relativo de la demanda media por estación entre la
última semana y la anterior, umbral 25%) y decide reentrenar si el accuracy
promedio cae más de 5 puntos frente a la corrida anterior registrada. Todo
(corrida, predicciones, métricas, señales de drift) se persiste en Supabase
— ver `docs/er-diagram.md` para el esquema completo.

## Cómo correr el pipeline localmente

```bash
python -m venv .venv && source .venv/bin/activate   # o .venv\Scripts\Activate.ps1 en Windows
pip install -r requirements.txt
pytest -q
python scripts/fetch_data.py        # baja data/ (o usa el corte ya versionado en el repo)
python -m src.train                 # entrena baselines + modelo, guarda artifacts/
python -m src.predict               # genera artifacts/predictions_latest.csv
python -m src.monitor               # drift + decisión de reentrenamiento (+ Supabase si hay credenciales)
```

## Base de datos (Supabase)

Proyecto `pulso-transmi`, esquema documentado en
[`docs/er-diagram.md`](docs/er-diagram.md). Variables de entorno esperadas
por `src/monitor.py`:

```
SUPABASE_URL=https://azqudvlltukeeuypivcq.supabase.co
SUPABASE_KEY=<service_role key — como secret de GitHub Actions, nunca en el código>
```

## Estado del proyecto

- [x] SDK instalado, corte inicial descargado y verificado (checksums)
- [x] EDA completo (continuidad, duplicados, variable objetivo, correlaciones)
- [x] Dos baselines + modelo de ML, con validación temporal
- [x] Esquema de base de datos en Supabase + diagrama ER
- [ ] Integración completa del pipeline en GitHub Actions (train+predict+monitor con schedule)
- [ ] Carga inicial de `data/` hacia Supabase
- [ ] Bono: dashboard en Vercel
