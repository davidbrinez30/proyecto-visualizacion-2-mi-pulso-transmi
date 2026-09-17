# Análisis exploratorio de datos — Pulso TransMi

Corte de datos: `dataset generated_at 2026-09-16`, histórico del `2026-07-26` al
`2026-09-08` (45 días), 12 estaciones, frecuencia de 15 minutos.

## 1. Calidad e integridad de los datos

- **Nulos:** ninguno en `observations.csv` ni en `context.csv` (0 valores faltantes
  en las 51.840 y 4.320 filas respectivamente).
- **Duplicados:** 0 combinaciones repetidas de `(station_id, observed_at)`.
- **Continuidad temporal:** las 12 estaciones tienen exactamente 4.320 registros
  cada una (45 días × 96 periodos/día), sin huecos de 15 minutos en ninguna serie.
- **Rango de la variable objetivo:** `demand` va de 14 a 2.284 pasajeros por
  periodo, sin valores negativos ni ceros. Checksums SHA-256 de los tres archivos
  verificados contra `/v1/meta` — el dataset es el oficial y está íntegro.

Conclusión: el corte inicial es limpio; no hace falta imputación en esta etapa.
El monitoreo de continuidad sigue siendo necesario en la etapa incremental,
porque el reto explícitamente cambiará patrones y liberará datos nuevos.

## 2. Variable objetivo (`demand`)

- Distribución fuertemente asimétrica a la derecha (cola larga): media 356.5,
  mediana 263, máximo 2.284. La mayoría de los periodos tienen demanda baja
  (madrugada) y unos pocos periodos pico concentran valores altos.
- Ver `graficos/01_distribucion_demanda.png`.

## 3. Patrones temporales

- **Perfil horario** (`graficos/03_perfil_horario.png`): dos picos claros de
  demanda — uno entre las 6:00 y 8:00 a.m. (~700 pasajeros/15min en promedio) y
  otro más pronunciado entre las 5:00 y 7:00 p.m. (~730). Valle nocturno estable
  entre 10:00 a.m. y 3:00 p.m. Es el patrón típico de "hora pico" de un sistema
  de transporte masivo urbano.
- **Serie completa** (`graficos/02_serie_temporal_total.png`): el patrón diario
  de dos picos se repite consistentemente los 45 días, con una caída visible
  los fines de semana.
- **Día de la semana** (`graficos/04_perfil_dia_semana.png`): la demanda
  promedio entre semana es ~28% mayor que en fin de semana (380 vs. 298
  pasajeros/15min).

## 4. Diferencias entre estaciones

- **Por estación** (`graficos/05_demanda_por_estacion.png` y
  `06_boxplot_por_estacion.png`): la estación con más demanda (Ricaurte - NQS,
  683 en promedio) mueve ~3.1x más pasajeros que la de menor demanda (Portal
  Usme, 218). Ricaurte - NQS, Banderas y Portal El Dorado son consistentemente
  las de mayor carga y mayor variabilidad (rango intercuartílico más ancho).
- **Geografía** (`graficos/08_mapa_estaciones.png`): las estaciones de mayor
  demanda están concentradas en el corredor central/occidente (Ricaurte,
  Banderas, El Dorado), mientras los portales periféricos (Usme, Suba) tienen
  demanda más baja y estable.

## 5. Correlación con variables de contexto

(`graficos/07_correlacion_contexto.png`, calculado sobre demanda agregada de
las 12 estaciones cada 15 min vs. clima/eventos)

| Variable | Correlación con demanda |
|---|---:|
| temperatura (°C) | 0.34 |
| hora del día | 0.24 |
| intensidad de evento | 0.15 |
| lluvia (mm) | -0.02 |

- La correlación más fuerte es con **temperatura**, aunque moderada — probablemente
  actúa más como proxy de hora del día/estacionalidad que como causa directa.
- **Lluvia** prácticamente no muestra correlación lineal con la demanda agregada
  en este corte; no descarta un efecto no lineal o localizado por estación, que
  vale la pena revisar en el feature engineering.
- **Intensidad de evento** tiene una correlación baja-moderada; conviene mirarla
  por estación individual, ya que a nivel agregado un evento en una sola estación
  se diluye.

## 6. Implicaciones para el modelado

- La fuerte estacionalidad horaria/semanal sugiere que features de calendario
  (hora, minuto, día de la semana, fin de semana) van a aportar más que las
  variables de clima en este corte.
- La asimetría de `demand` sugiere evaluar transformación log o modelos
  robustos a colas largas.
- Dado el patrón repetitivo diario, un baseline de "demanda de hace 96 periodos
  (mismo horario, día anterior)" — como el que trae el SDK — es un punto de
  comparación razonable; el reto pide al menos un segundo baseline (por ejemplo,
  promedio por estación+hora+día de la semana).
- La validación debe ser estrictamente temporal (train con los primeros ~38 días,
  validación con los últimos ~7), nunca aleatoria.
