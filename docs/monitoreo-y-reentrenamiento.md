# Monitoreo, reentrenamiento y selección de versiones (fase de drift)

Responde a los seis puntos de evidencia de `docs/fase-drift.md` del reto.
Código: `src/ensemble.py` (pronóstico adaptativo), `src/train.py` (política
de reentrenamiento, campeón vs. retador, monitoreo), `src/monitor.py`
(persistencia en Supabase), `scripts/submit_first_prediction.py` (entrega).

## 1. Continuidad de la ingesta y de las submissions

Cada ~15 min `.github/workflows/auto-pipeline.yml` trae el stream, sincroniza
Supabase, reconstruye `data/*.csv` desde Supabase, entrena/evalúa, predice,
envía y evalúa. Evidencia por corrida:

| Qué | Dónde |
|---|---|
| Corrida, corte de datos, modelo desplegado, decisión de reentreno y motivo | `pipeline_runs` (`model_version`, `data_cutoff`, `retrained`, `retrain_reason`) |
| Huecos > 3 h en los datos de entrenamiento | `pipeline_runs.error_message` (`detect_data_gaps`) |
| Métricas del sistema desplegado y de las referencias (últimas 24 h) | `metrics` + `artifacts/metrics.json` |
| Pesos del ensamble por estación en cada predicción | `artifacts/ensemble_weights.json` |
| Historial de reentrenamientos y promociones | `artifacts/model_registry.json` |
| Predicciones y accuracy real contra el dato publicado | `predictions`, `submission_performance` |

## 2. Problema operativo vs. cambio en la demanda

- **Operativo**: falta una corrida en `pipeline_runs`, `error_message` reporta
  un hueco, el portal muestra un ciclo sin entrega, o `data_cutoff` deja de
  avanzar. El 28/09 hubo un caso real (hueco de 3 días en el CSV y luego un
  fallo de paginación) y se corrigió haciendo de Supabase la fuente de verdad.
- **Demanda**: el pipeline corre completo y entrega, pero el error real sube.
  Se ve por bloques de tiempo, por estación y con la relación
  predicho/real. Ejemplo medido: desde el 16/sep 12:00 (tiempo del reto) el
  accuracy real cayó de ~78% a ~56-62% con predicho/real 0.73-0.89, y el
  18/sep 06:00 a 34% (predicho/real 0.40), concentrado primero en 05100,
  03000, 05000 y 02300 y luego en un cambio de distribución horaria.

## 3. Qué dispara una evaluación o un entrenamiento

**Cada corrida (barato, sin reentrenar):** el ensamble recalcula, por
estación, el error de cada candidato en las últimas 2 h y reparte pesos
∝ (1/WAPE)³. Candidatos: perfil histórico × nivel de los últimos 30 min,
perfil de 7 días × nivel de 45 min, persistencia, tendencia amortiguada,
mismo horario de ayer escalado, un detector automático de ciclo (periodo dominante de 2-12 h por autocorrelación de las últimas 12 h; agregado el 1/oct cuando la demanda pasó a un ciclo de 4 h) y el GBR. Así el sistema se adapta a cambios
de nivel o de forma en pocas horas sin gastar cómputo.

**Reentrenamiento del GBR (costoso), solo si:**

1. no hay modelo compatible; o
2. hay **≥ 6 h de datos nuevos** desde el último entrenamiento (las fases de
   drift duran ≥ 6 h según `docs/drift-control.md`); o
3. **degradación sostenida**: el accuracy fuera de muestra del GBR cae ≥ 5
   puntos bajo su accuracy de validación en las **dos** últimas ventanas de
   3 h, y ya pasaron ≥ 2 h desde el último entrenamiento (histéresis: no se
   reacciona a un único resultado malo).

En otro caso se conserva el campeón. Resultado en la simulación: 14
reentrenamientos en 151 corridas (≈ 1 de cada 11) en vez de reentrenar las
151 veces.

## 4. Comparación temporal sin información futura

- Compiten dos retadores con distinta ventana de historia (toda la historia y
  solo los últimos 3 días), ambos entrenados solo con targets ≤ (último dato −
  6 h) y evaluados en esas 6 h junto al campeón. Gana el de mejor accuracy y se
  promueve solo si iguala o mejora al campeón; si se promueve, se reajusta con
  todos los datos de su ventana. En el drift, la ventana de 3 días ganó por
  7-8 puntos (57.7% vs 50.5% y 63.6% vs 55.7% en el holdout).
- Los pesos del ensamble usan solo errores de targets ya observados al
  momento de predecir; el GBR solo cuenta errores posteriores a su corte de
  entrenamiento.
- El monitoreo de 24 h (`metrics.json`) re-simula cada ancla con los datos
  disponibles en ese instante.
- En la entrega, si el GBR vigente se entrenó con datos posteriores al
  `data_cutoff` del ciclo, se excluye de esa entrega.

## 5. Evidencia para mantener, promover o retirar una versión

`artifacts/model_registry.json` guarda cada evaluación: fecha, corte de
datos, motivo del disparo, accuracy del retador y del campeón en el holdout,
si se promovió y qué versión quedó. `pipeline_runs.retrain_reason` lleva el
resumen de cada corrida y `metrics.json` compara el sistema desplegado contra
referencias fijas (mismo horario de ayer, promedio estacional fijo).

## 6. Antes, durante y después del cambio observado

Simulación paso a paso cada 30 min con datos reales (en cada paso solo se usa
lo observado hasta ese instante; la política decide si reentrenar), comparada
con el accuracy real que obtuvo el sistema anterior en `submission_performance`:

| Bloque (tiempo del reto) | Sistema anterior (real) | Sistema nuevo (simulado) |
|---|---|---|
| 15 sep 12-18 h | 73.0% | 82.7% |
| 16 sep 06-12 h | 78.1% | 89.1% |
| 16 sep 12-18 h (inicio del drift) | 65.7% | 86.8% |
| 16 sep 18-24 h | 61.6% | 88.5% |
| 17 sep 00-06 h | 58.9% | 88.6% |
| 17 sep 12-18 h | 55.8% | 88.0% |
| 18 sep 00-06 h | 52.9% | 82.3% |
| 18 sep 06-10 h (cambio fuerte de forma) | 33.9% | 53.4% |
| 18 sep 12-24 h (nuevo régimen: ciclo de 4 h) | ~55% (ensamble sin detector de ciclo, en producción) | 80.7% (con `ciclo_detectado`) |

El último bloque muestra el límite: ante un cambio brusco de distribución
horaria, todo pronosticador pierde precisión hasta que hay datos del nuevo
régimen; el ensamble reduce el daño cambiando de peso hacia persistencia y
tendencia.
