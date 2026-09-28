# Dashboard en vivo — Pulso TransMi

Panel de monitoreo del pipeline en producción, en vivo:
**https://pulso-transmi-dashboard-visual-2.vercel.app**

`index.html` es una página estática (sin framework, sin build) que consulta
directamente la API REST de Supabase con la clave pública de solo lectura y
se actualiza sola cada 25 segundos. Muestra:

- Accuracy real del modelo desplegado (contra el dato oficial del reto),
  con su variación frente a la corrida anterior.
- Estado del pipeline y del reentrenamiento (si reentrenó en la última
  corrida y por qué — según el umbral de accuracy definido en
  `src/train.py`).
- Drift por estación: magnitud real del cambio de demanda (semana vs.
  semana anterior) y si superó el umbral.
- Accuracy real por estación y las últimas predicciones enviadas.

Está desplegado directamente en Vercel (no requiere build ni backend
propio); este archivo queda en el repo como referencia y respaldo del
código fuente.
