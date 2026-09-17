# Diagrama entidad-relación — Supabase (Pulso TransMi)

Proyecto Supabase: **pulso-transmi** (`azqudvlltukeeuypivcq`, región `sa-east-1`).
URL: `https://azqudvlltukeeuypivcq.supabase.co`

## Diagrama

```mermaid
erDiagram
    STATIONS ||--o{ OBSERVATIONS : "tiene"
    STATIONS ||--o{ PREDICTIONS : "tiene"
    STATIONS ||--o{ METRICS : "tiene"
    STATIONS ||--o{ DRIFT_SIGNALS : "tiene"
    PIPELINE_RUNS ||--o{ PREDICTIONS : "genera"
    PIPELINE_RUNS ||--o{ METRICS : "genera"
    PIPELINE_RUNS ||--o{ DRIFT_SIGNALS : "genera"

    STATIONS {
        text station_id PK
        text station_name
        text corridor
        numeric latitude
        numeric longitude
    }
    OBSERVATIONS {
        bigint id PK
        text station_id FK
        timestamptz observed_at
        integer demand
    }
    CONTEXT {
        bigint id PK
        timestamptz observed_at
        numeric rain_mm
        numeric rain_forecast
        numeric temperature_c
        numeric temperature_forecast
        numeric event_intensity
    }
    PIPELINE_RUNS {
        bigint id PK
        timestamptz run_at
        text git_commit
        text model_version
        timestamptz data_cutoff
        text cursor_last_processed
        text status
        text error_message
        boolean retrained
        text retrain_reason
    }
    PREDICTIONS {
        bigint id PK
        bigint run_id FK
        text station_id FK
        timestamptz target_at
        integer horizon_steps
        numeric predicted_demand
        text model_version
    }
    METRICS {
        bigint id PK
        bigint run_id FK
        text station_id FK
        numeric wape
        numeric accuracy
        timestamptz window_start
        timestamptz window_end
    }
    DRIFT_SIGNALS {
        bigint id PK
        bigint run_id FK
        text station_id FK
        text signal_type
        numeric signal_value
        numeric threshold
        boolean triggered
    }
```

## Justificación del modelo

- **`stations`** replica el catálogo geográfico de la API (`/v1/stations`) — 12 filas fijas.
- **`observations`** replica la demanda observada (`/v1/observations`). Clave única
  `(station_id, observed_at)` para poder hacer upsert idempotente cuando el
  pipeline incremental traiga solo periodos nuevos.
- **`context`** replica clima/eventos (`/v1/context`). No tiene FK a estación
  porque el contexto no es por estación — se une por `observed_at`.
- **`pipeline_runs`** es el registro de cada ejecución del workflow de GitHub
  Actions: guarda el commit, el cutoff de datos usado, si hubo reentrenamiento
  y por qué, y si la corrida fue exitosa o falló (trazabilidad que pide
  `docs/student-project.md`).
- **`predictions`** guarda cada predicción emitida, ligada a la corrida que la
  generó (`run_id`) y a la estación, con el horizonte y la versión del modelo.
- **`metrics`** guarda WAPE/Accuracy calculados por corrida (agregados o por
  estación si `station_id` no es null).
- **`drift_signals`** guarda señales de data/concept drift detectadas por
  corrida, con su umbral y si se disparó o no.

Todas las tablas tienen RLS activado con una política de **lectura pública**
(para que el bono de dashboard en Vercel pueda leer directo con la
`anon key`) y sin política de escritura — la escritura solo se hace con la
`service_role key` desde GitHub Actions (nunca expuesta en el navegador).

## Variables de entorno para el pipeline

```
SUPABASE_URL=https://azqudvlltukeeuypivcq.supabase.co
SUPABASE_KEY=<service_role key, guardarla como secret en GitHub, nunca en el código>
```

La `anon`/`publishable key` (para lectura desde el dashboard) sí se puede
exponer en el frontend:

```
SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImF6cXVkdmxsdHVrZWV1eXBpdmNxIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODk2NjA4NzYsImV4cCI6MjEwNTIzNjg3Nn0.HOywCsVUhq8Bn_VuYNqrZr8TYgAXL-eFrgdFNCjhaD0
```
