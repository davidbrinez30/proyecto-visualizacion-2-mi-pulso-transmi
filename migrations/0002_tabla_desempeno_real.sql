-- Tabla para medir el desempeno REAL: compara cada prediccion que enviamos
-- contra el dato real que el profesor termina publicando para ese mismo
-- station_id + target_at (llenada por scripts/eval_performance.py).

create table if not exists public.submission_performance (
  id bigint generated always as identity primary key,
  station_id text not null references public.stations(station_id),
  target_at timestamptz not null,
  predicted_demand double precision not null,
  actual_demand double precision not null,
  abs_error double precision not null,
  wape double precision,
  accuracy double precision,
  evaluated_at timestamptz not null default now(),
  unique (station_id, target_at)
);

alter table public.submission_performance enable row level security;

drop policy if exists "lectura publica" on public.submission_performance;
create policy "lectura publica" on public.submission_performance
  for select using (true);

drop policy if exists "escritura solo con secret key" on public.submission_performance;
create policy "escritura solo con secret key" on public.submission_performance
  for all using (auth.role() = 'service_role') with check (auth.role() = 'service_role');
