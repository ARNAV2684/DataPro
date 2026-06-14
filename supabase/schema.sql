-- ===========================================================================
-- Garuda ML Pipeline / DataLab Pro — reproducible Supabase schema
-- ===========================================================================
-- This mirrors the live "Data Science" project (ref: gtozjiqzsdbuptweeyeg) and
-- lets you stand up a fresh Supabase project for the app in one paste:
--   Supabase Dashboard -> SQL Editor -> New query -> paste -> Run.
--
-- It is idempotent (safe to re-run): tables use IF NOT EXISTS, enum types are
-- guarded, and bucket inserts use ON CONFLICT DO NOTHING.
--
-- Note: RLS is left disabled to match current behaviour (the backend uses the
-- service_role key, which bypasses RLS). Harden before any public deployment.
-- ===========================================================================

-- ----- enum types -----
do $$ begin
  create type pipeline_stage as enum ('upload', 'preprocess', 'augment', 'eda', 'model');
exception when duplicate_object then null; end $$;

do $$ begin
  create type processing_status as enum ('pending', 'running', 'completed', 'failed');
exception when duplicate_object then null; end $$;

-- ----- core pipeline tables -----
create table if not exists public.datasets (
  id text primary key default (gen_random_uuid())::text,
  user_id text not null,
  bucket_key text not null,
  filename text not null,
  filetype text,
  data_type text,
  file_size bigint,
  status text default 'pending',
  description text,
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

create table if not exists public.pipeline_artifacts (
  id text primary key default (gen_random_uuid())::text,
  user_id text not null,
  dataset_id text references public.datasets(id) on delete cascade,
  stage pipeline_stage not null,
  operation text,
  bucket_key text,
  status processing_status default 'pending',
  input_artifact_id text references public.pipeline_artifacts(id),
  parameters jsonb default '{}',
  metadata jsonb default '{}',
  execution_time double precision,
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

-- Lighter-weight artifact table used by the EDA and image-model endpoints.
create table if not exists public.artifacts (
  id text primary key default (gen_random_uuid())::text,
  user_id text not null,
  dataset_id text,
  stage text,
  bucket_key text,
  meta jsonb default '{}',
  status text default 'pending',
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

create table if not exists public.data_samples (
  id text primary key default (gen_random_uuid())::text,
  artifact_id text references public.pipeline_artifacts(id) on delete cascade,
  sample_data jsonb not null,
  total_rows integer,
  total_columns integer,
  column_info jsonb default '{}',
  created_at timestamptz default now()
);

create table if not exists public.processing_logs (
  id text primary key default (gen_random_uuid())::text,
  artifact_id text references public.pipeline_artifacts(id) on delete cascade,
  log_level text default 'info',
  message text not null,
  details jsonb default '{}',
  created_at timestamptz default now()
);

create table if not exists public.data_quality_metrics (
  id text primary key default (gen_random_uuid())::text,
  artifact_id text references public.pipeline_artifacts(id) on delete cascade,
  missing_values jsonb default '{}',
  outliers jsonb default '{}',
  duplicates integer default 0,
  data_types jsonb default '{}',
  statistics jsonb default '{}',
  created_at timestamptz default now()
);

create table if not exists public.model_results (
  id text primary key default (gen_random_uuid())::text,
  user_id text not null,
  dataset_id text references public.datasets(id) on delete cascade,
  model_type text not null,
  hyperparameters jsonb default '{}',
  metrics jsonb default '{}',
  model_path text,
  status processing_status default 'pending',
  training_time double precision,
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

create table if not exists public.user_profiles (
  id text primary key default (gen_random_uuid())::text,
  user_id text unique not null,
  email text,
  full_name text,
  avatar_url text,
  subscription_tier text default 'free',
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

create table if not exists public.user_preferences (
  id uuid primary key default gen_random_uuid(),
  user_id uuid unique references auth.users(id),
  dataset_type text check (dataset_type = any (array['numeric','text','image','mixed'])),
  theme text default 'light' check (theme = any (array['light','dark','auto'])),
  notifications_enabled boolean default true,
  auto_save boolean default true,
  language text default 'en' check (language = any (array['en','es','fr','de','zh'])),
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

-- ----- legacy tables (used by the frontend FileService) -----
create table if not exists public.user_files (
  id text primary key default (gen_random_uuid())::text,
  user_id uuid references auth.users(id),
  filename text not null,
  original_name text not null,
  file_type text not null,
  storage_path text not null,
  file_size bigint not null,
  processing_stage text default 'uploaded',
  upload_date timestamptz default now()
);

create table if not exists public.augmented_files (
  id text primary key default (gen_random_uuid())::text,
  original_file_id text references public.user_files(id) on delete cascade,
  technique text not null,
  parameters jsonb,
  result_storage_path text not null,
  sample_count integer,
  created_at timestamptz default now()
);

-- ===========================================================================
-- Storage buckets
-- ===========================================================================
-- The `eda` bucket MUST be public so the frontend can render visualizations
-- via public URLs. The rest are private (served through the backend).
insert into storage.buckets (id, name, public) values
  ('datasets',     'datasets',     false),
  ('preprocessed', 'preprocessed', false),
  ('augmented',    'augmented',    false),
  ('eda',          'eda',          true),
  ('models',       'models',       false)
on conflict (id) do nothing;

-- Ensure the eda bucket is public even if it already existed as private.
update storage.buckets set public = true where id = 'eda';
