# Supabase Setup — Consolidated To-Do

> **All Supabase changes are collected here so they can be applied in one go.**
> The image pipeline code is complete and tested locally; the items below are the
> only Supabase-side actions needed to run it end-to-end against the live project
> (which currently lives in a separate Supabase account).

## ✅ Verified status (live project: "Data Science", ref `gtozjiqzsdbuptweeyeg`)

Checked against the live project via the Supabase MCP — it is **already provisioned**:

- ✅ All backend tables exist with the correct columns and enums (`datasets`,
  `pipeline_artifacts`, `artifacts`, `data_samples`, `processing_logs`,
  `data_quality_metrics`, `model_results`, `user_profiles`, `user_preferences`)
  and already contain data.
- ✅ All 5 storage buckets exist; **`eda` is public** (visualizations work).
- ✅ The image flow needs **no schema changes** (reuses existing tables).

**To run the app you only need to finish two things:**

1. **Service-role key** — paste it into `api/.env` as `SUPABASE_SERVICE_ROLE_KEY`
   (Dashboard → Project Settings → API → `service_role`). It is secret and is the
   only value that cannot be fetched automatically; the backend's `SupabaseManager`
   requires it.
2. **Auth redirect URLs** — Dashboard → Authentication → URL Configuration → add
   `http://localhost:3000` (and `http://localhost:5173` for local dev) so Google /
   email sign-in redirects back to the app.

`api/.env`, `frontenddata 1.1/project/.env`, and the repo-root `.env` have already
been created locally (gitignored) with the URL + anon key pre-filled.

> Setting up a **fresh / new** Supabase project instead? Paste
> [`supabase/schema.sql`](supabase/schema.sql) into the SQL Editor — it creates
> every table, enum and bucket in one go.

> ⚠️ **Security advisory (from Supabase):** RLS is disabled on `user_preferences`
> and `artifacts`, so anyone with the anon key can read/modify those rows. This was
> deprioritised per your request. To lock down later:
> ```sql
> ALTER TABLE public.user_preferences ENABLE ROW LEVEL SECURITY;
> ALTER TABLE public.artifacts ENABLE ROW LEVEL SECURITY;
> ```
> (Add policies afterwards, or those tables become inaccessible to the anon key.)
> Docs: https://supabase.com/docs/guides/database/postgres/row-level-security

## 0. Backend credentials

The backend reads these environment variables (see `api/.env.example`):

| Variable                    | Purpose |
|-----------------------------|---------|
| `SUPABASE_URL`              | Project URL (`https://<ref>.supabase.co`) |
| `SUPABASE_SERVICE_ROLE_KEY` | Service-role key used by the API (full access) |
| `SUPABASE_KEY`              | Legacy anon key used by `api/supabase_client.py` |

Create `api/.env` from `api/.env.example` and fill these in.

### Frontend credentials (build time)

The frontend's Supabase client (`src/lib/supabase.ts`) reads these **Vite** vars,
which are inlined into the bundle **at build time**:

| Variable                  | Purpose |
|---------------------------|---------|
| `VITE_SUPABASE_URL`       | Same project URL as above |
| `VITE_SUPABASE_ANON_KEY`  | Anon (public) key for browser auth/storage |

- Local dev: copy `frontenddata 1.1/project/.env.example` → `.env`.
- Docker: copy the repo-root `.env.example` → `.env` (docker compose passes these
  as build args to the frontend image). Rebuild the frontend after changing them.

## 1. Storage buckets (REQUIRED for the image flow)

The pipeline uses five buckets. Create any that don't exist:

| Bucket         | Used by | Access |
|----------------|---------|--------|
| `datasets`     | uploads (raw images / zip) | private (downloaded via backend) |
| `preprocessed` | preprocess output zips | private |
| `augmented`    | augment output zips | private |
| `eda`          | EDA visualization PNGs | **public read** ← see note |
| `models`       | trained model artifacts | private |

> **`eda` must allow public read.** The frontend renders EDA visualizations
> directly from `get_public_url(...)`. If `eda` is private, the image (and the
> existing numeric/text) EDA plots will not display. Either make the `eda`
> bucket public, or add a Storage policy granting public `SELECT` on it.

Buckets can be created in the Supabase dashboard (Storage → New bucket) or via
`SupabaseManager.ensure_buckets_exist()`.

## 2. Database schema

**No schema changes are required for the image flow.** Image artifacts reuse the
existing tables — `pipeline_artifacts` and `artifacts` — with the stage set to
`preprocess` / `augment` / `eda` / `model` and image-specific details stored in
the JSON `metadata` / `meta` columns (`data_type: "image"`, summaries, metrics).

Confirm these tables already exist (they back the numeric/text flow too):
`datasets`, `pipeline_artifacts`, `artifacts`, `data_samples`,
`processing_logs`, `data_quality_metrics`, `model_results`, `user_profiles`.

> Note: the SQL under `frontenddata 1.1/project/supabase/migrations/` defines a
> *different* set of tables (`user_files`, `augmented_files`, `processing_jobs`)
> that the current backend does not use. Those migrations are stale; rely on the
> table names above (the ones the code actually reads/writes).

## 3. Quick verification

After setting env vars + buckets, hit `GET /api/datasets/test-connection`
(existing endpoint) — it checks both DB and Storage connectivity.

## 4. Optional hardening (not required to run)

Deprioritised per request, but noted for later: the backend trusts a
client-supplied `user_id` and uses the service-role key (which bypasses RLS).
Before any real/public deployment, add JWT verification on the API and enable
RLS on the tables above. See prior analysis for details.
