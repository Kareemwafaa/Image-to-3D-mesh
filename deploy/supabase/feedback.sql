-- Facet feedback: one table and one private file bucket in a Supabase project.
-- Run once in the Supabase dashboard: SQL Editor -> New query -> paste -> Run.
-- Safe to run again.
--
-- The site uses the project's public key (anon / publishable). These policies
-- let that key ADD feedback and UPLOAD files, nothing else: visitors cannot
-- read, change or delete anyone's feedback. You read it in the dashboard,
-- which bypasses these rules.

-- 1. The table ---------------------------------------------------------------
create table if not exists public.feedback (
  id          uuid primary key,
  created_at  timestamptz not null default now(),
  rating      smallint check (rating between 1 and 5),
  message     text check (char_length(message) <= 4000),
  email       text check (char_length(email) <= 200),
  photo_path  text check (char_length(photo_path) <= 200),
  model_path  text check (char_length(model_path) <= 200),
  context     jsonb check (pg_column_size(context) <= 4000),
  page_url    text check (char_length(page_url) <= 500),
  -- An empty form is not feedback.
  constraint feedback_not_empty check (rating is not null or coalesce(message, '') <> '')
);

alter table public.feedback enable row level security;

-- Insert only. With no select/update/delete policy, those are refused.
drop policy if exists "Anyone can send feedback" on public.feedback;
create policy "Anyone can send feedback"
  on public.feedback for insert
  to anon, authenticated
  with check (true);

-- Newer projects may not expose new tables to the API by default.
grant insert on table public.feedback to anon, authenticated;

-- 2. The file bucket ---------------------------------------------------------
-- Private (no public links), 25 MB per file, only photos and GLB models.
insert into storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
values ('feedback', 'feedback', false, 26214400,
        array['image/png', 'image/jpeg', 'image/webp', 'model/gltf-binary'])
on conflict (id) do update
  set public = false,
      file_size_limit = excluded.file_size_limit,
      allowed_mime_types = excluded.allowed_mime_types;

-- Upload only, into this bucket only. No read, overwrite or delete.
drop policy if exists "Anyone can upload feedback files" on storage.objects;
create policy "Anyone can upload feedback files"
  on storage.objects for insert
  to anon, authenticated
  with check (bucket_id = 'feedback');
