# In-app feedback

The **Feedback** button at the bottom right of the 3D viewer, shown once a model is generated,
opens a small form: a 1 to 5 star rating, a message, an optional email, and an opt-in switch to
include the photo and the generated model. The switch is off by default, so nothing but the text is
sent unless the person turns it on.

There is no server of ours involved. The browser sends the form straight to a free
[Supabase](https://supabase.com) project: the text goes into a database table, the photo and `.glb`
into a private file bucket. Only the project owner can read them, in the Supabase dashboard.

Until it is set up, the button stays hidden, so forks of this repo show no broken form.

## How it stays private

The site has to carry a Supabase key, and anyone can read it from the page. That key is the
project's **public** key (Supabase calls it `anon` or "publishable"), which is designed to be shipped
to browsers. What it may do is decided by row-level security rules in
[`deploy/supabase/feedback.sql`](../deploy/supabase/feedback.sql):

| Who | Feedback table | Files |
| --- | --- | --- |
| Visitors (public key) | add a row | upload into `feedback/` |
| Visitors | cannot read, edit or delete | cannot list, download, overwrite or delete |
| You (dashboard) | everything | everything |

Database checks cap the message at 4,000 characters and refuse empty or out-of-range ratings; the
bucket accepts only PNG, JPEG, WebP and GLB files up to 25 MB. The form also has a hidden "honeypot"
field that drops simple bots. Never put the **service_role** / **secret** key in the site: that one
bypasses every rule.

## Set it up (about 10 minutes, free)

1. **Make a Supabase account.** Go to [supabase.com](https://supabase.com), click *Start your project*
   and sign in with GitHub.
2. **Create a project.** Click *New project*. Any name (e.g. `facet-feedback`), any strong database
   password (save it somewhere, you won't need it for this), the region closest to you, the *Free*
   plan. Wait a minute or two while it starts.
3. **Create the table and bucket.** In the left sidebar open *SQL Editor*, click *New query*, paste
   the whole of [`deploy/supabase/feedback.sql`](../deploy/supabase/feedback.sql) and click *Run*.
   It should say "Success. No rows returned". Running it again is harmless.
4. **Copy two values.** Click *Connect* at the top of the project page (or open *Project Settings →
   API Keys* / *Data API*) and copy:
   - the **Project URL**, like `https://abcdefghijkl.supabase.co`
   - the **anon public** key (a long text starting with `eyJ`) or the **publishable** key (starts
     with `sb_publishable_`). Either works. Not the secret / service_role key.
5. **Paste them into the site's config.** On GitHub, open `frontend/public/config.json`, click the
   pencil (Edit) icon and fill in the two empty values:

   ```json
   {
     "apiUrl": "http://localhost:8000",
     "feedback": {
       "supabaseUrl": "https://abcdefghijkl.supabase.co",
       "supabaseAnonKey": "eyJhbGciOi..."
     }
   }
   ```

   Click *Commit changes*. GitHub Pages redeploys in a couple of minutes and the Feedback button
   appears.

## Read the feedback

- **Messages:** Supabase dashboard → *Table Editor* → `feedback`. Newest last; click a column header
  to sort by `created_at`. `context` holds the run's settings (engine, background removal, job id,
  time taken, any error), so you can reproduce a bad result.
- **Photos and models:** *Storage* → `feedback` bucket. Each submission has a folder named after its
  `id`; `photo_path` and `model_path` in the table point to the files. Download a `.glb` and open it
  in any 3D viewer.

To be emailed on every new message, add a *Database Webhook* (Integrations → Webhooks) on inserts
into `feedback`, or simply check the table now and then.

## Limits

The free plan gives 500 MB of database and 1 GB of file storage, plenty for feedback (a photo plus a
model is usually 2 to 10 MB). Free projects are paused after a week with no activity; feedback
coming in counts as activity, and a paused project can be resumed from the dashboard in one click.
Because the public key can insert, a determined spammer could fill the bucket; if that ever happens,
delete the upload policy (or the whole bucket) in the dashboard and the form keeps working for text.
