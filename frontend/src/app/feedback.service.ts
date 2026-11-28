import { Injectable, signal } from '@angular/core';

// Sends feedback from the in-app form straight to a Supabase project, so the
// static site needs no server of its own. Set up once with
// deploy/supabase/feedback.sql and docs/feedback.md, then put the project URL
// and its public (anon / publishable) key in public/config.json:
//
//   "feedback": { "supabaseUrl": "https://xyz.supabase.co", "supabaseAnonKey": "..." }
//
// The key is meant to be public: row-level security lets it add feedback and
// upload files, but never read, change or delete anything. With no config the
// Feedback button stays hidden.

export interface FeedbackConfig {
  supabaseUrl: string;
  supabaseAnonKey: string;
}

export interface FeedbackInput {
  /** 1 to 5, or null if not given. */
  rating: number | null;
  message: string;
  /** Optional, for a reply. */
  email: string;
  /** Sent only when the person ticks the box. */
  photo?: File | null;
  model?: Blob | null;
  /** Settings and timings of the run, for reproducing it. */
  context?: Record<string, unknown>;
}

const BUCKET = 'feedback';
const TABLE = 'feedback';
export const MESSAGE_MAX = 4000;
export const EMAIL_MAX = 200;
const PHOTO_EXT: Record<string, string> = {
  'image/png': 'png',
  'image/jpeg': 'jpg',
  'image/webp': 'webp',
};

/** Reads the "feedback" block of config.json; null if it is missing or incomplete. */
export function parseFeedbackConfig(cfg: unknown): FeedbackConfig | null {
  const f = (cfg as { feedback?: Partial<FeedbackConfig> } | null)?.feedback;
  const key = f?.supabaseAnonKey?.trim();
  let url: URL;
  try {
    url = new URL(f?.supabaseUrl?.trim() ?? '');
  } catch {
    return null;
  }
  if (url.protocol !== 'https:' || !key) return null;
  return { supabaseUrl: url.origin, supabaseAnonKey: key };
}

@Injectable({ providedIn: 'root' })
export class FeedbackService {
  /** The Supabase project to send to; null until loaded, or when feedback is off. */
  readonly config = signal<FeedbackConfig | null>(null);
  private loaded?: Promise<void>;

  /** Loads the settings from config.json (once). */
  load(): Promise<void> {
    this.loaded ??= fetch('config.json', { cache: 'no-store' })
      .then((r) => (r.ok ? r.json() : null))
      .catch(() => null)
      .then((cfg) => this.config.set(parseFeedbackConfig(cfg)));
    return this.loaded;
  }

  async send(input: FeedbackInput): Promise<void> {
    await this.load();
    const cfg = this.config();
    if (!cfg) throw new Error('Feedback is not set up on this site.');

    const id = uuid();
    let photoPath: string | null = null;
    let modelPath: string | null = null;
    // Files first, so a row never points at a file that failed to upload.
    if (input.photo) {
      photoPath = `${id}/photo.${PHOTO_EXT[input.photo.type] ?? 'png'}`;
      await this.upload(cfg, photoPath, input.photo, input.photo.type || 'image/png');
    }
    if (input.model) {
      modelPath = `${id}/model.glb`;
      await this.upload(cfg, modelPath, input.model, 'model/gltf-binary');
    }

    const res = await fetch(`${cfg.supabaseUrl}/rest/v1/${TABLE}`, {
      method: 'POST',
      headers: {
        ...authHeaders(cfg),
        'Content-Type': 'application/json',
        // The public key may insert but not read, so don't ask for the row back.
        Prefer: 'return=minimal',
      },
      body: JSON.stringify({
        id,
        rating: input.rating,
        message: input.message.trim().slice(0, MESSAGE_MAX) || null,
        email: input.email.trim().slice(0, EMAIL_MAX) || null,
        photo_path: photoPath,
        model_path: modelPath,
        context: input.context ?? null,
        page_url: location.origin + location.pathname,
      }),
    });
    if (!res.ok) throw new Error(`Could not send your feedback (${res.status}). Please try again.`);
  }

  private async upload(cfg: FeedbackConfig, path: string, body: Blob, type: string) {
    const res = await fetch(`${cfg.supabaseUrl}/storage/v1/object/${BUCKET}/${path}`, {
      method: 'POST',
      headers: { ...authHeaders(cfg), 'Content-Type': type, 'x-upsert': 'false' },
      body,
    });
    if (!res.ok) {
      const what = path.endsWith('.glb') ? 'the 3D model' : 'the photo';
      throw new Error(
        res.status === 413
          ? `${capitalize(what)} is too large to attach. Untick the box to send just your message.`
          : `Could not upload ${what} (${res.status}). Untick the box to send just your message.`,
      );
    }
  }
}

function authHeaders(cfg: FeedbackConfig): Record<string, string> {
  // Same headers supabase-js sends for an anonymous visitor.
  return { apikey: cfg.supabaseAnonKey, Authorization: `Bearer ${cfg.supabaseAnonKey}` };
}

/** Random v4 UUID; crypto.randomUUID only exists on https pages, this works on plain http too. */
function uuid(): string {
  const b = crypto.getRandomValues(new Uint8Array(16));
  b[6] = (b[6] & 0x0f) | 0x40;
  b[8] = (b[8] & 0x3f) | 0x80;
  const h = Array.from(b, (x) => x.toString(16).padStart(2, '0')).join('');
  return `${h.slice(0, 8)}-${h.slice(8, 12)}-${h.slice(12, 16)}-${h.slice(16, 20)}-${h.slice(20)}`;
}

function capitalize(s: string) {
  return s.charAt(0).toUpperCase() + s.slice(1);
}
