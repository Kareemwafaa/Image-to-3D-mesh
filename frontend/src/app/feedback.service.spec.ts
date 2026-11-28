import { TestBed } from '@angular/core/testing';
import { FeedbackService, parseFeedbackConfig } from './feedback.service';

const CFG = { feedback: { supabaseUrl: 'https://abc.supabase.co/', supabaseAnonKey: 'anon-key' } };

describe('parseFeedbackConfig', () => {
  it('reads the Supabase URL and key', () => {
    expect(parseFeedbackConfig(CFG)).toEqual({
      supabaseUrl: 'https://abc.supabase.co',
      supabaseAnonKey: 'anon-key',
    });
  });

  it('is off when the block is missing, blank or not https', () => {
    expect(parseFeedbackConfig({})).toBeNull();
    expect(parseFeedbackConfig(null)).toBeNull();
    expect(parseFeedbackConfig({ feedback: { supabaseUrl: '', supabaseAnonKey: '' } })).toBeNull();
    expect(
      parseFeedbackConfig({
        feedback: { supabaseUrl: 'https://abc.supabase.co', supabaseAnonKey: ' ' },
      }),
    ).toBeNull();
    expect(
      parseFeedbackConfig({
        feedback: { supabaseUrl: 'http://abc.supabase.co', supabaseAnonKey: 'k' },
      }),
    ).toBeNull();
  });
});

describe('FeedbackService.send', () => {
  let calls: { url: string; init: RequestInit }[];

  beforeEach(() => {
    TestBed.resetTestingModule();
    calls = [];
    vi.stubGlobal(
      'fetch',
      vi.fn(async (url: string, init: RequestInit = {}) => {
        calls.push({ url, init });
        if (url === 'config.json') return new Response(JSON.stringify(CFG));
        return new Response(null, { status: 201 });
      }),
    );
  });

  afterEach(() => vi.unstubAllGlobals());

  it('sends only the text when nothing is attached', async () => {
    await TestBed.inject(FeedbackService).send({
      rating: 4,
      message: ' nice ',
      email: '',
      context: { engine: 'instantmesh' },
    });
    const posts = calls.filter((c) => c.url !== 'config.json');
    expect(posts.map((c) => c.url)).toEqual(['https://abc.supabase.co/rest/v1/feedback']);
    const headers = posts[0].init.headers as Record<string, string>;
    expect(headers['apikey']).toBe('anon-key');
    expect(headers['Prefer']).toBe('return=minimal');
    const row = JSON.parse(posts[0].init.body as string);
    expect(row).toMatchObject({
      rating: 4,
      message: 'nice',
      email: null,
      photo_path: null,
      model_path: null,
    });
    expect(row.context).toEqual({ engine: 'instantmesh' });
  });

  it('uploads the photo and model first, then saves the row pointing at them', async () => {
    const photo = new File(['img'], 'mug.jpg', { type: 'image/jpeg' });
    const model = new Blob(['glb']);
    await TestBed.inject(FeedbackService).send({
      rating: null,
      message: 'lumpy',
      email: 'a@b.c',
      photo,
      model,
    });
    const posts = calls.filter((c) => c.url !== 'config.json');
    expect(posts).toHaveLength(3);
    const id = JSON.parse(posts[2].init.body as string).id;
    expect(posts[0].url).toBe(`https://abc.supabase.co/storage/v1/object/feedback/${id}/photo.jpg`);
    expect(posts[1].url).toBe(`https://abc.supabase.co/storage/v1/object/feedback/${id}/model.glb`);
    expect((posts[1].init.headers as Record<string, string>)['Content-Type']).toBe(
      'model/gltf-binary',
    );
    expect(JSON.parse(posts[2].init.body as string)).toMatchObject({
      photo_path: `${id}/photo.jpg`,
      model_path: `${id}/model.glb`,
      email: 'a@b.c',
    });
  });

  it('does not save the row when an upload fails', async () => {
    (fetch as ReturnType<typeof vi.fn>).mockImplementation(
      async (url: string, init: RequestInit = {}) => {
        calls.push({ url, init });
        if (url === 'config.json') return new Response(JSON.stringify(CFG));
        return new Response(null, { status: url.includes('/storage/') ? 400 : 201 });
      },
    );
    const photo = new File(['img'], 'mug.png', { type: 'image/png' });
    await expect(
      TestBed.inject(FeedbackService).send({ rating: 2, message: '', email: '', photo }),
    ).rejects.toThrow(/Untick the box/);
    expect(calls.some((c) => c.url.includes('/rest/v1/'))).toBe(false);
  });
});
