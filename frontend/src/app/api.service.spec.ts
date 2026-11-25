import { TestBed } from '@angular/core/testing';
import { ApiService, normalizeApiUrl } from './api.service';

describe('normalizeApiUrl', () => {
  it('keeps http(s) URLs and drops trailing slashes', () => {
    expect(normalizeApiUrl('https://abc.trycloudflare.com/')).toBe('https://abc.trycloudflare.com');
    expect(normalizeApiUrl(' http://localhost:8000 ')).toBe('http://localhost:8000');
    expect(normalizeApiUrl('https://host.example/api/')).toBe('https://host.example/api');
  });

  it('assumes https when the scheme is missing', () => {
    expect(normalizeApiUrl('abc.trycloudflare.com')).toBe('https://abc.trycloudflare.com');
  });

  it('rejects empty and non-web values', () => {
    expect(normalizeApiUrl('')).toBeNull();
    expect(normalizeApiUrl(null)).toBeNull();
    expect(normalizeApiUrl('javascript:alert(1)')).toBeNull();
    expect(normalizeApiUrl('https://')).toBeNull();
  });
});

describe('ApiService backend URL', () => {
  const original = location.href;

  beforeEach(() => {
    localStorage.clear();
    TestBed.resetTestingModule();
  });

  afterEach(() => history.replaceState(null, '', original));

  it('takes ?api= from the address, remembers it and removes it from the address', () => {
    history.replaceState(null, '', '/?job=42&api=https%3A%2F%2Fdemo.trycloudflare.com%2F');
    const api = TestBed.inject(ApiService);
    expect(api.baseUrl()).toBe('https://demo.trycloudflare.com');
    expect(api.source()).toBe('custom');
    expect(localStorage.getItem('facet.apiUrl')).toBe('https://demo.trycloudflare.com');
    expect(location.search).toBe('?job=42');
  });

  it('uses the URL saved in this browser', () => {
    localStorage.setItem('facet.apiUrl', 'https://saved.example');
    expect(TestBed.inject(ApiService).baseUrl()).toBe('https://saved.example');
  });

  it('goes back to the deployment default when cleared', async () => {
    localStorage.setItem('facet.apiUrl', 'https://saved.example');
    const api = TestBed.inject(ApiService);
    vi.spyOn(api, 'defaultUrl').mockResolvedValue('http://localhost:8000');
    await api.setBaseUrl(null);
    expect(api.baseUrl()).toBe('http://localhost:8000');
    expect(api.source()).toBe('default');
    expect(localStorage.getItem('facet.apiUrl')).toBeNull();
  });
});
