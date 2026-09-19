/**
 * Drop-in replacement for fetch() that attaches the device's auth token
 * automatically. Same signature as native fetch — every service file just
 * swaps `fetch(` for `apiFetch(`, nothing else changes.
 *
 * On first use (no token in localStorage yet), it calls POST /auth/device
 * once to get a token, caches it, and reuses it on every request after
 * that — including across page reloads, since it lives in localStorage.
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const DEVICE_TOKEN_KEY = 'aifin_device_token';

// Avoids firing multiple concurrent /auth/device requests if several API
// calls happen before the first token request resolves (e.g. on page load).
let tokenPromise: Promise<string> | null = null;

async function requestNewToken(): Promise<string> {
  const response = await fetch(`${API_BASE_URL}/auth/device`, { method: 'POST' });
  if (!response.ok) {
    throw new Error('Failed to obtain device token');
  }
  const data = await response.json();
  localStorage.setItem(DEVICE_TOKEN_KEY, data.token);
  return data.token;
}

async function getDeviceToken(): Promise<string> {
  const cached = localStorage.getItem(DEVICE_TOKEN_KEY);
  if (cached) return cached;

  if (!tokenPromise) {
    tokenPromise = requestNewToken().finally(() => {
      tokenPromise = null;
    });
  }
  return tokenPromise;
}

export async function apiFetch(input: RequestInfo | URL, init: RequestInit = {}): Promise<Response> {
  const token = await getDeviceToken();
  const headers = new Headers(init.headers);
  headers.set('Authorization', `Bearer ${token}`);

  let response = await fetch(input, { ...init, headers });

  // Stored token was rejected (expired, or backend restarted with a new
  // JWT_SECRET_KEY) — clear it and retry once with a freshly issued token.
  if (response.status === 401) {
    localStorage.removeItem(DEVICE_TOKEN_KEY);
    const freshToken = await getDeviceToken();
    const retryHeaders = new Headers(init.headers);
    retryHeaders.set('Authorization', `Bearer ${freshToken}`);
    response = await fetch(input, { ...init, headers: retryHeaders });
  }

  return response;
}