import { Injectable, computed, signal } from '@angular/core';

@Injectable({ providedIn: 'root' })
export class AuthTokenService {
  // Strictly held in memory only — never written to localStorage or URL query parameters.
  private readonly _token = signal<string>('');
  private readonly _apiBaseUrl = signal<string>('');
  private readonly _authChallenge = signal<boolean>(false);

  readonly token = this._token.asReadonly();
  readonly apiBaseUrl = this._apiBaseUrl.asReadonly();
  readonly authChallenge = this._authChallenge.asReadonly();
  readonly hasToken = computed(() => this._token().trim().length > 0);

  setToken(token: string): void {
    const trimmed = token.trim();
    this._token.set(trimmed);
    if (trimmed.length > 0) {
      this._authChallenge.set(false);
    }
  }

  clearToken(): void {
    this._token.set('');
  }

  setApiBaseUrl(url: string): void {
    const cleaned = url.trim().replace(/\/+$/, '');
    this._apiBaseUrl.set(cleaned);
  }

  resolveUrl(path: string): string {
    const normalizedPath = path.startsWith('/') ? path : `/${path}`;
    const base = this._apiBaseUrl();
    return base ? `${base}${normalizedPath}` : normalizedPath;
  }

  markAuthChallenge(required = true): void {
    this._authChallenge.set(required);
  }

  clearAuthChallenge(): void {
    this._authChallenge.set(false);
  }
}
