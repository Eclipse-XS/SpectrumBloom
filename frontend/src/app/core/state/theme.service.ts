import { Injectable, effect, signal } from '@angular/core';
import { ThemeMode } from '../models/workspace.models';

const THEME_STORAGE_KEY = 'sb-theme';

@Injectable({ providedIn: 'root' })
export class ThemeService {
  readonly mode = signal<ThemeMode>(this.readInitialTheme());

  constructor() {
    effect(() => {
      const current = this.mode();
      document.documentElement.dataset['theme'] = current;
      try {
        localStorage.setItem(THEME_STORAGE_KEY, current);
      } catch {
        // Ignore storage errors in hardened/private environments
      }
    });
  }

  setMode(next: ThemeMode): void {
    this.mode.set(next);
  }

  toggle(): void {
    this.mode.update((prev) => (prev === 'dark' ? 'light' : 'dark'));
  }

  private readInitialTheme(): ThemeMode {
    try {
      const saved = localStorage.getItem(THEME_STORAGE_KEY);
      if (saved === 'dark' || saved === 'light') {
        return saved;
      }
    } catch {
      // Ignore storage access errors
    }
    return 'light';
  }
}
