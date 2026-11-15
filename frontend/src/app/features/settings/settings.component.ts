import { ChangeDetectionStrategy, Component, computed, effect, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { AuthTokenService } from '../../core/auth/auth-token.service';
import { LLMProvider } from '../../core/models/api.models';
import { ThemeService } from '../../core/state/theme.service';
import { WorkspaceStateService } from '../../core/state/workspace-state.service';
import { PageHeaderComponent } from '../../shared/components/page-header.component';
import { SbCardComponent } from '../../shared/components/sb-card.component';
import { StatusBadgeComponent } from '../../shared/components/status-badge.component';
import { SvgIconComponent } from '../../shared/components/svg-icon.component';

@Component({
  selector: 'sb-settings',
  standalone: true,
  imports: [FormsModule, PageHeaderComponent, SbCardComponent, StatusBadgeComponent, SvgIconComponent],
  changeDetection: ChangeDetectionStrategy.OnPush,
  styles: [
    `
      .settings-layout {
        display: grid;
        grid-template-columns: 180px 1fr;
        gap: 22px;
      }
      .settings-nav {
        display: flex;
        flex-direction: column;
        gap: 4px;
        position: sticky;
        top: 84px;
        align-self: start;
      }
      .settings-nav a {
        text-decoration: none;
        color: var(--muted);
        font-size: 12px;
        padding: 9px 10px;
        border-left: 2px solid transparent;
        border-radius: 0 6px 6px 0;
      }
      .settings-nav a:hover {
        color: var(--primary);
        background: var(--primary-soft);
      }
      .settings-content {
        display: flex;
        flex-direction: column;
        gap: 14px;
        max-width: 920px;
      }
      .settings-form {
        display: flex;
        flex-direction: column;
        gap: 13px;
        max-width: 560px;
      }
      .settings-form label {
        display: flex;
        flex-direction: column;
        gap: 5px;
      }
      .settings-form label > span {
        font-size: 11px;
        font-weight: 600;
      }
      .model-chips {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
      }
      .model-chip {
        border: 1px solid var(--border-strong);
        background: var(--surface-2);
        border-radius: 14px;
        padding: 4px 10px;
        font-size: 11px;
      }
      .model-chip:hover {
        border-color: var(--primary);
        color: var(--primary);
      }
      .inline-grid {
        display: grid;
        grid-template-columns: 1fr 1fr auto;
        gap: 12px;
        align-items: end;
      }
      .runtime-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1px;
        background: var(--border);
        border: 1px solid var(--border);
        border-radius: 8px;
        overflow: hidden;
      }
      .runtime-grid > div {
        background: var(--surface);
        padding: 14px;
        display: grid;
        grid-template-columns: 1fr auto;
        gap: 4px;
      }
      .runtime-grid small {
        grid-column: 1 / -1;
        font-size: 11px;
        color: var(--muted);
      }
      .theme-options {
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
      }
      .theme-btn {
        width: 170px;
        background: none;
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 8px;
        display: grid;
        grid-template-columns: 1fr auto;
        text-align: left;
        gap: 8px;
      }
      .theme-btn.selected {
        border-color: var(--primary);
        color: var(--primary);
      }
      .theme-preview {
        grid-column: 1 / -1;
        height: 68px;
        border-radius: 5px;
        border: 1px solid var(--border);
        background: linear-gradient(90deg, #13272c 0 26%, #f7f9fa 26%);
      }
      .theme-preview.dark {
        background: linear-gradient(90deg, #10191d 0 26%, #172328 26%);
      }
      @media (max-width: 800px) {
        .settings-layout {
          display: block;
        }
        .settings-nav {
          display: none;
        }
        .inline-grid,
        .runtime-grid {
          grid-template-columns: 1fr;
        }
      }
    `,
  ],
  template: `
    <div class="page">
      <sb-page-header
        eyebrow="Workspace preferences"
        title="Settings & system status"
        description="Configure active session LLM provider/model, runtime API authentication, backend connectivity, and appearance."
      />

      <div class="settings-layout">
        <aside class="settings-nav" aria-label="Settings sections">
          <a href="#model">Model configuration</a>
          <a href="#backend">Backend connection</a>
          <a href="#runtime">Runtime status</a>
          <a href="#appearance">Appearance</a>
        </aside>

        <div class="settings-content">
          <div id="model">
            <sb-card
              title="Model & provider configuration"
              subtitle="Applied to the active analysis session; optional external provider API keys are never persisted in browser storage."
            >
              <form class="settings-form" (ngSubmit)="applyProvider()">
                <label>
                  <span>Provider</span>
                  <select
                    class="field-control"
                    name="provider"
                    [ngModel]="provider()"
                    (ngModelChange)="provider.set($event)"
                  >
                    <option value="ollama">Ollama (local)</option>
                    <option value="llamacpp">llama.cpp server (local)</option>
                    <option value="openai">OpenAI-compatible</option>
                    <option value="gemini">Google Gemini</option>
                  </select>
                </label>

                <label>
                  <span>Model name</span>
                  <input
                    class="field-control"
                    name="modelName"
                    [ngModel]="modelName()"
                    (ngModelChange)="modelName.set($event)"
                    placeholder="e.g. qwen3:4b-instruct"
                    required
                  />
                </label>

                @if (availableOllamaModels().length > 0 && provider() === 'ollama') {
                  <div>
                    <small style="display: block; font-size: 10px; color: var(--muted); margin-bottom: 6px">
                      Detected local Ollama models:
                    </small>
                    <div class="model-chips">
                      @for (m of availableOllamaModels(); track m) {
                        <button type="button" class="model-chip mono" (click)="modelName.set(m)">
                          {{ m }}
                        </button>
                      }
                    </div>
                  </div>
                }

                @if (provider() === 'openai' || provider() === 'gemini') {
                  <label>
                    <span>Provider API key (optional override, held in memory only)</span>
                    <input
                      type="password"
                      class="field-control"
                      name="providerApiKey"
                      [ngModel]="providerApiKey()"
                      (ngModelChange)="providerApiKey.set($event)"
                      placeholder="Leave blank to use backend environment key"
                      autocomplete="off"
                    />
                  </label>
                }

                <div style="display: flex; gap: 10px; align-items: center">
                  <button
                    type="submit"
                    class="button primary"
                    [disabled]="applyingProvider() || !modelName().trim()"
                  >
                    {{ applyingProvider() ? 'Applying…' : 'Apply configuration' }}
                  </button>
                </div>
              </form>
            </sb-card>
          </div>

          <div id="backend">
            <sb-card
              title="Backend connection & authentication"
              subtitle="Default empty endpoint uses the same-origin proxy (/api and /health). Bearer tokens stay in JS memory only."
            >
              <div class="inline-grid">
                <label style="display: flex; flex-direction: column; gap: 5px; font-size: 11px; font-weight: 600">
                  <span>API base URL (blank = same-origin proxy)</span>
                  <input
                    class="field-control"
                    name="apiBaseUrl"
                    [ngModel]="apiBaseUrlInput()"
                    (ngModelChange)="apiBaseUrlInput.set($event)"
                    placeholder="Same-origin (/api/v1 & /health)"
                  />
                </label>

                <label style="display: flex; flex-direction: column; gap: 5px; font-size: 11px; font-weight: 600">
                  <span>Bearer auth token (memory only)</span>
                  <input
                    type="password"
                    class="field-control"
                    name="bearerToken"
                    [ngModel]="bearerTokenInput()"
                    (ngModelChange)="bearerTokenInput.set($event)"
                    placeholder="Optional unless backend requires auth"
                    autocomplete="off"
                  />
                </label>

                <div style="display: flex; gap: 6px">
                  <button type="button" class="button primary" (click)="saveAndTestConnection()">
                    Connect &amp; test
                  </button>
                  @if (auth.hasToken()) {
                    <button type="button" class="button" (click)="clearBearerToken()">Clear</button>
                  }
                </div>
              </div>

              <div style="display: flex; align-items: center; gap: 12px; margin-top: 14px; font-size: 11px; color: var(--muted)">
                <sb-status-badge [kind]="state.health() ? 'success' : 'error'">
                  {{ state.health() ? 'Backend reachable (v' + state.health()!.version + ')' : 'Not reachable' }}
                </sb-status-badge>
                <span>
                  @if (state.healthError()) {
                    {{ state.healthError() }}
                  } @else {
                    Verified via GET /health
                  }
                </span>
              </div>
            </sb-card>
          </div>

          <div id="runtime">
            <sb-card
              title="Live runtime capabilities"
              subtitle="Telemetry reported directly by GET /health"
            >
              <div class="runtime-grid">
                <div>
                  <span>API status</span>
                  <sb-status-badge [kind]="state.health()?.status === 'ok' ? 'success' : 'warning'">
                    {{ state.health()?.status ?? 'unknown' }}
                  </sb-status-badge>
                  <small>Repository: {{ state.health()?.session_repository ?? 'unknown' }}</small>
                </div>

                <div>
                  <span>LLM backend</span>
                  <sb-status-badge [kind]="state.health()?.llm?.status === 'connected' ? 'success' : 'warning'">
                    {{ state.health()?.llm?.provider ?? 'unknown' }} · {{ state.health()?.llm?.status ?? 'unavailable' }}
                  </sb-status-badge>
                  <small>Default model: {{ state.health()?.default_model ?? state.activeModel() }}</small>
                </div>

                <div>
                  <span>Execution sandbox</span>
                  <sb-status-badge [kind]="state.health()?.sandbox_dynamic_python_enabled ? 'success' : 'warning'">
                    {{ state.health()?.sandbox_backend ?? 'unknown' }}
                  </sb-status-badge>
                  <small>
                    Dynamic Python:
                    {{ state.health()?.sandbox_dynamic_python_enabled ? 'Enabled (fail-closed)' : 'Disabled (deterministic-only)' }}
                  </small>
                </div>

                <div>
                  <span>Security &amp; limits</span>
                  <sb-status-badge [kind]="state.health()?.auth_required ? 'success' : 'neutral'">
                    {{ state.health()?.auth_required ? 'Auth required' : 'Dev open mode' }}
                  </sb-status-badge>
                  <small>
                    Max upload: {{ state.maxUploadMb() }} MB · Storage:
                    {{ state.health()?.dataset_storage?.backend ?? 'local' }}
                  </small>
                </div>
              </div>
            </sb-card>
          </div>

          <div id="appearance">
            <sb-card title="Appearance" subtitle="Theme preference persists locally on this device">
              <div class="theme-options">
                <button
                  type="button"
                  class="theme-btn"
                  [class.selected]="theme.mode() === 'light'"
                  (click)="theme.setMode('light')"
                >
                  <span class="theme-preview"></span>
                  <b>Light</b>
                  @if (theme.mode() === 'light') {
                    <sb-svg-icon name="check" [size]="15" />
                  }
                </button>
                <button
                  type="button"
                  class="theme-btn"
                  [class.selected]="theme.mode() === 'dark'"
                  (click)="theme.setMode('dark')"
                >
                  <span class="theme-preview dark"></span>
                  <b>Dark</b>
                  @if (theme.mode() === 'dark') {
                    <sb-svg-icon name="check" [size]="15" />
                  }
                </button>
              </div>
            </sb-card>
          </div>
        </div>
      </div>
    </div>
  `,
})
export class SettingsComponent {
  readonly state = inject(WorkspaceStateService);
  readonly auth = inject(AuthTokenService);
  readonly theme = inject(ThemeService);

  readonly provider = signal<LLMProvider>('ollama');
  readonly modelName = signal<string>('qwen3:4b-instruct');
  readonly providerApiKey = signal<string>('');
  readonly applyingProvider = signal<boolean>(false);

  readonly apiBaseUrlInput = signal<string>(this.auth.apiBaseUrl());
  readonly bearerTokenInput = signal<string>('');

  readonly availableOllamaModels = computed(() => {
    const raw = this.state.health()?.llm?.available_models;
    return Array.isArray(raw) ? raw.filter((m): m is string => typeof m === 'string') : [];
  });

  constructor() {
    effect(() => {
      const p = this.state.activeProvider() as LLMProvider;
      const m = this.state.activeModel();
      if (p) this.provider.set(p);
      if (m) this.modelName.set(m);
    });
  }

  applyProvider(): void {
    const model = this.modelName().trim();
    if (!model) return;
    this.applyingProvider.set(true);
    const key = this.providerApiKey().trim();
    this.state.updateProvider(
      {
        provider: this.provider(),
        model,
        api_key: key || null,
      },
      () => {
        this.providerApiKey.set('');
        this.applyingProvider.set(false);
      },
    );
  }

  saveAndTestConnection(): void {
    this.auth.setApiBaseUrl(this.apiBaseUrlInput());
    if (this.bearerTokenInput().trim()) {
      this.auth.setToken(this.bearerTokenInput());
      this.bearerTokenInput.set('');
    }
    this.state.retryBootstrapAfterAuth();
    this.state.showToast('Connection settings updated; re-checking /health.');
  }

  clearBearerToken(): void {
    this.auth.clearToken();
    this.bearerTokenInput.set('');
    this.state.showToast('In-memory Bearer token cleared.');
  }
}
