import { ChangeDetectionStrategy, Component, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';
import { AuthTokenService } from '../../core/auth/auth-token.service';
import { WorkspaceStateService } from '../../core/state/workspace-state.service';
import { SvgIconComponent } from './svg-icon.component';

@Component({
  selector: 'sb-auth-banner',
  standalone: true,
  imports: [FormsModule, SvgIconComponent],
  changeDetection: ChangeDetectionStrategy.OnPush,
  styles: [
    `
      .auth-card {
        max-width: 520px;
        width: 100%;
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 24px;
        box-shadow: var(--shadow);
      }
      .auth-head {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 12px;
      }
      .auth-badge {
        width: 40px;
        height: 40px;
        border-radius: 9px;
        background: var(--primary-soft);
        color: var(--primary);
        display: grid;
        place-items: center;
      }
      .auth-head h2 {
        margin: 0 0 2px;
        font-size: 16px;
      }
      .auth-head p {
        margin: 0;
        font-size: 11px;
        color: var(--muted);
      }
      .auth-form {
        display: flex;
        flex-direction: column;
        gap: 12px;
      }
      .auth-form label {
        display: flex;
        flex-direction: column;
        gap: 6px;
        font-size: 11px;
        font-weight: 600;
      }
      .auth-note {
        font-size: 11px;
        color: var(--muted);
        margin: 0;
      }
      .auth-actions {
        display: flex;
        justify-content: flex-end;
        gap: 8px;
      }
    `,
  ],
  template: `
    @if (auth.authChallenge()) {
      <div class="modal-backdrop" role="presentation">
        <div class="auth-card" role="dialog" aria-modal="true" aria-labelledby="auth-dialog-title">
          <div class="auth-head">
            <div class="auth-badge">
              <sb-svg-icon name="lock" [size]="20" />
            </div>
            <div>
              <h2 id="auth-dialog-title">API authentication required</h2>
              <p>This SpectrumBloom FastAPI backend enforces Bearer authentication.</p>
            </div>
          </div>
          <form class="auth-form" (ngSubmit)="connect()">
            <label>
              <span>API Bearer token (stored in memory only)</span>
              <input
                type="password"
                class="field-control"
                name="apiToken"
                [(ngModel)]="tokenInput"
                placeholder="Paste your API_AUTH_TOKEN…"
                autocomplete="off"
                required
              />
            </label>
            <p class="auth-note">
              Tokens remain in session memory only and are never written to browser localStorage or
              bundled assets.
            </p>
            <div class="auth-actions">
              <button type="button" class="button" (click)="dismiss()">Configure in Settings</button>
              <button type="submit" class="button primary" [disabled]="!tokenInput().trim()">
                Unlock workspace
              </button>
            </div>
          </form>
        </div>
      </div>
    }
  `,
})
export class AuthBannerComponent {
  readonly auth = inject(AuthTokenService);
  private readonly state = inject(WorkspaceStateService);
  private readonly router = inject(Router);

  readonly tokenInput = signal<string>('');

  connect(): void {
    const val = this.tokenInput().trim();
    if (!val) {
      return;
    }
    this.auth.setToken(val);
    this.tokenInput.set('');
    this.state.retryBootstrapAfterAuth();
  }

  dismiss(): void {
    this.auth.clearAuthChallenge();
    void this.router.navigate(['/settings']);
  }
}
