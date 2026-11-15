import { ChangeDetectionStrategy, Component, inject } from '@angular/core';
import { Router } from '@angular/router';
import { WorkspaceStateService } from '../../core/state/workspace-state.service';
import { PageHeaderComponent } from '../../shared/components/page-header.component';
import { SbCardComponent } from '../../shared/components/sb-card.component';
import { StatusBadgeComponent } from '../../shared/components/status-badge.component';
import { SvgIconComponent } from '../../shared/components/svg-icon.component';
import { formatInteger, formatTimestamp } from '../../shared/utils/format.utils';

@Component({
  selector: 'sb-sessions',
  standalone: true,
  imports: [PageHeaderComponent, SbCardComponent, StatusBadgeComponent, SvgIconComponent],
  changeDetection: ChangeDetectionStrategy.OnPush,
  styles: [
    `
      .session-list {
        display: flex;
        flex-direction: column;
        gap: 10px;
      }
      .session-item {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 15px 18px;
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        gap: 14px;
        box-shadow: var(--shadow);
      }
      .session-item.active {
        border-color: var(--primary);
        background: color-mix(in srgb, var(--primary-soft) 45%, var(--surface));
      }
      .session-main {
        display: flex;
        align-items: center;
        gap: 12px;
        min-width: 0;
      }
      .session-avatar {
        width: 36px;
        height: 36px;
        border-radius: 8px;
        background: var(--primary-soft);
        color: var(--primary);
        display: grid;
        place-items: center;
        font-size: 11px;
        font-weight: 700;
        flex-shrink: 0;
      }
      .session-meta strong {
        display: block;
        font-size: 13px;
      }
      .session-meta small {
        display: block;
        font-size: 11px;
        color: var(--muted);
        margin-top: 2px;
      }
      .session-actions {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 8px;
      }
    `,
  ],
  template: `
    <div class="page">
      <sb-page-header
        eyebrow="Workspace state"
        title="Sessions"
        description="Manage backend analysis sessions stored in the configured session repository (in-memory or Redis)."
      >
        <button type="button" class="button" (click)="state.refreshSessions()">
          <sb-svg-icon name="refresh" [size]="15" />
          Refresh
        </button>
        <button type="button" class="button primary" (click)="createSession()">
          <sb-svg-icon name="plus" [size]="15" />
          New session
        </button>
      </sb-page-header>

      @if (state.sessions().length === 0) {
        <sb-card title="No active backend sessions" subtitle="Create a session to import a dataset and run analyses">
          <button type="button" class="button primary" (click)="createSession()">
            Create session
          </button>
        </sb-card>
      } @else {
        <div class="session-list">
          @for (sess of state.sessions(); track sess.session_id) {
            <article class="session-item" [class.active]="sess.session_id === state.activeSessionId()">
              <div class="session-main">
                <div class="session-avatar">SB</div>
                <div class="session-meta">
                  <strong>
                    {{ sess.dataset?.filename || 'Empty workspace session' }}
                    <span class="mono" style="font-size: 11px; color: var(--muted); font-weight: 400">
                      ({{ sess.session_id.slice(0, 8) }})
                    </span>
                  </strong>
                  <small>
                    @if (sess.dataset) {
                      {{ formatInteger(sess.dataset.rows) }} × {{ formatInteger(sess.dataset.columns) }} ·
                    }
                    {{ sess.provider }} / {{ sess.model }} · {{ sess.message_count }} msgs · Updated
                    {{ formatTimestamp(sess.updated_at) }}
                  </small>
                </div>
              </div>

              <div class="session-actions">
                @if (sess.session_id === state.activeSessionId()) {
                  <sb-status-badge kind="success">Active</sb-status-badge>
                  <button
                    type="button"
                    class="button sm"
                    [disabled]="sess.message_count === 0"
                    (click)="clearActive()"
                  >
                    Clear chat
                  </button>
                } @else {
                  <button type="button" class="button sm primary" (click)="switchTo(sess.session_id, sess.has_data)">
                    Switch to session
                  </button>
                }
                <button
                  type="button"
                  class="button sm danger"
                  (click)="confirmDelete(sess.session_id)"
                >
                  <sb-svg-icon name="trash" [size]="14" />
                  Delete
                </button>
              </div>
            </article>
          }
        </div>
      }
    </div>
  `,
})
export class SessionsComponent {
  readonly state = inject(WorkspaceStateService);
  private readonly router = inject(Router);

  readonly formatInteger = formatInteger;
  readonly formatTimestamp = formatTimestamp;

  createSession(): void {
    this.state.createNewSession({}, () => {
      void this.router.navigate(['/onboarding']);
    });
  }

  switchTo(sessionId: string, hasData: boolean): void {
    this.state.selectSession(sessionId, () => {
      void this.router.navigate([hasData ? '/overview' : '/onboarding']);
    });
  }

  clearActive(): void {
    if (typeof window !== 'undefined' && window.confirm('Clear chat history in the active session?')) {
      this.state.clearMessages();
    }
  }

  confirmDelete(sessionId: string): void {
    if (typeof window !== 'undefined' && window.confirm(`Delete session ${sessionId.slice(0, 8)} and its uploaded dataset?`)) {
      this.state.deleteSession(sessionId);
    }
  }
}
