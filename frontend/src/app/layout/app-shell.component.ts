import { ChangeDetectionStrategy, Component, computed, inject, signal } from '@angular/core';
import { Router, RouterLink, RouterLinkActive, RouterOutlet } from '@angular/router';
import { AuthTokenService } from '../core/auth/auth-token.service';
import { ThemeService } from '../core/state/theme.service';
import { WorkspaceStateService } from '../core/state/workspace-state.service';
import { AuthBannerComponent } from '../shared/components/auth-banner.component';
import { ChartModalComponent } from '../shared/components/chart-modal.component';
import { StatusBadgeComponent } from '../shared/components/status-badge.component';
import { IconName, SvgIconComponent } from '../shared/components/svg-icon.component';
import { getFileExtensionBadge } from '../shared/utils/format.utils';

interface NavEntry {
  route: string;
  label: string;
  icon: IconName;
  showDot?: boolean;
}

@Component({
  selector: 'sb-app-shell',
  standalone: true,
  imports: [
    RouterOutlet,
    RouterLink,
    RouterLinkActive,
    SvgIconComponent,
    StatusBadgeComponent,
    ChartModalComponent,
    AuthBannerComponent,
  ],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div class="app-shell">
      <aside class="sidebar" [class.open]="mobileNavOpen()" aria-label="Workspace sidebar">
        <div class="brand">
          <div class="brand-mark" aria-hidden="true">
            <span></span>
            <span></span>
            <span></span>
          </div>
          <div class="brand-text">
            <strong>SpectrumBloom</strong>
            <small>Data Science Workspace</small>
          </div>
          <button
            type="button"
            class="icon-button brand-close"
            aria-label="Close navigation"
            (click)="mobileNavOpen.set(false)"
          >
            <sb-svg-icon name="close" [size]="17" />
          </button>
        </div>

        <nav aria-label="Workspace views">
          <span class="nav-label">Workspace</span>
          @for (item of workspaceNav; track item.route) {
            <a
              [routerLink]="item.route"
              routerLinkActive="active"
              (click)="mobileNavOpen.set(false)"
            >
              <sb-svg-icon [name]="item.icon" [size]="18" />
              <span>{{ item.label }}</span>
              @if (item.showDot && state.hasData()) {
                <i class="nav-dot" aria-hidden="true"></i>
              }
            </a>
          }
        </nav>

        <nav class="utility" aria-label="Manage">
          <span class="nav-label">Manage</span>
          <a routerLink="/sessions" routerLinkActive="active" (click)="mobileNavOpen.set(false)">
            <sb-svg-icon name="clock" [size]="18" />
            <span>Sessions</span>
          </a>
          <a routerLink="/settings" routerLinkActive="active" (click)="mobileNavOpen.set(false)">
            <sb-svg-icon name="settings" [size]="18" />
            <span>Settings</span>
          </a>
        </nav>

        <div class="sidebar-bottom">
          <div
            class="system-mini"
            [class.offline]="!state.health()"
            [class.warn]="state.health()?.status === 'degraded'"
          >
            <span>
              <i aria-hidden="true"></i>
              {{ state.health() ? 'FastAPI connected' : 'Backend offline' }}
            </span>
            <small>
              @if (state.health(); as h) {
                Sandbox: {{ h.sandbox_backend }} · {{ h.llm.provider }}
              } @else {
                Check FastAPI connection
              }
            </small>
          </div>

          <button type="button" (click)="theme.toggle()">
            <sb-svg-icon [name]="theme.mode() === 'dark' ? 'sun' : 'moon'" [size]="17" />
            <span>{{ theme.mode() === 'dark' ? 'Light' : 'Dark' }} theme</span>
          </button>

          <a
            href="https://github.com/Eclipse-XS/SpectrumBloom#readme"
            target="_blank"
            rel="noopener noreferrer"
          >
            <sb-svg-icon name="external" [size]="17" />
            <span>Documentation</span>
          </a>
        </div>
      </aside>

      <div class="mobile-scrim" (click)="mobileNavOpen.set(false)"></div>

      <section class="workspace">
        <header class="topbar">
          <button
            type="button"
            class="mobile-menu"
            aria-label="Open navigation"
            (click)="mobileNavOpen.set(true)"
          >
            <sb-svg-icon name="menu" [size]="19" />
          </button>

          <div class="mobile-logo">
            <div class="brand-mark" aria-hidden="true">
              <span></span>
              <span></span>
              <span></span>
            </div>
            <b>SpectrumBloom</b>
          </div>

          @if (state.datasetMeta(); as ds) {
            <button
              type="button"
              class="dataset-select"
              title="Replace or switch dataset"
              (click)="goTo('/onboarding')"
            >
              <span class="file-cube">{{ fileExt() }}</span>
              <span class="dataset-select-label">
                <small>Active dataset</small>
                <b>{{ ds.filename }}</b>
              </span>
              <sb-svg-icon class="chevron-icon" name="chevron" [size]="14" />
            </button>

            <div class="topbar-right">
              <sb-status-badge [kind]="backendBadgeKind()">
                {{ backendBadgeText() }}
              </sb-status-badge>
              <span class="divider"></span>
              <div class="session-context">
                <small>Session</small>
                <b>{{ state.activeSessionId()?.slice(0, 8) || '—' }}</b>
              </div>
              <div class="model-context">
                <span class="model-pill">{{ providerAbbrev() }}</span>
                <div class="model-context-meta">
                  <small>Model</small>
                  <b>{{ state.activeModel() }}</b>
                </div>
              </div>
              <button
                type="button"
                class="icon-button"
                aria-label="Open settings"
                (click)="goTo('/settings')"
              >
                <sb-svg-icon name="settings" [size]="17" />
              </button>
            </div>
          } @else {
            <span class="empty-context">Local-first exploratory workspace</span>
            <div class="topbar-right">
              <sb-status-badge [kind]="backendBadgeKind()">
                {{ backendBadgeText() }}
              </sb-status-badge>
              <button
                type="button"
                class="icon-button"
                aria-label="Open settings"
                (click)="goTo('/settings')"
              >
                <sb-svg-icon name="settings" [size]="17" />
              </button>
            </div>
          }
        </header>

        <main class="main-scroll">
          <router-outlet />
        </main>
      </section>

      <sb-chart-modal />
      <sb-auth-banner />

      @if (state.toast(); as toast) {
        <div [class]="'toast ' + (toast.kind === 'error' ? 'error' : '')" role="status">
          <sb-svg-icon [name]="toast.kind === 'error' ? 'warning' : 'check'" [size]="16" />
          <span>{{ toast.text }}</span>
        </div>
      }
    </div>
  `,
})
export class AppShellComponent {
  readonly state = inject(WorkspaceStateService);
  readonly theme = inject(ThemeService);
  readonly auth = inject(AuthTokenService);
  private readonly router = inject(Router);

  readonly mobileNavOpen = signal<boolean>(false);

  readonly workspaceNav: NavEntry[] = [
    { route: '/overview', label: 'Overview', icon: 'grid' },
    { route: '/explorer', label: 'Data Explorer', icon: 'table' },
    { route: '/eda', label: 'Automated EDA', icon: 'chart', showDot: true },
    { route: '/analyst', label: 'AI Analyst', icon: 'spark' },
    { route: '/ml', label: 'ML Lab', icon: 'flask' },
    { route: '/reports', label: 'Reports & Export', icon: 'download' },
  ];

  readonly fileExt = computed(() => getFileExtensionBadge(this.state.datasetMeta()?.filename));
  readonly providerAbbrev = computed(() =>
    this.state.activeProvider().slice(0, 2).toUpperCase(),
  );

  readonly backendBadgeKind = computed(() => {
    if (this.auth.authChallenge()) return 'warning' as const;
    if (!this.state.health()) return 'error' as const;
    return this.state.health()!.status === 'ok' ? ('success' as const) : ('warning' as const);
  });

  readonly backendBadgeText = computed(() => {
    if (this.auth.authChallenge()) return 'Auth required';
    if (!this.state.health()) return 'API offline';
    return 'API connected';
  });

  goTo(path: string): void {
    this.mobileNavOpen.set(false);
    void this.router.navigate([path]);
  }
}
