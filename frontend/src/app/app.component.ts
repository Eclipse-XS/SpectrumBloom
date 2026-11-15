import { ChangeDetectionStrategy, Component, OnInit, effect, inject } from '@angular/core';
import { Router } from '@angular/router';
import { ThemeService } from './core/state/theme.service';
import { WorkspaceStateService } from './core/state/workspace-state.service';
import { AppShellComponent } from './layout/app-shell.component';

@Component({
  selector: 'sb-root',
  standalone: true,
  imports: [AppShellComponent],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `<sb-app-shell />`,
})
export class AppComponent implements OnInit {
  private readonly state = inject(WorkspaceStateService);
  private readonly router = inject(Router);
  private readonly _theme = inject(ThemeService);
  private routedInitialDataset = false;

  constructor() {
    effect(() => {
      if (this.state.initializing()) {
        return;
      }
      if (!this.routedInitialDataset && this.state.hasData() && this.router.url === '/onboarding') {
        this.routedInitialDataset = true;
        void this.router.navigate(['/overview']);
      }
    });
  }

  ngOnInit(): void {
    this.state.initialize();
  }
}
