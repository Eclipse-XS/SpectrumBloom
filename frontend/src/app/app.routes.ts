import { Routes } from '@angular/router';

export const routes: Routes = [
  {
    path: '',
    pathMatch: 'full',
    redirectTo: 'onboarding',
  },
  {
    path: 'onboarding',
    loadComponent: () =>
      import('./features/onboarding/onboarding.component').then((m) => m.OnboardingComponent),
    title: 'Upload Dataset — SpectrumBloom',
  },
  {
    path: 'overview',
    loadComponent: () =>
      import('./features/overview/overview.component').then((m) => m.OverviewComponent),
    title: 'Overview — SpectrumBloom',
  },
  {
    path: 'explorer',
    loadComponent: () =>
      import('./features/explorer/explorer.component').then((m) => m.ExplorerComponent),
    title: 'Data Explorer — SpectrumBloom',
  },
  {
    path: 'eda',
    loadComponent: () =>
      import('./features/eda/automated-eda.component').then((m) => m.AutomatedEdaComponent),
    title: 'Automated EDA — SpectrumBloom',
  },
  {
    path: 'analyst',
    loadComponent: () =>
      import('./features/analyst/ai-analyst.component').then((m) => m.AiAnalystComponent),
    title: 'AI Analyst — SpectrumBloom',
  },
  {
    path: 'ml',
    loadComponent: () =>
      import('./features/ml-lab/ml-lab.component').then((m) => m.MlLabComponent),
    title: 'ML Lab — SpectrumBloom',
  },
  {
    path: 'reports',
    loadComponent: () =>
      import('./features/reports/reports.component').then((m) => m.ReportsComponent),
    title: 'Reports & Export — SpectrumBloom',
  },
  {
    path: 'sessions',
    loadComponent: () =>
      import('./features/sessions/sessions.component').then((m) => m.SessionsComponent),
    title: 'Sessions — SpectrumBloom',
  },
  {
    path: 'settings',
    loadComponent: () =>
      import('./features/settings/settings.component').then((m) => m.SettingsComponent),
    title: 'Settings — SpectrumBloom',
  },
  {
    path: '**',
    redirectTo: 'overview',
  },
];
