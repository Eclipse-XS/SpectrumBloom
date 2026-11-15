export type ThemeMode = 'light' | 'dark';

export type StatusKind = 'success' | 'warning' | 'error' | 'neutral';

export type EdaTabId = 'quality' | 'distributions' | 'relationships' | 'outliers' | 'tests';

export interface ToastMessage {
  id: number;
  text: string;
  kind: 'success' | 'error';
}

export interface ExpandedChartState {
  title: string;
  subtitle?: string;
  figureJson: string;
  filename: string;
}

export interface TypeGroupSummary {
  label: string;
  count: number;
  pct: number;
  colorClass: string;
}
