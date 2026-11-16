import { ChangeDetectionStrategy, Component, computed, effect, inject, signal } from '@angular/core';
import { FormControl, FormGroup, ReactiveFormsModule, Validators } from '@angular/forms';
import { Router } from '@angular/router';
import { FigurePayload } from '../../core/models/api.models';
import { WorkspaceStateService } from '../../core/state/workspace-state.service';
import { PageHeaderComponent } from '../../shared/components/page-header.component';
import { PlotlyChartComponent } from '../../shared/components/plotly-chart.component';
import { SbCardComponent } from '../../shared/components/sb-card.component';
import { StatusBadgeComponent } from '../../shared/components/status-badge.component';
import { SvgIconComponent } from '../../shared/components/svg-icon.component';
import { downloadJson, downloadPlotlyHtml } from '../../shared/utils/export.utils';
import { formatInteger, formatNumber, slugifyFilename } from '../../shared/utils/format.utils';

@Component({
  selector: 'sb-ml-lab',
  standalone: true,
  imports: [
    ReactiveFormsModule,
    PageHeaderComponent,
    SbCardComponent,
    StatusBadgeComponent,
    PlotlyChartComponent,
    SvgIconComponent,
  ],
  changeDetection: ChangeDetectionStrategy.OnPush,
  styles: [
    `
      .form-grid {
        display: grid;
        grid-template-columns: 1.3fr 1fr 0.8fr 0.8fr auto;
        gap: 14px;
        align-items: end;
      }
      .form-grid label {
        display: flex;
        flex-direction: column;
        gap: 5px;
      }
      .form-grid label > span {
        font-size: 11px;
        font-weight: 600;
      }
      .form-grid small {
        font-size: 10px;
        color: var(--muted);
      }
      .feature-pills {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 14px;
        padding-top: 12px;
        border-top: 1px solid var(--border);
      }
      .feature-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 11px;
        padding: 4px 10px;
        border: 1px solid var(--border);
        border-radius: 14px;
        background: var(--surface-2);
        cursor: pointer;
      }
      .ml-state-box {
        text-align: center;
        padding: 64px 20px;
      }
      .ml-state-icon {
        width: 52px;
        height: 52px;
        border-radius: 11px;
        background: var(--primary-soft);
        color: var(--primary);
        display: grid;
        place-items: center;
        margin: 0 auto 14px;
      }
      .ml-checks {
        display: flex;
        justify-content: center;
        flex-wrap: wrap;
        gap: 20px;
        color: var(--success);
        font-size: 11px;
        margin-top: 16px;
      }
      .ml-checks span {
        display: inline-flex;
        align-items: center;
        gap: 5px;
      }
      .experiment-summary {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        border: 1px solid var(--border);
        background: var(--surface);
        border-radius: 10px;
        margin: 14px 0;
      }
      .experiment-summary > div {
        padding: 14px 18px;
        border-right: 1px solid var(--border);
        display: flex;
        flex-direction: column;
        gap: 4px;
      }
      .experiment-summary > div:last-child {
        border: 0;
      }
      .experiment-summary span {
        font-size: 10px;
        color: var(--muted);
      }
      .experiment-summary strong {
        font-size: 14px;
      }
      .notes-item {
        display: flex;
        gap: 9px;
        align-items: flex-start;
        padding: 9px 0;
        border-bottom: 1px solid var(--border);
        font-size: 12px;
        color: var(--muted);
      }
      .notes-item:last-child {
        border: 0;
      }
      .best-row {
        background: var(--primary-soft);
      }
      @media (max-width: 1100px) {
        .form-grid {
          grid-template-columns: 1fr 1fr;
        }
      }
      @media (max-width: 800px) {
        .experiment-summary {
          grid-template-columns: 1fr 1fr;
        }
        .experiment-summary > div:nth-child(2) {
          border-right: 0;
        }
        .experiment-summary > div:nth-child(-n + 2) {
          border-bottom: 1px solid var(--border);
        }
      }
      @media (max-width: 520px) {
        .form-grid,
        .experiment-summary {
          grid-template-columns: 1fr;
        }
        .experiment-summary > div {
          border-right: 0;
          border-bottom: 1px solid var(--border);
        }
      }
    `,
  ],
  template: `
    <div class="page">
      @if (!state.hasData() || !profile()) {
        <div class="empty-panel">
          <h2>Upload a dataset to configure an ML experiment</h2>
          <p>Baseline cross-validation experiments require an active session dataset.</p>
          <button type="button" class="button primary" (click)="goToOnboarding()">
            Upload dataset
          </button>
        </div>
      } @else {
        <sb-page-header
          title="ML Lab"
          description="Run leak-safe ColumnTransformer and scikit-learn baseline cross-validation with permutation importance and diagnostic plots."
        >
          <sb-status-badge [kind]="mlResult() ? 'success' : 'neutral'">
            {{ mlResult() ? 'Experiment complete' : 'Ready to evaluate' }}
          </sb-status-badge>
        </sb-page-header>

        <sb-card
          title="Experiment configuration"
          subtitle="Select a target column, optional feature subset, split ratio, and CV folds"
        >
          <form [formGroup]="form" (ngSubmit)="submit()">
            <div class="form-grid">
              <label>
                <span>Target column</span>
                <select class="field-control" formControlName="target_column">
                  @for (col of selectableTargets(); track col.name) {
                    <option [value]="col.name">{{ col.name }} ({{ col.semantic_type }})</option>
                  }
                </select>
                <small>Outcome variable to predict</small>
              </label>

              <label>
                <span>Task type</span>
                <select class="field-control" formControlName="task_type">
                  <option value="auto">Auto — recommended</option>
                  <option value="classification">Classification</option>
                  <option value="regression">Regression</option>
                </select>
                <small>Auto infers from target values</small>
              </label>

              <label>
                <span>CV folds (2–10)</span>
                <select class="field-control" formControlName="cv_folds">
                  @for (fold of foldOptions; track fold) {
                    <option [ngValue]="fold">{{ fold }} folds</option>
                  }
                </select>
                <small>Stratified for classification</small>
              </label>

              <label>
                <span>Held-out test split</span>
                <select class="field-control" formControlName="test_size">
                  <option [ngValue]="0.15">15% test</option>
                  <option [ngValue]="0.2">20% test</option>
                  <option [ngValue]="0.3">30% test</option>
                </select>
                <small>Isolated held-out evaluation</small>
              </label>

              <div>
                <button
                  type="submit"
                  class="button primary"
                  [disabled]="state.mlRunning() || form.invalid"
                >
                  <sb-svg-icon name="flask" [size]="15" />
                  {{ state.mlRunning() ? 'Running experiment…' : 'Run baseline experiment' }}
                </button>
              </div>
            </div>

            <div class="feature-pills">
              <span style="font-size: 11px; color: var(--muted); align-self: center">
                Features ({{ effectiveFeatureCount() }} selected; blank = auto):
              </span>
              @for (col of candidateFeatures(); track col) {
                <label class="feature-pill">
                  <input
                    type="checkbox"
                    [checked]="isFeatureSelected(col)"
                    (change)="toggleFeature(col)"
                  />
                  <span>{{ col }}</span>
                </label>
              }
            </div>
          </form>
        </sb-card>

        @if (state.mlError()) {
          <div class="notice error" style="margin-top: 14px" role="alert">
            <sb-svg-icon name="warning" [size]="17" />
            <div class="notice-content">
              <strong>Experiment could not run</strong>
              <span>{{ state.mlError() }}</span>
            </div>
          </div>
        }

        @if (state.mlRunning()) {
          <div class="ml-state-box">
            <div class="loader-ring" style="margin: 0 auto 14px"></div>
            <h2>Evaluating baseline pipelines</h2>
            <p style="font-size: 12px; color: var(--muted); max-width: 440px; margin: 0 auto 14px">
              Fitting Dummy, Linear, Random Forest, and HistGradientBoosting pipelines in isolated cross-validation folds…
            </p>
            <div class="indeterminate" style="max-width: 340px; margin: 0 auto"><i></i></div>
          </div>
        } @else if (!mlResult()) {
          <div class="ml-state-box">
            <div class="ml-state-icon"><sb-svg-icon name="flask" [size]="24" /></div>
            <h2>Ready to compare baselines</h2>
            <p style="font-size: 12px; color: var(--muted); max-width: 500px; margin: 0 auto">
              Run a baseline experiment to compare models via cross-validation, inspect held-out metrics, and generate permutation feature importances.
            </p>
            <div class="ml-checks">
              <span><sb-svg-icon name="check" [size]="14" /> Leak-safe ColumnTransformer</span>
              <span><sb-svg-icon name="check" [size]="14" /> Held-out test evaluation</span>
              <span><sb-svg-icon name="check" [size]="14" /> Permutation importance</span>
            </div>
          </div>
        } @else {
          <div class="experiment-summary">
            <div>
              <span>Detected task</span>
              <strong style="text-transform: capitalize">{{ mlResult()!.task_type }}</strong>
            </div>
            <div>
              <span>Best baseline</span>
              <strong>{{ mlResult()!.best_model }}</strong>
            </div>
            <div>
              <span>Train / test rows</span>
              <strong>{{ formatInteger(mlResult()!.train_rows) }} / {{ formatInteger(mlResult()!.test_rows) }}</strong>
            </div>
            <div>
              <span>Usable features</span>
              <strong>{{ formatInteger(mlResult()!.feature_count) }}</strong>
            </div>
          </div>

          <sb-card
            title="Model leaderboard"
            [subtitle]="'Target: ' + mlResult()!.target_column + ' · CV primary metric: ' + primaryMetricLabel()"
          >
            <button card-actions type="button" class="button sm" (click)="exportMlJson()">
              <sb-svg-icon name="download" [size]="14" />
              Export JSON
            </button>

            <div class="data-table-wrap">
              <table class="profile-table">
                <thead>
                  <tr>
                    <th>Rank &amp; model</th>
                    <th>CV {{ primaryMetricLabel() }}</th>
                    <th>CV std</th>
                    @for (metricKey of testMetricKeys(); track metricKey) {
                      <th>Test {{ metricKey }}</th>
                    }
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  @for (model of sortedModels(); track model.model_name; let i = $index) {
                    <tr [class.best-row]="model.model_name === mlResult()!.best_model">
                      <td>
                        <b>{{ i + 1 }}. {{ model.model_name }}</b>
                      </td>
                      <td>{{ formatNumber(model.cv_primary_mean, 4) }}</td>
                      <td>± {{ formatNumber(model.cv_primary_std, 4) }}</td>
                      @for (metricKey of testMetricKeys(); track metricKey) {
                        <td>{{ formatNumber(model.test_metrics[metricKey], 4) }}</td>
                      }
                      <td>
                        <sb-status-badge [kind]="model.model_name === mlResult()!.best_model ? 'success' : 'neutral'">
                          {{ model.model_name === mlResult()!.best_model ? 'Selected' : 'Completed' }}
                        </sb-status-badge>
                      </td>
                    </tr>
                  }
                </tbody>
              </table>
            </div>
          </sb-card>

          <div class="chart-grid" style="margin-top: 14px">
            @for (fig of mlResult()!.figures; track fig.id) {
              <sb-card [title]="fig.title" [subtitle]="'Chart kind: ' + fig.chart_kind">
                <button
                  card-actions
                  type="button"
                  class="icon-button"
                  aria-label="Download HTML"
                  (click)="downloadHtml(fig)"
                >
                  <sb-svg-icon name="download" [size]="15" />
                </button>
                <button
                  card-actions
                  type="button"
                  class="icon-button"
                  aria-label="Expand chart"
                  (click)="expandFigure(fig)"
                >
                  <sb-svg-icon name="expand" [size]="15" />
                </button>
                <sb-plotly-chart [figureJson]="fig.figure_json" [title]="fig.title" [height]="320" />
              </sb-card>
            }

            @if (mlResult()!.notes.length > 0) {
              <sb-card title="Experiment notes" subtitle="Important diagnostic context and caveats">
                @for (note of mlResult()!.notes; track note) {
                  <div class="notes-item">
                    <sb-svg-icon name="info" [size]="15" />
                    <span>{{ note }}</span>
                  </div>
                }
              </sb-card>
            }
          </div>
        }
      }
    </div>
  `,
})
export class MlLabComponent {
  readonly state = inject(WorkspaceStateService);
  private readonly router = inject(Router);

  readonly profile = this.state.profile;
  readonly mlResult = this.state.mlResult;
  readonly foldOptions = [2, 3, 5, 7, 10];
  readonly selectedFeatures = signal<Set<string> | null>(null);

  readonly formatInteger = formatInteger;
  readonly formatNumber = formatNumber;

  readonly form = new FormGroup({
    target_column: new FormControl<string>('', { nonNullable: true, validators: [Validators.required] }),
    task_type: new FormControl<'auto' | 'classification' | 'regression'>('auto', { nonNullable: true }),
    cv_folds: new FormControl<number>(5, { nonNullable: true }),
    test_size: new FormControl<number>(0.2, { nonNullable: true }),
  });

  readonly selectableTargets = computed(() =>
    (this.profile()?.column_profiles ?? []).filter((c) => c.semantic_type !== 'empty'),
  );

  readonly candidateFeatures = computed(() => {
    const target = this.form.controls.target_column.value;
    return (this.profile()?.column_profiles ?? [])
      .map((c) => c.name)
      .filter((name) => name !== target);
  });

  readonly effectiveFeatureCount = computed(() => {
    const selected = this.selectedFeatures();
    if (!selected) {
      return this.candidateFeatures().length;
    }
    return selected.size;
  });

  readonly sortedModels = computed(() => {
    const res = this.mlResult();
    if (!res) return [];
    const copy = [...res.models];
    copy.sort((a, b) => {
      if (a.model_name === res.best_model) return -1;
      if (b.model_name === res.best_model) return 1;
      return b.cv_primary_mean - a.cv_primary_mean;
    });
    return copy;
  });

  readonly testMetricKeys = computed(() => {
    const first = this.mlResult()?.models[0];
    return first ? Object.keys(first.test_metrics) : [];
  });

  readonly primaryMetricLabel = computed(
    () => this.mlResult()?.models[0]?.cv_primary_metric_name.toUpperCase() ?? 'SCORE',
  );

  constructor() {
    effect(() => {
      const targets = this.selectableTargets();
      if (targets.length === 0) {
        return;
      }
      const currentTarget = this.form.controls.target_column.value;
      if (!targets.some((t) => t.name === currentTarget)) {
        const preferred =
          targets.find((t) => t.semantic_type !== 'identifier' && t.semantic_type !== 'datetime') ??
          targets[0];
        this.form.controls.target_column.setValue(preferred.name);
        this.selectedFeatures.set(null);
      }
    });
  }

  isFeatureSelected(col: string): boolean {
    const selected = this.selectedFeatures();
    return selected === null ? true : selected.has(col);
  }

  toggleFeature(col: string): void {
    const candidates = this.candidateFeatures();
    const current = this.selectedFeatures() ?? new Set(candidates);
    const next = new Set(current);
    if (next.has(col)) {
      if (next.size > 1) {
        next.delete(col);
      }
    } else {
      next.add(col);
    }
    this.selectedFeatures.set(next);
  }

  submit(): void {
    if (this.form.invalid) return;
    const raw = this.form.getRawValue();
    const sel = this.selectedFeatures();
    const features =
      sel === null
        ? null
        : [...sel].filter((name) => name !== raw.target_column);

    this.state.runMlExperiment({
      target_column: raw.target_column,
      feature_columns: features,
      task_type: raw.task_type,
      cv_folds: raw.cv_folds,
      test_size: raw.test_size,
    });
  }

  expandFigure(fig: FigurePayload): void {
    this.state.openExpandedChart({
      title: fig.title,
      subtitle: 'ML baseline diagnostics',
      figureJson: fig.figure_json,
      filename: `${slugifyFilename(fig.title)}.html`,
    });
  }

  downloadHtml(fig: FigurePayload): void {
    downloadPlotlyHtml(fig, {
      sessionId: this.state.activeSessionId(),
      datasetFilename: this.state.datasetMeta()?.filename ?? null,
    });
  }

  exportMlJson(): void {
    const res = this.mlResult();
    const sid = this.state.activeSessionId();
    const ds = this.state.datasetMeta();
    if (!res || !sid) return;
    const base = slugifyFilename(res.target_column, 'ml_experiment');
    downloadJson(`ml_${base}_results.json`, {
      session_id: sid,
      dataset_sha256: ds?.sha256 ?? null,
      ...res,
    });
    this.state.showToast(`ml_${base}_results.json downloaded.`);
  }

  goToOnboarding(): void {
    void this.router.navigate(['/onboarding']);
  }
}
