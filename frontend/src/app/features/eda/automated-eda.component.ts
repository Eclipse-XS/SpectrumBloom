import { ChangeDetectionStrategy, Component, computed, inject, signal } from '@angular/core';
import { Router } from '@angular/router';
import { FigurePayload } from '../../core/models/api.models';
import { EdaTabId } from '../../core/models/workspace.models';
import { WorkspaceStateService } from '../../core/state/workspace-state.service';
import { PageHeaderComponent } from '../../shared/components/page-header.component';
import { PlotlyChartComponent } from '../../shared/components/plotly-chart.component';
import { SbCardComponent } from '../../shared/components/sb-card.component';
import { SvgIconComponent } from '../../shared/components/svg-icon.component';
import { downloadJson, downloadPlotlyHtml } from '../../shared/utils/export.utils';
import { formatInteger, formatNumber, slugifyFilename } from '../../shared/utils/format.utils';

@Component({
  selector: 'sb-automated-eda',
  standalone: true,
  imports: [PageHeaderComponent, SbCardComponent, PlotlyChartComponent, SvgIconComponent],
  changeDetection: ChangeDetectionStrategy.OnPush,
  styles: [
    `
      .eda-summary {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        border: 1px solid var(--border);
        background: var(--surface);
        border-radius: 10px;
        margin-bottom: 14px;
      }
      .eda-summary > div {
        padding: 14px 18px;
        border-right: 1px solid var(--border);
        display: grid;
        grid-template-columns: 1fr auto;
        gap: 4px;
      }
      .eda-summary > div:last-child {
        border: 0;
      }
      .eda-summary span {
        font-size: 11px;
        color: var(--muted);
      }
      .eda-summary strong {
        grid-row: 1 / 3;
        grid-column: 2;
        font-size: 19px;
      }
      .eda-summary small {
        font-size: 10px;
        color: var(--faint);
      }
      .method-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
      }
      .method-grid > div {
        background: var(--surface-2);
        border-radius: 8px;
        padding: 15px;
        display: grid;
        grid-template-columns: 1fr auto;
        gap: 4px;
      }
      .method-grid span {
        font-size: 11px;
        color: var(--muted);
      }
      .method-grid b {
        font-size: 20px;
        grid-row: 1 / 3;
        grid-column: 2;
      }
      .method-grid small {
        font-size: 10px;
        color: var(--faint);
      }
      .rel-row {
        display: grid;
        grid-template-columns: 28px minmax(140px, 240px) 1fr 64px;
        gap: 12px;
        align-items: center;
        padding: 10px 0;
        border-bottom: 1px solid var(--border);
        font-size: 12px;
      }
      .rel-row:last-child {
        border: 0;
      }
      .rel-row b {
        width: 22px;
        height: 22px;
        display: grid;
        place-items: center;
        background: var(--surface-2);
        border-radius: 5px;
        font-size: 11px;
      }
      .rel-row i {
        height: 8px;
        background: var(--surface-2);
        border-radius: 4px;
        overflow: hidden;
        display: block;
      }
      .rel-row em {
        display: block;
        height: 100%;
        background: var(--primary);
        border-radius: 4px;
      }
      .rel-row strong {
        text-align: right;
        font-variant-numeric: tabular-nums;
      }
      .findings-row {
        display: grid;
        grid-template-columns: 36px 1fr;
        gap: 12px;
        padding: 11px 0;
        border-bottom: 1px solid var(--border);
        align-items: center;
      }
      .findings-row:last-child {
        border: 0;
      }
      .finding-icon {
        width: 34px;
        height: 34px;
        border-radius: 7px;
        display: grid;
        place-items: center;
        color: var(--warning);
        background: color-mix(in srgb, var(--warm) 13%, var(--surface));
      }
      .stat-note {
        font-size: 11px;
        color: var(--muted);
        display: flex;
        align-items: center;
        gap: 7px;
        padding-top: 12px;
      }
      .badge-ok {
        color: var(--success);
        font-weight: 600;
      }
      .badge-warn {
        color: var(--warning);
        font-weight: 600;
      }
      @media (max-width: 800px) {
        .eda-summary {
          grid-template-columns: 1fr 1fr;
        }
        .eda-summary > div:nth-child(2) {
          border-right: 0;
        }
        .eda-summary > div:nth-child(-n + 2) {
          border-bottom: 1px solid var(--border);
        }
      }
      @media (max-width: 520px) {
        .eda-summary,
        .method-grid {
          grid-template-columns: 1fr;
        }
        .eda-summary > div {
          border-right: 0;
          border-bottom: 1px solid var(--border);
        }
        .rel-row {
          grid-template-columns: 24px 1fr 54px;
        }
        .rel-row i {
          grid-column: 2 / -1;
        }
      }
    `,
  ],
  template: `
    <div class="page">
      @if (!state.hasData() || !profile() || !statistics()) {
        <div class="empty-panel">
          <h2>No dataset profiled yet</h2>
          <p>Import a dataset to inspect deterministic statistical diagnostics and Plotly figures.</p>
          <button type="button" class="button primary" (click)="goToOnboarding()">
            Upload dataset
          </button>
        </div>
      } @else {
        <sb-page-header
          title="Automated EDA"
          description="Deterministic missingness, distributions, relationships, outlier detections, and hypothesis tests backed by real Plotly figures."
        >
          <button
            type="button"
            class="button"
            [disabled]="state.loadingDataset()"
            (click)="state.refreshEda()"
          >
            <sb-svg-icon name="refresh" [size]="15" />
            Refresh analysis
          </button>
          <button type="button" class="button primary" (click)="exportEdaReport()">
            <sb-svg-icon name="download" [size]="15" />
            Export EDA JSON
          </button>
        </sb-page-header>

        <div class="section-tabs" role="tablist" aria-label="Automated EDA sections">
          @for (tab of tabs; track tab.id; let idx = $index) {
            <button
              type="button"
              role="tab"
              [attr.aria-selected]="activeTab() === tab.id"
              (click)="activeTab.set(tab.id)"
            >
              <span>0{{ idx + 1 }}</span>
              {{ tab.label }}
              @if (tab.id === 'quality' && profile()!.warnings.length > 0) {
                <i>{{ profile()!.warnings.length }}</i>
              }
            </button>
          }
        </div>

        @switch (activeTab()) {
          @case ('quality') {
            <div class="eda-summary">
              <div>
                <span>Missing cells</span>
                <strong>{{ formatInteger(statistics()!.missingness.total_missing_cells) }}</strong>
                <small>{{ formatNumber(statistics()!.missingness.overall_missing_pct, 2) }}% of cells</small>
              </div>
              <div>
                <span>Rows with missing values</span>
                <strong>{{ formatInteger(statistics()!.missingness.rows_with_any_missing) }}</strong>
                <small>{{ formatNumber(statistics()!.missingness.rows_with_any_missing_pct, 2) }}% of rows</small>
              </div>
              <div>
                <span>Duplicate rows</span>
                <strong>{{ formatInteger(profile()!.duplicate_rows) }}</strong>
                <small>{{ formatNumber(profile()!.duplicate_pct, 2) }}% of dataset</small>
              </div>
              <div>
                <span>Quality warnings</span>
                <strong>{{ profile()!.warnings.length }}</strong>
                <small>{{ profile()!.warnings.length ? 'Review details below' : 'None triggered' }}</small>
              </div>
            </div>

            <div class="chart-grid">
              @if (missingnessFigure(); as missFig) {
                <sb-card
                  [title]="missFig.title"
                  subtitle="Interactive Plotly chart from backend EDA engine"
                  cardClass="wide"
                >
                  <button
                    card-actions
                    type="button"
                    class="icon-button"
                    aria-label="Download HTML"
                    (click)="downloadHtml(missFig)"
                  >
                    <sb-svg-icon name="download" [size]="15" />
                  </button>
                  <button
                    card-actions
                    type="button"
                    class="icon-button"
                    aria-label="Expand chart"
                    (click)="expandFigure(missFig, 'Missingness diagnostics')"
                  >
                    <sb-svg-icon name="expand" [size]="15" />
                  </button>
                  <sb-plotly-chart [figureJson]="missFig.figure_json" [title]="missFig.title" [height]="320" />
                </sb-card>
              } @else {
                <sb-card
                  title="Completeness check"
                  subtitle="No columns contain missing cells"
                  cardClass="wide"
                >
                  <div class="notice info">
                    <sb-svg-icon name="check" [size]="17" />
                    <div class="notice-content">
                      <strong>100% non-null values</strong>
                      <span>Zero missing cells across all {{ profile()!.columns }} columns.</span>
                    </div>
                  </div>
                </sb-card>
              }

              <sb-card
                title="Quality findings"
                subtitle="Deterministic warnings raised by the profiler"
                cardClass="wide"
              >
                @if (profile()!.warnings.length === 0) {
                  <p>No structural or missingness warnings detected.</p>
                } @else {
                  @for (warn of profile()!.warnings; track warn) {
                    <div class="findings-row">
                      <div class="finding-icon">
                        <sb-svg-icon name="warning" [size]="16" />
                      </div>
                      <div>
                        <strong>Profiler finding</strong>
                        <p style="margin: 2px 0 0; font-size: 12px; color: var(--muted)">{{ warn }}</p>
                      </div>
                    </div>
                  }
                }
              </sb-card>
            </div>
          }

          @case ('distributions') {
            @if (distributionFigures().length === 0) {
              <div class="empty-panel">
                No numeric or categorical distribution figures were produced for this schema.
              </div>
            } @else {
              <div class="chart-grid">
                @for (fig of distributionFigures(); track fig.id) {
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
                      (click)="expandFigure(fig, 'Automated distribution')"
                    >
                      <sb-svg-icon name="expand" [size]="15" />
                    </button>
                    <sb-plotly-chart [figureJson]="fig.figure_json" [title]="fig.title" [height]="320" />
                  </sb-card>
                }
              </div>
            }
          }

          @case ('relationships') {
            <div class="chart-grid">
              @if (correlationFigure(); as corrFig) {
                <sb-card
                  [title]="corrFig.title"
                  subtitle="Interactive Pearson correlation matrix (−1 to +1)"
                  cardClass="wide"
                >
                  <button
                    card-actions
                    type="button"
                    class="icon-button"
                    aria-label="Download HTML"
                    (click)="downloadHtml(corrFig)"
                  >
                    <sb-svg-icon name="download" [size]="15" />
                  </button>
                  <button
                    card-actions
                    type="button"
                    class="icon-button"
                    aria-label="Expand chart"
                    (click)="expandFigure(corrFig, 'Numeric correlation matrix')"
                  >
                    <sb-svg-icon name="expand" [size]="15" />
                  </button>
                  <sb-plotly-chart [figureJson]="corrFig.figure_json" [title]="corrFig.title" [height]="400" />
                </sb-card>
              }

              <sb-card
                title="Top numeric correlations"
                subtitle="Ranked by magnitude across numeric pairs"
              >
                @if (statistics()!.top_correlations.length === 0) {
                  <p>At least 2 numeric columns with variance are required for correlation analysis.</p>
                } @else {
                  @for (pair of statistics()!.top_correlations; track pair.column_a + ':' + pair.column_b; let i = $index) {
                    <div class="rel-row">
                      <b>{{ i + 1 }}</b>
                      <span>{{ pair.column_a }} ↔ {{ pair.column_b }}</span>
                      <i>
                        <em [style.width.%]="(absVal(pair.pearson_r ?? pair.spearman_rho)) * 100"></em>
                      </i>
                      <strong>r={{ formatNumber(pair.pearson_r, 3) }}</strong>
                    </div>
                  }
                }
              </sb-card>

              <sb-card
                title="Categorical associations"
                subtitle="Cramér's V effect sizes (≥ 0.15)"
              >
                @if (statistics()!.categorical_associations.length === 0) {
                  <p>No categorical pairs exceeded V ≥ 0.15.</p>
                } @else {
                  @for (assoc of statistics()!.categorical_associations; track assoc.column_a + ':' + assoc.column_b; let i = $index) {
                    <div class="rel-row">
                      <b>{{ i + 1 }}</b>
                      <span>{{ assoc.column_a }} ↔ {{ assoc.column_b }}</span>
                      <i><em [style.width.%]="assoc.cramers_v * 100"></em></i>
                      <strong>V={{ formatNumber(assoc.cramers_v, 3) }}</strong>
                    </div>
                  }
                }
              </sb-card>
            </div>
          }

          @case ('outliers') {
            <div class="chart-grid">
              <sb-card
                title="Detection method totals"
                subtitle="Univariate and multivariate flags across numeric columns"
              >
                <div class="method-grid">
                  <div>
                    <span>IQR flags</span>
                    <b>{{ outlierTotals().iqr }}</b>
                    <small>Outside Q1−1.5×IQR .. Q3+1.5×IQR</small>
                  </div>
                  <div>
                    <span>Z-score flags</span>
                    <b>{{ outlierTotals().zscore }}</b>
                    <small>|z| &gt; 3.0</small>
                  </div>
                  <div>
                    <span>Robust MAD flags</span>
                    <b>{{ outlierTotals().mad }}</b>
                    <small>Modified Z &gt; 3.5</small>
                  </div>
                  <div>
                    <span>Isolation Forest</span>
                    <b>{{ statistics()!.multivariate_anomaly_count }}</b>
                    <small>Multivariate anomalies</small>
                  </div>
                </div>
              </sb-card>

              @if (outliersFigure(); as outFig) {
                <sb-card [title]="outFig.title" subtitle="Interactive Plotly box plots">
                  <button
                    card-actions
                    type="button"
                    class="icon-button"
                    aria-label="Download HTML"
                    (click)="downloadHtml(outFig)"
                  >
                    <sb-svg-icon name="download" [size]="15" />
                  </button>
                  <button
                    card-actions
                    type="button"
                    class="icon-button"
                    aria-label="Expand chart"
                    (click)="expandFigure(outFig, 'Outlier box plots')"
                  >
                    <sb-svg-icon name="expand" [size]="15" />
                  </button>
                  <sb-plotly-chart [figureJson]="outFig.figure_json" [title]="outFig.title" [height]="320" />
                </sb-card>
              }

              <sb-card
                title="Per-column outlier bounds"
                subtitle="Exact IQR bounds and flagged observation counts"
                cardClass="wide"
              >
                <div class="data-table-wrap">
                  <table class="profile-table">
                    <thead>
                      <tr>
                        <th>Column</th>
                        <th>IQR lower</th>
                        <th>IQR upper</th>
                        <th>IQR outliers</th>
                        <th>Z-score outliers</th>
                        <th>MAD outliers</th>
                      </tr>
                    </thead>
                    <tbody>
                      @for (item of statistics()!.outliers; track item.column) {
                        <tr>
                          <td><b>{{ item.column }}</b></td>
                          <td>{{ formatNumber(item.iqr_lower, 3) }}</td>
                          <td>{{ formatNumber(item.iqr_upper, 3) }}</td>
                          <td>{{ formatInteger(item.iqr_outlier_count) }} ({{ formatNumber(item.iqr_outlier_pct, 1) }}%)</td>
                          <td>{{ formatInteger(item.zscore_outlier_count) }}</td>
                          <td>{{ formatInteger(item.mad_outlier_count) }}</td>
                        </tr>
                      }
                    </tbody>
                  </table>
                </div>
              </sb-card>
            </div>
          }

          @case ('tests') {
            <div class="chart-grid">
              <sb-card
                title="Normality diagnostics"
                subtitle="Shapiro-Wilk (N ≤ 5,000) / D'Agostino K² · α = 0.05"
                cardClass="wide"
              >
                @if (statistics()!.normality_checks.length === 0) {
                  <p>No eligible numeric columns with sample size ≥ 4 for normality testing.</p>
                } @else {
                  <div class="data-table-wrap">
                    <table class="profile-table">
                      <thead>
                        <tr>
                          <th>Column</th>
                          <th>Test</th>
                          <th>Sample</th>
                          <th>Statistic</th>
                          <th>p-value</th>
                          <th>Skewness</th>
                          <th>Assessment (α=0.05)</th>
                        </tr>
                      </thead>
                      <tbody>
                        @for (chk of statistics()!.normality_checks; track chk.column) {
                          <tr>
                            <td><b>{{ chk.column }}</b></td>
                            <td class="mono">{{ chk.test_name }}</td>
                            <td>{{ formatInteger(chk.sample_size) }}</td>
                            <td>{{ formatNumber(chk.statistic, 4) }}</td>
                            <td>
                              {{ chk.p_value !== null && chk.p_value < 0.001 ? '< 0.001' : formatNumber(chk.p_value, 4) }}
                            </td>
                            <td>{{ formatNumber(chk.skewness, 3) }}</td>
                            <td>
                              @if (chk.is_normal_alpha_05 === true) {
                                <span class="badge-ok">Consistent with normality</span>
                              } @else if (chk.is_normal_alpha_05 === false) {
                                <span class="badge-warn">Evidence against normality</span>
                              } @else {
                                <span>Inconclusive</span>
                              }
                            </td>
                          </tr>
                        }
                      </tbody>
                    </table>
                  </div>
                }
                <div class="stat-note">
                  <sb-svg-icon name="info" [size]="15" />
                  Statistical significance does not establish practical importance. Evaluate effect sizes and sample size together.
                </div>
              </sb-card>

              <sb-card
                title="Group comparison tests"
                subtitle="Kruskal-Wallis H tests across categorical × numeric pairs (p < 0.05)"
                cardClass="wide"
              >
                @if (statistics()!.group_tests.length === 0) {
                  <p>No statistically significant group differences (p &lt; 0.05) detected across tested pairs.</p>
                } @else {
                  <div class="data-table-wrap">
                    <table class="profile-table">
                      <thead>
                        <tr>
                          <th>Grouping column</th>
                          <th>Numeric metric</th>
                          <th>Groups</th>
                          <th>Test</th>
                          <th>H statistic</th>
                          <th>p-value</th>
                        </tr>
                      </thead>
                      <tbody>
                        @for (gt of statistics()!.group_tests; track gt.categorical_column + ':' + gt.numeric_column) {
                          <tr>
                            <td><b>{{ gt.categorical_column }}</b></td>
                            <td>{{ gt.numeric_column }}</td>
                            <td>{{ gt.group_count }}</td>
                            <td class="mono">{{ gt.test }}</td>
                            <td>{{ formatNumber(gt.statistic, 4) }}</td>
                            <td class="badge-ok">
                              {{ gt.p_value < 0.001 ? '< 0.001' : formatNumber(gt.p_value, 4) }}
                            </td>
                          </tr>
                        }
                      </tbody>
                    </table>
                  </div>
                }
              </sb-card>
            </div>
          }
        }
      }
    </div>
  `,
})
export class AutomatedEdaComponent {
  readonly state = inject(WorkspaceStateService);
  private readonly router = inject(Router);

  readonly profile = this.state.profile;
  readonly statistics = this.state.statistics;
  readonly figures = this.state.edaFigures;

  readonly activeTab = signal<EdaTabId>('quality');
  readonly tabs: { id: EdaTabId; label: string }[] = [
    { id: 'quality', label: 'Quality' },
    { id: 'distributions', label: 'Distributions' },
    { id: 'relationships', label: 'Relationships' },
    { id: 'outliers', label: 'Outliers' },
    { id: 'tests', label: 'Statistical tests' },
  ];

  readonly formatInteger = formatInteger;
  readonly formatNumber = formatNumber;

  readonly missingnessFigure = computed(() =>
    this.figures().find((f) => f.chart_kind === 'missingness') ?? null,
  );

  readonly correlationFigure = computed(() =>
    this.figures().find((f) => f.chart_kind === 'correlation_heatmap') ?? null,
  );

  readonly outliersFigure = computed(() =>
    this.figures().find((f) => f.chart_kind === 'outliers') ?? null,
  );

  readonly distributionFigures = computed(() =>
    this.figures().filter(
      (f) => f.chart_kind === 'distribution' || f.chart_kind === 'categorical',
    ),
  );

  readonly outlierTotals = computed(() => {
    const list = this.statistics()?.outliers ?? [];
    return {
      iqr: list.reduce((acc, x) => acc + x.iqr_outlier_count, 0),
      zscore: list.reduce((acc, x) => acc + x.zscore_outlier_count, 0),
      mad: list.reduce((acc, x) => acc + x.mad_outlier_count, 0),
    };
  });

  absVal(n: number | null | undefined): number {
    return n === null || n === undefined ? 0 : Math.min(1, Math.abs(n));
  }

  expandFigure(fig: FigurePayload, subtitle: string): void {
    this.state.openExpandedChart({
      title: fig.title,
      subtitle,
      figureJson: fig.figure_json,
      filename: `${slugifyFilename(fig.title)}.html`,
    });
  }

  downloadHtml(fig: FigurePayload): void {
    downloadPlotlyHtml(fig);
  }

  exportEdaReport(): void {
    const report = this.state.edaReport();
    if (!report) return;
    const base = slugifyFilename(report.profile.filename.replace(/\.[^.]+$/, ''), 'dataset');
    downloadJson(`${base}_eda_report.json`, report);
    this.state.showToast(`${base}_eda_report.json downloaded.`);
  }

  goToOnboarding(): void {
    void this.router.navigate(['/onboarding']);
  }
}
