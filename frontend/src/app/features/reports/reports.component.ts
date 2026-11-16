import { ChangeDetectionStrategy, Component, computed, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';
import { FigurePayload } from '../../core/models/api.models';
import { WorkspaceStateService } from '../../core/state/workspace-state.service';
import { PageHeaderComponent } from '../../shared/components/page-header.component';
import { StatusBadgeComponent } from '../../shared/components/status-badge.component';
import { SvgIconComponent } from '../../shared/components/svg-icon.component';
import {
  downloadJson,
  downloadPlotlyHtml,
  downloadPreviewCsv,
} from '../../shared/utils/export.utils';
import { slugifyFilename } from '../../shared/utils/format.utils';

@Component({
  selector: 'sb-reports',
  standalone: true,
  imports: [FormsModule, PageHeaderComponent, StatusBadgeComponent, SvgIconComponent],
  changeDetection: ChangeDetectionStrategy.OnPush,
  styles: [
    `
      .export-overview {
        min-height: 62px;
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 9px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        padding: 10px 17px;
        margin-bottom: 13px;
      }
      .export-overview-left {
        display: flex;
        align-items: center;
        gap: 12px;
      }
      .export-overview-icon {
        width: 34px;
        height: 34px;
        border-radius: 7px;
        background: var(--primary-soft);
        color: var(--primary);
        display: grid;
        place-items: center;
      }
      .export-overview-left strong {
        display: block;
        font-size: 12px;
      }
      .export-overview-left small {
        display: block;
        font-size: 11px;
        color: var(--muted);
      }
      .export-list {
        border: 1px solid var(--border);
        border-radius: 10px;
        background: var(--surface);
        overflow: hidden;
      }
      .export-row {
        min-height: 86px;
        padding: 14px 17px;
        display: grid;
        grid-template-columns: 50px 1fr auto 128px;
        align-items: center;
        gap: 15px;
        border-bottom: 1px solid var(--border);
      }
      .export-row:last-child {
        border: 0;
      }
      .export-row.disabled {
        background: var(--surface-2);
      }
      .format-icon {
        width: 44px;
        height: 50px;
        border: 1px solid var(--border);
        border-radius: 7px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 3px;
        color: var(--primary);
      }
      .format-icon b {
        font-size: 8px;
      }
      .export-row h3 {
        font-size: 13px;
        margin: 0 0 4px;
      }
      .export-row p {
        font-size: 11px;
        color: var(--muted);
        margin: 0;
      }
      .export-row .unavail-note {
        display: block;
        font-size: 10px;
        color: var(--warning);
        margin-top: 4px;
      }
      .figure-picker {
        margin-top: 8px;
        max-width: 340px;
      }
      @media (max-width: 800px) {
        .export-row {
          grid-template-columns: 45px 1fr auto;
        }
        .export-row > button {
          grid-column: 2 / -1;
          width: 100%;
        }
      }
      @media (max-width: 520px) {
        .export-row {
          grid-template-columns: 42px 1fr;
        }
        .export-row > sb-status-badge {
          grid-column: 2;
        }
      }
    `,
  ],
  template: `
    <div class="page">
      <sb-page-header
        title="Reports & Export"
        description="Download genuine analytical artifacts from the active session. Availability updates dynamically as workflows run."
      >
        <button type="button" class="button" (click)="refresh()">
          <sb-svg-icon name="refresh" [size]="15" />
          Refresh availability
        </button>
      </sb-page-header>

      <div class="export-overview">
        <div class="export-overview-left">
          <div class="export-overview-icon">
            <sb-svg-icon name="download" [size]="17" />
          </div>
          <div>
            <strong>{{ availableCount() }} of 5 artifact types available</strong>
            <small>
              @if (state.datasetMeta(); as ds) {
                Generated from {{ ds.filename }} ({{ ds.rows }} rows · session {{ state.activeSessionId()?.slice(0, 8) }})
              } @else {
                No dataset uploaded in the active session
              }
            </small>
          </div>
        </div>
        <sb-status-badge [kind]="state.hasData() ? 'success' : 'neutral'">
          {{ state.hasData() ? 'Session active' : 'No dataset' }}
        </sb-status-badge>
      </div>

      <div class="export-list">
        <!-- 1. EDA JSON -->
        <div class="export-row" [class.disabled]="!state.edaReport()">
          <div class="format-icon"><sb-svg-icon name="file" [size]="16" /><b>JSON</b></div>
          <div>
            <h3>Automated EDA report</h3>
            <p>Full profile, column stats, normality/group tests, outlier summaries, and figure references.</p>
            @if (!state.edaReport()) {
              <small class="unavail-note">Upload a dataset to generate the EDA report.</small>
            }
          </div>
          <sb-status-badge [kind]="state.edaReport() ? 'success' : 'neutral'">
            {{ state.edaReport() ? 'Available' : 'Unavailable' }}
          </sb-status-badge>
          <button
            type="button"
            class="button"
            [disabled]="!state.edaReport()"
            (click)="downloadEda()"
          >
            <sb-svg-icon name="download" [size]="14" />
            Download
          </button>
        </div>

        <!-- 2. Preview CSV -->
        <div class="export-row" [class.disabled]="!state.preview()">
          <div class="format-icon"><sb-svg-icon name="table" [size]="16" /><b>CSV</b></div>
          <div>
            <h3>Dataset preview</h3>
            <p>Formula-safe CSV containing the {{ state.preview()?.preview?.length ?? 0 }} loaded preview rows.</p>
            @if (!state.preview()) {
              <small class="unavail-note">Upload a dataset to enable preview export.</small>
            }
          </div>
          <sb-status-badge [kind]="state.preview() ? 'success' : 'neutral'">
            {{ state.preview() ? 'Available' : 'Unavailable' }}
          </sb-status-badge>
          <button
            type="button"
            class="button"
            [disabled]="!state.preview()"
            (click)="downloadCsv()"
          >
            <sb-svg-icon name="download" [size]="14" />
            Download
          </button>
        </div>

        <!-- 3. Conversation & Traces JSON -->
        <div class="export-row" [class.disabled]="state.messages().length === 0">
          <div class="format-icon"><sb-svg-icon name="code" [size]="16" /><b>JSON</b></div>
          <div>
            <h3>Conversation &amp; tool traces</h3>
            <p>AI Analyst prompts, answers, Python/SQL tool traces, and evaluation checks.</p>
            @if (state.messages().length === 0) {
              <small class="unavail-note">Ask at least one question in AI Analyst to populate conversation traces.</small>
            }
          </div>
          <sb-status-badge [kind]="state.messages().length > 0 ? 'success' : 'neutral'">
            {{ state.messages().length > 0 ? 'Available' : 'Unavailable' }}
          </sb-status-badge>
          <button
            type="button"
            class="button"
            [disabled]="state.messages().length === 0"
            (click)="downloadTraces()"
          >
            <sb-svg-icon name="download" [size]="14" />
            Download
          </button>
        </div>

        <!-- 4. ML Results JSON -->
        <div class="export-row" [class.disabled]="!state.mlResult()">
          <div class="format-icon"><sb-svg-icon name="flask" [size]="16" /><b>JSON</b></div>
          <div>
            <h3>ML baseline experiment</h3>
            <p>Cross-validation scores, held-out test metrics, permutation importance, and caveats.</p>
            @if (!state.mlResult()) {
              <small class="unavail-note">Run an experiment in ML Lab to enable this export.</small>
            }
          </div>
          <sb-status-badge [kind]="state.mlResult() ? 'success' : 'neutral'">
            {{ state.mlResult() ? 'Available' : 'Unavailable' }}
          </sb-status-badge>
          <button
            type="button"
            class="button"
            [disabled]="!state.mlResult()"
            (click)="downloadMl()"
          >
            <sb-svg-icon name="download" [size]="14" />
            Download
          </button>
        </div>

        <!-- 5. Interactive Plotly HTML -->
        <div class="export-row" [class.disabled]="figures().length === 0">
          <div class="format-icon"><sb-svg-icon name="chart" [size]="16" /><b>HTML</b></div>
          <div>
            <h3>Interactive Plotly figure</h3>
            <p>Standalone HTML viewer containing real Plotly traces from EDA, AI Analyst, or ML Lab (loads Plotly from CDN with built-in offline trace/JSON fallback).</p>
            @if (figures().length === 0) {
              <small class="unavail-note">No Plotly figures exist in the active session yet.</small>
            } @else {
              <select
                class="field-control figure-picker"
                aria-label="Choose figure to export"
                [ngModel]="activeFigureId()"
                (ngModelChange)="selectedFigureId.set($event)"
              >
                @for (fig of figures(); track fig.id) {
                  <option [value]="fig.id">{{ fig.title }} ({{ fig.chart_kind }})</option>
                }
              </select>
            }
          </div>
          <sb-status-badge [kind]="figures().length > 0 ? 'success' : 'neutral'">
            {{ figures().length > 0 ? figures().length + ' figures' : 'Unavailable' }}
          </sb-status-badge>
          <button
            type="button"
            class="button"
            [disabled]="!selectedFigure()"
            (click)="downloadFigureHtml()"
          >
            <sb-svg-icon name="download" [size]="14" />
            Download
          </button>
        </div>
      </div>

      <div class="notice info" style="margin-top: 14px">
        <sb-svg-icon name="info" [size]="17" />
        <div class="notice-content">
          <strong>Backend-supported formats only</strong>
          <span>
            PDF reports, multi-sheet Excel workbooks, and ZIP bundles are not exposed by the FastAPI
            backend and are intentionally omitted here.
          </span>
        </div>
      </div>
    </div>
  `,
})
export class ReportsComponent {
  readonly state = inject(WorkspaceStateService);
  private readonly router = inject(Router);

  readonly figures = this.state.allExportableFigures;
  readonly selectedFigureId = signal<string>('');

  readonly activeFigureId = computed(() => {
    const chosen = this.selectedFigureId();
    const list = this.figures();
    if (chosen && list.some((f) => f.id === chosen)) {
      return chosen;
    }
    return list[0]?.id ?? '';
  });

  readonly selectedFigure = computed<FigurePayload | null>(
    () => this.figures().find((f) => f.id === this.activeFigureId()) ?? null,
  );

  readonly availableCount = computed(() => {
    let count = 0;
    if (this.state.edaReport()) count += 1;
    if (this.state.preview()) count += 1;
    if (this.state.messages().length > 0) count += 1;
    if (this.state.mlResult()) count += 1;
    if (this.figures().length > 0) count += 1;
    return count;
  });

  refresh(): void {
    this.state.refreshSessions();
    if (this.state.hasData()) {
      this.state.refreshEda();
    } else {
      void this.router.navigate(['/onboarding']);
    }
  }

  downloadEda(): void {
    const report = this.state.edaReport();
    const sid = this.state.activeSessionId();
    const ds = this.state.datasetMeta();
    if (!report || !sid) return;
    const base = slugifyFilename(report.profile.filename.replace(/\.[^.]+$/, ''), 'dataset');
    downloadJson(`${base}_eda_report.json`, {
      session_id: sid,
      dataset_sha256: ds?.sha256 ?? null,
      ...report,
    });
    this.state.showToast(`${base}_eda_report.json downloaded.`);
  }

  downloadCsv(): void {
    const prev = this.state.preview();
    if (!prev) return;
    const base = slugifyFilename(prev.filename.replace(/\.[^.]+$/, ''), 'dataset');
    downloadPreviewCsv(`${base}_preview.csv`, prev.columns, prev.preview);
    this.state.showToast(`${base}_preview.csv downloaded.`);
  }

  downloadTraces(): void {
    const sid = this.state.activeSessionId();
    const msgs = this.state.messages();
    if (!sid || msgs.length === 0) return;
    downloadJson(`analyst_traces_${sid.slice(0, 8)}.json`, {
      session_id: sid,
      dataset: this.state.datasetMeta(),
      provider: this.state.activeProvider(),
      model: this.state.activeModel(),
      messages: msgs,
    });
    this.state.showToast('Conversation & traces JSON downloaded.');
  }

  downloadMl(): void {
    const res = this.state.mlResult();
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

  downloadFigureHtml(): void {
    const fig = this.selectedFigure();
    if (!fig) return;
    downloadPlotlyHtml(fig, {
      sessionId: this.state.activeSessionId(),
      datasetFilename: this.state.datasetMeta()?.filename ?? null,
    });
    this.state.showToast(`${slugifyFilename(fig.title)}.html downloaded.`);
  }
}
