import { ChangeDetectionStrategy, Component, computed, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';
import { ColumnProfile, SemanticType } from '../../core/models/api.models';
import { TypeGroupSummary } from '../../core/models/workspace.models';
import { WorkspaceStateService } from '../../core/state/workspace-state.service';
import { KpiCardComponent } from '../../shared/components/kpi-card.component';
import { PageHeaderComponent } from '../../shared/components/page-header.component';
import { SbCardComponent } from '../../shared/components/sb-card.component';
import { StatusBadgeComponent } from '../../shared/components/status-badge.component';
import { SvgIconComponent } from '../../shared/components/svg-icon.component';
import {
  formatBytes,
  formatInteger,
  formatNumber,
  formatTimestamp,
  getFileExtensionBadge,
} from '../../shared/utils/format.utils';

@Component({
  selector: 'sb-overview',
  standalone: true,
  imports: [
    FormsModule,
    PageHeaderComponent,
    KpiCardComponent,
    SbCardComponent,
    StatusBadgeComponent,
    SvgIconComponent,
  ],
  changeDetection: ChangeDetectionStrategy.OnPush,
  styles: [
    `
      .dataset-strip {
        min-height: 66px;
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 10px;
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 12px;
        padding: 10px 17px;
        margin-bottom: 14px;
        box-shadow: var(--shadow);
      }
      .file-token {
        display: flex;
        align-items: center;
        gap: 11px;
        margin-right: 12px;
      }
      .file-token-icon {
        width: 34px;
        height: 34px;
        color: var(--primary);
        background: var(--primary-soft);
        border-radius: 7px;
        display: grid;
        place-items: center;
      }
      .file-token-meta {
        display: flex;
        flex-direction: column;
        gap: 2px;
      }
      .file-token-meta strong {
        font-size: 13px;
      }
      .file-token-meta span {
        font-size: 10px;
        color: var(--muted);
      }
      .dataset-meta {
        display: flex;
        align-items: center;
        margin-left: auto;
        flex-wrap: wrap;
      }
      .dataset-meta span {
        font-size: 11px;
        color: var(--muted);
        padding: 0 14px;
        border-left: 1px solid var(--border);
      }
      .dataset-meta b {
        color: var(--text);
        margin-right: 3px;
      }
      .overview-grid {
        display: grid;
        grid-template-columns: 1.35fr 1fr;
        gap: 14px;
        margin-bottom: 14px;
      }
      .quality-score {
        display: grid;
        grid-template-columns: 145px 1fr;
        align-items: center;
        gap: 16px;
      }
      .donut {
        position: relative;
        width: 118px;
        height: 118px;
        border-radius: 50%;
        display: grid;
        place-items: center;
        margin: 0 auto;
      }
      .donut::before {
        content: '';
        position: absolute;
        width: 86px;
        height: 86px;
        border-radius: 50%;
        background: var(--surface);
      }
      .donut-center {
        position: relative;
        display: flex;
        flex-direction: column;
        align-items: center;
      }
      .donut-center strong {
        font-size: 19px;
      }
      .donut-center span {
        font-size: 10px;
        color: var(--muted);
      }
      .quality-summary > div {
        display: grid;
        grid-template-columns: 1fr auto;
        gap: 6px;
        border-bottom: 1px solid var(--border);
        padding: 9px 0;
        font-size: 11px;
      }
      .quality-summary > div:last-child {
        border: 0;
      }
      .quality-summary span {
        color: var(--muted);
        display: flex;
        align-items: center;
        gap: 7px;
      }
      .dot {
        width: 7px;
        height: 7px;
        border-radius: 2px;
        display: inline-block;
      }
      .dot.good {
        background: var(--primary);
      }
      .dot.miss {
        background: var(--warm);
      }
      .dot.neutral {
        background: var(--border-strong);
      }
      .quality-columns {
        border-top: 1px solid var(--border);
        padding-top: 12px;
        margin-top: 10px;
      }
      .quality-columns-row {
        display: grid;
        grid-template-columns: 140px 52px 1fr;
        align-items: center;
        gap: 10px;
        margin: 7px 0;
        font-size: 11px;
      }
      .quality-columns-row span {
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
      }
      .quality-columns-row b {
        text-align: right;
        font-variant-numeric: tabular-nums;
      }
      .quality-columns-row i {
        height: 5px;
        background: var(--surface-2);
        border-radius: 4px;
        overflow: hidden;
        display: block;
      }
      .quality-columns-row em {
        display: block;
        background: var(--primary);
        height: 100%;
        border-radius: 4px;
      }
      .type-bar {
        height: 11px;
        display: flex;
        border-radius: 4px;
        overflow: hidden;
        margin: 4px 0 16px;
        background: var(--surface-2);
      }
      .t-num {
        background: var(--primary);
      }
      .t-cat {
        background: var(--blue);
      }
      .t-id {
        background: var(--violet);
      }
      .t-bool {
        background: var(--warm);
      }
      .type-list {
        list-style: none;
        margin: 0;
        padding: 0;
      }
      .type-list li {
        display: grid;
        grid-template-columns: 16px 1fr auto 44px;
        align-items: center;
        gap: 6px;
        padding: 7px 0;
        font-size: 11px;
        border-bottom: 1px solid var(--border);
      }
      .type-list li:last-child {
        border: 0;
      }
      .type-list small {
        text-align: right;
        color: var(--muted);
      }
      .search-small {
        height: 32px;
        display: flex;
        align-items: center;
        gap: 6px;
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 0 9px;
      }
      .search-small input {
        border: 0;
        outline: 0;
        background: transparent;
        width: 140px;
        font-size: 11px;
        color: var(--text);
      }
      .table-footer {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-top: 16px;
        font-size: 11px;
        color: var(--muted);
      }
      @media (max-width: 1100px) {
        .overview-grid {
          grid-template-columns: 1fr;
        }
      }
      @media (max-width: 800px) {
        .dataset-meta {
          width: 100%;
          margin: 4px 0 0;
          padding-top: 8px;
          border-top: 1px solid var(--border);
        }
        .dataset-meta span:first-child {
          border-left: 0;
          padding-left: 0;
        }
        .quality-score {
          grid-template-columns: 1fr;
        }
      }
    `,
  ],
  template: `
    <div class="page">
      @if (!state.hasData() || !profile() || !statistics()) {
        <div class="empty-panel">
          <h2>No dataset loaded in the active session</h2>
          <p>Import a CSV or Excel dataset first to compute deterministic metrics and charts.</p>
          <button type="button" class="button primary" (click)="goTo('/onboarding')">
            Go to dataset upload
          </button>
        </div>
      } @else {
        <sb-page-header
          title="Dataset overview"
          description="Deterministic profile of dataset structure, completeness, inferred semantic roles, and preview rows."
        >
          <button type="button" class="button" (click)="goTo('/onboarding')">
            <sb-svg-icon name="upload" [size]="15" />
            Replace dataset
          </button>
          <button type="button" class="button primary" (click)="goTo('/analyst')">
            <sb-svg-icon name="spark" [size]="15" />
            Ask AI Analyst
          </button>
        </sb-page-header>

        <div class="dataset-strip">
          <div class="file-token">
            <div class="file-token-icon">
              <sb-svg-icon name="file" [size]="18" />
            </div>
            <div class="file-token-meta">
              <strong>{{ profile()!.filename }}</strong>
              <span>
                {{ fileExt() }} · {{ formatBytes(datasetMeta()?.size_bytes) }} · Uploaded
                {{ formatTimestamp(datasetMeta()?.uploaded_at) }}
              </span>
            </div>
          </div>
          <sb-status-badge [kind]="profile()!.completeness_pct >= 90 ? 'success' : 'warning'">
            {{ profile()!.completeness_pct >= 90 ? 'Profile complete' : 'Review missingness' }}
          </sb-status-badge>
          <div class="dataset-meta">
            <span><b>{{ formatInteger(profile()!.rows) }}</b> rows</span>
            <span><b>{{ formatInteger(profile()!.columns) }}</b> columns</span>
            <span><b>{{ formatNumber(profile()!.memory_mb, 3) }}</b> MB</span>
          </div>
        </div>

        <div class="kpi-grid">
          <sb-kpi-card
            label="Total rows"
            [value]="formatInteger(profile()!.rows)"
            [note]="formatInteger(profile()!.duplicate_rows) + ' duplicate rows'"
            icon="table"
          />
          <sb-kpi-card
            label="Total columns"
            [value]="formatInteger(profile()!.columns)"
            [note]="statistics()!.numeric_columns.length + ' numeric · ' + statistics()!.categorical_columns.length + ' categorical'"
            icon="columns"
          />
          <sb-kpi-card
            label="Completeness"
            [value]="formatNumber(profile()!.completeness_pct, 1) + '%'"
            [note]="formatInteger(statistics()!.missingness.total_missing_cells) + ' of ' + formatInteger(statistics()!.missingness.total_cells) + ' cells missing'"
            icon="check"
          />
          <sb-kpi-card
            label="Duplicate rows"
            [value]="formatInteger(profile()!.duplicate_rows)"
            [note]="formatNumber(profile()!.duplicate_pct, 2) + '% of records'"
            icon="copy"
          />
          <sb-kpi-card
            label="Memory"
            [value]="formatNumber(profile()!.memory_mb, 3) + ' MB'"
            [note]="'Storage: ' + (datasetMeta()?.storage_backend ?? 'local')"
            icon="database"
          />
        </div>

        <div class="overview-grid">
          <sb-card
            title="Data quality"
            [subtitle]="'Completeness across all ' + profile()!.columns + ' columns'"
          >
            <button
              card-actions
              type="button"
              class="button ghost sm"
              (click)="goTo('/eda')"
            >
              View details
              <sb-svg-icon name="chevron" [size]="14" />
            </button>

            <div class="quality-score">
              <div class="donut" [style.background]="donutGradient()">
                <div class="donut-center">
                  <strong>{{ formatNumber(profile()!.completeness_pct, 1) }}%</strong>
                  <span>complete</span>
                </div>
              </div>
              <div class="quality-summary">
                <div>
                  <span><i class="dot good"></i> Complete cells</span>
                  <b>{{ formatInteger(completeCells()) }}</b>
                </div>
                <div>
                  <span><i class="dot miss"></i> Missing cells</span>
                  <b>{{ formatInteger(statistics()!.missingness.total_missing_cells) }}</b>
                </div>
                <div>
                  <span><i class="dot neutral"></i> Rows affected</span>
                  <b>{{ formatInteger(statistics()!.missingness.rows_with_any_missing) }}</b>
                </div>
              </div>
            </div>

            <div class="quality-columns">
              @for (col of topCompletenessColumns(); track col.name) {
                <div class="quality-columns-row">
                  <span [title]="col.name">{{ col.name }}</span>
                  <b>{{ formatNumber(100 - col.null_pct, 1) }}%</b>
                  <i><em [style.width.%]="100 - col.null_pct"></em></i>
                </div>
              }
            </div>
          </sb-card>

          <sb-card title="Column types" subtitle="Semantic roles inferred deterministically">
            <div class="type-bar" aria-label="Column semantic type distribution">
              @for (group of typeGroups(); track group.label) {
                @if (group.count > 0) {
                  <i [class]="group.colorClass" [style.width.%]="group.pct" [title]="group.label"></i>
                }
              }
            </div>
            <ul class="type-list">
              @for (group of typeGroups(); track group.label) {
                <li>
                  <span [class]="'dot ' + group.colorClass"></span>
                  <span>{{ group.label }}</span>
                  <b>{{ group.count }}</b>
                  <small>{{ formatNumber(group.pct, 0) }}%</small>
                </li>
              }
            </ul>

            @if (statistics()!.missingness.columns_with_missing.length > 0 || profile()!.warnings.length > 0) {
              <div class="notice warning mt-14">
                <sb-svg-icon name="warning" [size]="17" />
                <div class="notice-content">
                  <strong>
                    @if (statistics()!.missingness.columns_with_missing.length > 0) {
                      {{ statistics()!.missingness.columns_with_missing.length }} column(s) contain missing values
                    } @else {
                      {{ profile()!.warnings.length }} data quality alert(s) detected
                    }
                  </strong>
                  <span>
                    {{ profile()!.warnings[0] || 'Review missingness patterns before running ML baselines.' }}
                  </span>
                </div>
              </div>
            } @else {
              <div class="notice info mt-14">
                <sb-svg-icon name="check" [size]="17" />
                <div class="notice-content">
                  <strong>No completeness warnings triggered</strong>
                  <span>All columns meet quality thresholds.</span>
                </div>
              </div>
            }
          </sb-card>
        </div>

        <sb-card
          title="Column profiles"
          subtitle="Physical dtypes, inferred semantic roles, completeness, unique counts, and descriptive stats"
          cardClass="table-card"
        >
          <div card-actions class="search-small">
            <sb-svg-icon name="search" [size]="14" />
            <input
              type="search"
              aria-label="Search column profiles"
              placeholder="Search columns"
              [ngModel]="columnSearch()"
              (ngModelChange)="columnSearch.set($event)"
            />
          </div>

          <div class="data-table-wrap">
            <table class="profile-table">
              <thead>
                <tr>
                  <th>Column</th>
                  <th>Physical type</th>
                  <th>Semantic role</th>
                  <th>Non-null</th>
                  <th>Missing</th>
                  <th>Unique</th>
                  <th>Mean</th>
                  <th>Median</th>
                </tr>
              </thead>
              <tbody>
                @for (col of filteredColumns(); track col.name; let i = $index) {
                  <tr>
                    <td>
                      <span [class]="'dtype dtype-' + (i % 4)">{{ typeAbbrev(col) }}</span>
                      <b>{{ col.name }}</b>
                    </td>
                    <td class="mono">{{ col.dtype }}</td>
                    <td><span class="role">{{ col.semantic_type }}</span></td>
                    <td>{{ formatInteger(col.non_null_count) }}</td>
                    <td>
                      {{ formatInteger(col.null_count) }}
                      <small>({{ formatNumber(col.null_pct, 1) }}%)</small>
                    </td>
                    <td>{{ formatInteger(col.unique_count) }}</td>
                    <td>{{ formatNumber(col.numeric_stats?.mean, 2) }}</td>
                    <td>{{ formatNumber(col.numeric_stats?.median, 2) }}</td>
                  </tr>
                }
              </tbody>
            </table>
          </div>

          <div class="table-footer">
            <span>Showing {{ filteredColumns().length }} of {{ profile()!.columns }} columns</span>
            <button type="button" class="button ghost sm" (click)="goTo('/explorer')">
              Open Data Explorer
              <sb-svg-icon name="chevron" [size]="14" />
            </button>
          </div>
        </sb-card>

        <div class="mt-14">
          <sb-card
            title="Dataset preview"
            subtitle="First 5 rows returned by the backend · formula-safe rendering"
          >
            <button card-actions type="button" class="button sm" (click)="goTo('/explorer')">
              Explore preview rows
            </button>

            @if (preview(); as prev) {
              <div class="data-table-wrap">
                <table class="profile-table">
                  <thead>
                    <tr>
                      <th>#</th>
                      @for (col of prev.columns; track col) {
                        <th>{{ col }}</th>
                      }
                    </tr>
                  </thead>
                  <tbody>
                    @for (row of prev.preview.slice(0, 5); track $index; let rowIdx = $index) {
                      <tr>
                        <td class="mono">{{ rowIdx + 1 }}</td>
                        @for (col of prev.columns; track col) {
                          <td>
                            @if (row[col] === null || row[col] === undefined) {
                              <span class="role">NULL</span>
                            } @else {
                              {{ row[col] }}
                            }
                          </td>
                        }
                      </tr>
                    }
                  </tbody>
                </table>
              </div>
            } @else {
              <p class="empty-panel">Loading preview rows…</p>
            }
          </sb-card>
        </div>
      }
    </div>
  `,
})
export class OverviewComponent {
  readonly state = inject(WorkspaceStateService);
  private readonly router = inject(Router);

  readonly profile = this.state.profile;
  readonly statistics = this.state.statistics;
  readonly datasetMeta = this.state.datasetMeta;
  readonly preview = this.state.preview;

  readonly columnSearch = signal<string>('');

  readonly formatInteger = formatInteger;
  readonly formatNumber = formatNumber;
  readonly formatBytes = formatBytes;
  readonly formatTimestamp = formatTimestamp;

  readonly fileExt = computed(() => getFileExtensionBadge(this.profile()?.filename));

  readonly completeCells = computed(() => {
    const miss = this.statistics()?.missingness;
    if (!miss) return 0;
    return Math.max(0, miss.total_cells - miss.total_missing_cells);
  });

  readonly donutGradient = computed(() => {
    const pct = Math.max(0, Math.min(100, this.profile()?.completeness_pct ?? 100));
    const deg = (pct * 3.6).toFixed(1);
    return `conic-gradient(var(--primary) 0 ${deg}deg, var(--surface-3) ${deg}deg)`;
  });

  readonly topCompletenessColumns = computed(() => {
    const cols = [...(this.profile()?.column_profiles ?? [])];
    cols.sort((a, b) => b.null_pct - a.null_pct);
    return cols.slice(0, 4);
  });

  readonly typeGroups = computed<TypeGroupSummary[]>(() => {
    const cols = this.profile()?.column_profiles ?? [];
    const total = cols.length || 1;
    let numeric = 0;
    let categorical = 0;
    let identifierDate = 0;
    let boolOther = 0;

    for (const c of cols) {
      if (c.semantic_type === 'numeric_continuous' || c.semantic_type === 'numeric_discrete') {
        numeric += 1;
      } else if (c.semantic_type === 'categorical' || c.semantic_type === 'text') {
        categorical += 1;
      } else if (c.semantic_type === 'identifier' || c.semantic_type === 'datetime') {
        identifierDate += 1;
      } else {
        boolOther += 1;
      }
    }

    return [
      { label: 'Numeric', count: numeric, pct: (numeric / total) * 100, colorClass: 't-num' },
      { label: 'Categorical & Text', count: categorical, pct: (categorical / total) * 100, colorClass: 't-cat' },
      {
        label: 'Identifier / Datetime',
        count: identifierDate,
        pct: (identifierDate / total) * 100,
        colorClass: 't-id',
      },
      { label: 'Boolean / Empty', count: boolOther, pct: (boolOther / total) * 100, colorClass: 't-bool' },
    ];
  });

  readonly filteredColumns = computed(() => {
    const term = this.columnSearch().trim().toLowerCase();
    const cols = this.profile()?.column_profiles ?? [];
    if (!term) {
      return cols;
    }
    return cols.filter(
      (c) =>
        c.name.toLowerCase().includes(term) ||
        c.dtype.toLowerCase().includes(term) ||
        c.semantic_type.toLowerCase().includes(term),
    );
  });

  typeAbbrev(col: ColumnProfile): string {
    const map: Record<SemanticType, string> = {
      numeric_continuous: '#',
      numeric_discrete: '#',
      categorical: 'C',
      boolean: 'B',
      datetime: 'D',
      text: 'T',
      identifier: 'ID',
      empty: '∅',
    };
    return map[col.semantic_type] ?? '?';
  }

  goTo(path: string): void {
    void this.router.navigate([path]);
  }
}
