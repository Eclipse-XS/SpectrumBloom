import {
  ChangeDetectionStrategy,
  Component,
  ElementRef,
  HostListener,
  computed,
  inject,
  signal,
  viewChild,
} from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';
import { ColumnProfile } from '../../core/models/api.models';
import { WorkspaceStateService } from '../../core/state/workspace-state.service';
import { PageHeaderComponent } from '../../shared/components/page-header.component';
import { SbCardComponent } from '../../shared/components/sb-card.component';
import { SvgIconComponent } from '../../shared/components/svg-icon.component';
import { downloadPreviewCsv } from '../../shared/utils/export.utils';
import { formatInteger, formatNumber, slugifyFilename } from '../../shared/utils/format.utils';

@Component({
  selector: 'sb-explorer',
  standalone: true,
  imports: [FormsModule, PageHeaderComponent, SbCardComponent, SvgIconComponent],
  changeDetection: ChangeDetectionStrategy.OnPush,
  styles: [
    `
      .explorer-toolbar {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 10px;
      }
      .search-field {
        height: 38px;
        width: 310px;
        max-width: 100%;
        background: var(--surface);
        border: 1px solid var(--border-strong);
        border-radius: 7px;
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 0 10px;
      }
      .search-field input {
        min-width: 0;
        flex: 1;
        border: 0;
        background: transparent;
        outline: 0;
        color: var(--text);
        font-size: 12px;
      }
      .search-field kbd {
        font-size: 9px;
        color: var(--faint);
        border: 1px solid var(--border);
        background: var(--surface-2);
        padding: 2px 5px;
        border-radius: 4px;
      }
      .toolbar-spacer {
        flex: 1;
      }
      .toolbar-count {
        font-size: 11px;
        color: var(--muted);
      }
      .columns-popover {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 9px;
        padding: 12px 14px;
        margin-bottom: 10px;
        display: flex;
        flex-wrap: wrap;
        gap: 10px 16px;
        align-items: center;
      }
      .columns-popover label {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 11px;
        cursor: pointer;
      }
      .explorer-table-wrap {
        overflow: auto;
        max-height: 540px;
      }
      .explorer-table {
        border-collapse: collapse;
        min-width: 960px;
        width: 100%;
      }
      .explorer-table th,
      .explorer-table td {
        height: 40px;
        padding: 0 12px;
        border-right: 1px solid var(--border);
        border-bottom: 1px solid var(--border);
        font-size: 12px;
        white-space: nowrap;
        text-align: left;
      }
      .explorer-table th {
        position: sticky;
        top: 0;
        z-index: 2;
        background: var(--surface-2);
        height: 38px;
        color: var(--muted);
        font-size: 11px;
        padding: 0;
      }
      .th-btn {
        width: 100%;
        height: 100%;
        padding: 0 12px;
        border: 0;
        background: transparent;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 6px;
        font-size: 11px;
        font-weight: 650;
        color: var(--muted);
      }
      .th-btn:hover {
        color: var(--primary);
      }
      .row-num {
        width: 44px;
        text-align: center !important;
        color: var(--faint);
        background: var(--surface-2);
      }
      .null-pill {
        color: var(--warning);
        background: color-mix(in srgb, var(--warm) 14%, var(--surface));
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        padding: 2px 6px;
        border-radius: 3px;
      }
      .grid-footer {
        min-height: 46px;
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        padding: 8px 14px;
        gap: 12px;
        color: var(--muted);
        font-size: 11px;
        border-top: 1px solid var(--border);
      }
      .pager {
        display: flex;
        align-items: center;
        gap: 6px;
      }
      .pager b {
        height: 28px;
        min-width: 28px;
        border-radius: 5px;
        display: grid;
        place-items: center;
        background: var(--primary);
        color: #fff;
        font-size: 11px;
        padding: 0 8px;
      }
      .inspector-card {
        margin-top: 14px;
      }
      .inspector-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
        gap: 12px;
      }
      .inspector-item {
        background: var(--surface-2);
        border-radius: 8px;
        padding: 11px 13px;
      }
      .inspector-item small {
        display: block;
        font-size: 10px;
        color: var(--muted);
        margin-bottom: 3px;
      }
      .inspector-item strong {
        font-size: 13px;
      }
      .inspector-select {
        height: 32px;
        border: 1px solid var(--border-strong);
        background: var(--surface);
        color: var(--text);
        border-radius: 6px;
        padding: 0 10px;
        font-size: 12px;
      }
      @media (max-width: 520px) {
        .search-field {
          width: 100%;
        }
      }
    `,
  ],
  template: `
    <div class="page">
      @if (!state.hasData() || !preview()) {
        <div class="empty-panel">
          <h2>No dataset preview available</h2>
          <p>Upload a dataset first to inspect safe tabular preview rows.</p>
          <button type="button" class="button primary" (click)="goToOnboarding()">
            Upload dataset
          </button>
        </div>
      } @else {
        <sb-page-header
          title="Data Explorer"
          description="Inspect up to 100 preview rows returned by the backend with client-side filtering scoped clearly to the loaded preview."
        >
          <button
            type="button"
            class="button"
            (click)="columnDrawerOpen.set(!columnDrawerOpen())"
          >
            <sb-svg-icon name="columns" [size]="15" />
            Columns <b>{{ visibleColumns().length }}/{{ preview()!.columns.length }}</b>
          </button>
          <button type="button" class="button primary" (click)="exportPreview()">
            <sb-svg-icon name="download" [size]="15" />
            Export preview CSV
          </button>
        </sb-page-header>

        <div class="notice info" style="margin-bottom: 12px">
          <sb-svg-icon name="info" [size]="17" />
          <div class="notice-content">
            <strong>Preview scope boundary</strong>
            <span>
              Search, sorting, and NULL filters apply only to the
              {{ preview()!.preview.length }} loaded preview rows—not the full
              {{ formatInteger(preview()!.shape[0]) }}-row dataset. Use Automated EDA or AI Analyst for
              full-dataset calculations.
            </span>
          </div>
        </div>

        @if (columnDrawerOpen()) {
          <div class="columns-popover" aria-label="Visible columns selector">
            @for (col of preview()!.columns; track col) {
              <label>
                <input
                  type="checkbox"
                  [checked]="!hiddenColumns().has(col)"
                  (change)="toggleColumnVisibility(col)"
                />
                <span>{{ col }}</span>
              </label>
            }
            <button type="button" class="button ghost sm" (click)="resetHiddenColumns()">
              Show all
            </button>
          </div>
        }

        <div class="explorer-toolbar">
          <label class="search-field">
            <sb-svg-icon name="search" [size]="15" />
            <input
              #searchInput
              type="search"
              aria-label="Search loaded preview rows"
              placeholder="Search loaded preview rows…"
              [ngModel]="searchText()"
              (ngModelChange)="onSearchChange($event)"
            />
            <kbd>⌘K</kbd>
          </label>
          <button
            type="button"
            class="button"
            [class.primary]="onlyMissingRows()"
            (click)="toggleMissingFilter()"
          >
            <sb-svg-icon name="filter" [size]="15" />
            Rows with NULLs
            @if (onlyMissingRows()) {
              <b>ON</b>
            }
          </button>
          @if (sortColumn()) {
            <button type="button" class="button sm" (click)="clearSort()">
              Clear sort ({{ sortColumn() }})
              <sb-svg-icon name="close" [size]="13" />
            </button>
          }
          <div class="toolbar-spacer"></div>
          <span class="toolbar-count">
            {{ filteredSortedRows().length }} matching of {{ preview()!.preview.length }} preview rows
          </span>
        </div>

        <sb-card>
          <div class="explorer-table-wrap">
            <table class="explorer-table">
              <thead>
                <tr>
                  <th class="row-num">#</th>
                  @for (col of visibleColumns(); track col) {
                    <th>
                      <button
                        type="button"
                        class="th-btn"
                        (click)="onHeaderClick(col)"
                        [title]="'Sort preview by ' + col + ' and inspect column'"
                      >
                        <span>{{ col }}</span>
                        <span class="mono">
                          @if (sortColumn() === col) {
                            {{ sortDir() === 'asc' ? '▲' : '▼' }}
                          }
                        </span>
                      </button>
                    </th>
                  }
                </tr>
              </thead>
              <tbody>
                @for (row of pagedRows(); track $index; let idx = $index) {
                  <tr>
                    <td class="row-num mono">{{ pageStartIndex() + idx + 1 }}</td>
                    @for (col of visibleColumns(); track col) {
                      <td>
                        @if (row[col] === null || row[col] === undefined || row[col] === '') {
                          <span class="null-pill">NULL</span>
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

          <div class="grid-footer">
            <span>
              Showing rows {{ filteredSortedRows().length ? pageStartIndex() + 1 : 0 }}–{{ pageEndIndex() }}
              of {{ filteredSortedRows().length }} preview rows · {{ formatInteger(preview()!.shape[0]) }} total
              dataset rows
            </span>
            <div class="pager">
              <button
                type="button"
                class="button sm"
                [disabled]="currentPage() <= 1"
                (click)="prevPage()"
              >
                Previous
              </button>
              <b>{{ currentPage() }} / {{ totalPages() }}</b>
              <button
                type="button"
                class="button sm"
                [disabled]="currentPage() >= totalPages()"
                (click)="nextPage()"
              >
                Next
              </button>
            </div>
          </div>
        </sb-card>

        <div class="inspector-card">
          <sb-card
            title="Column inspector"
            subtitle="Deterministic full-dataset profile metrics for the selected column"
          >
            <select
              card-actions
              class="inspector-select"
              aria-label="Inspect column"
              [ngModel]="activeInspectorColumn()"
              (ngModelChange)="selectedColumnName.set($event)"
            >
              @for (col of preview()!.columns; track col) {
                <option [value]="col">{{ col }}</option>
              }
            </select>

            @if (inspectedProfile(); as prof) {
              <div class="inspector-grid">
                <div class="inspector-item">
                  <small>Semantic role</small>
                  <strong>{{ prof.semantic_type }}</strong>
                </div>
                <div class="inspector-item">
                  <small>Physical dtype</small>
                  <strong class="mono">{{ prof.dtype }}</strong>
                </div>
                <div class="inspector-item">
                  <small>Missing values</small>
                  <strong>{{ formatInteger(prof.null_count) }} ({{ formatNumber(prof.null_pct, 1) }}%)</strong>
                </div>
                <div class="inspector-item">
                  <small>Unique values</small>
                  <strong>{{ formatInteger(prof.unique_count) }}</strong>
                </div>
                @if (prof.numeric_stats) {
                  <div class="inspector-item">
                    <small>Mean / Median</small>
                    <strong>
                      {{ formatNumber(prof.numeric_stats.mean, 2) }} /
                      {{ formatNumber(prof.numeric_stats.median, 2) }}
                    </strong>
                  </div>
                  <div class="inspector-item">
                    <small>Min → Max</small>
                    <strong>
                      {{ formatNumber(prof.numeric_stats.min, 2) }} →
                      {{ formatNumber(prof.numeric_stats.max, 2) }}
                    </strong>
                  </div>
                } @else if (prof.top_values?.length) {
                  <div class="inspector-item">
                    <small>Most frequent</small>
                    <strong>{{ prof.top_values![0].value }} ({{ prof.top_values![0].pct }}%)</strong>
                  </div>
                }
              </div>
            }
          </sb-card>
        </div>
      }
    </div>
  `,
})
export class ExplorerComponent {
  readonly state = inject(WorkspaceStateService);
  private readonly router = inject(Router);

  private readonly searchInput = viewChild<ElementRef<HTMLInputElement>>('searchInput');

  readonly preview = this.state.preview;
  readonly profile = this.state.profile;

  readonly searchText = signal<string>('');
  readonly onlyMissingRows = signal<boolean>(false);
  readonly sortColumn = signal<string | null>(null);
  readonly sortDir = signal<'asc' | 'desc'>('asc');
  readonly currentPage = signal<number>(1);
  readonly pageSize = 15;
  readonly columnDrawerOpen = signal<boolean>(false);
  readonly hiddenColumns = signal<Set<string>>(new Set());
  readonly selectedColumnName = signal<string>('');

  readonly formatInteger = formatInteger;
  readonly formatNumber = formatNumber;

  readonly visibleColumns = computed(() => {
    const cols = this.preview()?.columns ?? [];
    const hidden = this.hiddenColumns();
    const visible = cols.filter((c) => !hidden.has(c));
    return visible.length > 0 ? visible : cols;
  });

  readonly filteredSortedRows = computed(() => {
    const rows = this.preview()?.preview ?? [];
    const cols = this.visibleColumns();
    const term = this.searchText().trim().toLowerCase();
    const missingOnly = this.onlyMissingRows();

    let result = rows.filter((row) => {
      if (missingOnly) {
        const hasNull = cols.some(
          (col) => row[col] === null || row[col] === undefined || row[col] === '',
        );
        if (!hasNull) return false;
      }
      if (!term) return true;
      return cols.some((col) =>
        String(row[col] ?? '')
          .toLowerCase()
          .includes(term),
      );
    });

    const sortCol = this.sortColumn();
    if (sortCol) {
      const factor = this.sortDir() === 'asc' ? 1 : -1;
      result = [...result].sort((a, b) => {
        const va = a[sortCol];
        const vb = b[sortCol];
        if (va === null || va === undefined) return 1;
        if (vb === null || vb === undefined) return -1;
        if (typeof va === 'number' && typeof vb === 'number') {
          return (va - vb) * factor;
        }
        return String(va).localeCompare(String(vb), undefined, { numeric: true }) * factor;
      });
    }

    return result;
  });

  readonly totalPages = computed(() =>
    Math.max(1, Math.ceil(this.filteredSortedRows().length / this.pageSize)),
  );

  readonly pageStartIndex = computed(() => (this.currentPage() - 1) * this.pageSize);
  readonly pageEndIndex = computed(() =>
    Math.min(this.filteredSortedRows().length, this.pageStartIndex() + this.pageSize),
  );

  readonly pagedRows = computed(() =>
    this.filteredSortedRows().slice(this.pageStartIndex(), this.pageEndIndex()),
  );

  readonly activeInspectorColumn = computed(() => {
    const chosen = this.selectedColumnName();
    const cols = this.preview()?.columns ?? [];
    if (chosen && cols.includes(chosen)) {
      return chosen;
    }
    return cols[0] ?? '';
  });

  readonly inspectedProfile = computed<ColumnProfile | null>(() => {
    const col = this.activeInspectorColumn();
    return this.profile()?.column_profiles.find((p) => p.name === col) ?? null;
  });

  @HostListener('window:keydown', ['$event'])
  handleShortcut(event: KeyboardEvent): void {
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'k') {
      event.preventDefault();
      this.searchInput()?.nativeElement.focus();
    }
  }

  onSearchChange(value: string): void {
    this.searchText.set(value);
    this.currentPage.set(1);
  }

  toggleMissingFilter(): void {
    this.onlyMissingRows.update((v) => !v);
    this.currentPage.set(1);
  }

  onHeaderClick(col: string): void {
    this.selectedColumnName.set(col);
    if (this.sortColumn() === col) {
      this.sortDir.update((d) => (d === 'asc' ? 'desc' : 'asc'));
    } else {
      this.sortColumn.set(col);
      this.sortDir.set('asc');
    }
    this.currentPage.set(1);
  }

  clearSort(): void {
    this.sortColumn.set(null);
  }

  toggleColumnVisibility(col: string): void {
    this.hiddenColumns.update((prev) => {
      const next = new Set(prev);
      if (next.has(col)) {
        next.delete(col);
      } else if ((this.preview()?.columns.length ?? 0) - next.size > 1) {
        next.add(col);
      }
      return next;
    });
  }

  resetHiddenColumns(): void {
    this.hiddenColumns.set(new Set());
  }

  prevPage(): void {
    this.currentPage.update((p) => Math.max(1, p - 1));
  }

  nextPage(): void {
    this.currentPage.update((p) => Math.min(this.totalPages(), p + 1));
  }

  exportPreview(): void {
    const prev = this.preview();
    if (!prev) return;
    const base = slugifyFilename(prev.filename.replace(/\.[^.]+$/, ''), 'dataset');
    downloadPreviewCsv(`${base}_preview.csv`, this.visibleColumns(), this.filteredSortedRows());
    this.state.showToast(`${base}_preview.csv downloaded.`);
  }

  goToOnboarding(): void {
    void this.router.navigate(['/onboarding']);
  }
}
