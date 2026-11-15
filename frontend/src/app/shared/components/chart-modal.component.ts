import { ChangeDetectionStrategy, Component, HostListener, inject, viewChild } from '@angular/core';
import { WorkspaceStateService } from '../../core/state/workspace-state.service';
import { PlotlyChartComponent } from './plotly-chart.component';
import { SvgIconComponent } from './svg-icon.component';

@Component({
  selector: 'sb-chart-modal',
  standalone: true,
  imports: [PlotlyChartComponent, SvgIconComponent],
  changeDetection: ChangeDetectionStrategy.OnPush,
  styles: [
    `
      .modal-header {
        min-height: 68px;
        border-bottom: 1px solid var(--border);
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        padding: 12px 20px;
      }
      .modal-header h2 {
        font-size: 15px;
        margin: 0 0 3px;
      }
      .modal-header p {
        font-size: 11px;
        color: var(--muted);
        margin: 0;
      }
      .modal-header-actions {
        display: flex;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
      }
      .modal-body {
        padding: 18px 22px;
        overflow: auto;
      }
      @media (max-width: 520px) {
        .modal-header {
          padding: 12px;
        }
        .modal-header .hide-xs {
          display: none;
        }
        .modal-body {
          padding: 12px;
        }
      }
    `,
  ],
  template: `
    @if (state.expandedChart(); as chart) {
      <div
        class="modal-backdrop"
        role="presentation"
        (mousedown)="onBackdropMouseDown($event)"
      >
        <div class="modal" role="dialog" aria-modal="true" aria-labelledby="expanded-chart-title">
          <header class="modal-header">
            <div>
              <h2 id="expanded-chart-title">{{ chart.title }}</h2>
              @if (chart.subtitle) {
                <p>{{ chart.subtitle }}</p>
              }
            </div>
            <div class="modal-header-actions">
              <button type="button" class="button sm hide-xs" (click)="exportPng()">
                <sb-svg-icon name="download" [size]="14" />
                PNG
              </button>
              <button type="button" class="button sm hide-xs" (click)="exportHtml()">
                <sb-svg-icon name="external" [size]="14" />
                HTML
              </button>
              <button
                type="button"
                class="icon-button"
                aria-label="Close chart dialog"
                (click)="close()"
              >
                <sb-svg-icon name="close" [size]="18" />
              </button>
            </div>
          </header>
          <div class="modal-body">
            <sb-plotly-chart
              #modalPlot
              [figureJson]="chart.figureJson"
              [title]="chart.title"
              [height]="500"
            />
          </div>
        </div>
      </div>
    }
  `,
})
export class ChartModalComponent {
  readonly state = inject(WorkspaceStateService);
  private readonly modalPlot = viewChild<PlotlyChartComponent>('modalPlot');

  @HostListener('document:keydown.escape')
  onEscapeKey(): void {
    if (this.state.expandedChart()) {
      this.close();
    }
  }

  onBackdropMouseDown(event: MouseEvent): void {
    if (event.target === event.currentTarget) {
      this.close();
    }
  }

  close(): void {
    this.state.closeExpandedChart();
  }

  async exportPng(): Promise<void> {
    await this.modalPlot()?.downloadPng();
  }

  exportHtml(): void {
    this.modalPlot()?.downloadHtml();
  }
}
