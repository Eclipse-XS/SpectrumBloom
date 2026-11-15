import {
  ChangeDetectionStrategy,
  Component,
  DestroyRef,
  ElementRef,
  effect,
  inject,
  input,
  signal,
  viewChild,
} from '@angular/core';
import { ThemeService } from '../../core/state/theme.service';
import { ThemeMode } from '../../core/models/workspace.models';
import { downloadPlotlyHtml } from '../utils/export.utils';
import { slugifyFilename } from '../utils/format.utils';
import { SvgIconComponent } from './svg-icon.component';

type PlotlyModule = typeof import('plotly.js-dist-min');

let plotlyLoader: Promise<PlotlyModule> | null = null;

function loadPlotly(): Promise<PlotlyModule> {
  if (!plotlyLoader) {
    plotlyLoader = import('plotly.js-dist-min').then((mod) => {
      const candidate = (mod as unknown as { default?: PlotlyModule }).default;
      return candidate ?? mod;
    });
  }
  return plotlyLoader;
}

interface ParsedFigure {
  data: Record<string, unknown>[];
  layout: Record<string, unknown>;
}

@Component({
  selector: 'sb-plotly-chart',
  standalone: true,
  imports: [SvgIconComponent],
  changeDetection: ChangeDetectionStrategy.OnPush,
  styles: [
    `
      .plotly-wrap {
        position: relative;
        width: 100%;
        min-height: 220px;
      }
      .plotly-host {
        width: 100%;
      }
      .plotly-fallback {
        min-height: 200px;
        border: 1px dashed var(--border-strong);
        border-radius: 8px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 8px;
        padding: 20px;
        color: var(--muted);
        font-size: 12px;
        text-align: center;
      }
    `,
  ],
  template: `
    @if (renderError()) {
      <div class="plotly-fallback" role="alert">
        <sb-svg-icon name="warning" [size]="20" />
        <span>{{ renderError() }}</span>
      </div>
    } @else {
      <div class="plotly-wrap" [style.minHeight.px]="height()">
        @if (rendering()) {
          <div class="plotly-fallback" aria-live="polite">
            <div class="loader-ring"></div>
            <span>Rendering interactive chart…</span>
          </div>
        }
        <div #plotlyHost class="plotly-host" [style.height.px]="height()"></div>
      </div>
    }
  `,
})
export class PlotlyChartComponent {
  private readonly themeService = inject(ThemeService);
  private readonly destroyRef = inject(DestroyRef);

  readonly figureJson = input.required<string>();
  readonly title = input<string>('Chart');
  readonly height = input<number>(330);

  private readonly hostRef = viewChild<ElementRef<HTMLDivElement>>('plotlyHost');

  readonly rendering = signal<boolean>(true);
  readonly renderError = signal<string | null>(null);

  private resizeObserver: ResizeObserver | null = null;
  private resizeRaf: number | null = null;
  private plotlyModule: PlotlyModule | null = null;
  private renderGeneration = 0;

  constructor() {
    effect(() => {
      const json = this.figureJson();
      const mode = this.themeService.mode();
      const chartHeight = this.height();
      const el = this.hostRef()?.nativeElement;
      if (!el) {
        return;
      }
      void this.renderFigure(el, json, mode, chartHeight);
    });

    this.destroyRef.onDestroy(() => {
      this.cleanup();
    });
  }

  async downloadPng(): Promise<void> {
    const el = this.hostRef()?.nativeElement;
    if (!el || !this.plotlyModule) {
      return;
    }
    const filename = slugifyFilename(this.title(), 'spectrumbloom_chart');
    await this.plotlyModule.downloadImage(el, {
      format: 'png',
      width: 1100,
      height: 680,
      filename,
    });
  }

  downloadHtml(): void {
    downloadPlotlyHtml({
      title: this.title(),
      figure_json: this.figureJson(),
    });
  }

  private async renderFigure(
    hostEl: HTMLDivElement,
    rawJson: string,
    mode: ThemeMode,
    chartHeight: number,
  ): Promise<void> {
    const gen = ++this.renderGeneration;
    this.renderError.set(null);

    let parsed: ParsedFigure;
    try {
      const obj = JSON.parse(rawJson) as { data?: unknown; layout?: unknown };
      if (!obj || !Array.isArray(obj.data) || obj.data.length === 0) {
        this.rendering.set(false);
        this.renderError.set('No plot traces were returned for this figure.');
        return;
      }
      parsed = {
        data: structuredClone(obj.data) as Record<string, unknown>[],
        layout: structuredClone((obj.layout as Record<string, unknown>) ?? {}),
      };
    } catch {
      this.rendering.set(false);
      this.renderError.set('Invalid Plotly figure JSON payload.');
      return;
    }

    try {
      const Plotly = await loadPlotly();
      if (gen !== this.renderGeneration) {
        return;
      }
      this.plotlyModule = Plotly;

      const themedLayout = this.buildThemedLayout(parsed.layout, mode, chartHeight);
      await Plotly.react(hostEl, parsed.data as never[], themedLayout as never, {
        responsive: true,
        displaylogo: false,
        modeBarButtonsToRemove: ['lasso2d', 'select2d'],
      });
      if (gen !== this.renderGeneration) {
        return;
      }

      this.ensureResizeObserver(hostEl);
      this.rendering.set(false);
    } catch (err) {
      if (gen === this.renderGeneration) {
        this.rendering.set(false);
        this.renderError.set(
          err instanceof Error ? err.message : 'Unable to render interactive chart.',
        );
      }
    }
  }

  private buildThemedLayout(
    baseLayout: Record<string, unknown>,
    mode: ThemeMode,
    chartHeight: number,
  ): Record<string, unknown> {
    const dark = mode === 'dark';
    const textColor = dark ? '#dce8e8' : '#172b35';
    const gridColor = dark ? 'rgba(165, 182, 186, 0.16)' : 'rgba(23, 43, 53, 0.09)';

    const existingMargin =
      typeof baseLayout['margin'] === 'object' && baseLayout['margin'] !== null
        ? (baseLayout['margin'] as Record<string, unknown>)
        : {};

    const layout: Record<string, unknown> = {
      ...baseLayout,
      autosize: true,
      height: chartHeight,
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      font: {
        ...(typeof baseLayout['font'] === 'object' && baseLayout['font'] !== null
          ? (baseLayout['font'] as Record<string, unknown>)
          : {}),
        family: 'Inter, ui-sans-serif, system-ui, sans-serif',
        color: textColor,
      },
      margin: {
        l: existingMargin['l'] ?? 48,
        r: existingMargin['r'] ?? 24,
        t: existingMargin['t'] ?? 44,
        b: existingMargin['b'] ?? 46,
      },
    };

    for (const axisKey of ['xaxis', 'yaxis', 'xaxis2', 'yaxis2']) {
      if (axisKey in layout || axisKey === 'xaxis' || axisKey === 'yaxis') {
        const existingAxis =
          typeof layout[axisKey] === 'object' && layout[axisKey] !== null
            ? (layout[axisKey] as Record<string, unknown>)
            : {};
        layout[axisKey] = {
          ...existingAxis,
          gridcolor: gridColor,
          zerolinecolor: gridColor,
        };
      }
    }

    return layout;
  }

  private ensureResizeObserver(hostEl: HTMLDivElement): void {
    if (this.resizeObserver || typeof ResizeObserver === 'undefined') {
      return;
    }
    this.resizeObserver = new ResizeObserver(() => {
      if (this.resizeRaf !== null) {
        cancelAnimationFrame(this.resizeRaf);
      }
      this.resizeRaf = requestAnimationFrame(() => {
        if (this.plotlyModule && hostEl.offsetParent !== null) {
          this.plotlyModule.Plots.resize(hostEl);
        }
      });
    });
    this.resizeObserver.observe(hostEl);
  }

  private cleanup(): void {
    if (this.resizeRaf !== null) {
      cancelAnimationFrame(this.resizeRaf);
      this.resizeRaf = null;
    }
    if (this.resizeObserver) {
      this.resizeObserver.disconnect();
      this.resizeObserver = null;
    }
    const el = this.hostRef()?.nativeElement;
    if (el && this.plotlyModule) {
      this.plotlyModule.purge(el);
    }
  }
}
