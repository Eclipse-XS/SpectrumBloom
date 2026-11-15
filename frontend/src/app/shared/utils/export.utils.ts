import { FigurePayload } from '../../core/models/api.models';
import { sanitizeCsvCell, slugifyFilename } from './format.utils';

export function triggerBlobDownload(filename: string, content: BlobPart, mimeType: string): void {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  anchor.style.display = 'none';
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function downloadJson(filename: string, payload: unknown): void {
  triggerBlobDownload(filename, JSON.stringify(payload, null, 2), 'application/json;charset=utf-8');
}

export function downloadPreviewCsv(
  filename: string,
  columns: string[],
  rows: Record<string, unknown>[],
): void {
  const headerLine = columns.map((c) => sanitizeCsvCell(c)).join(',');
  const bodyLines = rows.map((row) =>
    columns.map((col) => sanitizeCsvCell(row[col])).join(','),
  );
  const csv = [headerLine, ...bodyLines].join('\n');
  triggerBlobDownload(filename, csv, 'text/csv;charset=utf-8');
}

export function buildStandalonePlotlyHtml(figure: Pick<FigurePayload, 'title' | 'figure_json'>): string {
  const safeTitle = (figure.title || 'SpectrumBloom Chart')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
  const escapedFigureJson = figure.figure_json
    .replace(/<\//g, '<\\/')
    .replace(/<!--/g, '<\\!--');

  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>${safeTitle} — SpectrumBloom</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
  <style>
    body { margin: 0; padding: 24px; font-family: Inter, system-ui, sans-serif; background: #f7f9fa; color: #172b35; }
    .card { max-width: 1120px; margin: 0 auto; background: #fff; border: 1px solid #dce4e7; border-radius: 10px; padding: 18px; }
    h1 { font-size: 16px; margin: 0 0 12px; }
    #chart { width: 100%; height: 560px; }
  </style>
</head>
<body>
  <div class="card">
    <h1>${safeTitle}</h1>
    <div id="chart"></div>
  </div>
  <script>
    const fig = ${escapedFigureJson};
    Plotly.newPlot('chart', fig.data || [], fig.layout || {}, { responsive: true, displaylogo: false });
  </script>
</body>
</html>`;
}

export function downloadPlotlyHtml(figure: Pick<FigurePayload, 'title' | 'figure_json'>): void {
  const html = buildStandalonePlotlyHtml(figure);
  const name = `${slugifyFilename(figure.title || 'plotly_figure', 'plotly_figure')}.html`;
  triggerBlobDownload(name, html, 'text/html;charset=utf-8');
}
