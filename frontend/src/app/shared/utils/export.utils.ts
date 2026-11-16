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

export function buildStandalonePlotlyHtml(
  figure: Pick<FigurePayload, 'title' | 'figure_json'>,
  meta?: { sessionId?: string | null; datasetFilename?: string | null },
): string {
  const safeTitle = (figure.title || 'SpectrumBloom Chart')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
  const escapedFigureJson = figure.figure_json
    .replace(/<\//g, '<\\/')
    .replace(/<!--/g, '<\\!--');
  const safeSession = (meta?.sessionId || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
  const safeDataset = (meta?.datasetFilename || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');

  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>${safeTitle} — SpectrumBloom</title>
  <style>
    body { margin: 0; padding: 24px; font-family: Inter, system-ui, sans-serif; background: #f7f9fa; color: #172b35; }
    .card { max-width: 1120px; margin: 0 auto; background: #fff; border: 1px solid #dce4e7; border-radius: 10px; padding: 18px; }
    .head { display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between; gap: 8px; margin-bottom: 10px; }
    h1 { font-size: 16px; margin: 0; }
    .meta { font-size: 11px; color: #546a76; }
    .disclosure { font-size: 11px; color: #546a76; background: #f0f5f7; border: 1px solid #dce4e7; border-radius: 6px; padding: 8px 10px; margin-bottom: 12px; }
    #chart { width: 100%; min-height: 540px; }
    .offline-fallback { display: none; border: 1px solid #e0a800; background: #fff9e6; border-radius: 8px; padding: 14px; margin-top: 12px; font-size: 12px; }
    .offline-fallback pre { max-height: 360px; overflow: auto; background: #fff; border: 1px solid #dce4e7; border-radius: 6px; padding: 10px; font-size: 11px; }
    .offline-table { width: 100%; border-collapse: collapse; margin: 10px 0; font-size: 12px; }
    .offline-table th, .offline-table td { border: 1px solid #dce4e7; padding: 6px 8px; text-align: left; }
  </style>
</head>
<body>
  <div class="card">
    <div class="head">
      <h1>${safeTitle}</h1>
      <span class="meta">${safeDataset ? `Dataset: ${safeDataset}` : ''}${safeSession ? ` · Session: ${safeSession}` : ''}</span>
    </div>
    <div class="disclosure" id="cdn-disclosure">
      Interactive chart rendering loads <code>plotly-2.35.2.min.js</code> from <code>https://cdn.plot.ly/plotly-2.35.2.min.js</code> (requires internet access unless cached). When opened offline without CDN access, the embedded figure specification and trace data summary remain viewable below.
    </div>
    <div id="chart"></div>
    <div id="offline-fallback" class="offline-fallback">
      <strong>Offline mode (Plotly CDN unreachable):</strong> Rendering embedded trace summary and full JSON specification.
      <div id="offline-summary"></div>
      <pre id="offline-json"></pre>
    </div>
  </div>
  <script>
    const fig = ${escapedFigureJson};
    function renderOfflineFallback() {
      const chartEl = document.getElementById('chart');
      if (chartEl) chartEl.style.display = 'none';
      const box = document.getElementById('offline-fallback');
      if (box) box.style.display = 'block';
      const summaryEl = document.getElementById('offline-summary');
      const traces = Array.isArray(fig.data) ? fig.data : [];
      if (summaryEl) {
        const rows = traces.map((t, idx) => {
          const tName = String(t.name || ('Trace ' + (idx + 1))).replace(/</g, '&lt;');
          const tType = String(t.type || 'scatter').replace(/</g, '&lt;');
          const count = Array.isArray(t.x) ? t.x.length : (Array.isArray(t.y) ? t.y.length : (Array.isArray(t.values) ? t.values.length : 0));
          return '<tr><td>' + tName + '</td><td>' + tType + '</td><td>' + count + '</td></tr>';
        }).join('');
        summaryEl.innerHTML = '<table class="offline-table"><thead><tr><th>Trace</th><th>Type</th><th>Points</th></tr></thead><tbody>' + rows + '</tbody></table>';
      }
      const pre = document.getElementById('offline-json');
      if (pre) pre.textContent = JSON.stringify(fig, null, 2);
    }
    function renderInteractiveChart() {
      if (typeof window.Plotly !== 'undefined' && window.Plotly.newPlot) {
        window.Plotly.newPlot('chart', fig.data || [], fig.layout || {}, { responsive: true, displaylogo: false });
      } else {
        renderOfflineFallback();
      }
    }
  </script>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8" onload="renderInteractiveChart()" onerror="renderOfflineFallback()"></script>
</body>
</html>`;
}

export function downloadPlotlyHtml(
  figure: Pick<FigurePayload, 'title' | 'figure_json'>,
  meta?: { sessionId?: string | null; datasetFilename?: string | null },
): void {
  const html = buildStandalonePlotlyHtml(figure, meta);
  const name = `${slugifyFilename(figure.title || 'plotly_figure', 'plotly_figure')}.html`;
  triggerBlobDownload(name, html, 'text/html;charset=utf-8');
}
