import { Download, expect, Page, Route, test } from '@playwright/test';
import * as fs from 'node:fs';

interface MockSessionState {
  session_id: string;
  thread_id: string;
  provider: 'ollama' | 'llamacpp' | 'openai' | 'gemini';
  model: string;
  has_data: boolean;
  message_count: number;
  figure_count: number;
  dataset: {
    filename: string;
    size_bytes: number;
    sha256: string;
    rows: number;
    columns: number;
    storage_backend: 'local';
    storage_ref: string;
    uploaded_at: string;
  } | null;
  created_at: string;
  updated_at: string;
  messages: Record<string, unknown>[];
  edaReport: Record<string, unknown> | null;
  preview: Record<string, unknown> | null;
}

async function readDownloadText(download: Download): Promise<string> {
  const filePath = await download.path();
  if (!filePath) {
    throw new Error('Download path unavailable');
  }
  return fs.readFileSync(filePath, 'utf-8');
}

async function assertNoHorizontalOverflow(page: Page): Promise<void> {
  const metrics = await page.evaluate(() => ({
    scrollWidth: document.documentElement.scrollWidth,
    clientWidth: document.documentElement.clientWidth,
  }));
  expect(metrics.scrollWidth).toBeLessThanOrEqual(metrics.clientWidth);
}

function buildCustomEdaReport(filename: string) {
  const distFigure = {
    data: [
      {
        type: 'histogram',
        name: 'monthly_spend',
        x: [120.5, 240.0, 85.25, 310.75, 415.0],
        y: [1, 1, 1, 1, 1],
      },
    ],
    layout: {
      title: { text: `Distribution of monthly_spend (${filename})` },
      xaxis: { title: { text: 'monthly_spend (USD)' } },
      yaxis: { title: { text: 'Count' } },
    },
  };

  const corrFigure = {
    data: [
      {
        type: 'heatmap',
        x: ['monthly_spend', 'tenure_months', 'support_tickets'],
        y: ['monthly_spend', 'tenure_months', 'support_tickets'],
        z: [
          [1.0, 0.82, -0.34],
          [0.82, 1.0, -0.29],
          [-0.34, -0.29, 1.0],
        ],
      },
    ],
    layout: {
      title: { text: 'Numeric Correlation Matrix' },
      xaxis: { title: { text: 'Feature' } },
      yaxis: { title: { text: 'Feature' } },
    },
  };

  const missFigure = {
    data: [
      {
        type: 'bar',
        x: ['monthly_spend', 'segment'],
        y: [1, 1],
      },
    ],
    layout: {
      title: { text: 'Missing Values by Column' },
      xaxis: { title: { text: 'Column' } },
      yaxis: { title: { text: 'Missing count' } },
    },
  };

  const outlierFigure = {
    data: [
      {
        type: 'box',
        name: 'monthly_spend',
        y: [85.25, 120.5, 240.0, 310.75, 415.0],
      },
    ],
    layout: {
      title: { text: 'Outlier Box Plot' },
      yaxis: { title: { text: 'monthly_spend' } },
    },
  };

  return {
    profile: {
      filename,
      rows: 6,
      columns: 5,
      duplicate_rows: 0,
      duplicate_pct: 0,
      memory_mb: 0.012,
      completeness_pct: 93.3,
      warnings: ['Column monthly_spend has 16.7% missing values (1 of 6 rows).'],
      column_profiles: [
        {
          name: 'customer_id',
          dtype: 'object',
          semantic_type: 'identifier',
          non_null_count: 6,
          null_count: 0,
          null_pct: 0,
          unique_count: 6,
          sample_values: ['C-101', 'C-102', 'C-103'],
          numeric_stats: null,
          top_values: [],
        },
        {
          name: 'segment',
          dtype: 'object',
          semantic_type: 'categorical',
          non_null_count: 5,
          null_count: 1,
          null_pct: 16.7,
          unique_count: 3,
          sample_values: ['Enterprise', 'SMB', 'Consumer'],
          numeric_stats: null,
          top_values: [
            { value: 'Enterprise', count: 2, pct: 33.3 },
            { value: 'SMB', count: 2, pct: 33.3 },
            { value: 'Consumer', count: 1, pct: 16.7 },
          ],
        },
        {
          name: 'monthly_spend',
          dtype: 'float64',
          semantic_type: 'numeric_continuous',
          non_null_count: 5,
          null_count: 1,
          null_pct: 16.7,
          unique_count: 5,
          sample_values: [120.5, 240.0, 85.25],
          numeric_stats: {
            mean: 234.3,
            std: 135.4,
            min: 85.25,
            q25: 120.5,
            median: 240.0,
            q75: 310.75,
            max: 415.0,
            iqr: 190.25,
            skewness: 0.19,
            kurtosis: -1.12,
            zeros_count: 0,
            negatives_count: 0,
          },
          top_values: [],
        },
        {
          name: 'tenure_months',
          dtype: 'int64',
          semantic_type: 'numeric_discrete',
          non_null_count: 6,
          null_count: 0,
          null_pct: 0,
          unique_count: 6,
          sample_values: [4, 12, 24],
          numeric_stats: {
            mean: 19.5,
            std: 11.2,
            min: 4,
            q25: 12,
            median: 19,
            q75: 28,
            max: 36,
            iqr: 16,
            skewness: 0.14,
            kurtosis: -1.2,
            zeros_count: 0,
            negatives_count: 0,
          },
          top_values: [],
        },
        {
          name: 'churn_flag',
          dtype: 'int64',
          semantic_type: 'numeric_discrete',
          non_null_count: 6,
          null_count: 0,
          null_pct: 0,
          unique_count: 2,
          sample_values: [0, 1],
          numeric_stats: {
            mean: 0.333,
            std: 0.516,
            min: 0,
            q25: 0,
            median: 0,
            q75: 1,
            max: 1,
            iqr: 1,
            skewness: 0.71,
            kurtosis: -1.5,
            zeros_count: 4,
            negatives_count: 0,
          },
          top_values: [],
        },
      ],
    },
    statistics: {
      missingness: {
        total_cells: 30,
        total_missing_cells: 2,
        overall_missing_pct: 6.67,
        rows_with_any_missing: 2,
        rows_with_any_missing_pct: 33.33,
        columns_with_missing: [
          { column: 'monthly_spend', missing_count: 1, missing_pct: 16.7 },
          { column: 'segment', missing_count: 1, missing_pct: 16.7 },
        ],
        missing_cooccurrence_pairs: [],
      },
      numeric_columns: ['monthly_spend', 'tenure_months', 'churn_flag'],
      categorical_columns: ['segment'],
      pearson_matrix: {},
      spearman_matrix: {},
      top_correlations: [
        {
          column_a: 'monthly_spend',
          column_b: 'tenure_months',
          pearson_r: 0.821,
          pearson_p: 0.0041,
          spearman_rho: 0.85,
          spearman_p: 0.0029,
        },
      ],
      categorical_associations: [
        {
          column_a: 'segment',
          column_b: 'customer_id',
          cramers_v: 0.412,
          chi2_p: 0.031,
        },
      ],
      outliers: [
        {
          column: 'monthly_spend',
          iqr_lower: -164.875,
          iqr_upper: 596.125,
          iqr_outlier_count: 0,
          iqr_outlier_pct: 0,
          zscore_outlier_count: 0,
          mad_outlier_count: 1,
        },
      ],
      multivariate_anomaly_count: 1,
      normality_checks: [
        {
          column: 'monthly_spend',
          test_name: 'Shapiro-Wilk',
          sample_size: 5,
          statistic: 0.9612,
          p_value: 0.814,
          is_normal_alpha_05: true,
          skewness: 0.19,
          kurtosis: -1.12,
        },
      ],
      group_tests: [
        {
          categorical_column: 'segment',
          numeric_column: 'monthly_spend',
          group_count: 3,
          test: 'Kruskal-Wallis',
          statistic: 7.421,
          p_value: 0.0245,
        },
      ],
    },
    figures: [
      {
        id: 'fig-miss-1',
        title: 'Missing Values by Column',
        chart_kind: 'missingness',
        figure_json: JSON.stringify(missFigure),
      },
      {
        id: 'fig-dist-1',
        title: `Distribution of monthly_spend (${filename})`,
        chart_kind: 'distribution',
        figure_json: JSON.stringify(distFigure),
      },
      {
        id: 'fig-corr-1',
        title: 'Numeric Correlation Matrix',
        chart_kind: 'correlation_heatmap',
        figure_json: JSON.stringify(corrFigure),
      },
      {
        id: 'fig-out-1',
        title: 'Outlier Box Plot',
        chart_kind: 'outliers',
        figure_json: JSON.stringify(outlierFigure),
      },
    ],
  };
}

function buildCustomPreview(filename: string) {
  return {
    filename,
    shape: [6, 5],
    columns: ['customer_id', 'segment', 'monthly_spend', 'tenure_months', 'churn_flag'],
    dtypes: {
      customer_id: 'object',
      segment: 'object',
      monthly_spend: 'float64',
      tenure_months: 'int64',
      churn_flag: 'int64',
    },
    preview: [
      {
        customer_id: 'C-101',
        segment: 'Enterprise',
        monthly_spend: 415.0,
        tenure_months: 36,
        churn_flag: 0,
      },
      {
        customer_id: 'C-102',
        segment: 'SMB',
        monthly_spend: 120.5,
        tenure_months: 12,
        churn_flag: 0,
      },
      {
        customer_id: 'C-103',
        segment: 'Consumer',
        monthly_spend: 85.25,
        tenure_months: 4,
        churn_flag: 1,
      },
      {
        customer_id: 'C-104',
        segment: null,
        monthly_spend: 240.0,
        tenure_months: 18,
        churn_flag: 0,
      },
      {
        customer_id: 'C-105',
        segment: 'Enterprise',
        monthly_spend: null,
        tenure_months: 27,
        churn_flag: 0,
      },
      {
        customer_id: '=SUM(1,2)',
        segment: 'SMB',
        monthly_spend: 310.75,
        tenure_months: 20,
        churn_flag: 1,
      },
    ],
  };
}

test.describe('SpectrumBloom Repeatable Deterministic Browser E2E Suite', () => {
  test('comprehensive arbitrary CSV upload, Explorer filter/sort/NULLs, Plotly trace inspection, multi-turn AI Analyst, ML Lab configuration/leaderboard, CSV/JSON/HTML exports, session lifecycle, and responsive matrix', async ({
    page,
  }) => {
    test.setTimeout(180_000);

    const sessions = new Map<string, MockSessionState>();
    let sessionSeq = 1;
    let simulateTimeoutOnNextQuery = false;

    const createSessionRecord = (provider: 'ollama' = 'ollama', model = 'qwen3:4b-instruct'): MockSessionState => {
      const sid = `sess_det_000${sessionSeq++}`;
      const now = '2026-11-15T18:00:00Z';
      const rec: MockSessionState = {
        session_id: sid,
        thread_id: `thread_${sid}`,
        provider,
        model,
        has_data: false,
        message_count: 0,
        figure_count: 0,
        dataset: null,
        created_at: now,
        updated_at: now,
        messages: [],
        edaReport: null,
        preview: null,
      };
      sessions.set(sid, rec);
      return rec;
    };

    const toStatus = (s: MockSessionState) => ({
      session_id: s.session_id,
      thread_id: s.thread_id,
      provider: s.provider,
      model: s.model,
      has_data: s.has_data,
      message_count: s.messages.length,
      figure_count: s.figure_count,
      dataset: s.dataset,
      created_at: s.created_at,
      updated_at: s.updated_at,
    });

    await page.route('**/health', async (route: Route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          status: 'ok',
          version: '0.2.0',
          session_repository: 'connected',
          dataset_storage: { backend: 'local', status: 'connected' },
          sandbox_backend: 'docker',
          sandbox_dynamic_python_enabled: true,
          auth_required: false,
          max_upload_mb: 25,
          default_provider: 'ollama',
          default_model: 'qwen3:4b-instruct',
          llm: {
            provider: 'ollama',
            status: 'connected',
            configured_model: 'qwen3:4b-instruct',
            available_models: ['qwen3:4b-instruct', 'qwen2.5-coder:3b-instruct-q4_K_M'],
          },
          observability: { opentelemetry: true, langfuse: false },
        }),
      });
    });

    await page.route('**/api/v1/sessions**', async (route: Route) => {
      const req = route.request();
      const url = new URL(req.url());
      const pathname = url.pathname;
      const method = req.method();

      // GET / POST /api/v1/sessions
      if (pathname === '/api/v1/sessions') {
        if (method === 'GET') {
          await route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify([...sessions.values()].map(toStatus)),
          });
          return;
        }
        if (method === 'POST') {
          const body = req.postDataJSON() || {};
          const created = createSessionRecord(body.provider || 'ollama', body.model || 'qwen3:4b-instruct');
          await route.fulfill({
            status: 201,
            contentType: 'application/json',
            body: JSON.stringify({
              session_id: created.session_id,
              thread_id: created.thread_id,
              provider: created.provider,
              model: created.model,
            }),
          });
          return;
        }
      }

      const match = pathname.match(/^\/api\/v1\/sessions\/([^/]+)(?:\/(.*))?$/);
      if (!match) {
        await route.fallback();
        return;
      }
      const sid = decodeURIComponent(match[1]);
      const sub = match[2] || '';
      const sess = sessions.get(sid);
      if (!sess) {
        await route.fulfill({
          status: 404,
          contentType: 'application/json',
          body: JSON.stringify({ detail: `Session ${sid} not found` }),
        });
        return;
      }

      if (sub === '' && method === 'GET') {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(toStatus(sess)),
        });
        return;
      }

      if (sub === '' && method === 'DELETE') {
        sessions.delete(sid);
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ detail: 'Deleted' }),
        });
        return;
      }

      if (sub === 'clear' && method === 'POST') {
        sess.messages = [];
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(toStatus(sess)),
        });
        return;
      }

      if (sub === 'provider' && method === 'PATCH') {
        const body = req.postDataJSON() || {};
        sess.provider = body.provider || sess.provider;
        sess.model = body.model || sess.model;
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(toStatus(sess)),
        });
        return;
      }

      if (sub === 'history' && method === 'GET') {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            session_id: sid,
            messages: sess.messages,
            last_tool_traces: [],
          }),
        });
        return;
      }

      if (sub === 'upload' && method === 'POST') {
        const filename = 'custom_b2b_accounts.csv';
        sess.has_data = true;
        sess.figure_count = 4;
        sess.dataset = {
          filename,
          size_bytes: 1840,
          sha256: `sha_${sid}_${Date.now()}`,
          rows: 6,
          columns: 5,
          storage_backend: 'local',
          storage_ref: `/tmp/${filename}`,
          uploaded_at: '2026-11-15T18:05:00Z',
        };
        sess.messages = [];
        sess.edaReport = buildCustomEdaReport(filename);
        sess.preview = buildCustomPreview(filename);
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(sess.edaReport),
        });
        return;
      }

      if (sub === 'eda' && method === 'GET') {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(sess.edaReport),
        });
        return;
      }

      if (sub === 'data/preview' && method === 'GET') {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(sess.preview),
        });
        return;
      }

      if (sub === 'query' && method === 'POST') {
        const body = req.postDataJSON() || {};
        const queryText = String(body.query || '');
        const turnId = body.client_turn_id || `turn_${Date.now()}`;

        const analystChart = {
          id: `fig_analyst_${sess.messages.length + 1}`,
          title: 'Mean Monthly Spend by Segment',
          chart_kind: 'bar',
          figure_json: JSON.stringify({
            data: [
              {
                type: 'bar',
                name: 'Mean Spend',
                x: ['Enterprise', 'SMB', 'Consumer'],
                y: [415.0, 215.625, 85.25],
              },
            ],
            layout: {
              title: { text: 'Mean Monthly Spend by Segment' },
              xaxis: { title: { text: 'segment' } },
              yaxis: { title: { text: 'Mean monthly_spend (USD)' } },
            },
          }),
        };

        const userMsg = {
          role: 'user',
          content: queryText,
          timestamp: '2026-11-15T18:06:00Z',
          figures: [],
          tool_traces: [],
          evaluation: null,
          status: 'success',
          error: null,
          latency_ms: null,
          retries_used: 0,
          client_turn_id: turnId,
        };
        const answerText = queryText.toLowerCase().includes('spearman')
          ? 'The Spearman correlation between `monthly_spend` and `tenure_months` is **0.8500** ($p = 0.0029$).'
          : 'Across segments, **Enterprise** averages **$415.00**, **SMB** averages **$215.63**, and **Consumer** averages **$85.25**.';
        const assistantMsg = {
          role: 'assistant',
          content: answerText,
          timestamp: '2026-11-15T18:06:02Z',
          figures: [analystChart],
          tool_traces: [
            {
              tool: 'aggregate_data',
              code: "aggregate_data(metric_column='monthly_spend', agg_func='mean', group_by='segment')",
              stdout: 'Enterprise: 415.0, SMB: 215.625, Consumer: 85.25',
              result: { Enterprise: 415.0, SMB: 215.625, Consumer: 85.25 },
              error: null,
              duration_ms: 18.4,
            },
          ],
          evaluation: {
            passed: true,
            tool_execution_ok: true,
            execution_pass_rate: 1.0,
            numeric_grounding_ratio: 1.0,
            numeric_grounding_passed: true,
            chart_validity_ok: true,
            judge_score: null,
            notes: [],
          },
          status: 'success',
          error: null,
          latency_ms: 842.0,
          retries_used: 0,
          client_turn_id: turnId,
        };

        // Record in backend history even if gateway times out on the wire!
        sess.messages.push(userMsg, assistantMsg);
        sess.figure_count += 1;

        if (simulateTimeoutOnNextQuery) {
          simulateTimeoutOnNextQuery = false;
          await route.fulfill({
            status: 504,
            contentType: 'application/json',
            body: JSON.stringify({ detail: 'Gateway timeout waiting for upstream proxy' }),
          });
          return;
        }

        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            session_id: sid,
            thread_id: sess.thread_id,
            query: queryText,
            answer: answerText,
            status: 'success',
            error: null,
            figures: [analystChart],
            tool_traces: assistantMsg.tool_traces,
            evaluation: assistantMsg.evaluation,
            latency_ms: 842.0,
            retries_used: 0,
          }),
        });
        return;
      }

      if (sub === 'ml/experiment' && method === 'POST') {
        const body = req.postDataJSON() || {};
        const targetCol = body.target_column || 'churn_flag';
        const mlFig = {
          id: 'fig-ml-imp-1',
          title: `Permutation Feature Importance (${targetCol})`,
          chart_kind: 'feature_importance',
          figure_json: JSON.stringify({
            data: [
              {
                type: 'bar',
                orientation: 'h',
                x: [0.342, 0.218, 0.095],
                y: ['monthly_spend', 'tenure_months', 'segment'],
              },
            ],
            layout: {
              title: { text: `Permutation Feature Importance (${targetCol})` },
              xaxis: { title: { text: 'Importance Mean' } },
              yaxis: { title: { text: 'Feature' } },
            },
          }),
        };
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            target_column: targetCol,
            task_type: body.task_type === 'regression' ? 'regression' : 'classification',
            train_rows: 4,
            test_rows: 2,
            feature_count: (body.feature_columns && body.feature_columns.length) || 3,
            best_model: 'RandomForest',
            models: [
              {
                model_name: 'RandomForest',
                cv_primary_metric_name: 'f1_macro',
                cv_primary_mean: 0.9125,
                cv_primary_std: 0.0312,
                test_metrics: { f1_macro: 0.92, roc_auc: 0.95, accuracy: 0.9167 },
              },
              {
                model_name: 'HistGradientBoosting',
                cv_primary_metric_name: 'f1_macro',
                cv_primary_mean: 0.884,
                cv_primary_std: 0.041,
                test_metrics: { f1_macro: 0.89, roc_auc: 0.91, accuracy: 0.8833 },
              },
              {
                model_name: 'DummyBaseline',
                cv_primary_metric_name: 'f1_macro',
                cv_primary_mean: 0.45,
                cv_primary_std: 0.01,
                test_metrics: { f1_macro: 0.45, roc_auc: 0.5, accuracy: 0.6667 },
              },
            ],
            feature_importances: [
              { feature: 'monthly_spend', importance_mean: 0.342, importance_std: 0.021 },
              { feature: 'tenure_months', importance_mean: 0.218, importance_std: 0.015 },
              { feature: 'segment', importance_mean: 0.095, importance_std: 0.009 },
            ],
            figures: [mlFig],
            notes: [
              'Identifier column customer_id was automatically excluded to prevent target leakage.',
            ],
          }),
        });
        return;
      }

      await route.fallback();
    });

    await page.setViewportSize({ width: 1440, height: 900 });
    await page.goto('/onboarding');

    // 1. Upload an arbitrary custom CSV dataset (not the bundled sample)
    const customCsvContent = [
      'customer_id,segment,monthly_spend,tenure_months,churn_flag',
      'C-101,Enterprise,415.0,36,0',
      'C-102,SMB,120.5,12,0',
      'C-103,Consumer,85.25,4,1',
      'C-104,,240.0,18,0',
      'C-105,Enterprise,,27,0',
      '=SUM(1,2),SMB,310.75,20,1',
    ].join('\n');

    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles({
      name: 'custom_b2b_accounts.csv',
      mimeType: 'text/csv',
      buffer: Buffer.from(customCsvContent, 'utf-8'),
    });

    await page.waitForURL('**/overview');
    await expect(page.locator('.dataset-strip')).toContainText('custom_b2b_accounts.csv');
    await expect(page.locator('.dataset-strip')).toContainText('6');
    await expect(page.getByText('93.3%').first()).toBeVisible();

    // 2. Data Explorer: search/filter, NULL values filter, column sorting, column visibility, CSV export content verification
    await page.getByRole('link', { name: 'Data Explorer' }).click();
    await page.waitForURL('**/explorer');
    await expect(page.locator('.toolbar-count')).toContainText('6 matching of 6 preview rows');
    await expect(page.locator('.null-pill')).toHaveCount(2);

    // Filter by search term
    const searchInput = page.getByLabel('Search loaded preview rows');
    await searchInput.fill('Enterprise');
    await expect(page.locator('.toolbar-count')).toContainText('2 matching of 6 preview rows');
    await searchInput.fill('');
    await expect(page.locator('.toolbar-count')).toContainText('6 matching of 6 preview rows');

    // Filter rows with NULLs
    await page.getByRole('button', { name: /Rows with NULLs/i }).click();
    await expect(page.locator('.toolbar-count')).toContainText('2 matching of 6 preview rows');
    await page.getByRole('button', { name: /Rows with NULLs/i }).click();
    await expect(page.locator('.toolbar-count')).toContainText('6 matching of 6 preview rows');

    // Sort by monthly_spend ascending then descending and verify actual row order
    const spendHeader = page.locator('button.th-btn', { hasText: 'monthly_spend' });
    await spendHeader.click();
    await expect(page.locator('.explorer-table tbody tr').first()).toContainText('85.25');
    await spendHeader.click();
    await expect(page.locator('.explorer-table tbody tr').first()).toContainText('415');
    await page.getByRole('button', { name: /Clear sort/i }).click();

    // Verify Column Inspector updates when selecting monthly_spend
    await expect(page.locator('.inspector-card')).toContainText('numeric_continuous');
    await expect(page.locator('.inspector-card')).toContainText('234.3');

    // Export preview CSV and verify actual downloaded file contents (including formula-injection neutralization!)
    const [previewCsvDownload] = await Promise.all([
      page.waitForEvent('download'),
      page.getByRole('button', { name: 'Export preview CSV' }).click(),
    ]);
    const csvText = await readDownloadText(previewCsvDownload);
    expect(csvText).toContain('customer_id,segment,monthly_spend,tenure_months,churn_flag');
    expect(csvText).toContain('C-101,Enterprise,415,36,0');
    expect(csvText).toContain("'=SUM(1,2)");

    // 3. Automated EDA: verify Plotly data and trace correctness on DOM element (not just visibility)
    await page.getByRole('link', { name: 'Automated EDA' }).click();
    await page.waitForURL('**/eda');
    await page.getByRole('tab', { name: /02\s*Distributions/i }).click();
    const edaPlot = page.locator('.js-plotly-plot').first();
    await expect(edaPlot).toBeVisible();

    const plotDataCheck = await edaPlot.evaluate((el) => {
      const gd = el as unknown as {
        data?: { type?: string; x?: number[]; name?: string }[];
        layout?: { xaxis?: { title?: { text?: string } }; yaxis?: { title?: { text?: string } } };
      };
      return {
        traceCount: gd.data?.length ?? 0,
        traceType: gd.data?.[0]?.type ?? '',
        xValues: gd.data?.[0]?.x ?? [],
        xAxisTitle: gd.layout?.xaxis?.title?.text ?? '',
        yAxisTitle: gd.layout?.yaxis?.title?.text ?? '',
      };
    });
    expect(plotDataCheck.traceCount).toBe(1);
    expect(plotDataCheck.traceType).toBe('histogram');
    expect(plotDataCheck.xValues).toEqual([120.5, 240.0, 85.25, 310.75, 415.0]);
    expect(plotDataCheck.xAxisTitle).toBe('monthly_spend (USD)');
    expect(plotDataCheck.yAxisTitle).toBe('Count');

    // 4. AI Analyst: multi-turn query, Plotly canvas trace verification, timeout reconciliation, and reload persistence
    await page.getByRole('link', { name: 'AI Analyst' }).click();
    await page.waitForURL('**/analyst');

    const composer = page.getByLabel('Ask a question about this dataset');
    await composer.fill('Compare average monthly_spend across segment and plot a bar chart.');
    await page.locator('button.send-btn').click();

    await expect(page.locator('.message.assistant .markdown-body').first()).toContainText('415.00');
    const canvasPlot = page.locator('.analysis-canvas .js-plotly-plot').first();
    await expect(canvasPlot).toBeVisible();
    const analystTrace = await canvasPlot.evaluate((el) => {
      const gd = el as unknown as { data?: { type?: string; x?: string[]; y?: number[] }[] };
      return {
        type: gd.data?.[0]?.type,
        x: gd.data?.[0]?.x,
        y: gd.data?.[0]?.y,
      };
    });
    expect(analystTrace.type).toBe('bar');
    expect(analystTrace.x).toEqual(['Enterprise', 'SMB', 'Consumer']);
    expect(analystTrace.y).toEqual([415.0, 215.625, 85.25]);

    // Inspect evidence and verify honest trace provenance badge (no misleading "Verification passed" claim)
    await page.getByRole('button', { name: 'Inspect evidence' }).first().click();
    await expect(page.getByText('Trace checks passed')).toBeVisible();
    await expect(page.getByText('Trace match: 100%')).toBeVisible();

    // Turn 2: simulate HTTP 504 timeout after backend completed request -> verify automatic history reconciliation
    simulateTimeoutOnNextQuery = true;
    await composer.fill('What is the Spearman correlation between monthly_spend and tenure_months?');
    await page.locator('button.send-btn').click();
    await expect(page.locator('.message.assistant .markdown-body').nth(1)).toContainText('0.8500');

    // Reload page and confirm multi-turn history is preserved
    await page.reload();
    await expect(page.locator('.message.assistant .markdown-body')).toHaveCount(2);

    // 5. ML Lab: configure target/task/folds/features, run experiment, verify leaderboard & real result fields
    await page.getByRole('link', { name: 'ML Lab' }).click();
    await page.waitForURL('**/ml');
    await page.locator('select[formcontrolname="target_column"]').selectOption('churn_flag');
    await page.locator('select[formcontrolname="task_type"]').selectOption('classification');
    await page.locator('select[formcontrolname="cv_folds"]').selectOption({ index: 1 });
    await page.getByRole('button', { name: 'Run baseline experiment' }).click();

    await expect(page.locator('.experiment-summary')).toContainText('RandomForest');
    await expect(page.locator('.experiment-summary')).toContainText('4 / 2');
    await expect(page.locator('.profile-table')).toContainText('0.9125');
    await expect(page.locator('.profile-table')).toContainText('HistGradientBoosting');
    await expect(page.getByText(/customer_id was automatically excluded/i)).toBeVisible();

    const mlPlot = page.locator('.js-plotly-plot').first();
    await expect(mlPlot).toBeVisible();
    const mlTrace = await mlPlot.evaluate((el) => {
      const gd = el as unknown as { data?: { type?: string; x?: number[]; y?: string[] }[] };
      return { x: gd.data?.[0]?.x, y: gd.data?.[0]?.y };
    });
    expect(mlTrace.x).toEqual([0.342, 0.218, 0.095]);
    expect(mlTrace.y).toEqual(['monthly_spend', 'tenure_months', 'segment']);

    // 6. Reports & Export: verify all 5 artifact types are available and inspect exported JSON/CSV/HTML contents
    await page.getByRole('link', { name: 'Reports & Export' }).click();
    await page.waitForURL('**/reports');
    await expect(page.getByText('5 of 5 artifact types available')).toBeVisible();

    const activeSid = [...sessions.keys()][0];

    // Export 1: EDA JSON
    const [edaJsonDownload] = await Promise.all([
      page.waitForEvent('download'),
      page.locator('.export-row').nth(0).getByRole('button', { name: 'Download' }).click(),
    ]);
    const edaJson = JSON.parse(await readDownloadText(edaJsonDownload));
    expect(edaJson.session_id).toBe(activeSid);
    expect(edaJson.profile.filename).toBe('custom_b2b_accounts.csv');
    expect(edaJson.profile.rows).toBe(6);

    // Export 3: Conversation & Traces JSON
    const [tracesJsonDownload] = await Promise.all([
      page.waitForEvent('download'),
      page.locator('.export-row').nth(2).getByRole('button', { name: 'Download' }).click(),
    ]);
    const tracesJson = JSON.parse(await readDownloadText(tracesJsonDownload));
    expect(tracesJson.session_id).toBe(activeSid);
    expect(tracesJson.messages.length).toBe(4);

    // Export 4: ML Results JSON
    const [mlJsonDownload] = await Promise.all([
      page.waitForEvent('download'),
      page.locator('.export-row').nth(3).getByRole('button', { name: 'Download' }).click(),
    ]);
    const mlJson = JSON.parse(await readDownloadText(mlJsonDownload));
    expect(mlJson.session_id).toBe(activeSid);
    expect(mlJson.best_model).toBe('RandomForest');

    // Export 5: Interactive Plotly HTML (verify embedded figure JSON, session metadata, CDN disclosure, and offline fallback)
    const [htmlDownload] = await Promise.all([
      page.waitForEvent('download'),
      page.locator('.export-row').nth(4).getByRole('button', { name: 'Download' }).click(),
    ]);
    const htmlContent = await readDownloadText(htmlDownload);
    expect(htmlContent).toContain('https://cdn.plot.ly/plotly-2.35.2.min.js');
    expect(htmlContent).toContain('renderOfflineFallback()');
    expect(htmlContent).toContain('custom_b2b_accounts.csv');
    expect(htmlContent).toContain(activeSid);

    // 7. Sessions: create new session, switch between sessions, clear chat, and delete session
    await page.getByRole('link', { name: 'Sessions' }).click();
    await page.waitForURL('**/sessions');
    await expect(page.locator('.session-item')).toHaveCount(1);

    await page.getByRole('button', { name: 'New session' }).click();
    await page.waitForURL('**/onboarding');
    await page.getByRole('link', { name: 'Sessions' }).click();
    await page.waitForURL('**/sessions');
    await expect(page.locator('.session-item')).toHaveCount(2);

    // Switch back to the first session with data
    await page.getByRole('button', { name: 'Switch to session' }).first().click();
    await page.waitForURL('**/overview');
    await expect(page.locator('.dataset-strip')).toContainText('custom_b2b_accounts.csv');

    // Clear chat on active session via inline two-click confirmation
    await page.getByRole('link', { name: 'Sessions' }).click();
    await page.waitForURL('**/sessions');
    await page.getByRole('button', { name: 'Clear chat' }).click();
    await page.getByRole('button', { name: 'Confirm clear' }).click();
    await expect(page.locator('.session-item.active')).toContainText('0 msgs');

    // Delete the second empty session via inline two-click confirmation
    const nonActiveCard = page.locator('.session-item:not(.active)').first();
    await nonActiveCard.getByRole('button', { name: 'Delete' }).click();
    await nonActiveCard.getByRole('button', { name: 'Confirm delete' }).click();
    await expect(page.locator('.session-item')).toHaveCount(1);

    // 8. Theme switching (Light / Dark) & Responsive layout matrix across all major screens at 1920, 1440, 1280, 834, 390, 360 px
    await page.getByRole('link', { name: 'Settings' }).click();
    await page.waitForURL('**/settings');
    await page.locator('button.theme-btn', { hasText: 'Dark' }).click();
    await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
    await page.locator('button.theme-btn', { hasText: 'Light' }).click();
    await expect(page.locator('html')).toHaveAttribute('data-theme', 'light');

    const screens = ['/overview', '/explorer', '/eda', '/analyst', '/ml', '/reports', '/sessions', '/settings'];
    const widths = [1920, 1440, 1280, 834, 390, 360];

    for (const screen of screens) {
      await page.goto(screen);
      for (const width of widths) {
        await page.setViewportSize({ width, height: 860 });
        await assertNoHorizontalOverflow(page);
      }
    }
  });

  test('displays clear error states when backend is disconnected or returns server errors', async ({
    page,
  }) => {
    await page.route('**/health', async (route: Route) => {
      await route.fulfill({
        status: 503,
        contentType: 'application/json',
        body: JSON.stringify({ detail: 'FastAPI backend temporarily unavailable' }),
      });
    });

    await page.goto('/settings');
    await expect(page.getByText('API offline')).toBeVisible();
    await expect(page.getByText('Not reachable')).toBeVisible();
    await expect(page.getByText('FastAPI backend temporarily unavailable')).toBeVisible();
  });
});
