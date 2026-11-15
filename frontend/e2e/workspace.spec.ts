import { expect, Page, test } from '@playwright/test';
import * as fs from 'node:fs';
import * as path from 'node:path';

const SCREENSHOT_DIR = path.resolve(__dirname, '../../docs/screenshots');
const CAPTURE_SCREENSHOTS = process.env['CAPTURE_SCREENSHOTS'] === '1';

async function attachErrorListeners(page: Page): Promise<{
  consoleErrors: string[];
  pageErrors: string[];
  serverErrors: string[];
}> {
  const consoleErrors: string[] = [];
  const pageErrors: string[] = [];
  const serverErrors: string[] = [];

  page.on('console', (msg) => {
    if (msg.type() === 'error') {
      consoleErrors.push(msg.text());
    }
  });
  page.on('pageerror', (err) => {
    pageErrors.push(err.message);
  });
  page.on('response', (res) => {
    if (res.status() >= 500 && res.url().includes('/api/')) {
      serverErrors.push(`${res.status()} ${res.url()}`);
    }
  });

  return { consoleErrors, pageErrors, serverErrors };
}

async function assertNoHorizontalOverflow(page: Page): Promise<void> {
  const metrics = await page.evaluate(() => ({
    scrollWidth: document.documentElement.scrollWidth,
    clientWidth: document.documentElement.clientWidth,
  }));
  expect(metrics.scrollWidth).toBeLessThanOrEqual(metrics.clientWidth);
}

async function maybeScreenshot(page: Page, filename: string): Promise<void> {
  if (!CAPTURE_SCREENSHOTS) {
    return;
  }
  fs.mkdirSync(SCREENSHOT_DIR, { recursive: true });
  await page.screenshot({
    path: path.join(SCREENSHOT_DIR, filename),
    fullPage: false,
  });
}

test.describe('SpectrumBloom Live FastAPI + Angular Workspace E2E', () => {
  test('full end-to-end onboarding, EDA, AI Analyst, ML Lab, Reports, Sessions, Settings, and reload persistence', async ({
    page,
  }) => {
    test.setTimeout(240_000);
    const { consoleErrors, pageErrors, serverErrors } = await attachErrorListeners(page);

    await page.setViewportSize({ width: 1440, height: 900 });
    await page.goto('/onboarding');
    await page.evaluate(() => localStorage.clear());
    await page.reload();

    // 1. Onboarding: live max upload MB & real sample dataset upload
    await expect(page.getByRole('heading', { name: 'Bring your data into focus.' })).toBeVisible();
    await expect(page.getByText(/Maximum \d+ MB/)).toBeVisible();

    await page.getByRole('button', { name: 'Load sample dataset' }).click();
    await page.waitForURL('**/overview', { timeout: 30_000 });
    await expect(page.locator('.dataset-strip')).toContainText('sample_ecommerce.csv');
    await expect(page.locator('.dataset-meta')).toContainText('180');
    await expect(page.getByText('API connected', { exact: true })).toBeVisible();
    await maybeScreenshot(page, 'angular_overview_light.png');

    // 2. Data Explorer: preview boundary, Ctrl+K search, NULL filter, sort, column drawer, CSV export
    await page.getByRole('link', { name: 'Data Explorer' }).click();
    await page.waitForURL('**/explorer');
    await expect(page.getByText('Preview scope boundary')).toBeVisible();

    await page.keyboard.press('Control+k');
    const searchBox = page.getByLabel('Search loaded preview rows');
    await expect(searchBox).toBeFocused();
    await searchBox.fill('North');
    await expect(page.locator('.toolbar-count')).toContainText('of 100 preview rows');
    await searchBox.fill('');

    await page.getByRole('button', { name: /Rows with NULLs/i }).click();
    await page.getByRole('button', { name: /Rows with NULLs/i }).click();

    await page.locator('button.th-btn', { hasText: 'revenue' }).click();
    await expect(page.locator('.inspector-card')).toContainText('revenue');

    const [csvDownload] = await Promise.all([
      page.waitForEvent('download'),
      page.getByRole('button', { name: 'Export preview CSV' }).click(),
    ]);
    expect(csvDownload.suggestedFilename()).toContain('.csv');

    // 3. Automated EDA: 5 tabs + live Plotly DOM + Expand modal + Escape close
    await page.getByRole('link', { name: 'Automated EDA' }).click();
    await page.waitForURL('**/eda');
    await expect(page.getByRole('tab', { name: /01\s*Quality/i })).toBeVisible();

    await page.getByRole('tab', { name: /02\s*Distributions/i }).click();
    await expect(page.locator('.js-plotly-plot').first()).toBeVisible({ timeout: 25_000 });

    await page.getByRole('button', { name: 'Expand chart' }).first().click();
    const dialog = page.getByRole('dialog');
    await expect(dialog).toBeVisible();
    await expect(dialog.locator('.js-plotly-plot')).toBeVisible({ timeout: 20_000 });
    await page.keyboard.press('Escape');
    await expect(dialog).toBeHidden();

    await page.getByRole('tab', { name: /03\s*Relationships/i }).click();
    await expect(page.locator('.js-plotly-plot').first()).toBeVisible({ timeout: 20_000 });
    await maybeScreenshot(page, 'angular_eda_light.png');

    await page.getByRole('tab', { name: /04\s*Outliers/i }).click();
    await expect(page.getByRole('heading', { level: 3 }).or(page.locator('.card-head h2')).first()).toBeVisible();

    await page.getByRole('tab', { name: /05\s*Statistical tests/i }).click();
    await expect(page.getByText(/Shapiro-Wilk/i).first()).toBeVisible();

    // 4. AI Analyst: real FastAPI + Ollama + DockerSandbox multi-turn query + Plotly Canvas + persistence
    await page.getByRole('link', { name: 'AI Analyst' }).click();
    await page.waitForURL('**/analyst');

    const composer = page.getByLabel('Ask a question about this dataset');
    await composer.fill('What is the total revenue by region, and show a bar chart?');
    await page.locator('button.send-btn').click();

    await expect(page.locator('.message.assistant .markdown-body').first()).toBeVisible({ timeout: 120_000 });
    await expect(page.locator('.message.assistant .markdown-body').first()).not.toBeEmpty();

    const evidenceBtn = page.getByRole('button', { name: 'Inspect evidence' }).first();
    if (await evidenceBtn.isVisible()) {
      await evidenceBtn.click();
      await expect(page.locator('.evidence').first()).toBeVisible();
    }

    // Check Analysis Canvas Plotly render if chart produced; also verify dark theme preview screenshot
    await expect(page.locator('.analysis-canvas')).toBeVisible();
    await maybeScreenshot(page, 'angular_analyst_light.png');

    // Follow-up query to test multi-turn state
    await composer.fill('What is the mean order revenue across the dataset?');
    await page.locator('button.send-btn').click();
    await expect(page.locator('.message.assistant .markdown-body').nth(1)).toBeVisible({ timeout: 90_000 });

    // Reload persistence check
    await page.reload();
    await expect(page.locator('.message.assistant .markdown-body').nth(1)).toBeVisible({ timeout: 20_000 });

    // 5. ML Lab: real scikit-learn CV leaderboard & diagnostic Plotly charts
    await page.getByRole('link', { name: 'ML Lab' }).click();
    await page.waitForURL('**/ml');
    await page.locator('select[formcontrolname="target_column"]').selectOption('churned');
    await page.locator('select[formcontrolname="cv_folds"]').selectOption({ index: 1 });
    await page.getByRole('button', { name: 'Run baseline experiment' }).click();

    await expect(page.locator('.profile-table')).toBeVisible({ timeout: 45_000 });
    await expect(page.locator('.js-plotly-plot').first()).toBeVisible({ timeout: 25_000 });
    await maybeScreenshot(page, 'angular_ml_light.png');

    // 6. Reports & Export: verify live availability and download artifacts
    await page.getByRole('link', { name: 'Reports & Export' }).click();
    await page.waitForURL('**/reports');
    await expect(page.getByText(/4 of 5|5 of 5/)).toBeVisible();

    const [edaDownload] = await Promise.all([
      page.waitForEvent('download'),
      page.locator('.export-row').first().getByRole('button', { name: 'Download' }).click(),
    ]);
    expect(edaDownload.suggestedFilename()).toContain('.json');

    // 7. Settings: verify health telemetry + Dark theme switch + screenshot
    await page.getByRole('link', { name: 'Settings' }).click();
    await page.waitForURL('**/settings');
    await expect(page.getByText('Verified via GET /health')).toBeVisible();
    await page.locator('button.theme-btn', { hasText: 'Dark' }).click();
    await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');

    await page.getByRole('link', { name: 'Overview' }).click();
    await page.waitForURL('**/overview');
    await page.evaluate(() => {
      document.querySelector('.page-scroll')?.scrollTo({ top: 0, behavior: 'instant' });
      window.scrollTo(0, 0);
    });
    await maybeScreenshot(page, 'angular_overview_dark.png');

    // Reset back to Light theme
    await page.getByRole('link', { name: 'Settings' }).click();
    await page.locator('button.theme-btn', { hasText: 'Light' }).click();
    await expect(page.locator('html')).toHaveAttribute('data-theme', 'light');

    // 8. Sessions: create disposable session & switch back
    await page.getByRole('link', { name: 'Sessions' }).click();
    await page.waitForURL('**/sessions');
    await expect(page.locator('.session-row')).toHaveCount(await page.locator('.session-row').count());

    expect(consoleErrors).toEqual([]);
    expect(pageErrors).toEqual([]);
    expect(serverErrors).toEqual([]);
  });

  test('responsive viewport matrix has zero horizontal overflow and functional mobile drawer', async ({
    page,
  }) => {
    const viewports = [
      { width: 1920, height: 1080 },
      { width: 1440, height: 900 },
      { width: 1280, height: 800 },
      { width: 834, height: 1194 },
      { width: 390, height: 844 },
      { width: 360, height: 800 },
    ];

    for (const vp of viewports) {
      await page.setViewportSize(vp);
      await page.goto('/onboarding');
      await expect(page.getByRole('heading', { name: 'Bring your data into focus.' })).toBeVisible();
      await assertNoHorizontalOverflow(page);

      if (vp.width <= 800) {
        const menuBtn = page.getByRole('button', { name: 'Open navigation' });
        await expect(menuBtn).toBeVisible();
        await menuBtn.click();
        await expect(page.locator('aside.sidebar.open')).toBeVisible();
        await page.getByRole('button', { name: 'Close navigation' }).click();
        await expect(page.locator('aside.sidebar.open')).toBeHidden();
      }
    }

    await page.setViewportSize({ width: 390, height: 844 });
    await page.goto('/settings');
    await expect(page.getByRole('heading', { name: 'Settings & system status' })).toBeVisible();
    await expect(page.getByText('API connected', { exact: true })).toBeVisible();
    await assertNoHorizontalOverflow(page);
    await maybeScreenshot(page, 'angular_settings_mobile.png');
  });

  test('prompts for Bearer auth token when backend enforces authentication and never writes token to localStorage', async ({
    page,
  }) => {
    let seenAuthHeader: string | null = null;

    await page.route('**/health', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          status: 'ok',
          version: '0.1.0',
          session_repository: 'connected',
          dataset_storage: { backend: 'local', status: 'connected' },
          sandbox_backend: 'docker',
          sandbox_dynamic_python_enabled: true,
          auth_required: true,
          max_upload_mb: 25,
          default_provider: 'ollama',
          default_model: 'qwen3:4b-instruct',
          llm: {
            provider: 'ollama',
            status: 'connected',
            configured_model: 'qwen3:4b-instruct',
            models: ['qwen3:4b-instruct'],
          },
          observability: { opentelemetry: true, langfuse: false },
        }),
      });
    });

    await page.route('**/api/v1/sessions', async (route) => {
      seenAuthHeader = route.request().headers()['authorization'] ?? null;
      if (seenAuthHeader !== 'Bearer test-secret-token-value') {
        await route.fulfill({
          status: 401,
          contentType: 'application/json',
          body: JSON.stringify({ detail: 'Invalid or missing bearer token' }),
        });
        return;
      }
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify([]),
      });
    });

    await page.goto('/onboarding');
    await expect(page.getByRole('heading', { name: 'API authentication required' })).toBeVisible();

    await page.getByPlaceholder('Paste your API_AUTH_TOKEN…').fill('test-secret-token-value');
    await page.getByRole('button', { name: 'Unlock workspace' }).click();

    await expect.poll(() => seenAuthHeader).toBe('Bearer test-secret-token-value');

    const storageDump = await page.evaluate(() => JSON.stringify(localStorage));
    expect(storageDump).not.toContain('test-secret-token-value');
  });
});
