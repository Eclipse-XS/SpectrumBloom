import { defineConfig } from '@playwright/test';

const baseURL = process.env['PLAYWRIGHT_BASE_URL'] || 'http://localhost:4200';
const browserChannel = process.env['PLAYWRIGHT_CHANNEL'] || (process.platform === 'win32' ? 'chrome' : undefined);

export default defineConfig({
  testDir: './e2e',
  timeout: 90_000,
  expect: {
    timeout: 15_000,
  },
  fullyParallel: false,
  workers: 1,
  retries: process.env['CI'] ? 1 : 0,
  reporter: [['list']],
  webServer: process.env['PLAYWRIGHT_SKIP_WEBSERVER'] === '1'
    ? undefined
    : {
        command: 'npx ng serve --port 4200 --proxy-config proxy.conf.json',
        url: baseURL,
        reuseExistingServer: true,
        timeout: 120_000,
      },
  use: {
    baseURL,
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    ...(browserChannel ? { channel: browserChannel } : {}),
    launchOptions: {
      args: ['--no-sandbox', '--disable-dev-shm-usage'],
    },
  },
});
