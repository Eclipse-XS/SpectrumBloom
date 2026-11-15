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
