import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: 'tests/e2e',
  timeout: 60_000,
  use: {
    baseURL: 'http://localhost:4173',
    // Set PW_CHROME to reuse an installed Chromium instead of `npx playwright install`.
    launchOptions: process.env.PW_CHROME ? { executablePath: process.env.PW_CHROME } : {},
    permissions: ['clipboard-read', 'clipboard-write'],
  },
  webServer: {
    command: 'npm run build -w apps/web && npm run preview -w apps/web -- --port 4173 --strictPort',
    url: 'http://localhost:4173',
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
});
