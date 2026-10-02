import { defineConfig } from '@playwright/test';
export default defineConfig({
  testDir: './e2e', workers: 1, fullyParallel: false, retries: 0,
  timeout: 180000,
  outputDir: '../artifacts/nano_lab/browser_mvp_20261002/ci/browser-tests',
  reporter: [['list'], ['json', { outputFile: '../artifacts/nano_lab/browser_mvp_20261002/ci/playwright.json' }]],
  use: { baseURL: 'http://127.0.0.1:4187', viewport: { width: 1200, height: 960 },
    screenshot: 'only-on-failure', trace: 'retain-on-failure' },
  webServer: { command: 'npm run preview', url: 'http://127.0.0.1:4187', reuseExistingServer: false, timeout: 20000 },
});
