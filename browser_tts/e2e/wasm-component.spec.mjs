import { test, expect } from '@playwright/test';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { resolve } from 'node:path';

test('real browser WASM embedding outputs match pinned lossless rows without WebGPU', async ({ page, browser }) => {
  const group = await readFile('/proc/self/cgroup', 'utf8');
  if (!group.includes('nano-lab-model')) throw Error('Use the existing process-tree resource guard.');
  const out = resolve('../artifacts/nano_lab/browser_mvp_20261002/ci'); await mkdir(out, { recursive: true });
  const requests = []; page.on('request', (request) => requests.push(request.url()));
  await page.goto('/wasm-component.html');
  let result;
  try { result = await page.evaluate(() => window.voiceWasmComponent.run()); }
  finally {
    const snapshot = await page.evaluate(() => window.voiceWasmComponent.snapshot());
    await writeFile(resolve(out, 'wasm-embedding.json'), JSON.stringify({ browserVersion: browser.version(), ...snapshot, requests }, null, 2));
    await page.screenshot({ path: resolve(out, 'wasm-embedding.png'), fullPage: true });
  }
  expect(result.status).toBe('passed'); expect(result.cases).toHaveLength(3);
  expect(result.fullNanoInferenceRun).toBe(false); expect(result.speechGenerated).toBe(false);
  expect(requests.some((url) => /language_model|conditional_decoder|speech_encoder|\/voice\//.test(url))).toBe(false);
});
