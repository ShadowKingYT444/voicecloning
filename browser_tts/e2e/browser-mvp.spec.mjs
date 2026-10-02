import { test, expect } from '@playwright/test';
import { readFile, writeFile, mkdir, open } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { resolve } from 'node:path';

const out = resolve('../artifacts/nano_lab/browser_mvp_20261002/ci');
const workerSource = await readFile(new URL('./fixture-worker.js', import.meta.url), 'utf8');
test.beforeAll(async () => {
  const cgroup = await readFile('/proc/self/cgroup', 'utf8');
  if (!cgroup.includes('nano-lab-model')) throw Error('Browser checks require the existing systemd process-tree resource guard.');
  await mkdir(out, { recursive: true });
});

async function fixture(page, { failFirstLoad = false } = {}) {
  let workers = 0;
  // These stubs permit control-flow checks only. The preceding capability test
  // runs on the real browser APIs without any stubs or special GPU flags.
  await page.addInitScript(() => {
    Object.defineProperty(navigator, 'gpu', { configurable: true, value: {
      async requestAdapter() { return { features: new Set(['shader-f16']) }; },
    } });
    if (typeof WebAssembly.Suspending !== 'function') WebAssembly.Suspending = function() {};
    if (typeof WebAssembly.promising !== 'function') WebAssembly.promising = function() {};
  });
  const modelRequests = [];
  page.on('request', (request) => { if (/\.onnx|huggingface\.co|\/models\/.+onnx/.test(request.url())) modelRequests.push(request.url()); });
  await page.route('**/assets/worker-*.js', (route) => {
    workers += 1;
    return route.fulfill({ contentType: 'text/javascript', body: `globalThis.FIXTURE_FAIL_LOAD=${failFirstLoad && workers === 1};\n${workerSource}` });
  });
  await page.goto('/');
  await page.evaluate(() => {
    const label = document.createElement('p'); label.textContent = 'MODEL-FREE TEST · Synthetic PCM tones. No speech inference or performance validation.';
    label.style.cssText = 'background:#fff3c8;padding:12px;margin:0;position:relative;z-index:99;text-align:center';
    document.body.prepend(label);
  });
  await expect(page.locator('#load-model')).toBeEnabled();
  return modelRequests;
}

test('actual Chromium capabilities are captured without model requests', async ({ page, browser }) => {
  const requests = [];
  page.on('request', (request) => requests.push(request.url()));
  await page.goto('/capabilities.html');
  await page.getByRole('button', { name: 'Check browser', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Save report' })).toBeEnabled();
  const report = await page.evaluate(() => window.voiceCapabilities.snapshot());
  expect(report.modelsDownloaded).toBe(false); expect(report.gpuDeviceRequested).toBe(false); expect(report.inferenceRun).toBe(false);
  expect(requests.some((url) => /\.onnx|\/voice\/|huggingface|\/models\//.test(url))).toBe(false);
  await writeFile(resolve(out, 'chromium-capabilities.json'), JSON.stringify({ browserVersion: browser.version(), ...report, requests }, null, 2));
  await page.screenshot({ path: resolve(out, 'capabilities-desktop.png'), fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(390);
  await page.screenshot({ path: resolve(out, 'capabilities-mobile.png'), fullPage: true });
  for (const [name, expected] of [
    ['LICENSE', '7d6744b52e52960d42c69b29b55b4b9e0b4656cf52569ab710093f7707f7a067'],
    ['NOTICE.md', '8eea800cd6d2d93c2987953aa4706cd2df42f81161da314f1c2fb478d1783f07'],
  ]) {
    const response = await page.request.get(`/licenses/chatterbox-nano-ONNX/${name}`);
    expect(response.ok()).toBe(true);
    expect(createHash('sha256').update(await response.body()).digest('hex')).toBe(expected);
  }
});

test('model-free reader load failure, retry, stop, restart and export controls', async ({ page }) => {
  const requests = await fixture(page, { failFirstLoad: true });
  await page.locator('#load-model').click();
  await expect(page.locator('#load-model')).toContainText('Retry model load');
  expect(await page.evaluate(() => window.voiceStudy.snapshot().modelReady)).toBe(false);
  await page.locator('#load-model').click();
  await expect(page.locator('#load-model')).toContainText('Voice prepared');
  await page.locator('#passage').fill(Array(140).fill('Synthetic fixture text.').join(' '));
  await page.locator('#read-button').click();
  await expect.poll(() => page.evaluate(() => window.voiceStudy.snapshot().chunks.length)).toBeGreaterThan(0);
  await page.locator('#stop-button').click();
  await expect.poll(() => page.evaluate(() => window.voiceStudy.snapshot().readingOutcome)).toBe('stopped');
  expect(await page.evaluate(() => window.voiceStudy.snapshot().metrics.scheduledQueueSize)).toBe(0);
  const short = await page.evaluate(() => window.voiceStudy.readText('A synthetic tone checks the playback controls.', { seed: 1337 }));
  expect(short.readingOutcome).toBe('complete'); expect(short.metrics.maximumScheduledQueueSize).toBeLessThanOrEqual(2);
  expect(short.outputContract.releaseReady).toBe(false);
  expect(short.outputContract.applicationLayerWatermarkApplied).toBe(false);
  await page.evaluate(() => window.voiceStudy.openWavExport());
  await expect(page.locator('#read-button')).toBeDisabled();
  expect(await page.evaluate(async () => {
    try { await window.voiceStudy.readText('Blocked during export.'); return false; }
    catch (error) { return /export/.test(error.message); }
  })).toBe(true);
  await page.evaluate(() => window.voiceStudy.cancelWavExport());
  await expect(page.locator('#read-button')).toBeEnabled();
  expect(requests).toEqual([]);
  await page.screenshot({ path: resolve(out, 'reader-controls-desktop.png'), fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(390);
  await page.screenshot({ path: resolve(out, 'reader-controls-mobile.png'), fullPage: true });
});

test('visible Save holds the reading lock through picker, backpressure and failure', async ({ page }) => {
  await fixture(page);
  await page.evaluate(() => window.voiceStudy.prepareVoice());
  await page.evaluate(() => window.voiceStudy.readText('A synthetic export concurrency check.', { seed: 1337 }));
  await page.evaluate(() => {
    window.saveFixture = { calls: 0, writes: [], closed: false, aborted: false };
    window.showSaveFilePicker = () => {
      window.saveFixture.calls += 1;
      return new Promise((resolve, reject) => {
        window.saveFixture.cancelPicker = () => reject(new DOMException('Cancelled', 'AbortError'));
        window.saveFixture.acceptPicker = () => resolve({ async createWritable() {
          return {
            async write(bytes) {
              window.saveFixture.writes.push(Array.from(bytes));
              if (window.saveFixture.failWrite) throw Error('Synthetic disk failure');
              if (window.saveFixture.writes.length === 1) await new Promise((done) => { window.saveFixture.releaseWrite = done; });
            },
            async close() { window.saveFixture.closed = true; },
            async abort() { window.saveFixture.aborted = true; },
          };
        } });
      });
    };
  });
  const blocked = () => page.evaluate(async () => {
    try { await window.voiceStudy.readText('Conflicting reading.'); return false; }
    catch (error) { return /export/.test(error.message); }
  });
  await page.locator('#download-reading').click();
  await expect(page.locator('#read-button')).toBeDisabled(); expect(await blocked()).toBe(true);
  await page.evaluate(() => window.saveFixture.cancelPicker());
  await expect(page.locator('#read-button')).toBeEnabled();

  await page.locator('#download-reading').click();
  await page.evaluate(() => window.saveFixture.acceptPicker());
  await expect.poll(() => page.evaluate(() => window.saveFixture.writes.length)).toBe(1);
  expect(await blocked()).toBe(true);
  await page.evaluate(() => window.voiceStudy.cancelWavExport()); // Must not unlock a visible file export.
  await expect(page.locator('#read-button')).toBeDisabled();
  await page.evaluate(() => window.saveFixture.releaseWrite());
  await expect(page.locator('#read-button')).toBeEnabled();
  const saved = await page.evaluate(() => window.saveFixture);
  expect(saved.calls).toBe(2); expect(saved.closed).toBe(true); expect(saved.aborted).toBe(false);
  const bytes = Buffer.concat(saved.writes.map((part) => Buffer.from(part)));
  expect(bytes.readUInt32LE(40)).toBe(bytes.length - 44);

  await page.evaluate(() => { window.saveFixture.failWrite = true; });
  await page.locator('#download-reading').click();
  await page.evaluate(() => window.saveFixture.acceptPicker());
  await expect(page.locator('#read-button')).toBeEnabled();
  expect(await page.evaluate(() => window.saveFixture.aborted)).toBe(true);
  await expect(page.locator('#model-status-text')).toContainText('Audio export failed');
});

test('reload cleanup removes abandoned transient audio and preserves live tabs and unrelated keys', async ({ page, context }) => {
  await fixture(page); await page.evaluate(() => window.voiceStudy.prepareVoice());
  await page.evaluate(() => window.voiceStudy.readText('First temporary fixture recording.'));
  const keys = (target) => target.evaluate(async () => {
    const db = await new Promise((resolve, reject) => {
      const request = indexedDB.open('voice-study-reading', 1);
      request.onsuccess = () => resolve(request.result); request.onerror = () => reject(request.error);
    });
    try { return await new Promise((resolve, reject) => {
      const request = db.transaction('passages').objectStore('passages').getAllKeys();
      request.onsuccess = () => resolve(request.result); request.onerror = () => reject(request.error);
    }); } finally { db.close(); }
  });
  const oldSession = (await keys(page))[0][0];
  await page.evaluate(async () => {
    const db = await new Promise((resolve) => { const request = indexedDB.open('voice-study-reading', 1); request.onsuccess = () => resolve(request.result); });
    await new Promise((resolve, reject) => {
      const tx = db.transaction('passages', 'readwrite'); tx.oncomplete = resolve; tx.onabort = () => reject(tx.error);
      tx.objectStore('passages').put(new Blob(['kept note']), 'user-kept-note');
      tx.objectStore('passages').put(new Blob(['library item']), ['library-file', 0]);
    }); db.close();
  });
  const second = await context.newPage();
  await fixture(second); await second.evaluate(() => window.voiceStudy.prepareVoice());
  await second.evaluate(() => window.voiceStudy.readText('Live second tab fixture recording.'));
  const liveKeys = await keys(second);
  expect(liveKeys.some((key) => Array.isArray(key) && key[0] === oldSession)).toBe(true);
  const otherSession = liveKeys.find((key) => Array.isArray(key) && key[0] !== oldSession && key[0] !== 'library-file')[0];
  await page.reload(); await page.evaluate(() => window.voiceStudy.prepareVoice());
  await page.evaluate(() => window.voiceStudy.readText('New recording after reload.'));
  const cleaned = await keys(page);
  expect(cleaned.some((key) => Array.isArray(key) && key[0] === oldSession)).toBe(false);
  expect(cleaned.some((key) => Array.isArray(key) && key[0] === otherSession)).toBe(true);
  expect(cleaned).toContainEqual('user-kept-note'); expect(cleaned).toContainEqual(['library-file', 0]);
  expect(await page.evaluate(() => window.voiceStudy.snapshot().storageRetention.removedSessions)).toBeGreaterThan(0);
  await second.close();
});

test('full-paper synthetic playback keeps two credits and streams more than 60 seconds to disk', async ({ page }) => {
  const requests = await fixture(page);
  await page.locator('#load-model').click();
  await expect(page.locator('#load-model')).toContainText('Voice prepared');
  const snapshot = await page.evaluate(async () => {
    const text = await (await fetch('/federalist-no-10.txt')).text();
    return window.voiceStudy.readText(text, { chunkWords: 15, seed: 1337 });
  });
  expect(snapshot.readingOutcome).toBe('complete'); expect(snapshot.metrics.audioSeconds).toBeGreaterThan(60);
  expect(snapshot.metrics.maximumScheduledQueueSize).toBeLessThanOrEqual(2);
  expect(snapshot.metrics.scheduledQueueSize).toBe(0);
  const info = await page.evaluate(() => window.voiceStudy.openWavExport());
  const file = await open(resolve(out, 'synthetic-full-paper.wav'), 'w');
  const digest = createHash('sha256'); let written = 0; let maximum = 0;
  try {
    while (true) {
      const row = await page.evaluate(() => window.voiceStudy.readWavExport());
      expect(row.offsetBytes).toBe(written);
      if (row.done) break;
      const bytes = Buffer.from(row.base64, 'base64');
      expect(bytes.length).toBeLessThanOrEqual(65536);
      await file.write(bytes); digest.update(bytes); written += bytes.length; maximum = Math.max(maximum, bytes.length);
    }
  } finally { await file.close(); }
  expect(written).toBe(info.bytes); expect(await page.evaluate(() => window.voiceStudy.snapshot().exporting)).toBe(false);
  expect(requests).toEqual([]);
  await writeFile(resolve(out, 'synthetic-controls.json'), JSON.stringify({
    kind: 'model-free-control-flow-only', inferenceRun: false, speechQualityValidated: false,
    memoryTargetValidated: false, latencyTargetValidated: false,
    note: 'Synthetic tones replace the worker. Metrics below describe fixture scheduling, not TTS inference.',
    snapshot, export: { ...info, writtenBytes: written, maximumTransferBytes: maximum, sha256: digest.digest('hex') },
  }, null, 2));
});
