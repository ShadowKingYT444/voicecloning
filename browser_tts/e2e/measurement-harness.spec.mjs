import { test, expect } from '@playwright/test';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';

test('isolated model-free Python harness reports browser PSS and JS heap separately', async ({ playwright }) => {
  const group = await readFile('/proc/self/cgroup', 'utf8');
  if (!group.includes('nano-lab-model')) throw Error('Use the existing process-tree resource guard.');
  const out = resolve('../artifacts/nano_lab/browser_mvp_20261002/ci/harness-ui-only');
  await promisify(execFile)('python3', [
    'scripts/measure-browser.py', '--chrome', playwright.chromium.executablePath(),
    '--ui-only', '--timeout-seconds', '75', '--startup-timeout-seconds', '20',
    '--output-dir', out,
  ], { timeout: 85000, maxBuffer: 128 * 1024 });
  const report = JSON.parse(await readFile(resolve(out, 'measurement.json'), 'utf8'));
  expect(report.status).toBe('complete'); expect(report.mode).toBe('ui-only');
  expect(report.guard.bounded_job_unit_detected).toBe(true);
  const samples = Object.values(report.stages).flatMap((stage) => stage.samples);
  expect(samples.some((sample) => sample.javascriptHeap.observedTargetUsedBytes > 0)).toBe(true);
  expect(samples.every((sample) => sample.hostProcessTree.pssMiB > 0)).toBe(true);
  expect(report.networkRequests.some((request) => /\.onnx|huggingface\.co/.test(request.url))).toBe(false);
});
