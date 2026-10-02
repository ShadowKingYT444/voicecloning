import test from 'node:test';
import assert from 'node:assert/strict';
import { checkCapabilities } from '../public/capabilities.js';

function browser({ fallback = false, features = ['shader-f16'], vendor = 'intel', secure = true } = {}) {
  const requests = [];
  const scope = { isSecureContext: secure, WebAssembly: { Suspending() {}, promising() {} },
    AudioContext() {}, indexedDB: {}, navigator: { userAgent: 'test', gpu: {
      async requestAdapter(options) { requests.push(options); return {
        info: { vendor, isFallbackAdapter: fallback }, features: new Set(features),
        limits: { maxBufferSize: 268435456 },
        requestDevice() { throw new Error('No device may be requested by the capability check.'); },
      }; },
    } } };
  return { scope, requests };
}
test('model-free check preserves preference and distinguishes prerequisites from inference', async () => {
  const { scope, requests } = browser();
  const report = await checkCapabilities({ scope, powerPreference: 'high-performance' });
  assert.deepEqual(requests, [{ powerPreference: 'high-performance' }]);
  assert.equal(report.capabilityPrerequisitesPassed, true);
  assert.equal(report.hardwareAdapterVerified, true);
  assert.equal(report.inferenceRun, false);
  assert.equal(report.modelsDownloaded, false);
  assert.equal(report.gpuDeviceRequested, false);
  assert.equal(report.adapter.limits.maxBufferSize, 268435456);
});
test('software, missing features, unknown identity and insecure contexts do not pass hardware/runtime gates', async () => {
  for (const options of [{ fallback: true }, { vendor: 'SwiftShader' }, { vendor: '' }]) {
    assert.equal((await checkCapabilities(browser(options))).hardwareAdapterVerified, false);
  }
  for (const options of [{ features: [] }, { secure: false }]) {
    assert.equal((await checkCapabilities(browser(options))).capabilityPrerequisitesPassed, false);
  }
  const { scope } = browser(); scope.navigator.gpu.requestAdapter = async () => null;
  assert.equal((await checkCapabilities({ scope })).capabilityPrerequisitesPassed, false);
});
test('missing WebGPU and adapter errors produce inspectable reports', async () => {
  assert.equal((await checkCapabilities({ scope: {} })).runtime.webgpu, false);
  const { scope } = browser(); scope.navigator.gpu.requestAdapter = async () => { throw Error('policy'); };
  assert.equal((await checkCapabilities({ scope })).adapterError, 'policy');
  await assert.rejects(checkCapabilities({ powerPreference: 'fast' }), /Select/);
});
