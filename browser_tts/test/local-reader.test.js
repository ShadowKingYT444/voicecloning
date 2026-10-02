import test from 'node:test';
import assert from 'node:assert/strict';
import { LocalReaderClient, selectReaderBackend } from '../src/local-reader.js';

const cpuStatus = () => Response.json({ schema: 'voice-study.local-reader/v1', backend: 'CPU' });
const scope = () => ({ AudioContext() {}, indexedDB: {}, navigator: {}, WebAssembly: {} });

test('native fetch retains its Window receiver when the client calls it', async () => {
  const original = globalThis.fetch;
  globalThis.fetch = function () { assert.equal(this, globalThis); return Promise.resolve(Response.json({ backend: 'Local CPU' })); };
  try { await new LocalReaderClient().prepare({}); }
  finally { globalThis.fetch = original; }
});

test('CPU service works without WebGPU or JSPI, including refused adapters', async () => {
  for (const gpu of [undefined, { requestAdapter: async () => null }, { requestAdapter: async () => { throw Error('policy'); } }]) {
    const browser = scope(); browser.navigator.gpu = gpu;
    browser.WebAssembly = { Suspending() {}, promising() {} };
    assert.equal(await selectReaderBackend({ scope: browser, fetcher: cpuStatus }), 'cpu');
  }
});
test('the guarded local service is used when browser GPU support is also present', async () => {
  const browser = scope(); browser.WebAssembly = { Suspending() {}, promising() {} };
  browser.navigator.gpu = { requestAdapter() { throw Error('CPU service must be used first'); } };
  assert.equal(await selectReaderBackend({ scope: browser, fetcher: cpuStatus }), 'cpu');
});
test('FP16 WebGPU stays available and missing CPU service produces an actionable error', async () => {
  const browser = scope(); browser.WebAssembly = { Suspending() {}, promising() {} };
  browser.navigator.gpu = { requestAdapter: async () => ({ features: new Set(['shader-f16']) }) };
  assert.equal(await selectReaderBackend({ scope: browser, fetcher: () => { throw Error('must not fetch'); } }), 'webgpu');
  await assert.rejects(selectReaderBackend({ scope: scope(), fetcher: async () => Response.json({}, { status: 404 }) }), /local reader service/);
});
function audioResponse(seed = 42) {
  return new Response(new Float32Array([0, 0.25, 0]).buffer, { headers: {
    'X-Reader-Metadata': JSON.stringify({ sampleRate: 24000, synthesisSeconds: 1, speechTokens: [1], nextSeed: seed }),
  } });
}
test('CPU transport keeps passage order, sampler state and a two-passage queue', async () => {
  const requests = []; const messages = [];
  const client = new LocalReaderClient(async (url, options) => {
    requests.push({ url, body: JSON.parse(options.body) }); return audioResponse();
  });
  client.onmessage = ({ data }) => messages.push(data);
  const reading = client.read({ runId: 1, seed: 1337, chunks: ['First.', 'Second.', 'Third.'] });
  await new Promise((resolve) => setTimeout(resolve, 10));
  assert.equal(requests.length, 2); assert.equal(requests[1].body.seed, 42);
  client.postMessage({ type: 'consumed', runId: 999 }); assert.equal(requests.length, 2);
  client.postMessage({ type: 'consumed', runId: 1 }); await reading;
  assert.deepEqual(messages.filter((m) => m.type === 'chunk').map((m) => m.index), [0, 1, 2]);
  assert.equal(messages.at(-1).type, 'done');
});
test('Stop cancels active service request, drops late audio, then permits restart', async () => {
  let finish; const calls = []; const messages = [];
  const client = new LocalReaderClient(async (url, options) => {
    calls.push({ url, body: JSON.parse(options.body) });
    if (url.endsWith('/stop')) return Response.json({ stopped: true });
    return new Promise((resolve) => { finish = resolve; });
  });
  client.onmessage = ({ data }) => messages.push(data);
  const reading = client.read({ runId: 2, chunks: ['First.'], seed: 1 });
  client.postMessage({ type: 'stop', runId: 2 });
  assert.equal(calls[1].body.requestId, calls[0].body.requestId);
  finish(audioResponse()); await reading;
  assert.deepEqual(messages.map((m) => m.type), ['stopped']);
  const restart = client.read({ runId: 3, chunks: ['Again.'], seed: 1 });
  finish(audioResponse()); await restart;
  assert.deepEqual(messages.map((m) => m.type), ['stopped', 'chunk', 'done']);
});
test('CPU transport rejects browser experiment overrides', async () => {
  const client = new LocalReaderClient(() => { throw Error('must not fetch'); });
  await assert.rejects(client.prepare({ losslessEmbeddingManifestUrl: '/test.json' }), /overrides require WebGPU/);
});
