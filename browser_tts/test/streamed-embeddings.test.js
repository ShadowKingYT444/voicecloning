import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { StreamedEmbeddings, LOSSLESS_MANIFEST_SHA256 } from '../src/streamed-embeddings.js';
import { halfToFloat } from '../src/numeric.js';

const digest = (data) => createHash('sha256').update(data).digest('hex');
function fixture() {
  const files = new Map();
  const manifest = { tables: { text: { rows: 128, shards: [] }, speech: { rows: 6563, shards: [] } } };
  for (const [table, starts] of [['text', [0, 64]], ['speech', [0, 6528]]]) {
    for (const first of starts) {
      const rows = Math.min(64, manifest.tables[table].rows - first);
      const bytes = new Uint8Array(rows * 768 * 2); const view = new DataView(bytes.buffer);
      for (let row = 0; row < rows; row += 1) for (let col = 0; col < 768; col += 1) {
        // Include signed zero/subnormal/normal values without NaN/Infinity.
        view.setUint16((row * 768 + col) * 2, ((first + row + col) % 0x7c00) | (col % 2 ? 0x8000 : 0), true);
      }
      const file = `${table}-${first}.fp16`; files.set(file, bytes);
      manifest.tables[table].shards[Math.floor(first / 64)] = { file, first_row: first, rows, bytes: bytes.length, sha256: digest(bytes) };
    }
  }
  const fetched = [];
  const fetchImpl = async (url) => {
    const name = new URL(url).pathname.split('/').at(-1); fetched.push(name);
    return { ok: true, blob: async () => new Blob([files.get(name)]) };
  };
  return { manifest, files, fetched, fetchImpl };
}
test('lossless lookup preserves the hybrid text/speech tail and exact FP16-to-FP32 bits', async () => {
  const f = fixture(); const lookup = new StreamedEmbeddings(f.manifest, 'https://app.test/shards/manifest.json', f);
  const output = await lookup.lookup([0, 64, 50256, 50256]);
  const expected = new Float32Array(4 * 768);
  for (const [row, token] of [0, 64, 6561, 6561].entries()) {
    for (let col = 0; col < 768; col += 1) expected[row * 768 + col] = halfToFloat(((token + col) % 0x7c00) | (col % 2 ? 0x8000 : 0));
  }
  assert.deepEqual(new Uint8Array(output.buffer), new Uint8Array(expected.buffer));
  assert.deepEqual(f.fetched, ['text-0.fp16', 'text-64.fp16', 'speech-6528.fp16']);
  assert.equal(lookup.snapshot().cacheHits, 1);
  assert.equal((await lookup.lookup([0])).length, 768); // Single ID is a speech lookup, as in ONNX Slice(-2).
});
test('cache stays within its fixed budget and evicts before loading another shard', async () => {
  const f = fixture(); const lookup = new StreamedEmbeddings(f.manifest, 'https://app.test/shards/manifest.json', { ...f, cacheBytes: 98304 });
  await lookup.lookup([0, 64, 50256, 50256]);
  await lookup.lookup([0, 50256, 50256]);
  assert.ok(lookup.snapshot().cachePeakBytes <= 98304);
  assert.ok(lookup.snapshot().shardRequests > 3);
  lookup.clear(); assert.equal(lookup.snapshot().cacheBytes, 0);
});
test('invalid IDs fail before shard fetch; corrupted bytes never enter the cache', async () => {
  const f = fixture(); const lookup = new StreamedEmbeddings(f.manifest, 'https://app.test/shards/manifest.json', f);
  for (const ids of [[], [-1], [6563], [1.2], [NaN], [129, 50256, 50256]]) await assert.rejects(lookup.lookup(ids));
  assert.equal(f.fetched.length, 0);
  f.files.get('speech-0.fp16')[0] ^= 1;
  await assert.rejects(lookup.lookup([0]), /hash check/);
  assert.equal(lookup.snapshot().cacheBytes, 0);
});
test('manifest is independently pinned and rejects changed bytes and cross-origin URLs', async () => {
  const location = { href: 'https://app.test/', origin: 'https://app.test' };
  const data = await readFile(new URL('../../artifacts/nano_lab/browser_mvp_20261002/lossless-manifest.json', import.meta.url));
  assert.equal(digest(data), LOSSLESS_MANIFEST_SHA256);
  const fetchImpl = async () => ({ ok: true, blob: async () => new Blob([data]) });
  const loaded = await StreamedEmbeddings.load('/shards/manifest.json', { location, fetchImpl });
  assert.equal(loaded.snapshot().shardRequests, 0);
  await assert.rejects(StreamedEmbeddings.load('https://elsewhere.test/a.json', { location, fetchImpl }), /app origin/);
  await assert.rejects(StreamedEmbeddings.load('/a.json', { location,
    fetchImpl: async () => ({ ok: true, blob: async () => new Blob([data, ' ']) }) }), /hash\/size/);
});
