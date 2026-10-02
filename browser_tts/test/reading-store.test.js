import test from 'node:test';
import assert from 'node:assert/strict';
import { ReadingStore, isTemporaryReadingKey } from '../src/reading-store.js';

function fixture(sizes = [3, 70000, 11]) {
  const store = new ReadingStore();
  const bytes = sizes.map((size, index) => Uint8Array.from({ length: size * 2 }, (_, offset) => (index * 17 + offset) % 256));
  store.count = bytes.length; store.frames = sizes.reduce((sum, size) => sum + size, 0);
  store.get = async (index) => new Blob([bytes[index]]);
  return { store, bytes };
}
test('transient cleanup recognizes only app UUID/passage keys, leaving other keys alone', () => {
  const id = '684ecbd6-a6a8-4103-aef4-de0415d628d8';
  assert.equal(isTemporaryReadingKey([id, 0]), true);
  for (const key of ['user-kept-note', ['library-file', 0], [id, -1], [id, 'saved'], [id, 0, 'keep'], null]) {
    assert.equal(isTemporaryReadingKey(key), false);
  }
});
test('long WAV exports use bounded chunks with exact header, ordering, and PCM bytes', async () => {
  const { store, bytes } = fixture(); const output = [];
  for await (const chunk of store.wavChunks()) {
    assert.ok(chunk.byteLength <= 65536); output.push(chunk);
  }
  const all = Buffer.concat(output);
  assert.equal(all.readUInt32LE(40), store.frames * 2);
  assert.equal(all.readUInt32LE(24), 24000);
  assert.deepEqual(all.subarray(44), Buffer.concat(bytes));
  assert.equal(all.length, 44 + store.frames * 2);
});
test('streamed file writes await backpressure and never send whole long passages', async () => {
  const { store, bytes } = fixture(); let pending = false; const output = [];
  await store.writeTo({ async write(data) {
    assert.equal(pending, false); pending = true;
    assert.ok(data.byteLength <= 65536);
    await Promise.resolve(); output.push(data); pending = false;
  } });
  assert.deepEqual(Buffer.concat(output).subarray(44), Buffer.concat(bytes));
});
test('missing, changed, or truncated stored audio fails export instead of declaring success', async () => {
  const a = fixture().store; a.get = async () => null;
  await assert.rejects(a.writeTo({ write() {} }), /missing/);
  const b = fixture().store; b.get = async () => new Blob([new Uint8Array(2)]);
  await assert.rejects(b.writeTo({ write() {} }), /size changed/);
  const c = fixture().store;
  await assert.rejects(c.writeTo({ write() { c.count += 1; } }), /changed during export/);
  await assert.rejects(fixture().store.wavChunks(100000).next(), /65536/);
});
