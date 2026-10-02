import assert from 'node:assert/strict';
import test from 'node:test';
import { wavHeader, encodePcmMono16, fadeEdges } from '../src/audio.js';

test('PCM WAV header describes the actual mono payload', () => {
  const frames = 17;
  const header = wavHeader(frames);
  const view = new DataView(header);
  const ascii = (offset, length) => String.fromCharCode(...new Uint8Array(header, offset, length));
  assert.equal(ascii(0, 4), 'RIFF'); assert.equal(ascii(8, 4), 'WAVE');
  assert.equal(view.getUint32(4, true), 36 + frames * 2);
  assert.equal(view.getUint16(20, true), 1); assert.equal(view.getUint16(22, true), 1);
  assert.equal(view.getUint32(24, true), 24000); assert.equal(view.getUint16(34, true), 16);
  assert.equal(view.getUint32(40, true), frames * 2);
});

test('PCM encoding clips only values outside its representable range', () => {
  const values = [-2, -1, -0.5, 0, 0.5, 1, 2];
  const encoded = new DataView(encodePcmMono16(values));
  assert.deepEqual(values.map((_, i) => encoded.getInt16(i * 2, true)), [-32768, -32768, -16384, 0, 16383, 32767, 32767]);
});

test('edge fades do not change interior samples, including short passages', () => {
  const samples = new Float32Array(1000).fill(0.5);
  assert.equal(fadeEdges(samples), samples);
  assert.equal(samples[0], 0); assert.equal(samples.at(-1), 0); assert.equal(samples[300], 0.5);
  const short = fadeEdges(new Float32Array(10).fill(1));
  assert.ok(short.every(Number.isFinite)); assert.equal(short[0], 0); assert.equal(short.at(-1), 0);
});

test('invalid and oversized WAV lengths fail before allocation', () => {
  for (const count of [0, -1, 1.5, NaN, 0x80000000]) assert.throws(() => wavHeader(count));
});
