import assert from 'node:assert/strict';
import test from 'node:test';
import { floatToHalf, halfToFloat } from '../src/numeric.js';

test('all finite half values and infinities survive a float32 round trip', () => {
  for (let bits = 0; bits <= 0xffff; bits += 1) {
    if ((bits & 0x7c00) === 0x7c00 && (bits & 0x03ff)) continue;
    assert.equal(floatToHalf(halfToFloat(bits)), bits, `half bits ${bits.toString(16)}`);
  }
});

test('rounding carries into the next exponent and uses ties to even', () => {
  assert.equal(floatToHalf(1.9999), 0x4000);
  assert.equal(floatToHalf(-1.9999), 0xc000);
  assert.equal(floatToHalf(1 + 2 ** -11), 0x3c00);
  assert.equal(floatToHalf(1 + 3 * 2 ** -11), 0x3c02);
  assert.equal(floatToHalf(2 ** -25), 0);
  assert.equal(floatToHalf(3 * 2 ** -25), 2);
  assert.equal(floatToHalf(65520), 0x7c00);
  assert.ok(Number.isNaN(halfToFloat(floatToHalf(NaN))));
});
