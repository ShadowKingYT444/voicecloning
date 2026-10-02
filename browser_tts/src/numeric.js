const floatWord = new Float32Array(1);
const integerWord = new Uint32Array(floatWord.buffer);

export function halfToFloat(value) {
  const sign = (value & 0x8000) ? -1 : 1;
  const exponent = (value >> 10) & 31; const fraction = value & 1023;
  if (exponent === 0) return sign * 2 ** -14 * (fraction / 1024);
  if (exponent === 31) return fraction ? NaN : sign * Infinity;
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

export function floatToHalf(value) {
  floatWord[0] = value;
  const bits = integerWord[0]; const sign = (bits >>> 16) & 0x8000;
  let mantissa = bits & 0x7fffff;
  const rawExponent = (bits >>> 23) & 0xff;
  if (rawExponent === 255) return sign | (mantissa ? 0x7e00 : 0x7c00);
  let exponent = rawExponent - 127 + 15;
  if (exponent >= 31) return sign | 0x7c00;
  if (exponent <= 0) {
    if (exponent < -10) return sign;
    mantissa |= 0x800000;
    const shift = 14 - exponent;
    let rounded = mantissa >>> shift;
    const remainder = mantissa & ((1 << shift) - 1);
    const midpoint = 1 << (shift - 1);
    if (remainder > midpoint || (remainder === midpoint && (rounded & 1))) rounded += 1;
    return sign | rounded;
  }
  let rounded = mantissa >>> 13;
  const remainder = mantissa & 0x1fff;
  if (remainder > 0x1000 || (remainder === 0x1000 && (rounded & 1))) rounded += 1;
  if (rounded === 1024) { rounded = 0; exponent += 1; }
  return sign | (exponent << 10) | rounded;
}

export function values(tensor) {
  const data = tensor.data;
  if (tensor.type === 'float16') return Float32Array.from(data, halfToFloat);
  return Float32Array.from(data);
}
