export function fadeEdges(samples, sampleRate = 24000, fadeMs = 9) {
  const fadeFrames = Math.round(sampleRate * fadeMs / 1000);
  for (let index = 0; index < samples.length; index += 1) {
    const edge = Math.min(index, samples.length - index - 1);
    samples[index] *= fadeFrames ? Math.min(1, edge / fadeFrames) : 1;
  }
  return samples;
}

export function wavHeader(sampleCount, sampleRate = 24000) {
  if (!Number.isSafeInteger(sampleCount) || sampleCount <= 0 || sampleCount * 2 > 0xffffffff - 36) {
    throw new Error('The generated reading cannot be exported as a PCM WAV.');
  }
  const bytes = new ArrayBuffer(44); const view = new DataView(bytes);
  const text = (offset, value) => { for (let i = 0; i < value.length; i += 1) view.setUint8(offset + i, value.charCodeAt(i)); };
  text(0, 'RIFF'); view.setUint32(4, 36 + sampleCount * 2, true); text(8, 'WAVE'); text(12, 'fmt ');
  view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true); view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  text(36, 'data'); view.setUint32(40, sampleCount * 2, true);
  return bytes;
}

export function encodePcmMono16(samples) {
  const bytes = new ArrayBuffer(samples.length * 2); const view = new DataView(bytes);
  for (let index = 0; index < samples.length; index += 1) {
    const value = Math.max(-1, Math.min(1, samples[index]));
    view.setInt16(index * 2, value < 0 ? value * 32768 : value * 32767, true);
  }
  return bytes;
}
