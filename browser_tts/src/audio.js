export function encodeWavMono16(chunks, sampleRate = 24000, fadeMs = 9) {
  if (!chunks.length) throw new Error('No synthesized audio is available.');
  const sampleCount = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  if (!sampleCount || sampleCount > 0x7ffffff0) throw new Error('The generated reading is too large to export as WAV.');
  const bytes = new ArrayBuffer(44 + sampleCount * 2); const view = new DataView(bytes);
  const writeText = (offset, value) => { for (let i = 0; i < value.length; i += 1) view.setUint8(offset + i, value.charCodeAt(i)); };
  writeText(0, 'RIFF'); view.setUint32(4, 36 + sampleCount * 2, true); writeText(8, 'WAVE');
  writeText(12, 'fmt '); view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true); view.setUint32(28, sampleRate * 2, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  writeText(36, 'data'); view.setUint32(40, sampleCount * 2, true);
  const fadeFrames = Math.max(0, Math.round(sampleRate * fadeMs / 1000)); let offset = 44;
  for (const chunk of chunks) for (let index = 0; index < chunk.length; index += 1) {
    const edgeDistance = Math.min(index, chunk.length - index - 1);
    const gain = fadeFrames ? Math.min(1, edgeDistance / fadeFrames) : 1;
    const clipped = Math.max(-1, Math.min(1, chunk[index] * gain));
    view.setInt16(offset, clipped < 0 ? clipped * 32768 : clipped * 32767, true); offset += 2;
  }
  return bytes;
}
