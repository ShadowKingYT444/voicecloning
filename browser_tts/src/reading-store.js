import { wavHeader, encodePcmMono16 } from './audio.js';

const requestResult = (request) => new Promise((resolve, reject) => {
  request.onsuccess = () => resolve(request.result);
  request.onerror = () => reject(request.error);
});

export class ReadingStore {
  async open() {
    if (this.db) return;
    this.sessionId ||= crypto.randomUUID();
    const request = indexedDB.open('voice-study-reading', 1);
    request.onupgradeneeded = () => request.result.createObjectStore('passages');
    this.db = await requestResult(request);
  }
  async transaction(mode, operation) {
    await this.open();
    const tx = this.db.transaction('passages', mode);
    const finished = new Promise((resolve, reject) => {
      tx.oncomplete = resolve;
      tx.onabort = () => reject(tx.error || new Error('Audio storage was interrupted.'));
      tx.onerror = () => reject(tx.error);
    });
    const result = operation(tx.objectStore('passages'));
    await finished;
    return result;
  }
  async clear() { await this.transaction('readwrite', (store) => store.delete(IDBKeyRange.bound([this.sessionId, 0], [this.sessionId, Number.MAX_SAFE_INTEGER]))); this.frames = 0; this.count = 0; }
  async append(samples) {
    const pcm = new Blob([encodePcmMono16(samples)]);
    await this.transaction('readwrite', (store) => store.put(pcm, [this.sessionId, this.count]));
    this.frames += samples.length; this.count += 1;
  }
  async get(index) {
    await this.open();
    return requestResult(this.db.transaction('passages').objectStore('passages').get([this.sessionId, index]));
  }
  async writeTo(stream) {
    for await (const bytes of this.wavChunks()) await stream.write(bytes);
  }
  async *wavChunks(maxBytes = 65536) {
    if (!Number.isSafeInteger(maxBytes) || maxBytes < 44 || maxBytes > 65536) {
      throw new Error('WAV export chunks must contain between 44 and 65536 bytes.');
    }
    const count = this.count; const frames = this.frames;
    let pcmBytes = 0;
    yield new Uint8Array(wavHeader(frames));
    for (let index = 0; index < count; index += 1) {
      if (this.count !== count || this.frames !== frames) throw new Error('The reading changed during export.');
      const part = await this.get(index);
      if (!part) throw new Error(`Saved audio passage ${index + 1} is missing.`);
      pcmBytes += part.size;
      for (let offset = 0; offset < part.size; offset += maxBytes) {
        yield new Uint8Array(await part.slice(offset, offset + maxBytes).arrayBuffer());
      }
    }
    if (this.count !== count || this.frames !== frames || pcmBytes !== frames * 2) {
      throw new Error('Saved audio size changed or does not match the WAV header.');
    }
  }
  async blob() {
    // Used only after inference, for browsers without a file stream API.
    const parts = [wavHeader(this.frames)];
    for (let index = 0; index < this.count; index += 1) {
      const part = await this.get(index);
      if (!part) throw new Error(`Saved audio passage ${index + 1} is missing.`);
      parts.push(part);
    }
    return new Blob(parts, { type: 'audio/wav' });
  }
}
