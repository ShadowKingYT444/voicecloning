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
    await stream.write(wavHeader(this.frames));
    for (let index = 0; index < this.count; index += 1) {
      const part = await this.get(index);
      if (!part) throw new Error(`Saved audio passage ${index + 1} is missing.`);
      await stream.write(part);
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
