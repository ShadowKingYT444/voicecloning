import { wavHeader, encodePcmMono16 } from './audio.js';

const requestResult = (request) => new Promise((resolve, reject) => {
  request.onsuccess = () => resolve(request.result);
  request.onerror = () => reject(request.error);
});

export function isTemporaryReadingKey(key) {
  return Array.isArray(key) && key.length === 2
    && typeof key[0] === 'string' && /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(key[0])
    && Number.isSafeInteger(key[1]) && key[1] >= 0;
}
const sessionLock = (id) => `voice-study-reading:${id}`;

export class ReadingStore {
  async open() {
    if (this.db) return;
    if (this.opening) return this.opening;
    this.sessionId ||= crypto.randomUUID();
    this.opening = this.openSession();
    try { await this.opening; } finally { this.opening = null; }
  }
  async openSession() {
    const locks = globalThis.navigator?.locks;
    if (locks) {
      // The browser releases this lease when its document dies. Stale sessions
      // can then be removed without racing a live tab, picker or file writer.
      await new Promise((ready, reject) => {
        this.lease = locks.request(sessionLock(this.sessionId), async () => {
          ready(); await new Promise((release) => { this.releaseLease = release; });
        }).catch(reject);
      });
    }
    const request = indexedDB.open('voice-study-reading', 1);
    request.onupgradeneeded = () => request.result.createObjectStore('passages');
    try {
      this.db = await requestResult(request);
      this.retention = { mode: locks ? 'web-lock-scoped-transient-cleanup' : 'cleanup-skipped-no-web-locks', removedSessions: 0 };
      if (locks) await this.cleanupStaleSessions(locks);
    } catch (error) { this.releaseLease?.(); this.db?.close(); this.db = null; throw error; }
  }
  async cleanupStaleSessions(locks) {
    const sessions = new Set();
    await new Promise((resolve, reject) => {
      const tx = this.db.transaction('passages');
      tx.oncomplete = resolve; tx.onabort = () => reject(tx.error);
      const cursor = tx.objectStore('passages').openKeyCursor();
      cursor.onerror = () => reject(cursor.error);
      cursor.onsuccess = () => {
        const item = cursor.result;
        if (!item) return;
        if (isTemporaryReadingKey(item.key) && item.key[0] !== this.sessionId) sessions.add(item.key[0]);
        item.continue();
      };
    });
    for (const id of sessions) {
      await locks.request(sessionLock(id), { ifAvailable: true }, async (lock) => {
        if (!lock) return;
        await this.transaction('readwrite', (store) => store.delete(IDBKeyRange.bound([id, 0], [id, Number.MAX_SAFE_INTEGER])));
        this.retention.removedSessions += 1;
      });
    }
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
