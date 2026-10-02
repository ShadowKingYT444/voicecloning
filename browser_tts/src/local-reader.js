// The local service runs the same pinned voice on CPU. No browser GPU is needed.
export async function localReaderStatus(fetcher = fetch) {
  try {
    const response = await fetcher('/api/reader/status');
    if (!response.ok) return null;
    const value = await response.json();
    return value.schema === 'voice-study.local-reader/v1' && value.backend === 'CPU' ? value : null;
  } catch { return null; }
}

export async function selectReaderBackend({ scope = globalThis, fetcher = fetch } = {}) {
  if (!scope.AudioContext || !scope.indexedDB) throw new Error('This browser needs audio playback and local storage support.');
  if (await localReaderStatus(fetcher)) return 'cpu';
  try {
    if (scope.navigator?.gpu && typeof scope.WebAssembly?.Suspending === 'function'
        && typeof scope.WebAssembly?.promising === 'function') {
      const adapter = await scope.navigator.gpu.requestAdapter({ powerPreference: 'low-power' });
      if (adapter?.features.has('shader-f16')) return 'webgpu';
    }
  } catch { /* A browser GPU policy can refuse the adapter. Try the local service. */ }
  throw new Error('Browser GPU inference is unavailable. Start the local reader service to use CPU speech.');
}

export class LocalReaderClient {
  constructor(fetcher = (...args) => globalThis.fetch(...args)) {
    this.fetcher = fetcher;
    this.onmessage = null;
    this.onerror = null;
    this.closed = false;
    this.active = null;
    this.wake = null;
    this.session = globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random()}`;
  }
  emit(type, extra = {}) { if (!this.closed) this.onmessage?.({ data: { type, ...extra } }); }
  async request(path, body) {
    const response = await this.fetcher(`/api/reader/${path}`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    });
    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.error || `The local reader returned HTTP ${response.status}.`);
    }
    return response;
  }
  postMessage(data) {
    if (this.closed) return;
    if (data.type === 'stop' && data.runId === this.active?.runId) {
      this.active.stopped = true;
      this.wake?.();
      if (this.active.requestId) this.request('stop', { requestId: this.active.requestId }).catch(() => {});
    } else if (data.type === 'consumed' && data.runId === this.active?.runId) {
      this.active.credits = Math.min(2, this.active.credits + 1); this.wake?.();
    } else if (data.type === 'load') {
      this.prepare(data).catch((error) => this.emit('error', { message: error.message }));
    } else if (data.type === 'read' && !this.active) {
      this.read(data).catch((error) => this.emit('error', { runId: data.runId, message: error.message }));
    }
  }
  async prepare(options) {
    if (options.embeddingManifestUrl || options.losslessEmbeddingManifestUrl || options.voiceStateManifestUrl || options.traceInference) {
      throw new Error('Local CPU speech uses the pinned default voice. Browser experiment overrides require WebGPU.');
    }
    this.emit('progress', { message: 'Preparing the selected voice on local CPU…', value: 10 });
    const response = await this.request('prepare', {});
    this.emit('ready', await response.json());
  }
  async read(data) {
    const run = { runId: data.runId, stopped: false, credits: 2, requestId: null };
    this.active = run;
    let seed = (data.seed ?? 1337) >>> 0;
    try {
      for (let index = 0; index < data.chunks.length; index += 1) {
        while (!run.stopped && !run.credits) await new Promise((resolve) => { this.wake = resolve; });
        this.wake = null;
        if (run.stopped || this.closed) break;
        run.credits -= 1;
        run.requestId = `${this.session}:${run.runId}:${index}`;
        const response = await this.request('synthesize', { text: data.chunks[index], seed, requestId: run.requestId });
        const metadata = JSON.parse(response.headers.get('X-Reader-Metadata'));
        const audio = await response.arrayBuffer();
        run.requestId = null;
        if (run.stopped || this.closed) break;
        if (!metadata || metadata.sampleRate !== 24000 || !audio.byteLength || audio.byteLength % 4) {
          throw new Error('The local reader returned invalid audio.');
        }
        seed = metadata.nextSeed >>> 0;
        this.emit('chunk', { ...metadata, runId: run.runId, index, text: data.chunks[index], audio });
      }
      this.emit(run.stopped ? 'stopped' : 'done', { runId: run.runId });
    } catch (error) {
      if (run.stopped) this.emit('stopped', { runId: run.runId });
      else throw error;
    } finally { this.active = null; this.wake = null; }
  }
  terminate() {
    if (this.active) this.postMessage({ type: 'stop', runId: this.active.runId });
    this.closed = true;
  }
}
