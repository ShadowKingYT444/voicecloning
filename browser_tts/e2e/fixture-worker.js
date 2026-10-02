// Test-only transport: synthetic tones, no tokenizer/model/voice-state import.
const failLoad = globalThis.FIXTURE_FAIL_LOAD === true;
let running = false; let stopped = false; let active = null; let credits = 0; let wake;
self.onmessage = async ({ data }) => {
  if (data.type === 'load') {
    self.postMessage(failLoad ? { type: 'error', message: 'Synthetic fixture load failure.' }
      : { type: 'ready', backend: 'SYNTHETIC PCM TEST FIXTURE', loadSeconds: 0,
        modelIdentity: { fixture: true, inferenceRun: false, fittedAdapterApplied: false } });
  } else if (data.type === 'stop' && data.runId === active) { stopped = true; wake?.(); }
  else if (data.type === 'consumed' && data.runId === active) { credits = Math.min(2, credits + 1); wake?.(); }
  else if (data.type === 'read' && !running) {
    running = true; stopped = false; active = data.runId; credits = 2;
    try {
      for (let index = 0; index < data.chunks.length; index += 1) {
        while (!stopped && !credits) await new Promise((resolve) => { wake = resolve; });
        if (stopped) break;
        credits -= 1;
        const frames = data.chunks.length > 10 ? 9600 : 2400;
        const audio = Float32Array.from({ length: frames }, (_, frame) => 0.01 * Math.sin(frame * Math.PI * 2 * 440 / 24000));
        self.postMessage({ type: 'chunk', runId: active, index, text: data.chunks[index], sampleRate: 24000,
          synthesisSeconds: 0, speechTokens: [42], truncated: false, audio: audio.buffer }, [audio.buffer]);
      }
      self.postMessage({ type: stopped ? 'stopped' : 'done', runId: active });
    } finally { running = false; }
  }
};
