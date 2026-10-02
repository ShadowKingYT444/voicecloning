import './style.css';
import { splitIntoPassages, wordCount } from './chunker.js';
import { fadeEdges } from './audio.js';
import { ReadingStore } from './reading-store.js';
import { REVISION } from './model-loader.js';

const $ = (id) => document.getElementById(id);
const ui = {
  reference: $('reference-player'), playReference: $('play-reference'), wave: document.querySelector('.voice-wave'),
  status: $('model-status'), statusText: $('model-status-text'), load: $('load-model'), progressWrap: $('load-progress-wrap'),
  loadProgress: $('load-progress'), loadLabel: $('load-progress-label'), passage: $('passage'), words: $('word-count'),
  chunks: $('chunk-count'), chunkSize: $('chunk-size'), read: $('read-button'), stop: $('stop-button'), readState: $('read-state'),
  readingWrap: $('reading-progress-wrap'), readingProgress: $('reading-progress'), readingLabel: $('reading-progress-label'),
  readingTime: $('reading-time-label'), now: $('now-reading'), currentText: $('current-text'), duration: $('current-duration'),
  download: $('download-reading'),
};
const store = new ReadingStore();
const localModelBase = '/models/chatterbox-nano-browser/';
const localModelDiscovery = fetch(`${localModelBase}download_manifest.json`)
  .then(async (response) => {
    if (!response.ok) return undefined;
    const manifest = await response.json();
    return manifest.repo_id === 'owensong/chatterbox-nano-ONNX' && manifest.revision === REVISION
      ? localModelBase : undefined;
  }).catch(() => undefined);
let worker; let audioContext; const sources = new Set(); const labelTimers = new Set();
let nextPlayTime = 0; let runStarted = 0; let firstAudio = null; let loadSeconds = null;
let synthSeconds = 0; let audioSeconds = 0; let completed = 0; let total = 0;
let busy = false; let receivingDone = false; let modelReady = false; let stopping = false;
let status = 'checking'; let lastError = null; let runId = 0; let seed = 1337;
let modelIdentity = null;
let gaps = []; let chunkRecords = []; let loadWait; let readWait; let messageChain = Promise.resolve();

function setStatus(text, state = '') { ui.statusText.textContent = text; ui.status.dataset.state = state; }
function fmt(seconds) { return `${seconds.toFixed(2)} s`; }
function passages() { return splitIntoPassages(ui.passage.value, Number(ui.chunkSize.value)); }
function updateTextStats() {
  ui.words.textContent = `${wordCount(ui.passage.value).toLocaleString()} words`;
  ui.chunks.textContent = `${passages().length.toLocaleString()} passages`;
  ui.read.disabled = !modelReady || !passages().length || busy;
  ui.passage.disabled = busy; ui.chunkSize.disabled = busy;
}
function metric(id, value) { $(id).textContent = value; }
function snapshot() {
  return { status, error: lastError, modelReady, busy, loadSeconds, seed, modelIdentity,
    chunks: chunkRecords.map((chunk) => ({ ...chunk })),
    metrics: { firstAudioSeconds: firstAudio, totalSynthesisSeconds: synthSeconds, audioSeconds,
      rtf: audioSeconds ? synthSeconds / audioSeconds : null, playbackGapsSeconds: [...gaps],
      maximumPlaybackGapSeconds: Math.max(0, ...gaps), scheduledQueueSize: sources.size } };
}
function deferred() {
  let resolve; let reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

fetch('/federalist-no-10.txt').then((r) => { if (!r.ok) throw new Error('Text file could not load.'); return r.text(); })
  .then((text) => { ui.passage.value = text.trim(); updateTextStats(); })
  .catch((error) => { ui.words.textContent = error.message; });

if (!navigator.gpu || typeof WebAssembly.Suspending !== 'function' || typeof WebAssembly.promising !== 'function') {
  status = 'error'; lastError = 'This reader requires WebGPU and WebAssembly JSPI support.';
  setStatus(lastError, 'error');
} else {
  navigator.gpu.requestAdapter({ powerPreference: 'low-power' }).then((adapter) => {
    if (!adapter) throw new Error('No WebGPU adapter is available.');
    if (!adapter.features.has('shader-f16')) throw new Error('This reader requires a WebGPU adapter with FP16 shader support.');
    status = 'idle'; setStatus('Prepare the selected voice to begin.', 'ready'); ui.load.disabled = false;
  }).catch((error) => { status = 'error'; lastError = error.message; setStatus(lastError, 'error'); });
}
ui.passage.addEventListener('input', updateTextStats);
ui.chunkSize.addEventListener('change', updateTextStats);
ui.playReference.addEventListener('click', async () => {
  try {
    if (ui.reference.paused) { await ui.reference.play(); ui.wave.classList.add('playing'); }
    else { ui.reference.pause(); ui.wave.classList.remove('playing'); }
  } catch (error) { setStatus(error.message, 'error'); }
});
ui.reference.addEventListener('ended', () => ui.wave.classList.remove('playing'));

function prepareVoice(options = {}) {
  if (modelReady) return Promise.resolve(snapshot());
  if (loadWait) return loadWait.promise;
  loadWait = deferred(); status = 'loading'; lastError = null;
  ui.load.disabled = true; ui.load.querySelector('span:first-child').textContent = 'Loading model…';
  ui.progressWrap.hidden = false; setStatus('Loading the saved voice and three model graphs…', 'loading');
  worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
  const loadingWorker = worker;
  worker.onmessage = ({ data }) => {
    // Serialize storage writes and terminal messages to preserve passage order.
    messageChain = messageChain.then(() => {
      if (worker === loadingWorker) return handleWorkerMessage(data);
    }).catch((error) => { if (worker === loadingWorker) fail(error); });
  };
  worker.onerror = (event) => { if (worker === loadingWorker) fail(new Error(`Worker error: ${event.message}`)); };
  localModelDiscovery.then((discoveredBase) => {
    if (worker !== loadingWorker) return;
    loadingWorker.postMessage({ type: 'load', embeddingManifestUrl: options.embeddingManifestUrl,
      modelBaseUrl: options.modelBaseUrl ?? discoveredBase, voiceStateManifestUrl: options.voiceStateManifestUrl,
      powerPreference: options.powerPreference });
  }).catch((error) => { if (worker === loadingWorker) fail(error); });
  return loadWait.promise;
}
ui.load.addEventListener('click', () => { prepareVoice().catch(() => {}); });

async function readText(text, options = {}) {
  if (!modelReady) throw new Error('Prepare the voice before reading.');
  if (busy) throw new Error('A reading is already active.');
  const chunks = splitIntoPassages(text, options.chunkWords ?? Number(ui.chunkSize.value));
  if (!chunks.length) throw new Error('Enter text to read.');
  busy = true; status = 'reading'; stopping = false; lastError = null; updateTextStats();
  try {
    await store.clear();
    ui.reference.pause();
    audioContext ||= new AudioContext(); await audioContext.resume();
  } catch (error) { fail(error); throw error; }
  readWait = deferred(); runId += 1; seed = (options.seed ?? 1337) >>> 0;
  completed = 0; total = chunks.length; firstAudio = null; synthSeconds = 0; audioSeconds = 0;
  gaps = []; chunkRecords = []; receivingDone = false;
  runStarted = performance.now(); nextPlayTime = 0;
  ui.download.hidden = true; ui.stop.disabled = false; ui.readState.textContent = 'Generating the first passage…';
  ui.readingWrap.hidden = false; ui.now.hidden = true; ui.readingProgress.style.width = '0%';
  ui.readingLabel.textContent = `Passage 0 of ${total}`; ui.readingTime.textContent = '';
  metric('metric-first', '—'); metric('metric-last', '—'); metric('metric-audio', '—'); metric('metric-rtf', '—');
  metric('metric-gap', '—');
  worker.postMessage({ type: 'read', runId, seed, chunks });
  return readWait.promise;
}
ui.read.addEventListener('click', () => { readText(ui.passage.value).catch(() => {}); });

function cancelSources() {
  for (const timer of labelTimers) clearTimeout(timer);
  labelTimers.clear();
  for (const source of sources) { source.onended = null; try { source.stop(); } catch {} source.disconnect(); }
  sources.clear(); ui.wave.classList.remove('playing');
}
function stop() {
  if (!busy) return Promise.resolve(snapshot());
  stopping = true; status = 'stopping'; receivingDone = false;
  worker?.postMessage({ type: 'stop', runId }); cancelSources();
  ui.stop.disabled = true; ui.readState.textContent = 'Stopping after the active model call…';
  // Synthesis may already be done while scheduled playback is still active.
  if (readWait && completed === total) finishReading(true);
  return readWait?.promise || Promise.resolve(snapshot());
}
ui.stop.addEventListener('click', () => { stop().catch(() => {}); });

async function playChunk(message) {
  if (stopping || message.runId !== runId) return;
  const samples = fadeEdges(new Float32Array(message.audio), message.sampleRate);
  await store.append(samples);
  if (stopping) return;
  const buffer = audioContext.createBuffer(1, samples.length, message.sampleRate);
  buffer.copyToChannel(samples, 0);
  const source = audioContext.createBufferSource(); source.buffer = buffer;
  source.connect(audioContext.destination);
  const start = Math.max(audioContext.currentTime + 0.025, nextPlayTime);
  if (completed > 0) gaps.push(Math.max(0, start - nextPlayTime));
  const end = start + buffer.duration; nextPlayTime = end;
  sources.add(source);
  source.onended = () => {
    sources.delete(source); source.disconnect(); source.buffer = null;
    worker?.postMessage({ type: 'consumed', runId });
    if (receivingDone && sources.size === 0) finishReading();
  };
  source.start(start);
  audioSeconds += buffer.duration; synthSeconds += message.synthesisSeconds; completed += 1;
  chunkRecords.push({ index: message.index, text: message.text, sampleRate: message.sampleRate,
    synthesisSeconds: message.synthesisSeconds, audioSeconds: buffer.duration,
    speechTokens: message.speechTokens, truncated: message.truncated, scheduledStartSeconds: start });
  if (firstAudio === null) {
    firstAudio = (performance.now() - runStarted) / 1000 + Math.max(0, start - audioContext.currentTime);
    metric('metric-first', fmt(firstAudio));
  }
  metric('metric-last', fmt(message.synthesisSeconds)); metric('metric-audio', fmt(audioSeconds));
  metric('metric-rtf', (synthSeconds / audioSeconds).toFixed(2));
  metric('metric-gap', fmt(Math.max(0, ...gaps)));
  ui.readingTime.textContent = `Generated ${fmt(synthSeconds)} · ${fmt(audioSeconds)} audio`;
  const timer = setTimeout(() => {
    labelTimers.delete(timer);
    if (stopping) return;
    ui.now.hidden = false;
    ui.currentText.textContent = message.text; ui.duration.textContent = fmt(buffer.duration);
    ui.readState.textContent = `Playing passage ${message.index + 1} of ${total}`;
    ui.readingProgress.style.width = `${Math.round((message.index + 1) / total * 100)}%`;
    ui.readingLabel.textContent = `Passage ${message.index + 1} of ${total}`;
    ui.wave.classList.add('playing');
  }, Math.max(0, (start - audioContext.currentTime) * 1000));
  labelTimers.add(timer);
}

async function handleWorkerMessage(data) {
  if (data.runId != null && (data.runId !== runId || !busy)) return;
  if (data.type === 'progress') {
    setStatus(data.message, 'loading'); ui.loadLabel.textContent = data.message;
    if (data.value != null) ui.loadProgress.style.width = `${Math.max(0, Math.min(100, data.value))}%`;
  } else if (data.type === 'ready') {
    modelReady = true; status = 'ready'; loadSeconds = data.loadSeconds; modelIdentity = data.modelIdentity;
    metric('metric-load', fmt(loadSeconds)); setStatus(`Voice ready · ${data.backend}`, 'ready');
    ui.load.querySelector('span:first-child').textContent = 'Voice prepared'; ui.progressWrap.hidden = true;
    updateTextStats(); loadWait?.resolve(snapshot()); loadWait = null;
  } else if (data.type === 'chunk') {
    await playChunk(data);
  } else if (data.type === 'done') {
    receivingDone = true;
    if (stopping || sources.size === 0) finishReading(stopping);
  } else if (data.type === 'stopped') {
    finishReading(true);
  } else if (data.type === 'error') {
    fail(new Error(data.message));
  }
}
function finishReading(wasStopped = false) {
  receivingDone = false; busy = false; status = 'ready'; ui.stop.disabled = true;
  ui.readState.textContent = wasStopped ? 'Stopped.' : `Finished in ${fmt((performance.now() - runStarted) / 1000)}.`;
  ui.wave.classList.remove('playing');
  for (const timer of labelTimers) clearTimeout(timer); labelTimers.clear();
  showDownload(); updateTextStats();
  readWait?.resolve(snapshot()); readWait = null;
}
function fail(error) {
  const preparing = !!loadWait;
  lastError = error?.message || String(error); status = 'error'; busy = false; receivingDone = false;
  cancelSources(); ui.stop.disabled = true;
  setStatus(preparing && /fixed voice state/i.test(lastError)
    ? 'The saved voice could not load. Check the voice files, then retry.' : lastError, 'error');
  ui.progressWrap.hidden = true;
  ui.readState.textContent = preparing ? 'The voice could not be prepared.' : 'The reading could not complete.';
  // A model runtime failure invalidates the session. A retry loads clean sessions.
  modelReady = false; worker?.terminate(); worker = null;
  ui.load.disabled = false; ui.load.querySelector('span:first-child').textContent = 'Retry model load';
  loadWait?.reject(new Error(lastError)); loadWait = null;
  readWait?.reject(new Error(lastError)); readWait = null;
  showDownload(); updateTextStats();
}
function showDownload() {
  ui.download.hidden = !store.count;
  ui.download.textContent = `Save ${store.count || 0} passage${store.count === 1 ? '' : 's'}`;
}
ui.download.addEventListener('click', async (event) => {
  event.preventDefault(); if (busy || !store.count) return;
  const name = store.count === total ? 'federalist-no-10-asmr.wav' : 'federalist-no-10-asmr-partial.wav';
  try {
    if (window.showSaveFilePicker) {
      const file = await window.showSaveFilePicker({ suggestedName: name, types: [{ description: 'WAV audio', accept: { 'audio/wav': ['.wav'] } }] });
      const stream = await file.createWritable();
      try { await store.writeTo(stream); await stream.close(); } catch (error) { await stream.abort(); throw error; }
    } else {
      const url = URL.createObjectURL(await store.blob());
      const link = document.createElement('a'); link.href = url; link.download = name; link.click();
      setTimeout(() => URL.revokeObjectURL(url), 60000);
    }
  } catch (error) { if (error.name !== 'AbortError') setStatus(`Audio export failed: ${error.message}`, 'error'); }
});

window.voiceStudy = {
  prepareVoice, readText, stop, snapshot,
  async exportWavBase64() {
    if (busy) throw new Error('Wait for the reading to finish before exporting.');
    if (audioSeconds > 60) throw new Error('Use the Save control to stream a long reading to disk.');
    const bytes = new Uint8Array(await (await store.blob()).arrayBuffer());
    let binary = '';
    for (let start = 0; start < bytes.length; start += 8192) binary += String.fromCharCode(...bytes.subarray(start, start + 8192));
    return btoa(binary);
  },
};
