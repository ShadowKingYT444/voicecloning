import './style.css';
import { splitIntoPassages, wordCount } from './chunker.js';
import { encodeWavMono16 } from './audio.js';

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
let worker; let audioContext; let sources = new Set(); let nextPlayTime = 0; let runStarted = 0; let firstAudio = null;
let synthSeconds = 0; let audioSeconds = 0; let completed = 0; let total = 0; let busy = false; let receivingDone = false; let modelReady = false;
let recordedChunks = []; let downloadUrl;

function setStatus(text, state = '') { ui.statusText.textContent = text; ui.status.dataset.state = state; }
function fmt(seconds) { return `${seconds.toFixed(2)} s`; }
function passages() { return splitIntoPassages(ui.passage.value, Number(ui.chunkSize.value)); }
function updateTextStats() {
  const chunks = passages();
  ui.words.textContent = `${wordCount(ui.passage.value).toLocaleString()} words`;
  ui.chunks.textContent = `${chunks.length.toLocaleString()} passages`;
  ui.read.disabled = !worker || !chunks.length || busy;
}
function updateMetric(id, value) { $(id).textContent = value; }

fetch('/federalist-no-10.txt').then((r) => { if (!r.ok) throw new Error('Text file could not load.'); return r.text(); })
  .then((text) => { ui.passage.value = text.trim(); updateTextStats(); })
  .catch((error) => { ui.words.textContent = error.message; });

navigator.gpu?.requestAdapter().then((adapter) => {
  if (!adapter) { setStatus('WebGPU is not available in this browser.', 'error'); return; }
  setStatus('WebGPU is available. Prepare the voice to load the model.', 'ready'); ui.load.disabled = false;
}).catch(() => setStatus('WebGPU could not start. Check browser and GPU support.', 'error'));

ui.passage.addEventListener('input', updateTextStats);
ui.chunkSize.addEventListener('change', updateTextStats);
ui.playReference.addEventListener('click', async () => {
  if (ui.reference.paused) { await ui.reference.play(); ui.wave.classList.add('playing'); }
  else { ui.reference.pause(); ui.wave.classList.remove('playing'); }
});
ui.reference.addEventListener('ended', () => ui.wave.classList.remove('playing'));

ui.load.addEventListener('click', () => {
  if (worker) return;
  ui.load.disabled = true; ui.load.querySelector('span:first-child').textContent = 'Loading model…';
  ui.progressWrap.hidden = false; setStatus('Downloading and preparing four model graphs…', 'loading');
  worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
  worker.onmessage = handleWorkerMessage;
  worker.onerror = (event) => {
    setStatus(`Worker error: ${event.message}`, 'error');
    if (!modelReady) { worker?.terminate(); worker = null; ui.load.disabled = false; ui.load.querySelector('span:first-child').textContent = 'Retry model load'; }
    updateTextStats();
  };
  worker.postMessage({ type: 'load' });
});

ui.read.addEventListener('click', async () => {
  if (!worker || busy) return;
  const chunks = passages(); if (!chunks.length) return;
  audioContext ||= new AudioContext(); await audioContext.resume();
  busy = true; completed = 0; total = chunks.length; firstAudio = null; synthSeconds = 0; audioSeconds = 0;
  runStarted = performance.now(); nextPlayTime = audioContext.currentTime + 0.12; sources.clear();
  receivingDone = false;
  if (downloadUrl) { URL.revokeObjectURL(downloadUrl); downloadUrl = undefined; }
  recordedChunks = []; ui.download.hidden = true;
  ui.read.disabled = true; ui.stop.disabled = false; ui.readState.textContent = 'Generating the first passage…';
  ui.readingWrap.hidden = false; ui.now.hidden = false; ui.readingProgress.style.width = '0%';
  ui.readingLabel.textContent = `Passage 0 of ${total}`; ui.readingTime.textContent = '';
  worker.postMessage({ type: 'read', chunks });
});

ui.stop.addEventListener('click', () => {
  worker?.postMessage({ type: 'stop' });
  for (const source of sources) { try { source.stop(); } catch {} }
  sources.clear(); if (audioContext) nextPlayTime = audioContext.currentTime;
  receivingDone = false; ui.stop.disabled = true; ui.readState.textContent = 'Stopping after the active model call…'; ui.wave.classList.remove('playing');
});

function playChunk(message) {
  const samples = new Float32Array(message.audio);
  recordedChunks.push(samples);
  const buffer = audioContext.createBuffer(1, samples.length, message.sampleRate);
  buffer.copyToChannel(samples, 0);
  const source = audioContext.createBufferSource(); source.buffer = buffer;
  const gain = audioContext.createGain(); source.connect(gain).connect(audioContext.destination);
  const start = Math.max(audioContext.currentTime + 0.025, nextPlayTime - 0.008);
  const end = start + buffer.duration;
  gain.gain.setValueAtTime(0, start); gain.gain.linearRampToValueAtTime(1, start + 0.009);
  gain.gain.setValueAtTime(1, Math.max(start + 0.01, end - 0.012)); gain.gain.linearRampToValueAtTime(0, end);
  source.start(start); source.stop(end); sources.add(source); source.onended = () => { sources.delete(source); if (receivingDone && sources.size === 0) finishPlayback(); };
  nextPlayTime = end; audioSeconds += buffer.duration; synthSeconds += message.synthesisSeconds;
  completed += 1;
  if (firstAudio === null) {
    firstAudio = (performance.now() - runStarted + Math.max(0, (start - audioContext.currentTime) * 1000)) / 1000;
    updateMetric('metric-first', fmt(firstAudio));
  }
  updateMetric('metric-last', fmt(message.synthesisSeconds)); updateMetric('metric-audio', fmt(audioSeconds));
  updateMetric('metric-rtf', (synthSeconds / audioSeconds).toFixed(2));
  ui.currentText.textContent = message.text; ui.duration.textContent = fmt(buffer.duration);
  ui.readState.textContent = `Playing passage ${message.index + 1} of ${total}`;
  ui.readingProgress.style.width = `${Math.round(completed / total * 100)}%`;
  ui.readingLabel.textContent = `Passage ${completed} of ${total}`;
  ui.readingTime.textContent = `Generated ${fmt(synthSeconds)} · ${fmt(audioSeconds)} audio`;
  ui.wave.classList.add('playing');
}

function handleWorkerMessage({ data }) {
  if (data.type === 'progress') {
    setStatus(data.message, 'loading'); ui.loadLabel.textContent = data.message;
    if (data.value != null) ui.loadProgress.style.width = `${Math.max(0, Math.min(100, data.value))}%`;
  } else if (data.type === 'ready') {
    modelReady = true;
    updateMetric('metric-load', fmt(data.loadSeconds));
    setStatus(`Voice ready · ${data.backend}`, 'ready'); ui.load.querySelector('span:first-child').textContent = 'Voice prepared';
    ui.progressWrap.hidden = true; ui.read.disabled = false; updateTextStats();
  } else if (data.type === 'chunk') {
    playChunk(data);
  } else if (data.type === 'done') {
    prepareDownload();
    receivingDone = true;
    if (sources.size === 0) finishPlayback();
    else ui.readState.textContent = 'All passages generated. Playback is finishing.';
  } else if (data.type === 'stopped') {
    prepareDownload();
    receivingDone = false; busy = false; ui.stop.disabled = true; ui.readState.textContent = 'Stopped.'; updateTextStats();
  } else if (data.type === 'error') {
    prepareDownload();
    receivingDone = false; busy = false; ui.stop.disabled = true; setStatus(data.message, 'error'); ui.readState.textContent = 'The browser could not complete inference.';
    if (!modelReady) { worker?.terminate(); worker = null; ui.load.disabled = false; ui.load.querySelector('span:first-child').textContent = 'Retry model load'; }
    updateTextStats();
  }
}

function prepareDownload() {
  if (!recordedChunks.length) return;
  if (downloadUrl) URL.revokeObjectURL(downloadUrl);
  downloadUrl = URL.createObjectURL(new Blob([encodeWavMono16(recordedChunks)], { type: 'audio/wav' }));
  ui.download.href = downloadUrl;
  ui.download.download = recordedChunks.length === total ? 'federalist-no-10-asmr.wav' : 'federalist-no-10-asmr-partial.wav';
  ui.download.textContent = `Download ${recordedChunks.length} passage${recordedChunks.length === 1 ? '' : 's'}`; ui.download.hidden = false;
}

function finishPlayback() {
  receivingDone = false; busy = false; ui.stop.disabled = true;
  ui.readState.textContent = `Finished in ${fmt((performance.now() - runStarted) / 1000)}.`;
  ui.wave.classList.remove('playing'); updateTextStats();
}
