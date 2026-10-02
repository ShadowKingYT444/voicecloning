import * as ort from 'onnxruntime-web/webgpu';
import { Tokenizer } from '@huggingface/tokenizers';

const REVISION = '4a66d7dab72a9e98f24b515d49a1d7a81632df2e';
const ROOT = `https://huggingface.co/owensong/chatterbox-nano-ONNX/resolve/${REVISION}/`;
const GRAPH = (name) => `onnx/${name}.onnx`;
const START_SPEECH = 6561; const STOP_SPEECH = 6562; const SILENCE = 4299;
const sessions = {}; let tokenizer; let stopped = false; let running = false; let refAudio; let speakerConditioning; let randomState = 1337;

function post(type, extra = {}) { self.postMessage({ type, ...extra }); }
function halfToFloat(value) {
  const sign = (value & 0x8000) ? -1 : 1; const exponent = (value >> 10) & 31; const fraction = value & 1023;
  if (exponent === 0) return sign * Math.pow(2, -14) * (fraction / 1024);
  if (exponent === 31) return fraction ? NaN : sign * Infinity;
  return sign * Math.pow(2, exponent - 15) * (1 + fraction / 1024);
}
function values(tensor) {
  const data = tensor.data;
  if (data instanceof Uint16Array || tensor.type === 'float16') return Float32Array.from(data, halfToFloat);
  return Float32Array.from(data);
}
function floatToHalf(value) {
  const f = new Float32Array(1); const u = new Uint32Array(f.buffer); f[0] = value;
  const x = u[0]; const sign = (x >>> 16) & 0x8000; let mantissa = x & 0x7fffff; let exponent = ((x >>> 23) & 0xff) - 127 + 15;
  if (exponent <= 0) { if (exponent < -10) return sign; mantissa = (mantissa | 0x800000) >> (1 - exponent); return sign | ((mantissa + 0x1000) >> 13); }
  if (exponent >= 31) return sign | 0x7c00;
  return sign | (exponent << 10) | ((mantissa + 0x1000) >> 13);
}
function audioFromWav(buffer) {
  const view = new DataView(buffer); const id = (offset) => String.fromCharCode(...new Uint8Array(buffer, offset, 4));
  if (id(0) !== 'RIFF' || id(8) !== 'WAVE') throw new Error('Reference file is not a WAV file.');
  let offset = 12; let format; let channels; let rate; let bits; let dataOffset; let dataSize;
  while (offset + 8 <= view.byteLength) {
    const name = id(offset); const size = view.getUint32(offset + 4, true);
    if (name === 'fmt ') { format = view.getUint16(offset + 8, true); channels = view.getUint16(offset + 10, true); rate = view.getUint32(offset + 12, true); bits = view.getUint16(offset + 22, true); }
    if (name === 'data') { dataOffset = offset + 8; dataSize = size; break; }
    offset += 8 + size + (size & 1);
  }
  if (rate !== 24000 || !dataOffset || ![1, 3].includes(format) || ![16, 24, 32].includes(bits)) throw new Error('Reference WAV must be mono 24 kHz PCM.');
  const frames = dataSize / (channels * bits / 8); const mono = new Float32Array(frames); const bytes = bits / 8;
  for (let frame = 0; frame < frames; frame += 1) {
    let sum = 0;
    for (let channel = 0; channel < channels; channel += 1) {
      const p = dataOffset + (frame * channels + channel) * bytes; let value;
      if (format === 3 && bits === 32) value = view.getFloat32(p, true);
      else if (bits === 16) value = view.getInt16(p, true) / 32768;
      else if (bits === 24) { let integer = view.getUint8(p) | (view.getUint8(p + 1) << 8) | (view.getUint8(p + 2) << 16); if (integer & 0x800000) integer |= 0xff000000; value = integer / 8388608; }
      else value = view.getInt32(p, true) / 2147483648;
      sum += value;
    }
    mono[frame] = sum / channels;
  }
  return mono;
}
async function fetchJson(path) { const response = await fetch(ROOT + path); if (!response.ok) throw new Error(`Could not download ${path}: ${response.status}`); return response.json(); }
async function createGraph(name, outputLocations = {}) {
  const path = GRAPH(name);
  return ort.InferenceSession.create(ROOT + path, {
    executionProviders: ['webgpu'], graphOptimizationLevel: 'all',
    externalData: [{ path: `${name}.onnx_data`, data: ROOT + `${path}_data` }],
    preferredOutputLocation: outputLocations,
  });
}
async function load() {
  if (!navigator.gpu) throw new Error('WebGPU is unavailable in this browser.');
  const adapter = await navigator.gpu.requestAdapter(); if (!adapter) throw new Error('No WebGPU adapter was found.');
  post('progress', { value: 3, message: 'Loading tokenizer and reference voice…' });
  const [tokenizerJson, tokenizerConfig, wavResponse] = await Promise.all([
    fetchJson('tokenizer.json'), fetchJson('tokenizer_config.json'), fetch('/voice/asmr_t3_seed47_fit.wav'),
  ]);
  if (!wavResponse.ok) throw new Error('The selected reference voice could not be loaded.');
  refAudio = audioFromWav(await wavResponse.arrayBuffer());
  tokenizer = new Tokenizer(tokenizerJson, tokenizerConfig);
  const names = ['embed_tokens_fp16', 'speech_encoder_q4f16', 'language_model_q4f16', 'conditional_decoder_q4'];
  for (let index = 0; index < names.length; index += 1) {
    const name = names[index]; post('progress', { value: 8 + index * 21, message: `Loading graph ${index + 1} of 4 · ${name.replaceAll('_', ' ')}` });
    if (name === 'language_model_q4f16') {
      const locations = {};
      // Keep recurrent KV outputs on the WebGPU device. Only logits are sampled on CPU.
      locations.logits = 'cpu';
      for (let layer = 0; layer < 12; layer += 1) {
        locations[`present.${layer}.key`] = 'gpu-buffer';
        locations[`present.${layer}.value`] = 'gpu-buffer';
      }
      sessions[name] = await createGraph(name, locations);
    } else sessions[name] = await createGraph(name);
  }
  post('progress', { value: 94, message: 'Encoding the reference voice…' });
  const referenceOutputs = await sessions.speech_encoder_q4f16.run({ audio_values: tensor('float32', refAudio, [1, refAudio.length]) });
  const [audioFeatures, audioTokens, speakerEmbeddings, speakerFeatures] = sessions.speech_encoder_q4f16.outputNames.map((name) => referenceOutputs[name]);
  speakerConditioning = { audioFeatures, audioTokens, speakerEmbeddings, speakerFeatures };
  refAudio = null;
  post('progress', { value: 100, message: 'Reference voice is ready.' });
  return adapter;
}
function tensor(type, data, dims) { return new ort.Tensor(type, data, dims); }
function int64(data, dims) { return tensor('int64', BigInt64Array.from(data, (value) => BigInt(value)), dims); }
function dispose(t) { try { t?.dispose(); } catch {} }
function random() { randomState = (Math.imul(randomState, 1664525) + 1013904223) >>> 0; return randomState / 0x100000000; }
function sessionTensor(session, name, data, dims, fallback = 'float16') {
  const meta = session.inputMetadata?.find((item) => item.name === name); const type = meta?.type || fallback;
  if (type === 'float16') return tensor(type, data instanceof Uint16Array ? data : Uint16Array.from(data, floatToHalf), dims);
  return tensor(type, data instanceof Float32Array ? data : Float32Array.from(data, (x) => typeof x === 'bigint' ? Number(x) : x), dims);
}
function sampleLogits(logits, history) {
  const allScores = values(logits); const vocabSize = logits.dims.at(-1); const scores = allScores.subarray(allScores.length - vocabSize);
  const repeats = new Map(); for (const token of history) repeats.set(token, (repeats.get(token) || 0) + 1);
  const adjusted = Array.from(scores, (score, id) => {
    let value = score / 0.8; if (repeats.has(id)) value = value < 0 ? value * 1.2 : value / 1.2;
    return [value, id];
  }).sort((a, b) => b[0] - a[0]).slice(0, 1000);
  const max = adjusted[0]?.[0] ?? 0; const exp = adjusted.map(([score, id]) => [Math.exp(score - max), id]); const total = exp.reduce((sum, item) => sum + item[0], 0);
  let cumulative = 0; const nucleus = [];
  for (const item of exp) { nucleus.push(item); cumulative += item[0] / total; if (cumulative >= 0.95) break; }
  const mass = nucleus.reduce((sum, item) => sum + item[0], 0); let choice = random() * mass;
  for (const [probability, id] of nucleus) { choice -= probability; if (choice <= 0) return id; }
  return nucleus.at(-1)?.[1] ?? STOP_SPEECH;
}
async function synthesize(text, index) {
  const t0 = performance.now();
  const embed = sessions.embed_tokens_fp16; const lm = sessions.language_model_q4f16; const decoder = sessions.conditional_decoder_q4;
  const { audioFeatures, audioTokens, speakerEmbeddings, speakerFeatures } = speakerConditioning;
  const ids = tokenizer.encode(text).ids; const inputIds = int64(ids, [1, ids.length]);
  const embedded = await embed.run({ input_ids: inputIds }); const textEmbeddings = embedded[embed.outputNames[0]];
  const audioData = values(audioFeatures); const textData = values(textEmbeddings);
  const featureDim = audioFeatures.dims.at(-1); const textDim = textEmbeddings.dims.at(-1);
  if (featureDim !== textDim) throw new Error(`Embedding dimensions differ (${featureDim} and ${textDim}).`);
  const sequence = audioFeatures.dims.at(-2) + textEmbeddings.dims.at(-2); const combined = new Float32Array(sequence * featureDim);
  combined.set(audioData); combined.set(textData, audioData.length);
  const embedsName = lm.inputNames.find((name) => name.includes('inputs_embeds')) || 'inputs_embeds';
  const feeds = { [embedsName]: sessionTensor(lm, embedsName, combined, [1, sequence, featureDim]) };
  const maskName = lm.inputNames.find((name) => name === 'attention_mask'); const posName = lm.inputNames.find((name) => name === 'position_ids');
  if (maskName) feeds[maskName] = int64(Array(sequence).fill(1), [1, sequence]);
  if (posName) feeds[posName] = int64(Array.from({ length: sequence }, (_, i) => i), [1, sequence]);
  const cacheInputs = lm.inputNames.filter((name) => name.includes('past_key_values'));
  const cacheOutputs = lm.outputNames.filter((name) => name.includes('present') || name.includes('past_key_values'));
  if (cacheInputs.length !== 24 || cacheOutputs.length !== 24) throw new Error(`Expected 24 cache inputs and outputs, got ${cacheInputs.length} and ${cacheOutputs.length}.`);
  for (const name of cacheInputs) feeds[name] = sessionTensor(lm, name, new Float32Array(0), [1, 12, 0, 64]);
  const history = [START_SPEECH]; let result;
  const tokenLimit = Math.min(256, Math.max(96, wordCountLocal(text) * 8));
  try {
    result = await lm.run(feeds);
    let cache = cacheOutputs.map((name) => result[name]);
    let logits = result[lm.outputNames[0]];
    for (let step = 0; step < tokenLimit; step += 1) {
      if (stopped) break;
      const tokenId = sampleLogits(logits, history); if (tokenId === STOP_SPEECH) break;
      history.push(tokenId);
      if (step + 1 >= tokenLimit) break;
      const tokenResult = await embed.run({ input_ids: int64([tokenId], [1, 1]) });
      const tokenVector = tokenResult[embed.outputNames[0]];
      const past = step === 0 ? sequence : sequence + step;
      const nextFeeds = {
        [embedsName]: sessionTensor(lm, embedsName, values(tokenVector), [1, 1, featureDim]),
      };
      if (maskName) nextFeeds[maskName] = int64(Array(past + 1).fill(1), [1, past + 1]);
      if (posName) nextFeeds[posName] = int64([past], [1, 1]);
      for (let i = 0; i < cacheInputs.length; i += 1) nextFeeds[cacheInputs[i]] = cache[i];
      const nextResult = await lm.run(nextFeeds);
      for (const [name, value] of Object.entries(nextFeeds)) if (!cacheInputs.includes(name)) dispose(value);
      dispose(logits);
      for (const item of cache) dispose(item);
      cache = cacheOutputs.map((name) => nextResult[name]); logits = nextResult[lm.outputNames[0]];
      for (const key of Object.keys(tokenResult)) dispose(tokenResult[key]);
      result = nextResult;
    }
    for (const item of cache) dispose(item);
    if (stopped) {
      for (const key of Object.keys(embedded)) dispose(embedded[key]);
      return;
    }
    const speechTokens = history.slice(1); if (!speechTokens.length) throw new Error(`No speech tokens were generated for passage ${index + 1}.`);
    const prefixTokens = Array.from(audioTokens.data, Number);
    const decoderTokens = [...prefixTokens, ...speechTokens, SILENCE, SILENCE, SILENCE];
    const decodeResult = await decoder.run({
      speech_tokens: int64(decoderTokens, [1, decoderTokens.length]),
      speaker_embeddings: speakerEmbeddings, speaker_features: speakerFeatures,
    });
    const waveform = decodeResult[decoder.outputNames[0]]; const audio = values(waveform);
    const synthesisSeconds = (performance.now() - t0) / 1000;
    if (!stopped) {
      const output = { type: 'chunk', index, text, sampleRate: 24000, synthesisSeconds, audio: audio.buffer };
      self.postMessage(output, [audio.buffer]);
    }
    for (const key of Object.keys(embedded)) dispose(embedded[key]);
    for (const key of Object.keys(decodeResult)) dispose(decodeResult[key]);
  } finally {
    if (result) for (const key of Object.keys(result)) dispose(result[key]);
    for (const value of Object.values(feeds)) dispose(value);
  }
}
function wordCountLocal(text) { return text.trim().split(/\s+/).filter(Boolean).length; }

self.onmessage = async ({ data }) => {
  if (data.type === 'load') {
    try { const t0 = performance.now(); const adapter = await load(); post('progress', { value: 100, message: 'Model is ready.' }); post('ready', { backend: adapter.features ? 'WebGPU · adapter features detected' : 'WebGPU' , loadSeconds: (performance.now() - t0) / 1000 }); }
    catch (error) { post('error', { message: error?.message || String(error) }); }
  } else if (data.type === 'stop') { stopped = true; }
  else if (data.type === 'read' && !running) {
    running = true; stopped = false; randomState = 1337;
    try { for (let index = 0; index < data.chunks.length; index += 1) { if (stopped) break; await synthesize(data.chunks[index], index); } post(stopped ? 'stopped' : 'done'); }
    catch (error) { post('error', { message: error?.message || String(error) }); }
    finally { running = false; }
  }
};
