import * as ort from 'onnxruntime-web/jspi';
import {
  createGraph,
  createExperimentalEmbeddingGraph,
  normalizeSameOriginModelBaseUrl,
  resolveModelAssetUrl,
  ROOT,
  REVISION,
} from './model-loader.js';
import { loadFixedVoiceState } from './voice-state.js';
import { floatToHalf, values } from './numeric.js';
import { Tokenizer } from '@huggingface/tokenizers';
import { StreamedEmbeddings } from './streamed-embeddings.js';

const START_SPEECH = 6561; const STOP_SPEECH = 6562; const SILENCE = 4299;
const sessions = {}; let tokenizer; let stopped = false; let running = false; let speakerConditioning; let randomState = 1337;
let streamedEmbeddings = null;

function post(type, extra = {}) { self.postMessage({ type, ...extra }); }
async function fetchJson(path, modelBaseUrl) {
  const url = resolveModelAssetUrl(modelBaseUrl, path);
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not download ${path}: ${response.status}`);
  return response.json();
}
async function load(options = {}) {
  const modelBaseUrl = options.modelBaseUrl == null
    ? ROOT
    : normalizeSameOriginModelBaseUrl(options.modelBaseUrl, self.location.href);
  const voiceStateManifestUrl = new URL(options.voiceStateManifestUrl ?? '/voice/asmr-state.json', self.location.href);
  if (voiceStateManifestUrl.origin !== self.location.origin || voiceStateManifestUrl.username || voiceStateManifestUrl.password) {
    throw new Error('A voice-state manifest override must use the browser app origin.');
  }
  // Check the small voice artifact before downloading any weights.
  speakerConditioning = await loadFixedVoiceState({ ort, manifestUrl: voiceStateManifestUrl.href, dataUrl: '/voice/asmr-state.bin' });
  if (!navigator.gpu) throw new Error('WebGPU is unavailable in this browser.');
  const powerPreference = options.powerPreference ?? 'low-power';
  if (!['low-power', 'high-performance'].includes(powerPreference)) throw new Error('The GPU power preference is invalid.');
  ort.env.webgpu.powerPreference = powerPreference;
  const adapter = await navigator.gpu.requestAdapter({ powerPreference });
  if (!adapter) throw new Error('No WebGPU adapter was found.');
  if (!adapter.features.has('shader-f16')) throw new Error('The selected WebGPU adapter does not support FP16 shaders.');
  // Use the checked adapter. ORT 1.30 requests its own device with the required
  // features and limits. A second adapter request could select another GPU.
  ort.env.webgpu.adapter = adapter;
  ort.env.wasm.numThreads = 1;
  post('progress', { value: 3, message: 'Loading the tokenizer…' });
  const [tokenizerJson, tokenizerConfig] = await Promise.all([
    fetchJson('tokenizer.json', modelBaseUrl),
    fetchJson('tokenizer_config.json', modelBaseUrl),
  ]);
  tokenizer = new Tokenizer(tokenizerJson, tokenizerConfig);
  if (options.embeddingManifestUrl && options.losslessEmbeddingManifestUrl) {
    throw new Error('Select either Q4 or lossless streamed embeddings.');
  }
  if (options.losslessEmbeddingManifestUrl) {
    streamedEmbeddings = await StreamedEmbeddings.load(options.losslessEmbeddingManifestUrl);
  }
  const names = streamedEmbeddings
    ? ['language_model_q4f16', 'conditional_decoder_q4']
    : ['embed_tokens_fp16', 'language_model_q4f16', 'conditional_decoder_q4'];
  for (let index = 0; index < names.length; index += 1) {
    const name = names[index];
    post('progress', { value: 8 + index * (90 / names.length), message: `Loading graph ${index + 1} of ${names.length}…` });
    const locations = {};
    if (name === 'language_model_q4f16') {
      locations.logits = 'cpu';
      for (let layer = 0; layer < 12; layer += 1) {
        locations[`present.${layer}.key`] = 'gpu-buffer';
        locations[`present.${layer}.value`] = 'gpu-buffer';
      }
    }
    sessions[name] = name === 'embed_tokens_fp16' && options.embeddingManifestUrl
      ? await createExperimentalEmbeddingGraph(ort, options.embeddingManifestUrl)
      : await createGraph(ort, name, locations, { baseUrl: modelBaseUrl });
  }
  return { modelBaseUrl, voiceStateManifestUrl: voiceStateManifestUrl.href, powerPreference };
}
async function releaseModel() {
  streamedEmbeddings?.clear(); streamedEmbeddings = null;
  for (const session of Object.values(sessions)) { try { await session.release(); } catch {} }
  for (const name of Object.keys(sessions)) delete sessions[name];
  for (const value of Object.values(speakerConditioning || {})) dispose(value);
  speakerConditioning = null;
}
function tensor(type, data, dims) { return new ort.Tensor(type, data, dims); }
function int64(data, dims) { return tensor('int64', BigInt64Array.from(data, (value) => BigInt(value)), dims); }
function dispose(t) { try { t?.dispose(); } catch {} }
function random() { randomState = (Math.imul(randomState, 1664525) + 1013904223) >>> 0; return randomState / 0x100000000; }
function sessionTensor(session, name, data, dims, fallback = 'float16') {
  const meta = session.inputMetadata?.find((item) => item.name === name); const type = meta?.isTensor ? meta.type : fallback;
  if (type === 'float16') return tensor(type, data instanceof Uint16Array ? data : Uint16Array.from(data, floatToHalf), dims);
  return tensor(type, data instanceof Float32Array ? data : Float32Array.from(data, (x) => typeof x === 'bigint' ? Number(x) : x), dims);
}
function sampleLogits(logits, history) {
  const allScores = values(logits); const vocabSize = logits.dims.at(-1); const scores = allScores.subarray(allScores.length - vocabSize);
  if (vocabSize !== STOP_SPEECH + 1 || scores.length !== vocabSize
    || scores.some((score) => Number.isNaN(score) || score === Infinity)
    || !scores.some(Number.isFinite)) throw new Error('The language model returned invalid speech logits.');
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
  const owned = new Set();
  const keep = (value) => { owned.add(value); return value; };
  const free = (value) => { if (owned.delete(value)) dispose(value); };
  const run = async (session, feeds) => {
    const outputs = await session.run(feeds);
    Object.values(outputs).forEach(keep);
    return outputs;
  };
  const integers = (data, dims) => keep(int64(data, dims));
  const floats = (session, name, data, dims) => keep(sessionTensor(session, name, data, dims));
  try {
    const embed = sessions.embed_tokens_fp16; const lm = sessions.language_model_q4f16; const decoder = sessions.conditional_decoder_q4;
    const embedIds = async (ids) => {
      if (streamedEmbeddings) {
        return keep(tensor('float32', await streamedEmbeddings.lookup(ids), [1, ids.length, 768]));
      }
      const input = integers(ids, [1, ids.length]);
      const output = await run(embed, { input_ids: input }); free(input);
      return output[embed.outputNames[0]];
    };
    const { audioFeatures, audioTokens, speakerEmbeddings, speakerFeatures } = speakerConditioning;
    const ids = tokenizer.encode(text, { add_special_tokens: true }).ids;
    if (ids.length < 3 || ids.at(-1) !== 50256 || ids.at(-2) !== 50256) {
      throw new Error('The tokenizer did not supply the embedding graph’s two speech-start markers.');
    }
    const textEmbeddings = await embedIds(ids);
    const audioData = values(audioFeatures); const textData = values(textEmbeddings);
    const featureDim = audioFeatures.dims.at(-1);
    if (featureDim !== textEmbeddings.dims.at(-1)) throw new Error('Voice and text embedding dimensions differ.');
    const sequence = audioFeatures.dims.at(-2) + textEmbeddings.dims.at(-2);
    const combined = new Float32Array(sequence * featureDim);
    combined.set(audioData); combined.set(textData, audioData.length);
    const embedsName = 'inputs_embeds';
    const cacheInputs = lm.inputNames.filter((name) => name.startsWith('past_key_values.'));
    const cacheOutputs = cacheInputs.map((name) => name.replace('past_key_values.', 'present.'));
    if (cacheInputs.length !== 24 || cacheOutputs.some((name) => !lm.outputNames.includes(name))) throw new Error('The language model has an unexpected KV cache contract.');
    let feeds = {
      [embedsName]: floats(lm, embedsName, combined, [1, sequence, featureDim]),
      attention_mask: integers(Array(sequence).fill(1), [1, sequence]),
      position_ids: integers(Array.from({ length: sequence }, (_, i) => i), [1, sequence]),
    };
    for (const name of cacheInputs) feeds[name] = floats(lm, name, new Float32Array(0), [1, 12, 0, 64]);
    let result = await run(lm, feeds);
    Object.values(feeds).forEach(free);
    free(textEmbeddings);
    const history = [START_SPEECH];
    const tokenLimit = 256;
    let reachedEos = false;
    for (let step = 0; step < tokenLimit; step += 1) {
      if (stopped) return null;
      const tokenId = sampleLogits(result.logits, history);
      if (tokenId === STOP_SPEECH) { reachedEos = true; break; }
      history.push(tokenId);
      if (step + 1 === tokenLimit) break;
      const tokenVector = await embedIds([tokenId]);
      const past = sequence + step;
      feeds = {
        [embedsName]: floats(lm, embedsName, values(tokenVector), [1, 1, featureDim]),
        attention_mask: integers(Array(past + 1).fill(1), [1, past + 1]),
        position_ids: integers([past], [1, 1]),
      };
      for (let i = 0; i < cacheInputs.length; i += 1) feeds[cacheInputs[i]] = result[cacheOutputs[i]];
      const nextResult = await run(lm, feeds);
      Object.values(feeds).forEach(free);
      Object.values(result).forEach(free);
      free(tokenVector);
      result = nextResult;
    }
    Object.values(result).forEach(free);
    if (stopped) return null;
    // A truncated passage cannot be presented as a complete reading.
    if (!reachedEos) throw new Error(`Passage ${index + 1} reached the 256-token limit. Use shorter passages.`);
    const speechTokens = history.slice(1);
    if (!speechTokens.length) throw new Error(`No speech tokens were generated for passage ${index + 1}.`);
    const decoderTokens = [...Array.from(audioTokens.data, Number), ...speechTokens, SILENCE, SILENCE, SILENCE];
    const decodeResult = await run(decoder, {
      speech_tokens: integers(decoderTokens, [1, decoderTokens.length]),
      speaker_embeddings: speakerEmbeddings, speaker_features: speakerFeatures,
    });
    const audio = values(decodeResult[decoder.outputNames[0]]);
    if (!audio.length || audio.some((value) => !Number.isFinite(value))) throw new Error('The decoder returned invalid audio.');
    return { index, text, sampleRate: 24000, synthesisSeconds: (performance.now() - t0) / 1000,
      speechTokens, truncated: false, embeddingLookup: streamedEmbeddings?.snapshot() || null, audio: audio.buffer };
  } finally {
    for (const value of owned) dispose(value);
  }
}

let activeRun = null; let credits = 0; let wakeCredit;
function wake() { wakeCredit?.(); wakeCredit = undefined; }
async function waitForCredit() {
  while (!stopped && credits === 0) await new Promise((resolve) => { wakeCredit = resolve; });
  if (stopped) return false;
  credits -= 1; return true;
}
self.onmessage = async ({ data }) => {
  if (data.type === 'load') {
    try {
      const t0 = performance.now(); const { modelBaseUrl, voiceStateManifestUrl, powerPreference } = await load(data);
      post('ready', { backend: 'WebGPU', loadSeconds: (performance.now() - t0) / 1000,
        modelIdentity: { revision: REVISION, modelBaseUrl, powerPreference,
          embedding: data.losslessEmbeddingManifestUrl || data.embeddingManifestUrl || 'embed_tokens_fp16',
          embeddingMode: streamedEmbeddings ? 'lossless-fp16-shards' : data.embeddingManifestUrl ? 'experimental-q4' : 'fp16-onnx',
          voiceState: voiceStateManifestUrl, fittedAdapterApplied: false } });
    } catch (error) { await releaseModel(); post('error', { message: error?.message || String(error) }); }
  } else if (data.type === 'stop' && data.runId === activeRun) {
    stopped = true; wake();
  } else if (data.type === 'consumed' && data.runId === activeRun) {
    credits = Math.min(2, credits + 1); wake();
  } else if (data.type === 'read' && !running) {
    running = true; stopped = false; activeRun = data.runId;
    credits = 2; randomState = (data.seed ?? 1337) >>> 0;
    try {
      for (let index = 0; index < data.chunks.length; index += 1) {
        if (!await waitForCredit()) break;
        const output = await synthesize(data.chunks[index], index);
        if (stopped) break;
        self.postMessage({ type: 'chunk', runId: activeRun, ...output }, [output.audio]);
      }
      post(stopped ? 'stopped' : 'done', { runId: activeRun });
    } catch (error) { post('error', { runId: activeRun, message: error?.message || String(error) }); }
    finally { running = false; }
  }
};
