import {
  getPinnedAsset,
  REFERENCE_SHA256,
  REVISION,
  SHA256SUMS_SHA256,
  sha256Blob,
} from './model-loader.js';

const SCHEMA_VERSION = 1;
const FORMAT = 'chatterbox-nano-reference-state-v1';
const MODEL_REPOSITORY = 'owensong/chatterbox-nano-ONNX';
const REFERENCE_PATH = 'browser_tts/public/voice/asmr_t3_seed47_fit.wav';
const MAX_STATE_BYTES = 16 * 1024 * 1024;
const DATA_FILE = 'asmr-state.bin';

const ENCODER_ASSET = getPinnedAsset('speech_encoder_q4f16');
const TENSOR_SPEC = Object.freeze([
  Object.freeze({ role: 'audio_features', name: 'audio_features', type: 'float32', rank: 3, lastDim: 768 }),
  Object.freeze({ role: 'audio_tokens', name: 'audio_tokens', type: 'int64', rank: 2 }),
  Object.freeze({ role: 'speaker_embeddings', name: 'speaker_embeddings', type: 'float32', rank: 2, lastDim: 192 }),
  Object.freeze({ role: 'speaker_features', name: 'speaker_features', type: 'float32', rank: 3, lastDim: 80 }),
]);

const ELEMENT_BYTES = Object.freeze({ float32: 4, int64: 8 });

function invalid(message) {
  throw new Error(`Invalid fixed voice state: ${message}`);
}

function isSha256(value) {
  return typeof value === 'string' && /^[a-f0-9]{64}$/.test(value);
}

function requireObject(value, label) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) invalid(`${label} must be an object.`);
}

function validateManifest(manifest) {
  requireObject(manifest, 'manifest');
  if (manifest.schema_version !== SCHEMA_VERSION) invalid(`unsupported schema version ${manifest.schema_version}.`);
  if (manifest.format !== FORMAT) invalid(`unsupported format ${manifest.format}.`);

  requireObject(manifest.model, 'model provenance');
  if (manifest.model.repo_id !== MODEL_REPOSITORY) invalid('model repository does not match the pinned conversion.');
  if (manifest.model.revision !== REVISION) invalid(`graph revision must be ${REVISION}.`);
  if (manifest.model.sha256sums_sha256 !== SHA256SUMS_SHA256) invalid('upstream SHA256SUMS digest does not match the pinned conversion.');

  const graph = manifest.model.encoder_graph;
  const weights = manifest.model.encoder_external_weights;
  requireObject(graph, 'encoder graph provenance');
  requireObject(weights, 'encoder weight provenance');
  if (graph.path !== ENCODER_ASSET.graphPath || graph.sha256 !== ENCODER_ASSET.graphSha256) {
    invalid('conditioning graph does not match the pinned speech encoder.');
  }
  if (weights.path !== `onnx/${ENCODER_ASSET.externalPath}` || weights.sha256 !== ENCODER_ASSET.externalSha256) {
    invalid('conditioning weights do not match the pinned speech encoder.');
  }

  requireObject(manifest.reference, 'reference provenance');
  if (manifest.reference.path !== REFERENCE_PATH || manifest.reference.sha256 !== REFERENCE_SHA256) {
    invalid('reference audio does not match the pinned ASMR recording.');
  }
  if (manifest.reference.sample_rate !== 24000 || manifest.reference.channels !== 1) {
    invalid('reference audio metadata must be mono at 24 kHz.');
  }

  requireObject(manifest.encoder_execution, 'encoder execution provenance');
  if (manifest.encoder_execution.provider !== 'CPUExecutionProvider'
      || manifest.encoder_execution.onnxruntime_version !== '1.29.0'
      || manifest.encoder_execution.graph_optimization_level !== 'ORT_ENABLE_ALL') {
    invalid('conditioning must record the pinned CPU ONNX Runtime export settings.');
  }

  requireObject(manifest.conditioning, 'conditioning provenance');
  if (manifest.conditioning.source !== 'generated_reference'
      || manifest.conditioning.zero_shot !== false
      || manifest.conditioning.speaker_specific !== true
      || manifest.conditioning.fitted_adapter_applied !== false) {
    invalid('conditioning provenance must identify speaker-specific generated-reference features without a fitted adapter.');
  }

  requireObject(manifest.data, 'binary data metadata');
  if (manifest.data.file !== DATA_FILE) invalid(`binary filename must be ${DATA_FILE}.`);
  if (!Number.isSafeInteger(manifest.data.size_bytes)
      || manifest.data.size_bytes <= 0
      || manifest.data.size_bytes > MAX_STATE_BYTES) {
    invalid(`binary size must be between 1 byte and ${MAX_STATE_BYTES} bytes.`);
  }
  if (!isSha256(manifest.data.sha256)) invalid('binary SHA-256 must contain 64 lowercase hexadecimal characters.');

  if (!Array.isArray(manifest.tensors) || manifest.tensors.length !== TENSOR_SPEC.length) {
    invalid(`exactly ${TENSOR_SPEC.length} tensors are required.`);
  }

  let nextOffset = 0;
  for (let index = 0; index < TENSOR_SPEC.length; index += 1) {
    const expected = TENSOR_SPEC[index];
    const tensor = manifest.tensors[index];
    requireObject(tensor, `tensor ${expected.role}`);
    if (tensor.role !== expected.role || tensor.name !== expected.name) {
      invalid(`tensor ${index + 1} must be ${expected.name} (${expected.role}).`);
    }
    if (tensor.type !== expected.type) invalid(`${expected.name} must use ONNX tensor type ${expected.type}.`);
    if (!Array.isArray(tensor.dims)
        || tensor.dims.length !== expected.rank
        || tensor.dims.some((dim) => !Number.isSafeInteger(dim) || dim <= 0)) {
      invalid(`${expected.name} must have ${expected.rank} positive integer dimensions.`);
    }
    if (tensor.dims[0] !== 1) invalid(`${expected.name} must have batch size 1.`);
    if (expected.lastDim && tensor.dims.at(-1) !== expected.lastDim) {
      invalid(`${expected.name} must have last dimension ${expected.lastDim}.`);
    }
    const elementCount = tensor.dims.reduce((product, dim) => product * dim, 1);
    if (!Number.isSafeInteger(elementCount)) invalid(`${expected.name} has an unsafe element count.`);
    const expectedBytes = elementCount * ELEMENT_BYTES[expected.type];
    if (!Number.isSafeInteger(expectedBytes) || tensor.byte_length !== expectedBytes) {
      invalid(`${expected.name} byte length does not match its type and dimensions.`);
    }
    if (tensor.offset !== nextOffset) invalid(`${expected.name} offset must be ${nextOffset}.`);
    if (tensor.serialization_round_trip_exact !== true) invalid(`${expected.name} lacks an exact CPU serialization check.`);
    nextOffset += expectedBytes;
  }

  if (manifest.data.size_bytes !== nextOffset) invalid('binary size does not equal the tensor byte ranges.');
  return manifest;
}

function createTypedArray(type, buffer, offset, byteLength) {
  const bytes = buffer.slice(offset, offset + byteLength);
  if (type === 'float32') return new Float32Array(bytes);
  if (type === 'int64') return new BigInt64Array(bytes);
  invalid(`unsupported ONNX tensor type ${type}.`);
}

async function responseBlob(fetchImpl, url, label) {
  const response = await fetchImpl(url);
  if (!response.ok) throw new Error(`Could not load ${label}: HTTP ${response.status}.`);
  return response.blob();
}

export async function loadFixedVoiceState({
  ort,
  fetchImpl = globalThis.fetch,
  manifestUrl = '/voice/asmr-state.json',
  dataUrl = '/voice/asmr-state.bin',
} = {}) {
  if (!ort?.Tensor) throw new TypeError('An ONNX Runtime module is required to load the fixed voice state.');
  if (typeof fetchImpl !== 'function') throw new Error('Fetch is unavailable in this worker.');

  const manifestBlob = await responseBlob(fetchImpl, manifestUrl, 'the fixed voice state manifest');
  if (manifestBlob.size > 256 * 1024) invalid('manifest is larger than 256 KiB.');

  let manifest;
  try {
    manifest = JSON.parse(await manifestBlob.text());
  } catch (error) {
    throw new Error(`Could not parse the fixed voice state manifest: ${error?.message || String(error)}.`);
  }
  validateManifest(manifest);

  const dataBlob = await responseBlob(fetchImpl, dataUrl, 'the fixed voice state binary');
  if (dataBlob.size !== manifest.data.size_bytes) invalid('binary size does not match the manifest.');
  const actualSha256 = await sha256Blob(dataBlob);
  if (actualSha256 !== manifest.data.sha256) invalid(`binary SHA-256 mismatch: expected ${manifest.data.sha256}, received ${actualSha256}.`);

  const bytes = await dataBlob.arrayBuffer();
  const tensors = {};
  try {
    for (let index = 0; index < TENSOR_SPEC.length; index += 1) {
      const spec = TENSOR_SPEC[index];
      const entry = manifest.tensors[index];
      const data = createTypedArray(entry.type, bytes, entry.offset, entry.byte_length);
      if (entry.type === 'float32' && data.some((value) => !Number.isFinite(value))) invalid(`${spec.name} contains non-finite values.`);
      if (entry.type === 'int64' && data.some((value) => value < 0n || value >= 6563n)) invalid('audio_tokens contains an invalid speech token.');
      tensors[spec.role] = new ort.Tensor(entry.type, data, entry.dims);
    }
  } catch (error) {
    for (const tensor of Object.values(tensors)) {
      try { tensor.dispose?.(); } catch {}
    }
    throw error;
  }

  return {
    audioFeatures: tensors.audio_features,
    audioTokens: tensors.audio_tokens,
    speakerEmbeddings: tensors.speaker_embeddings,
    speakerFeatures: tensors.speaker_features,
  };
}

export { validateManifest as validateFixedVoiceStateManifest };
