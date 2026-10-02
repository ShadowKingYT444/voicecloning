export const REVISION = '4a66d7dab72a9e98f24b515d49a1d7a81632df2e';
export const ROOT = `https://huggingface.co/owensong/chatterbox-nano-ONNX/resolve/${REVISION}/`;
export const SHA256SUMS_SHA256 = 'ecd0b0ee1fde0f9342564fda72fe1ebb4907baa6c9d013c09052bcad5220dcf4';
export const REFERENCE_SHA256 = '67f94a868976b22a47bce8fd00a873d4c5b7085ba6eedd698f8a898a26ce76c0';

const ASSETS = Object.freeze({
  embed_tokens_fp16: Object.freeze({
    graphPath: 'onnx/embed_tokens_fp16.onnx',
    graphSha256: '019d257243774091d78c2ad91c2c0f61e4e442740cb7b3b00b5a89109417b18d',
    externalPath: 'embed_tokens_fp16.onnx_data',
    externalSha256: 'bcd7b35ae4f206932e2491cb60b42ebb80f6d8facfdb53ba7d7449ad00a3237b',
  }),
  speech_encoder_q4f16: Object.freeze({
    graphPath: 'onnx/speech_encoder_q4f16.onnx',
    graphSha256: '29e249f59eaf95015527588b955e5286c7ee4524e7bb54a2f8d589b838e8aed2',
    externalPath: 'speech_encoder_q4f16.onnx_data',
    externalSha256: '55d89bd87fd36be48b2e831c99c2e309d432169119049066f9f22fcfe517798d',
  }),
  language_model_q4f16: Object.freeze({
    graphPath: 'onnx/language_model_q4f16.onnx',
    graphSha256: '8fe9620856d86b8a0235041d7fd2a8a47292da50828044bab5740b246e66e65b',
    externalPath: 'language_model_q4f16.onnx_data',
    externalSha256: '8f2fdc616373c9ccaebbf2db3210dc80c860922a672898bca111eb75f6d3abd2',
  }),
  conditional_decoder_q4: Object.freeze({
    graphPath: 'onnx/conditional_decoder_q4.onnx',
    graphSha256: '745faa9c2e2494a81e47f2a72601ac248639fd417ae53d999c5068bc441d1f97',
    externalPath: 'conditional_decoder_q4.onnx_data',
    externalSha256: 'b5c5317e0b79a1a19dd3d5e2b2091ea06b15716716ab801a54eaeb906c6971ec',
  }),
});

const RUNTIME_GRAPHS = new Set([
  'embed_tokens_fp16',
  'language_model_q4f16',
  'conditional_decoder_q4',
]);

const EMBEDDING_Q4_CANDIDATE_SCHEMA = 'browser_tts.embedding-q4-candidate/v1';
const EMBEDDING_Q4_SOURCE_FILES = Object.freeze({
  'onnx/embed_tokens_fp16.onnx': Object.freeze({
    sha256: '019d257243774091d78c2ad91c2c0f61e4e442740cb7b3b00b5a89109417b18d',
    bytes: 1520,
  }),
  'onnx/embed_tokens_fp16.onnx_data': Object.freeze({
    sha256: 'bcd7b35ae4f206932e2491cb60b42ebb80f6d8facfdb53ba7d7449ad00a3237b',
    bytes: 87304704,
  }),
});
const EMBEDDING_Q4_TARGET_GRAPH = 'embed_tokens_gather_q4.onnx';
const EMBEDDING_Q4_TARGET_DATA = 'embed_tokens_gather_q4.onnx.data';

const SHA256_K = Uint32Array.from([
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]);

const SHA256_INITIAL = Uint32Array.from([
  0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
  0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
]);

function rotateRight(value, count) {
  return (value >>> count) | (value << (32 - count));
}

class Sha256 {
  constructor() {
    this.state = SHA256_INITIAL.slice();
    this.buffer = new Uint8Array(64);
    this.bufferLength = 0;
    this.byteLength = 0n;
    this.words = new Uint32Array(64);
  }

  update(bytes) {
    if (!(bytes instanceof Uint8Array)) throw new TypeError('SHA-256 input must be a Uint8Array.');
    this.byteLength += BigInt(bytes.byteLength);
    let offset = 0;

    if (this.bufferLength) {
      const copied = Math.min(64 - this.bufferLength, bytes.byteLength);
      this.buffer.set(bytes.subarray(0, copied), this.bufferLength);
      this.bufferLength += copied;
      offset = copied;
      if (this.bufferLength === 64) {
        this.compress(this.buffer, 0);
        this.bufferLength = 0;
      }
    }

    while (offset + 64 <= bytes.byteLength) {
      this.compress(bytes, offset);
      offset += 64;
    }

    if (offset < bytes.byteLength) {
      this.buffer.set(bytes.subarray(offset), 0);
      this.bufferLength = bytes.byteLength - offset;
    }
  }

  compress(bytes, offset) {
    const view = new DataView(bytes.buffer, bytes.byteOffset + offset, 64);
    const words = this.words;
    for (let index = 0; index < 16; index += 1) words[index] = view.getUint32(index * 4, false);
    for (let index = 16; index < 64; index += 1) {
      const x = words[index - 15];
      const y = words[index - 2];
      const s0 = rotateRight(x, 7) ^ rotateRight(x, 18) ^ (x >>> 3);
      const s1 = rotateRight(y, 17) ^ rotateRight(y, 19) ^ (y >>> 10);
      words[index] = (words[index - 16] + s0 + words[index - 7] + s1) >>> 0;
    }

    let a = this.state[0];
    let b = this.state[1];
    let c = this.state[2];
    let d = this.state[3];
    let e = this.state[4];
    let f = this.state[5];
    let g = this.state[6];
    let h = this.state[7];
    for (let index = 0; index < 64; index += 1) {
      const sum1 = rotateRight(e, 6) ^ rotateRight(e, 11) ^ rotateRight(e, 25);
      const choose = (e & f) ^ (~e & g);
      const first = (h + sum1 + choose + SHA256_K[index] + words[index]) >>> 0;
      const sum0 = rotateRight(a, 2) ^ rotateRight(a, 13) ^ rotateRight(a, 22);
      const majority = (a & b) ^ (a & c) ^ (b & c);
      const second = (sum0 + majority) >>> 0;
      const nextA = (first + second) >>> 0;
      const nextE = (d + first) >>> 0;
      h = g;
      g = f;
      f = e;
      e = nextE;
      d = c;
      c = b;
      b = a;
      a = nextA;
    }

    this.state[0] = (this.state[0] + a) >>> 0;
    this.state[1] = (this.state[1] + b) >>> 0;
    this.state[2] = (this.state[2] + c) >>> 0;
    this.state[3] = (this.state[3] + d) >>> 0;
    this.state[4] = (this.state[4] + e) >>> 0;
    this.state[5] = (this.state[5] + f) >>> 0;
    this.state[6] = (this.state[6] + g) >>> 0;
    this.state[7] = (this.state[7] + h) >>> 0;
  }

  digestHex() {
    const bitLength = this.byteLength * 8n;
    const paddedLength = this.bufferLength < 56 ? 64 : 128;
    const padded = new Uint8Array(paddedLength);
    padded.set(this.buffer.subarray(0, this.bufferLength));
    padded[this.bufferLength] = 0x80;
    const view = new DataView(padded.buffer);
    view.setUint32(paddedLength - 8, Number((bitLength >> 32n) & 0xffffffffn), false);
    view.setUint32(paddedLength - 4, Number(bitLength & 0xffffffffn), false);
    for (let offset = 0; offset < paddedLength; offset += 64) this.compress(padded, offset);
    return Array.from(this.state, (word) => word.toString(16).padStart(8, '0')).join('');
  }
}

export function sha256Bytes(bytes) {
  const digest = new Sha256();
  digest.update(bytes);
  return digest.digestHex();
}

export async function sha256Blob(blob) {
  if (!blob || typeof blob.stream !== 'function') {
    throw new TypeError('This browser cannot stream a Blob for SHA-256 verification.');
  }

  const digest = new Sha256();
  const reader = blob.stream().getReader();
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      digest.update(value);
    }
  } finally {
    reader.releaseLock?.();
  }
  return digest.digestHex();
}

export function assertJspiSupport(webAssembly = globalThis.WebAssembly) {
  if (typeof webAssembly?.Suspending !== 'function' || typeof webAssembly?.promising !== 'function') {
    throw new Error('This browser does not support WebAssembly JSPI. Use a browser with the JSPI APIs enabled.');
  }
}

function directoryUrl(baseUrl, referenceUrl = globalThis.location?.href || ROOT) {
  if (typeof baseUrl !== 'string' || !baseUrl.trim()) {
    throw new TypeError('A non-empty model base URL is required.');
  }
  const base = new URL(baseUrl, referenceUrl);
  if (base.search || base.hash) throw new Error('A model base URL cannot contain a query or fragment.');
  if (base.username || base.password) throw new Error('A model base URL cannot contain URL credentials.');
  if (!base.pathname.endsWith('/')) base.pathname += '/';
  return base;
}

export function normalizeSameOriginModelBaseUrl(modelBaseUrl, pageUrl = globalThis.location?.href) {
  if (typeof pageUrl !== 'string' || !pageUrl) {
    throw new Error('The page URL is required to validate a local model base URL.');
  }
  const page = new URL(pageUrl);
  const base = directoryUrl(modelBaseUrl, `${page.origin}/`);
  if (base.origin !== page.origin || (base.protocol !== 'http:' && base.protocol !== 'https:')) {
    throw new Error('A model base URL override must use the same HTTP or HTTPS origin as the browser app.');
  }
  return base.href;
}

export function resolveModelAssetUrl(baseUrl, assetPath) {
  if (typeof assetPath !== 'string' || !assetPath || assetPath.startsWith('/')) {
    throw new TypeError('A model asset path must be a non-empty relative path.');
  }
  return new URL(assetPath, directoryUrl(baseUrl)).href;
}

export function resolvePinnedRuntimeAsset(name, baseUrl = ROOT) {
  if (!RUNTIME_GRAPHS.has(name)) {
    throw new Error(`Unknown pinned runtime ONNX graph: ${name}.`);
  }
  const asset = ASSETS[name];
  return Object.freeze({
    graphUrl: resolveModelAssetUrl(baseUrl, asset.graphPath),
    graphSha256: asset.graphSha256,
    externalUrl: resolveModelAssetUrl(baseUrl, `onnx/${asset.externalPath}`),
    externalPath: asset.externalPath,
    externalSha256: asset.externalSha256,
  });
}

async function checkedResponse(fetchImpl, url, label) {
  let response;
  try { response = await fetchImpl(url); }
  catch (error) {
    throw new Error(`Could not download pinned ${label}: ${error?.message || String(error)}.`, { cause: error });
  }
  if (!response.ok) throw new Error(`Could not download pinned ${label}: HTTP ${response.status}.`);
  return response;
}

async function checkedGraph(fetchImpl, asset, graphUrl) {
  const response = await checkedResponse(fetchImpl, graphUrl, `graph ${asset.graphPath}`);
  const bytes = new Uint8Array(await response.arrayBuffer());
  const actualSha256 = sha256Bytes(bytes);
  if (actualSha256 !== asset.graphSha256) {
    throw new Error(`Pinned graph hash mismatch for ${asset.graphPath}: expected ${asset.graphSha256}, received ${actualSha256}.`);
  }
  return bytes;
}

async function checkedExternalData(fetchImpl, asset, externalUrl) {
  const path = `onnx/${asset.externalPath}`;
  const response = await checkedResponse(fetchImpl, externalUrl, `external weights ${path}`);
  const blob = await response.blob();
  const actualSha256 = await sha256Blob(blob);
  if (actualSha256 !== asset.externalSha256) {
    throw new Error(`Pinned external-weight hash mismatch for ${path}: expected ${asset.externalSha256}, received ${actualSha256}.`);
  }
  return blob;
}

function getExecutionProviders(executionProvider, gpuDevice) {
  if (!['webgpu', 'wasm'].includes(executionProvider)) {
    throw new Error('Only the explicit WebGPU or WASM provider is supported.');
  }
  if (gpuDevice && executionProvider !== 'webgpu') {
    throw new Error('A supplied GPU device requires the WebGPU execution provider.');
  }
  return gpuDevice ? [{ name: 'webgpu', device: gpuDevice }] : [executionProvider];
}

export async function createGraph(ort, name, outputLocations = {}, options = {}) {
  if (!ort?.InferenceSession?.create) throw new TypeError('An ONNX Runtime module is required.');
  if (!RUNTIME_GRAPHS.has(name)) {
    if (name === 'speech_encoder_q4f16') {
      throw new Error('The speech encoder is offline-only. Load the fixed precomputed voice state instead.');
    }
    throw new Error(`Unknown pinned ONNX graph: ${name}.`);
  }

  assertJspiSupport(options.webAssembly);
  const executionProvider = options.executionProvider ?? 'webgpu';
  const executionProviders = getExecutionProviders(executionProvider, options.gpuDevice);
  const asset = ASSETS[name];
  const fetchImpl = options.fetchImpl || globalThis.fetch;
  if (typeof fetchImpl !== 'function') throw new Error('Fetch is unavailable in this worker.');
  const locations = resolvePinnedRuntimeAsset(name, options.baseUrl || ROOT);

  const [graph, externalData] = await Promise.all([
    checkedGraph(fetchImpl, asset, locations.graphUrl),
    checkedExternalData(fetchImpl, asset, locations.externalUrl),
  ]);

  return ort.InferenceSession.create(graph, {
    executionProviders,
    graphOptimizationLevel: 'all',
    externalData: [{ path: asset.externalPath, data: externalData }],
    preferredOutputLocation: outputLocations,
  });
}

function sameNumbers(left, right) {
  return Array.isArray(left)
    && left.length === right.length
    && left.every((value, index) => value === right[index]);
}

function candidateObject(value, label) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`Invalid experimental embedding candidate: ${label} must be an object.`);
  }
}

function candidateMetric(value, label) {
  candidateObject(value, label);
  for (const key of ['elements_compared', 'max_absolute_error', 'mean_absolute_error', 'root_mean_square_error']) {
    if (!Number.isFinite(value[key]) || value[key] < 0) {
      throw new Error(`Invalid experimental embedding candidate: ${label}.${key} must be finite and non-negative.`);
    }
  }
  if (!Number.isSafeInteger(value.elements_compared) || value.elements_compared <= 0) {
    throw new Error(`Invalid experimental embedding candidate: ${label}.elements_compared must be a positive integer.`);
  }
}

export function validateExperimentalEmbeddingManifest(manifest) {
  candidateObject(manifest, 'manifest');
  if (manifest.schema !== EMBEDDING_Q4_CANDIDATE_SCHEMA) {
    throw new Error(`Invalid experimental embedding candidate: schema must be ${EMBEDDING_Q4_CANDIDATE_SCHEMA}.`);
  }
  if (manifest.candidate_status !== 'experimental_unpromoted' || manifest.production_gate !== false) {
    throw new Error('Invalid experimental embedding candidate: only an unpromoted candidate with production_gate=false can load.');
  }

  candidateObject(manifest.source, 'source');
  const source = manifest.source;
  if (source.repo_id !== 'owensong/chatterbox-nano-ONNX' || source.revision !== REVISION) {
    throw new Error('Invalid experimental embedding candidate: source repository or revision is not pinned.');
  }
  candidateObject(source.sha256sums, 'source.sha256sums');
  if (source.sha256sums.path !== 'SHA256SUMS' || source.sha256sums.sha256 !== SHA256SUMS_SHA256) {
    throw new Error('Invalid experimental embedding candidate: source SHA256SUMS pin does not match.');
  }
  candidateObject(source.files, 'source.files');
  if (Object.keys(source.files).length !== Object.keys(EMBEDDING_Q4_SOURCE_FILES).length) {
    throw new Error('Invalid experimental embedding candidate: source file manifest must contain only the two pinned embedding assets.');
  }
  for (const [path, expected] of Object.entries(EMBEDDING_Q4_SOURCE_FILES)) {
    candidateObject(source.files[path], `source.files[${path}]`);
    if (source.files[path].sha256 !== expected.sha256
        || source.files[path].bytes !== expected.bytes) {
      throw new Error(`Invalid experimental embedding candidate: source file pin does not match ${path}.`);
    }
  }

  candidateObject(source.graph, 'source.graph');
  const sourceGraph = source.graph;
  if (sourceGraph.input_name !== 'input_ids'
      || sourceGraph.output_name !== 'inputs_embeds'
      || sourceGraph.input_type !== 'int64'
      || sourceGraph.output_type !== 'float'
      || sourceGraph.embedding_dim !== 768
      || !Array.isArray(sourceGraph.gather_tables)
      || sourceGraph.gather_tables.length !== 2) {
    throw new Error('Invalid experimental embedding candidate: source graph contract is not the pinned embedding graph.');
  }
  const expectedGatherTables = [
    { initializer: 'text_emb.weight', shape: [50276, 768], dtype: 'float16', gather_axis: 0 },
    { initializer: 'speech_emb.weight', shape: [6563, 768], dtype: 'float16', gather_axis: 0 },
  ];
  for (let index = 0; index < expectedGatherTables.length; index += 1) {
    const expected = expectedGatherTables[index];
    const matches = sourceGraph.gather_tables.filter((table) => table?.initializer === expected.initializer);
    if (matches.length !== 1) throw new Error(`Invalid experimental embedding candidate: expected one ${expected.initializer} table.`);
    const actual = matches[0];
    candidateObject(actual, `source.graph.gather_tables[${index}]`);
    if (actual.initializer !== expected.initializer
        || !sameNumbers(actual.shape, expected.shape)
        || actual.dtype !== expected.dtype
        || actual.gather_axis !== expected.gather_axis) {
      throw new Error(`Invalid experimental embedding candidate: source Gather table ${expected.initializer} does not match.`);
    }
  }
  candidateObject(sourceGraph.hybrid_tail_contract, 'source.graph.hybrid_tail_contract');
  const tail = sourceGraph.hybrid_tail_contract;
  if (tail.text_slice_end_exclusive !== -2
      || tail.speech_slice_start !== -2
      || tail.sentinel_token_id !== 50256
      || tail.speech_replacement_token_id !== 6561) {
    throw new Error('Invalid experimental embedding candidate: source hybrid-token tail contract is not pinned.');
  }

  candidateObject(manifest.quantization, 'quantization');
  const quantization = manifest.quantization;
  if (quantization.api !== 'onnxruntime.quantization.matmul_nbits_quantizer.MatMulNBitsQuantizer'
      || quantization.ort_version !== '1.29.0'
      || quantization.algorithm !== 'DEFAULT'
      || quantization.bits !== 4
      || quantization.block_size !== 128
      || quantization.is_symmetric !== false
      || quantization.quantized_dtype !== 'UINT4'
      || quantization.quant_format !== 'QOperator'
      || !sameNumbers(quantization.op_types_to_quantize, ['Gather'])
      || quantization.quant_axes?.Gather !== 1
      || quantization.source_gather_axis !== 0
      || quantization.target_operator !== 'com.microsoft::GatherBlockQuantized') {
    throw new Error('Invalid experimental embedding candidate: quantization settings are not the reviewed Gather Q4 profile.');
  }

  candidateObject(manifest.target, 'target');
  const graph = manifest.target.graph;
  const externalData = manifest.target.external_data;
  for (const [entry, path, label] of [
    [graph, EMBEDDING_Q4_TARGET_GRAPH, 'target.graph'],
    [externalData, EMBEDDING_Q4_TARGET_DATA, 'target.external_data'],
  ]) {
    candidateObject(entry, label);
    if (entry.path !== path
        || !/^[a-f0-9]{64}$/.test(entry.sha256 || '')
        || !Number.isSafeInteger(entry.bytes)
        || entry.bytes <= 0) {
      throw new Error(`Invalid experimental embedding candidate: ${label} path, byte count, or SHA-256 is invalid.`);
    }
  }
  if (graph.bytes > 64 * 1024 * 1024 || externalData.bytes > 1024 * 1024 * 1024) {
    throw new Error('Invalid experimental embedding candidate: target files exceed the diagnostic loader size limits.');
  }

  candidateObject(manifest.cpu_component_check, 'cpu_component_check');
  const cpuCheck = manifest.cpu_component_check;
  if (cpuCheck.status !== 'completed'
      || cpuCheck.provider !== 'CPUExecutionProvider'
      || cpuCheck.method !== 'sequential_streamed_full_lookup'
      || cpuCheck.acceptance !== 'descriptive_only_no_gate'
      || cpuCheck.fixed_probe_seed !== 20261001
      || cpuCheck.batch_size !== 32
      || cpuCheck.failure_reason != null
      || cpuCheck.quality_or_promotion_gate !== false) {
    throw new Error('Invalid experimental embedding candidate: the descriptive CPU component check is incomplete.');
  }
  candidateObject(cpuCheck.full_lookup, 'cpu_component_check.full_lookup');
  for (const name of ['text_emb', 'speech_emb']) {
    candidateObject(cpuCheck.full_lookup[name], `cpu_component_check.full_lookup.${name}`);
    const expectedTokens = name === 'text_emb' ? 50276 : 6563;
    if (cpuCheck.full_lookup[name].token_count !== expectedTokens
        || cpuCheck.full_lookup[name].elements_compared !== expectedTokens * 768) {
      throw new Error(`Invalid experimental embedding candidate: ${name} token_count is invalid.`);
    }
    candidateMetric(cpuCheck.full_lookup[name], `cpu_component_check.full_lookup.${name}`);
  }

  candidateObject(cpuCheck.fixed_probes, 'cpu_component_check.fixed_probes');
  const fixed = cpuCheck.fixed_probes;
  if (!Array.isArray(fixed.text?.token_ids)
      || fixed.text.token_ids.length === 0
      || fixed.text.token_ids.some((id) => !Number.isSafeInteger(id) || id < 0 || id >= 50276)
      || !Array.isArray(fixed.speech?.token_ids)
      || fixed.speech.token_ids.length === 0
      || fixed.speech.token_ids.some((id) => !Number.isSafeInteger(id) || id < 0 || id >= 6563)) {
    throw new Error('Invalid experimental embedding candidate: fixed probe token IDs are invalid.');
  }
  for (const label of ['text', 'speech']) {
    if (fixed[label].elements_compared !== fixed[label].token_ids.length * 768) {
      throw new Error(`Invalid experimental embedding candidate: ${label} probe element count is invalid.`);
    }
  }
  candidateObject(fixed.initial_tail, 'cpu_component_check.fixed_probes.initial_tail');
  if (!sameNumbers(fixed.initial_tail.token_ids, [50256, 50256])) {
    throw new Error('Invalid experimental embedding candidate: the tokenizer initial-tail probe is missing.');
  }
  if (fixed.initial_tail.elements_compared !== 1536) {
    throw new Error('Invalid experimental embedding candidate: the initial-tail probe must compare both 768-wide rows.');
  }
  for (const label of ['text', 'speech', 'initial_tail']) {
    candidateMetric(fixed[label], `cpu_component_check.fixed_probes.${label}`);
  }
  if (!(typeof manifest.limitations === 'string' && manifest.limitations.trim())
      && !(Array.isArray(manifest.limitations) && manifest.limitations.length > 0)) {
    throw new Error('Invalid experimental embedding candidate: limitations must be recorded.');
  }

  return manifest;
}

async function fetchCandidateBlob(fetchImpl, url, label, expected) {
  const response = await checkedResponse(fetchImpl, url, label);
  const blob = await response.blob();
  if (blob.size !== expected.bytes) {
    throw new Error(`Experimental embedding candidate ${label} size mismatch: expected ${expected.bytes}, received ${blob.size}.`);
  }
  const actualSha256 = await sha256Blob(blob);
  if (actualSha256 !== expected.sha256) {
    throw new Error(`Experimental embedding candidate ${label} SHA-256 mismatch: expected ${expected.sha256}, received ${actualSha256}.`);
  }
  return blob;
}

export async function createExperimentalEmbeddingGraph(ort, manifestUrl, options = {}) {
  if (!ort?.InferenceSession?.create) throw new TypeError('An ONNX Runtime module is required.');
  if (typeof manifestUrl !== 'string' || !manifestUrl) throw new TypeError('A diagnostic candidate manifest URL is required.');
  assertJspiSupport(options.webAssembly);
  const fetchImpl = options.fetchImpl || globalThis.fetch;
  if (typeof fetchImpl !== 'function') throw new Error('Fetch is unavailable in this worker.');

  const manifestResponse = await checkedResponse(fetchImpl, manifestUrl, 'experimental embedding manifest');
  const manifestBlob = await manifestResponse.blob();
  if (manifestBlob.size > 256 * 1024) throw new Error('Experimental embedding manifest is larger than 256 KiB.');
  let manifest;
  try {
    manifest = JSON.parse(await manifestBlob.text());
  } catch (error) {
    throw new Error(`Could not parse experimental embedding manifest: ${error?.message || String(error)}.`);
  }
  validateExperimentalEmbeddingManifest(manifest);

  const manifestBase = new URL(manifestUrl, globalThis.location?.href || 'http://localhost/');
  const [graphBlob, externalDataBlob] = await Promise.all([
    fetchCandidateBlob(fetchImpl, new URL(manifest.target.graph.path, manifestBase).href, 'graph', manifest.target.graph),
    fetchCandidateBlob(fetchImpl, new URL(manifest.target.external_data.path, manifestBase).href, 'external weights', manifest.target.external_data),
  ]);
  const graph = new Uint8Array(await graphBlob.arrayBuffer());

  return ort.InferenceSession.create(graph, {
    executionProviders: getExecutionProviders('webgpu', options.gpuDevice),
    graphOptimizationLevel: 'all',
    externalData: [{ path: manifest.target.external_data.path, data: externalDataBlob }],
  });
}

export function getPinnedAsset(name) {
  return ASSETS[name];
}
