import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import test from 'node:test';

import {
  assertJspiSupport,
  getPinnedAsset,
  normalizeSameOriginModelBaseUrl,
  REFERENCE_SHA256,
  REVISION,
  SHA256SUMS_SHA256,
  createGraph,
  createExperimentalEmbeddingGraph,
  resolveModelAssetUrl,
  resolvePinnedRuntimeAsset,
  sha256Blob,
  sha256Bytes,
  validateExperimentalEmbeddingManifest,
} from '../src/model-loader.js';
import {
  getFixedVoiceStateIdentity,
  loadFixedVoiceState,
  removeDuplicateDonorSpeechStartEmbedding,
  resolveVoiceStateDataUrl,
  validateFixedVoiceStateManifest as validateFixedVoiceStateManifestFromLoader,
} from '../src/voice-state.js';

test('the retained real Q4 export manifest passes browser validation', () => {
  const path = new URL('../../artifacts/nano_lab/browser_embedding_q4_20261001/manifest.json', import.meta.url);
  const manifest = JSON.parse(readFileSync(path, 'utf8'));
  assert.equal(validateExperimentalEmbeddingManifest(manifest), manifest);
  assert.equal(manifest.production_gate, false);
});

function referenceHash(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}

function makeChunkedBlob(bytes, chunkSizes) {
  let offset = 0;
  let chunkIndex = 0;
  return {
    size: bytes.byteLength,
    stream() {
      return {
        getReader() {
          return {
            async read() {
              if (offset >= bytes.byteLength && chunkIndex >= chunkSizes.length) return { done: true };
              const requested = chunkSizes[chunkIndex] ?? bytes.byteLength;
              chunkIndex += 1;
              const next = Math.min(bytes.byteLength, offset + requested);
              const value = bytes.subarray(offset, next);
              offset = next;
              return { done: false, value };
            },
            releaseLock() {},
          };
        },
      };
    },
  };
}

function addNativeDonorProvenance(manifest) {
  manifest.conditioning.source = 'native_t3_donor';
  manifest.data.sha256 = '77a2955a6b50122c581cf211f86d7c624381b7dfa8bb8ce61af3fb6359e16496';
  manifest.conditioning.native_t3_donor = {
    profile: 'clean_asmr_original',
    cache: {
      path: 'artifacts/nano_lab/decoder_embedding_fit/conditionals.pt',
      sha256: '3c78b8bedb9b5ac94d5aaf900d509162cb4c5274f59cf7fb525117498786c40c',
    },
    checkpoint: {
      repo_id: 'ResembleAI/chatterbox-nano',
      revision: '71ccd1d0081b430592cea481f4307e764e07bc64',
      path: 'models/chatterbox-nano/t3_nano_v1.safetensors',
      sha256: '72b110185087d945dbdf54dee4e333848e1811bdd5fd6cb16ceb8da50006f0c9',
    },
    prefix: {
      path: 'artifacts/nano_lab/browser_measurements/local_asmr_cache_prefix_20261002/prefix.npy',
      sha256: '6d27b0087b442abb8264d0fea19425be649f663519830e9ca28bdb57c80fe311',
      frames: 334,
      prompt_tokens: 333,
      processing: 'cond_enc.spkr_enc(speaker_emb) concatenated with speech_emb(cond_prompt_speech_tokens)',
    },
    execution: { provider: 'CPU', framework: 'PyTorch', version: '2.11.0+cpu', threads: 2 },
  };
  manifest.encoder_execution.scope = 'three retained decoder-conditioning tensors copied bitwise from source_public_state; does not describe native T3 prefix';
  manifest.source_public_state = {
    path: 'browser_tts/public/voice/asmr-state.bin',
    sha256: '6f56c0dd844ecc038e5b60debff22d1e16b44c9d13386d0f7a91156bfca1b38e',
  };
}

function readNativeDonorStateFixture() {
  const directory = new URL('../public/voice/native-asmr-donor/', import.meta.url);
  return {
    manifest: JSON.parse(readFileSync(new URL('asmr-state.json', directory), 'utf8')),
    binary: readFileSync(new URL('asmr-state.bin', directory)),
  };
}

function makeStateFixture({ source = 'generated_reference', featureFrames = 1 } = {}) {
  const specifications = [
    { role: 'audio_features', name: 'audio_features', type: 'float32', dims: [1, featureFrames, 768], bytes: featureFrames * 768 * 4 },
    { role: 'audio_tokens', name: 'audio_tokens', type: 'int64', dims: [1, 88], bytes: 88 * 8 },
    { role: 'speaker_embeddings', name: 'speaker_embeddings', type: 'float32', dims: [1, 192], bytes: 192 * 4 },
    { role: 'speaker_features', name: 'speaker_features', type: 'float32', dims: [1, 176, 80], bytes: 176 * 80 * 4 },
  ];
  let offset = 0;
  const tensors = specifications.map((specification) => {
    const entry = {
      role: specification.role,
      name: specification.name,
      type: specification.type,
      dims: specification.dims,
      offset,
      byte_length: specification.bytes,
      serialization_round_trip_exact: true,
    };
    offset += specification.bytes;
    return entry;
  });
  const binary = Buffer.alloc(offset);
  const encoder = getPinnedAsset('speech_encoder_q4f16');
  const manifest = {
    schema_version: 1,
    format: 'chatterbox-nano-reference-state-v1',
    model: {
      repo_id: 'owensong/chatterbox-nano-ONNX',
      revision: REVISION,
      sha256sums_sha256: SHA256SUMS_SHA256,
      encoder_graph: { path: encoder.graphPath, sha256: encoder.graphSha256 },
      encoder_external_weights: { path: `onnx/${encoder.externalPath}`, sha256: encoder.externalSha256 },
    },
    reference: {
      path: 'browser_tts/public/voice/asmr_t3_seed47_fit.wav',
      sha256: REFERENCE_SHA256,
      sample_rate: 24000,
      channels: 1,
      wav_encoding: 'PCM-24',
    },
    conditioning: {
      source: 'generated_reference',
      zero_shot: false,
      speaker_specific: true,
      fitted_adapter_applied: false,
      description: 'Fixture only.',
    },
    encoder_execution: {
      provider: 'CPUExecutionProvider',
      onnxruntime_version: '1.29.0',
      graph_optimization_level: 'ORT_ENABLE_ALL',
    },
    data: { file: 'asmr-state.bin', size_bytes: binary.byteLength, sha256: referenceHash(binary) },
    tensors,
  };
  if (source === 'native_t3_donor') addNativeDonorProvenance(manifest);
  return { manifest, binary };
}

function makeFetch(manifest, binary, requests = []) {
  return async (url) => {
    requests.push(url);
    const pathname = new URL(url, 'http://voice-state.invalid/').pathname;
    if (pathname.endsWith('/asmr-state.json')) {
      return { ok: true, status: 200, blob: async () => new Blob([JSON.stringify(manifest)]) };
    }
    if (pathname.endsWith('/asmr-state.bin')) {
      return { ok: true, status: 200, blob: async () => new Blob([binary]) };
    }
    return { ok: false, status: 404, blob: async () => new Blob() };
  };
}

class FakeTensor {
  constructor(type, data, dims) {
    this.type = type;
    this.data = data;
    this.dims = dims;
    this.disposed = false;
  }

  dispose() {
    this.disposed = true;
  }
}

function makeEmbeddingCandidateManifest() {
  const metric = (elementsCompared) => ({
    elements_compared: elementsCompared,
    max_absolute_error: 0.125,
    mean_absolute_error: 0.025,
    root_mean_square_error: 0.04,
  });
  return {
    schema: 'browser_tts.embedding-q4-candidate/v1',
    candidate_status: 'experimental_unpromoted',
    production_gate: false,
    source: {
      repo_id: 'owensong/chatterbox-nano-ONNX',
      revision: REVISION,
      sha256sums: { path: 'SHA256SUMS', sha256: SHA256SUMS_SHA256 },
      files: {
        'onnx/embed_tokens_fp16.onnx': {
          sha256: '019d257243774091d78c2ad91c2c0f61e4e442740cb7b3b00b5a89109417b18d',
          bytes: 1_520,
        },
        'onnx/embed_tokens_fp16.onnx_data': {
          sha256: 'bcd7b35ae4f206932e2491cb60b42ebb80f6d8facfdb53ba7d7449ad00a3237b',
          bytes: 87_304_704,
        },
      },
      graph: {
        input_name: 'input_ids',
        output_name: 'inputs_embeds',
        input_type: 'int64',
        output_type: 'float',
        embedding_dim: 768,
        gather_tables: [
          { initializer: 'text_emb.weight', shape: [50_276, 768], dtype: 'float16', gather_axis: 0 },
          { initializer: 'speech_emb.weight', shape: [6_563, 768], dtype: 'float16', gather_axis: 0 },
        ],
        hybrid_tail_contract: {
          text_slice_end_exclusive: -2,
          speech_slice_start: -2,
          sentinel_token_id: 50256,
          speech_replacement_token_id: 6561,
        },
      },
    },
    quantization: {
      api: 'onnxruntime.quantization.matmul_nbits_quantizer.MatMulNBitsQuantizer',
      ort_version: '1.29.0',
      algorithm: 'DEFAULT',
      bits: 4,
      block_size: 128,
      is_symmetric: false,
      quantized_dtype: 'UINT4',
      quant_format: 'QOperator',
      op_types_to_quantize: ['Gather'],
      quant_axes: { Gather: 1 },
      source_gather_axis: 0,
      target_operator: 'com.microsoft::GatherBlockQuantized',
    },
    target: {
      graph: { path: 'embed_tokens_gather_q4.onnx', sha256: 'a'.repeat(64), bytes: 100 },
      external_data: { path: 'embed_tokens_gather_q4.onnx.data', sha256: 'b'.repeat(64), bytes: 200 },
    },
    cpu_component_check: {
      status: 'completed',
      provider: 'CPUExecutionProvider',
      method: 'sequential_streamed_full_lookup',
      quality_or_promotion_gate: false,
      acceptance: 'descriptive_only_no_gate',
      fixed_probe_seed: 20261001,
      batch_size: 32,
      full_lookup: {
        text_emb: { token_count: 50276, ...metric(50276 * 768) },
        speech_emb: { token_count: 6563, ...metric(6563 * 768) },
      },
      fixed_probes: {
        text: { token_ids: [1], ...metric(768) },
        speech: { token_ids: [2], ...metric(768) },
        initial_tail: { token_ids: [50256, 50256], ...metric(1536) },
      },
      failure_reason: null,
    },
    limitations: ['A descriptive CPU component check does not authorize WebGPU or quality claims.'],
  };
}

test('SHA-256 matches Node crypto at padding and block boundaries', () => {
  for (const length of [0, 1, 55, 56, 63, 64, 65, 1024, 100_003]) {
    const bytes = Buffer.alloc(length);
    for (let index = 0; index < bytes.length; index += 1) bytes[index] = (index * 29 + length) & 0xff;
    assert.equal(sha256Bytes(bytes), referenceHash(bytes), `length ${length}`);
  }
});

test('SHA-256 handles a Blob stream split across arbitrary chunk boundaries', async () => {
  const bytes = Buffer.alloc(100_003);
  for (let index = 0; index < bytes.length; index += 1) bytes[index] = (index * 17 + 3) & 0xff;
  const chunked = makeChunkedBlob(bytes, [1, 63, 64, 65, 0, 512, 4097, 15_001, 33_333]);
  assert.equal(await sha256Blob(chunked), referenceHash(bytes));
});

test('fixed state loader verifies provenance and creates the four raw ORT tensors', async () => {
  const { manifest, binary } = makeStateFixture();
  const state = await loadFixedVoiceState({
    ort: { Tensor: FakeTensor },
    fetchImpl: makeFetch(manifest, binary),
  });

  assert.deepEqual(Object.keys(state), [
    'audioFeatures',
    'audioTokens',
    'speakerEmbeddings',
    'speakerFeatures',
  ]);
  assert.equal(state.audioFeatures.type, 'float32');
  assert.deepEqual(state.audioFeatures.dims, [1, 1, 768]);
  assert.ok(state.audioFeatures.data instanceof Float32Array);
  assert.equal(state.audioTokens.type, 'int64');
  assert.ok(state.audioTokens.data instanceof BigInt64Array);
  assert.deepEqual(state.speakerEmbeddings.dims, [1, 192]);
  assert.deepEqual(state.speakerFeatures.dims, [1, 176, 80]);
  assert.deepEqual(getFixedVoiceStateIdentity(state), {
    source: 'generated_reference', speechStartCount: 2, nativeT3Donor: null,
  });
});

test('native donor state validates its pinned provenance and fetches its binary beside its manifest', async () => {
  const { manifest, binary } = readNativeDonorStateFixture();
  const manifestUrl = '/voice/native-asmr-donor/asmr-state.json';
  const requests = [];
  assert.equal(validateFixedVoiceStateManifestFromLoader(manifest), manifest);
  assert.equal(resolveVoiceStateDataUrl(manifestUrl), '/voice/native-asmr-donor/asmr-state.bin');

  const state = await loadFixedVoiceState({
    ort: { Tensor: FakeTensor },
    fetchImpl: makeFetch(manifest, binary, requests),
    manifestUrl,
  });
  assert.deepEqual(requests, [manifestUrl, '/voice/native-asmr-donor/asmr-state.bin']);
  assert.deepEqual(state.audioFeatures.dims, [1, 334, 768]);
  assert.deepEqual(state.audioTokens.dims, [1, 88]);
  assert.deepEqual(state.speakerEmbeddings.dims, [1, 192]);
  assert.deepEqual(state.speakerFeatures.dims, [1, 176, 80]);
  assert.deepEqual(getFixedVoiceStateIdentity(state), {
    source: 'native_t3_donor',
    speechStartCount: 1,
    nativeT3Donor: {
      profile: 'clean_asmr_original',
      cacheSha256: '3c78b8bedb9b5ac94d5aaf900d509162cb4c5274f59cf7fb525117498786c40c',
      revision: '71ccd1d0081b430592cea481f4307e764e07bc64',
      checkpointSha256: '72b110185087d945dbdf54dee4e333848e1811bdd5fd6cb16ceb8da50006f0c9',
      prefixSha256: '6d27b0087b442abb8264d0fea19425be649f663519830e9ca28bdb57c80fe311',
      prefixFrames: 334,
      promptTokens: 333,
    },
  });
});

test('voice-state binary URL stays beside the same-origin manifest', () => {
  assert.equal(
    resolveVoiceStateDataUrl(
      'https://voice.example/voice/native/asmr-state.json',
      'asmr-state.bin',
      'https://voice.example/app/index.html',
    ),
    'https://voice.example/voice/native/asmr-state.bin',
  );
  assert.throws(
    () => resolveVoiceStateDataUrl('https://cdn.example/native/asmr-state.json', 'asmr-state.bin', 'https://voice.example/'),
    /browser app origin/,
  );
  assert.throws(() => resolveVoiceStateDataUrl('/voice/state.json', '../asmr-state.bin'), /binary filename/);
});

test('native donor provenance and prefix shape reject unpinned variants', () => {
  const { manifest } = makeStateFixture({ source: 'native_t3_donor', featureFrames: 334 });
  manifest.conditioning.native_t3_donor.prefix.sha256 = '0'.repeat(64);
  assert.throws(() => validateFixedVoiceStateManifestFromLoader(manifest), /prefix provenance/);

  const wrongShape = makeStateFixture({ source: 'native_t3_donor', featureFrames: 333 });
  assert.throws(() => validateFixedVoiceStateManifestFromLoader(wrongShape.manifest), /334 frames/);

  const wrongRetainedState = makeStateFixture({ source: 'native_t3_donor', featureFrames: 334 });
  wrongRetainedState.manifest.tensors[3].dims[1] = 175;
  assert.throws(() => validateFixedVoiceStateManifestFromLoader(wrongRetainedState.manifest), /retained public decoder state/);
});

test('native donor rejects a self-hashed but unpinned binary before fetching it', async () => {
  const { manifest, binary } = makeStateFixture({ source: 'native_t3_donor', featureFrames: 334 });
  manifest.data.sha256 = referenceHash(binary);
  const requests = [];
  await assert.rejects(loadFixedVoiceState({
    ort: { Tensor: FakeTensor },
    fetchImpl: makeFetch(manifest, binary, requests),
    manifestUrl: '/voice/native-asmr-donor/asmr-state.json',
  }), /native T3 donor binary SHA-256/);
  assert.deepEqual(requests, ['/voice/native-asmr-donor/asmr-state.json']);
});

test('native donor trims exactly one identical trailing speech-start embedding', () => {
  const ids = [7, 50256, 50256];
  const embeddings = Float32Array.from([1, 2, 9, 10, 9, 10]);
  assert.deepEqual(removeDuplicateDonorSpeechStartEmbedding(embeddings, ids, 2), Float32Array.from([1, 2, 9, 10]));
  assert.throws(() => removeDuplicateDonorSpeechStartEmbedding(Float32Array.from([1, 2, 9, 10, 8, 10]), ids, 2), /different rows/);
  assert.throws(() => removeDuplicateDonorSpeechStartEmbedding(embeddings, [7, 50256, 8], 2), /two speech-start markers/);
});

test('fixed state loader rejects an unpinned revision before it fetches tensor bytes', async () => {
  const { manifest, binary } = makeStateFixture();
  manifest.model.revision = 'unverified-revision';
  const requests = [];

  await assert.rejects(
    loadFixedVoiceState({
      ort: { Tensor: FakeTensor },
      fetchImpl: makeFetch(manifest, binary, requests),
    }),
    /graph revision/,
  );
  assert.deepEqual(requests, ['/voice/asmr-state.json']);
});

test('fixed state loader rejects a binary hash mismatch before creating tensors', async () => {
  const { manifest, binary } = makeStateFixture();
  const changed = Buffer.from(binary);
  changed[0] ^= 0xff;
  let tensorCount = 0;

  await assert.rejects(
    loadFixedVoiceState({
      ort: { Tensor: class extends FakeTensor { constructor(...args) { super(...args); tensorCount += 1; } } },
      fetchImpl: makeFetch(manifest, changed),
    }),
    /binary SHA-256 mismatch/,
  );
  assert.equal(tensorCount, 0);
});

test('fixed state loader rejects an invalid output type or shape', async () => {
  const { manifest, binary } = makeStateFixture();
  manifest.tensors[0].type = 'float16';

  await assert.rejects(
    loadFixedVoiceState({ ort: { Tensor: FakeTensor }, fetchImpl: makeFetch(manifest, binary) }),
    /must use ONNX tensor type float32/,
  );
});

test('JSPI preflight gives a clear error when either JSPI API is absent', () => {
  assert.throws(() => assertJspiSupport({}), /WebAssembly JSPI/);
  assert.throws(() => assertJspiSupport({ Suspending() {} }), /WebAssembly JSPI/);
  assert.doesNotThrow(() => assertJspiSupport({ Suspending() {}, promising() {} }));
});

test('local model base resolves as a directory and keeps the pinned graph hashes', () => {
  const baseUrl = normalizeSameOriginModelBaseUrl(
    '/models/chatterbox-nano-browser',
    'https://voice.example/app/assets/worker.js',
  );
  assert.equal(baseUrl, 'https://voice.example/models/chatterbox-nano-browser/');
  assert.equal(resolveModelAssetUrl(baseUrl, 'tokenizer.json'), `${baseUrl}tokenizer.json`);

  const decoder = resolvePinnedRuntimeAsset('conditional_decoder_q4', baseUrl);
  assert.equal(decoder.graphUrl, `${baseUrl}onnx/conditional_decoder_q4.onnx`);
  assert.equal(decoder.externalUrl, `${baseUrl}onnx/conditional_decoder_q4.onnx_data`);
  assert.equal(decoder.graphSha256, '745faa9c2e2494a81e47f2a72601ac248639fd417ae53d999c5068bc441d1f97');
  assert.equal(decoder.externalSha256, 'b5c5317e0b79a1a19dd3d5e2b2091ea06b15716716ab801a54eaeb906c6971ec');
});

test('worker model override rejects cross-origin and ambiguous base URLs', () => {
  assert.throws(
    () => normalizeSameOriginModelBaseUrl('https://cdn.example/models/nano', 'https://voice.example/'),
    /same HTTP or HTTPS origin/,
  );
  assert.throws(
    () => normalizeSameOriginModelBaseUrl('/models/nano?version=1', 'https://voice.example/'),
    /query or fragment/,
  );
});

test('experimental embedding candidate accepts only the pinned descriptive contract', () => {
  const manifest = makeEmbeddingCandidateManifest();
  assert.equal(validateExperimentalEmbeddingManifest(manifest), manifest);
  manifest.quantization.quant_axes.Gather = 0;
  assert.throws(() => validateExperimentalEmbeddingManifest(manifest), /quantization settings/);
});

test('runtime graph loader refuses the offline-only encoder', async () => {
  const ort = { InferenceSession: { create: async () => assert.fail('No runtime session should be created.') } };
  await assert.rejects(createGraph(ort, 'speech_encoder_q4f16'), /offline-only/);
});

test('runtime graph loader rejects an implicit or unsupported provider before fetching weights', async () => {
  const ort = { InferenceSession: { create: async () => assert.fail('No runtime session should be created.') } };
  await assert.rejects(createGraph(ort, 'embed_tokens_fp16', {}, {
    webAssembly: { Suspending() {}, promising() {} },
    executionProvider: 'auto',
    fetchImpl: async () => assert.fail('Invalid provider must not fetch weights.'),
  }), /explicit WebGPU or WASM/);
});

test('a supplied device reaches the WebGPU EP and cannot be paired with WASM', async () => {
  const graphBytes = Buffer.from('diagnostic graph bytes');
  const weightBytes = Buffer.from('diagnostic weight bytes');
  const manifest = makeEmbeddingCandidateManifest();
  manifest.target.graph.bytes = graphBytes.byteLength;
  manifest.target.graph.sha256 = referenceHash(graphBytes);
  manifest.target.external_data.bytes = weightBytes.byteLength;
  manifest.target.external_data.sha256 = referenceHash(weightBytes);

  const manifestUrl = 'https://voice.example/experiments/q4/manifest.json';
  const graphUrl = 'https://voice.example/experiments/q4/embed_tokens_gather_q4.onnx';
  const weightsUrl = 'https://voice.example/experiments/q4/embed_tokens_gather_q4.onnx.data';
  const requests = [];
  const fetchImpl = async (url) => {
    requests.push(url);
    const bytes = url === manifestUrl
      ? Buffer.from(JSON.stringify(manifest))
      : url === graphUrl ? graphBytes : url === weightsUrl ? weightBytes : null;
    return bytes
      ? { ok: true, status: 200, blob: async () => new Blob([bytes]) }
      : { ok: false, status: 404, blob: async () => new Blob() };
  };
  const device = { name: 'selected-adapter-device' };
  let captured;
  const ort = {
    InferenceSession: {
      create: async (graph, options) => {
        captured = { graph, options };
        return { release: async () => {} };
      },
    },
  };

  await createExperimentalEmbeddingGraph(ort, manifestUrl, {
    gpuDevice: device,
    fetchImpl,
    webAssembly: { Suspending() {}, promising() {} },
  });
  assert.equal(Buffer.from(captured.graph).compare(graphBytes), 0);
  assert.deepEqual(captured.options.executionProviders, [{ name: 'webgpu', device }]);
  assert.deepEqual(requests, [manifestUrl, graphUrl, weightsUrl]);

  let forbiddenFetches = 0;
  await assert.rejects(createGraph(ort, 'embed_tokens_fp16', {}, {
    webAssembly: { Suspending() {}, promising() {} },
    executionProvider: 'wasm',
    gpuDevice: device,
    fetchImpl: async () => { forbiddenFetches += 1; throw new Error('fetch must not start'); },
  }), /GPU device requires the WebGPU execution provider/);
  assert.equal(forbiddenFetches, 0);
});

 test('fixed state rejects non-finite features even with a matching binary hash', async () => {
  const { manifest, binary } = makeStateFixture();
  binary.writeFloatLE(Infinity, 0); manifest.data.sha256 = referenceHash(binary);
  await assert.rejects(loadFixedVoiceState({ ort: { Tensor: FakeTensor }, fetchImpl: makeFetch(manifest, binary) }), /non-finite/);
});

test('fixed state rejects speech tokens outside the model vocabulary', async () => {
  const { manifest, binary } = makeStateFixture();
  binary.writeBigInt64LE(6563n, manifest.tensors[1].offset); manifest.data.sha256 = referenceHash(binary);
  await assert.rejects(loadFixedVoiceState({ ort: { Tensor: FakeTensor }, fetchImpl: makeFetch(manifest, binary) }), /invalid speech token/);
});

test('a partial table scan cannot be labeled a complete diagnostic check', () => {
  const manifest = makeEmbeddingCandidateManifest(); manifest.cpu_component_check.full_lookup.text_emb.token_count = 2;
  assert.throws(() => validateExperimentalEmbeddingManifest(manifest), /token_count/);
});
