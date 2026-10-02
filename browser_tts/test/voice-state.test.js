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
  resolveModelAssetUrl,
  resolvePinnedRuntimeAsset,
  sha256Blob,
  sha256Bytes,
  validateExperimentalEmbeddingManifest,
} from '../src/model-loader.js';
import { loadFixedVoiceState } from '../src/voice-state.js';

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

function makeStateFixture() {
  const specifications = [
    { role: 'audio_features', name: 'audio_features', type: 'float32', dims: [1, 1, 768], bytes: 1 * 1 * 768 * 4 },
    { role: 'audio_tokens', name: 'audio_tokens', type: 'int64', dims: [1, 2], bytes: 2 * 8 },
    { role: 'speaker_embeddings', name: 'speaker_embeddings', type: 'float32', dims: [1, 192], bytes: 192 * 4 },
    { role: 'speaker_features', name: 'speaker_features', type: 'float32', dims: [1, 1, 80], bytes: 80 * 4 },
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
  return { manifest, binary };
}

function makeFetch(manifest, binary, requests = []) {
  return async (url) => {
    requests.push(url);
    if (url === '/voice/asmr-state.json') {
      return { ok: true, status: 200, blob: async () => new Blob([JSON.stringify(manifest)]) };
    }
    if (url === '/voice/asmr-state.bin') {
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
  assert.deepEqual(state.speakerFeatures.dims, [1, 1, 80]);
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
