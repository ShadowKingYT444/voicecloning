import * as ort from 'onnxruntime-web/jspi';
import { createGraph, REVISION } from './model-loader.js';
import { StreamedEmbeddings } from './streamed-embeddings.js';

let result = null; let running = false;
async function run() {
  if (running) throw Error('The component check is already active.');
  running = true; document.getElementById('run').disabled = true;
  let session; let lookup;
  const status = document.getElementById('status');
  result = { schema: 'voice-study.wasm-embedding-component/v1', provider: 'wasm', ortVersion: '1.30.0',
    revision: REVISION, embeddingPackageBytes: 87306224, fullNanoInferenceRun: false,
    speechGenerated: false, voiceQualityVerified: false, memoryTargetVerified: false,
    cases: [], status: 'running' };
  try {
    ort.env.wasm.numThreads = 1; ort.env.wasm.proxy = false;
    status.textContent = 'Loading independently verified lossless shard metadata…';
    lookup = await StreamedEmbeddings.load('/experiments/embedding_lossless/manifest.json');
    status.textContent = 'Loading the pinned embedding graph on CPU/WASM…';
    session = await createGraph(ort, 'embed_tokens_fp16', {}, {
      baseUrl: new URL('/models/chatterbox-nano-browser/', location.href).href,
      executionProvider: 'wasm',
    });
    for (const ids of [[1, 50275, 50256, 50256], [0], [6562, 50256]]) {
      const input = new ort.Tensor('int64', BigInt64Array.from(ids, BigInt), [1, ids.length]);
      let outputs;
      try {
        outputs = await session.run({ input_ids: input });
        const actual = outputs.inputs_embeds.data;
        const expected = await lookup.lookup(ids);
        const actualBits = new Uint32Array(actual.buffer, actual.byteOffset, actual.length);
        const expectedBits = new Uint32Array(expected.buffer);
        const exact = actual.length === expected.length && actualBits.every((value, index) => value === expectedBits[index]);
        if (!exact || !actual.every(Number.isFinite)) throw Error('WASM embedding output differs from the pinned lossless lookup.');
        result.cases.push({ ids, float32Values: actual.length, bitwiseEqual: true, finite: true });
      } finally {
        input.dispose(); for (const tensor of Object.values(outputs || {})) tensor.dispose();
      }
    }
    result.embeddingLookup = lookup.snapshot(); result.status = 'passed';
    status.textContent = 'CPU/WASM embedding component passed. Full Nano speech remains untested.';
  } catch (error) {
    result.status = 'failed'; result.error = error.message;
    status.textContent = `Component failed: ${error.message}`;
    throw error;
  } finally {
    try { await session?.release(); } catch {}
    lookup?.clear(); running = false; document.getElementById('run').disabled = false;
    document.getElementById('report').textContent = JSON.stringify(result, null, 2);
  }
  return result;
}
document.getElementById('run').addEventListener('click', () => run().catch(() => {}));
window.voiceWasmComponent = { run, snapshot: () => result };
