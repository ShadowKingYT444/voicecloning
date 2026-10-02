# Cloud browser MVP continuation, 2026-10-02

Branch: `codex/browser-streaming-mvp-20261002`. Base:
`09ba969b28b8856c87d74e4c83aafad8843990e1` (`work`, same as remote `main`).
Draft: https://github.com/ShadowKingYT444/voicecloning/pull/1.
Work stops no later than 2026-10-02 10:42 UTC. Work/Cloud only; the user's
computer is excluded. No training, paid compute, credential creation, or public
deployment is authorized.

## Recovery and voice boundary

Ten browser files were restored from the pinned public conversion and verified:
395,880,313 bytes (377.54 MiB), excluding voice state and runtime assets.
The selected generated reference and state match their documented hashes:

- Reference: `67f94a868976b22a47bce8fd00a873d4c5b7085ba6eedd698f8a898a26ce76c0`.
- State, 331,200 bytes: `6f56c0dd844ecc038e5b60debff22d1e16b44c9d13386d0f7a91156bfca1b38e`.

This preserves those **input artifacts**, not a demonstrated accepted voice.
The community Nano preset uses a generated 3.52-second reference, CPU-exported
base encoder conditioning, and no folded local fitted adapter. Terry praised
voices on October 2, but the accepted sample/profile/seed/adapter has not been
identified. No browser output exists to compare to that feedback.

The tracked references and profile JSONs are present. Historical fitted adapters,
conditioning/donor caches, fitted stage weights and most listening WAVs are
absent. Examples include `adapter_aligned_all_attn.pt`,
`decoder_embedding_fit/conditionals.pt`, `decoder_attention_fit/best_attention.pt`,
and `t3_reference_matrix_caches/prompt.conds.pt`. The per-profile inventory is
[fitted-assets.json](../artifacts/nano_lab/browser_mvp_20261002/fitted-assets.json).
These profiles are experimental, not identified accepted controls. Authorized
Git assets do not contain their excluded binary inputs. Public base downloads do
not recreate them. Recovering them needs an existing authorized asset store;
do not request the excluded laptop or fit replacement adapters.

`browser_cpu_components_20261002/RESULTS.md` from task
`01a0fb43-45e0-7072-9546-83117adf801a` is absent here. The reported earlier Python
158.53 → 45.77 MiB result is historical, not reproduced by this continuation.
Q4 provenance reports are tracked, but Q4 binary candidates remain absent.

## Browser capability evidence

This Codex executor has Chromium and no visible GPU devices. Its systemd user
manager is offline and cgroup v2 is read-only. Browser workloads cannot run here
through the required process-tree guard.

Separately, the parent reported a live Work-browser visit to
`https://webgpureport.org/`: WebGPU appeared disabled, no fallback adapter was
supported, and rgba16float canvas support was absent. That observation concerns
the separate browser, not this executor. It does not authorize changing browser
security or GPU flags. WGSL language features alone do not establish an adapter.

`/capabilities.html` checks actual adapter identity/fallback status, `shader-f16`,
limits, JSPI and browser APIs without model/reference fetches, device requests or
inference. A standalone copy is in
[capability-check.html](../artifacts/nano_lab/browser_mvp_20261002/capability-check.html).
No native private preview is exposed here. Sites publishing provisions a source
write credential, which conflicts with this task's constraint; no Site was
created. GitHub Actions provides an authorized cloud-only browser test route.

## Optional lossless embedding runtime

The worker can replace the 87,304,704-byte FP16 embedding ONNX session with
independently hash-verified 64-row FP16 shards. It preserves the exact graph's
hybrid text/speech-tail lookup and FP16-to-FP32 conversion. The manifest itself
is pinned, so shard hashes cannot be replaced by an arbitrary same-origin
manifest. The LRU holds at most 4 MiB of shard ArrayBuffers; output tensors,
temporary response Blobs, the browser HTTP cache, other graphs and runtime
allocations are additional. This cache bound is not a browser memory measurement.

No quantization, sampling change, fitted adapter or default promotion is applied.
Package bytes are unchanged for the FP16 tables, plus 129,432 bytes of manifest
metadata and HTTP/file overhead. Random shard requests may hurt TTFA. Measure
that tradeoff rather than inferring a speedup from the smaller resident cache.

Generate and verify serially from the repository root:

```bash
python3 scripts/nano_lab/bounded_job.py --backend single-process --max-memory-mib 640 -- \
  python3 browser_tts/scripts/pack-lossless-embeddings.py
python3 scripts/nano_lab/bounded_job.py --backend single-process --small-job -- \
  .venv-browser-cpu/bin/python browser_tts/scripts/verify-lossless-embeddings.py \
  --report artifacts/nano_lab/browser_mvp_20261002/lossless_cpu.json
```

The verifier needs `onnxruntime==1.29.0` and `numpy==1.26.4`. It compares every
text and speech row to CPU ONNX: **43,652,352 Float32 values matched bitwise**.
Its measured process peak RSS is 148.73 MiB, including the CPU ONNX session.
That is a component comparison, not an embedding-only streaming-memory benchmark
and not browser/audio parity.

## Playback and measurement harness

The retained fixed voice state still omits the reference encoder. Playback keeps
two passage credits. Snapshots now retain the maximum queue and requested audio
buffer bytes, generation completion and playback completion separately. TTFA is
request-to-first-scheduled-WebAudio-start and includes storage/context preparation;
it excludes model loading. Timeline gaps are scheduled gaps, not audible-gap
measurements. Listen at boundaries before making a continuity claim.

The export API returns at most 65,536 PCM bytes per call. It awaits disk/browser
backpressure and rejects missing, changed, duplicated, reordered or truncated
data. The harness streams long WAVs to disk, checks exact header/payload size,
and records hashes. It no longer transfers the entire paper as one base64 string.

On a cloud browser host with the existing enforceable systemd guard, rebuild and
serve `dist`, then run:

```bash
python3 scripts/nano_lab/bounded_job.py --max-memory-mib 640 -- npm --prefix browser_tts run build:low-copy
npm --prefix browser_tts run preview
python3 scripts/nano_lab/bounded_job.py -- \
  python3 browser_tts/scripts/measure-browser.py --chrome chromium \
  --hardware-webgpu --power-preference low-power --model-base /models/chatterbox-nano-browser/ \
  --full-paper --timeout-seconds 3600 --full-reading-timeout-seconds 3000 \
  --output-dir artifacts/nano_lab/browser_measurements/cloud_fp16_full
```

Repeat in a separate run with the same text/seed/provider and
`--lossless-embedding-manifest /experiments/embedding_lossless/manifest.json`.
The new mode is opt-in and unpromoted. Q4 remains a separate experiment.

Report package bytes, CPU Python RSS, absolute browser RSS/PSS, incremental PSS
above clean Chrome, JS heap, requested GPU buffers and physical GPU evidence
separately. GPU descriptor sizes are not VRAM residency. The clean profile does
not flush OS/HTTP caches. Cold load, first request, warm request, playback and
export are separate stages. This harness does not perform a word/listening audit.

## Cloud CPU browser controls

The draft includes a serial, 15-minute GitHub Actions workflow on the standard
public-repository Ubuntu runner. It establishes the existing systemd guard,
then runs guarded components, build and Chromium tests. It downloads no models.
The real capability page is tested without API stubs. Other integration tests
replace the model worker with **synthetic tones**, exercise failure/retry,
stop/restart, full-paper two-credit playback and >60-second streamed export,
and capture desktop/mobile screenshots. Their metrics describe fixture control
flow, never TTS speed, voice quality, speech accuracy or the memory target.

The current Work browser cannot run the WebGPU-only model path. A CPU/WASM or
cloud-service inference fallback has not been verified or integrated; the UI
tests do not create one. No architecture change is justified by fixture timing.

## Remaining speech gates

1. A permitted cloud browser with JSPI and a verified non-fallback FP16 adapter,
   or a separately validated CPU/WASM/service provider with enforceable guard.
2. Matched first/warm generation with actual WAVs/tokens and process/JS/GPU
   traces; include cold load and export peaks.
3. Independent transcripts, raw/controlled listening and fixed-reference
   identity checks. Identify the praised voice and recover its exact artifacts.
4. FP16 versus lossless browser logits/tokens/audio parity and measured tradeoffs.
5. Real full-paper continuity, complete saved audio, stop/restart and memory
   drift under actual inference. Synthetic integration cannot satisfy these.

No voice is promoted and no target performance is claimed.
