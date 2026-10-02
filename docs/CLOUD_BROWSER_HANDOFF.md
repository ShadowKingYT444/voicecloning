# Cloud handoff: fixed ASMR browser reader

Updated 2026-10-01. Repository: `ShadowKingYT444/voicecloning`, branch `main`.
The user requested this push so a Codex cloud agent can continue on a 16 GiB VM.
Local experimentation stops at this handoff. Do not treat publication as model
or product completion.

## Outcome to deliver

Make the selected Chatterbox Nano voice read Federalist No. 10 in the browser
with correct words, clean and realistic ASMR speech, quick first audio, smooth
passage transitions, and measured low memory. Preserve the strict target below
500 MiB. Report absolute Chrome RSS, absolute PSS, PSS above a clean browser,
and GPU allocation evidence separately. Do not change the memory definition to
make a result pass. Requested GPU buffer bytes do not measure physical VRAM.

The user rejected earlier voice samples as unconvincing. No current voice or
runtime is promoted. Keep the separate Harvey research track and its existing
numerical gates. Speaker-specific fits are not general zero-shot improvements.

The continuation brief asked for these steps before an architecture decision:

1. Remove the reference encoder from fixed-voice runtime. Precompute its state.
2. Use ONNX Runtime Web 1.30 JSPI with Blob external data.
3. Quantize the embedding lookup tables to Q4 and compare downstream output.
4. Measure actual browser host memory, GPU allocations, load peaks, first and
   warm latency, words, and listening artifacts.
5. Change the decoder or model only after measuring the actual bottleneck.

These changes are implemented or exported. Step 4 is not complete. Native
memory measurements and model download size do not prove Nano cannot fit in
the browser.

## Read first

- `AGENTS.md`, `README.md`, `docs/EXPERIMENT_HANDOFF.md`.
- `scripts/nano_lab/AGENTS.md` for the resource guard.
- `docs/BROWSER_CONTINUATION.md` for retained evidence and provenance.
- `browser_tts/MEASUREMENT.md` for runner commands and measurement limits.
- `browser_tts/EMBEDDING_Q4.md` for the unpromoted candidate.

Use short, precise sentences. Inspect the actual rendered page and audio.
Do not declare success from a build, automatic score, or short sample alone.

## State of the code

| Area | Current implementation |
|---|---|
| `src/worker.js` | Three graphs, fixed voice state, GPU KV outputs, seeded token sampling, tensor disposal, explicit 256-token failure, two playback credits |
| `src/model-loader.js` | Pinned revision and SHA-256 checks, JSPI support check, Blob external data, same-origin local model override, strict experimental Q4 manifest validation |
| `src/voice-state.js` | Provenance, hash, shape, dtype, finite-value, and speech-token checks |
| `src/main.js` | Model-ready gate, worker retry protection, sequential storage writes, scheduled adjacent playback, stop, metrics, debug API |
| `src/reading-store.js` | Session-scoped IndexedDB PCM storage and streamed WAV export |
| `src/numeric.js`, `src/audio.js` | Correct FP16 conversion, PCM/WAV encoding, short edge fades |
| `scripts/download-model.py` | Pinned downloads and hash-checked same-origin staging; runtime-only mode omits the encoder |
| `scripts/export-voice-state.py` | CPU export of the exact reference encoder outputs |
| `scripts/quantize-embedding.py` | Q4 Gather candidate and streamed all-row CPU lookup comparisons |
| `scripts/measure-browser.py` | Owned clean Chrome profile, process-tree RSS/PSS, adapter checks, lazy GPU buffer hooks, matched short first/warm WAVs, static UI and adapter-only modes |
| `scripts/build-low-copy.mjs` | Optional build that hard-links immutable public model files instead of copying them |

All paths above are under `browser_tts/`. Dependencies are pinned to
`onnxruntime-web@1.30.0` and `@huggingface/tokenizers@0.1.3`.

Latest source changes need a new guarded build and component test run:

- The page and worker default to `low-power`. The worker requires `shader-f16`
  before tokenizer or model downloads and gives ORT the exact checked adapter.
  ORT creates the device with its required features and limits.
- The sampler rejects malformed, NaN, positive-infinite, or entirely non-finite
  speech logits. Negative-infinite masked logits remain valid.
- `npm run build:low-copy` has syntax checks only. Model hard links share an
  inode with their public source files. Keep source weights immutable. Its
  cross-filesystem copy fallback has not been exercised.

The most recent ordinary build stopped when local desktop headroom fell below
4 GiB. Local `dist` is incomplete. It is excluded from Git. Rebuild on the VM.
Do not start from the old preview and assume it contains the latest worker.

## Included and excluded assets

The selected reference WAV, `public/voice/asmr-state.json`, and the small
`public/voice/asmr-state.bin` are included. The two successful static UI
screenshots are also included. No reference encoder is needed for the first
browser baseline.

Reference SHA-256:
`67f94a868976b22a47bce8fd00a873d4c5b7085ba6eedd698f8a898a26ce76c0`.
State: 331,200 bytes. SHA-256:
`6f56c0dd844ecc038e5b60debff22d1e16b44c9d13386d0f7a91156bfca1b38e`.

The reference is a generated 3.52-second clip. The community base encoder
produced its state. Local fitted adapters were not folded into these public
graphs. This conditioning does not prove that the accepted fitted voice has
been reproduced. Voice quality is still unaccepted.

Large graph weights, Q4 candidate binaries, native fitted adapters, caches,
generated listening WAVs, dependencies, and local environments remain excluded.
The public browser model can be restored with the downloader below. Q4 can be
rebuilt with the exporter. Native fitted-state continuation needs the separate
asset transfer in `docs/ASSETS.md`; public base downloads do not recreate it.
Historical reports and relative stage symlinks are preserved as evidence.
Their local paths and referenced binary targets can be absent on a fresh VM.

## Verified evidence

- CPU ORT 1.29 exported all four conditioning tensors. Exact byte serialization
  round trips passed. This is not WebGPU encoder or final-audio equivalence.
- Earlier production builds and 21 component tests passed under the guard.
  They predate the final adapter, logit, and low-copy build changes.
- The three pinned graph/weight pairs total 374.14 MiB. Tokenizer/config files
  bring model assets to 377.54 MiB. This is file size, not runtime memory.
- Q4 external embedding weights fell from 87,304,704 to 22,678,761 bytes, a
  74.0% reduction. CPU maximum absolute errors were 0.0552063 for text and
  0.1845703 for speech. The all-row measurements are descriptive. No numerical
  or audio acceptance gate passed. Q4 remains unpromoted.
- `ui_fixed_voice_mobile_fixed` passed all 16 static/error checks. Parent
  inspected desktop and mobile screenshots. Mobile scroll width was 390 at a
  390-pixel viewport. No model/tokenizer download was observed on the controlled
  invalid-state path. This does not verify ready, playback, stop, or save states.
- Chrome 154 Headless Shell's default adapter exposed Intel `gen-12lp` and
  `shader-f16`. An explicit high-performance request exposed NVIDIA `lovelace`
  without `shader-f16`. These adapter-only checks loaded no speech model.
  The explicit low-power path and normal Chrome 152 model inference still need
  verification. Do not force the Dawn NVIDIA FP16 toggle to bypass this result.
- `fixed_voice_fp16_first` loaded embedding and language-model sessions using
  SwiftShader, then failed fetching the decoder. Baseline RSS/PSS were
  1,141.117/398.174 MiB. Sampled load peaks were 1,837.613/1,003.231 MiB.
  Incremental peak PSS was 605.057 MiB. This incomplete software load does not
  measure final inference or establish an architecture limit.

Reports are under `artifacts/nano_lab/browser_measurements/`. Q4 provenance is
under `artifacts/nano_lab/browser_embedding_q4_20261001/`. Runtime inventory is
under `artifacts/nano_lab/browser_runtime_20261001/`. Failed attempts are retained.
No browser-generated speech WAV has completed in this continuation.

## VM preparation and resource boundary

First inspect available RAM, browser version, GPU adapters, JSPI support,
`systemctl --user`, and cgroup v2. The local machine had hardware GPUs and a
systemd user manager. A cloud VM with 16 GiB RAM can still lack both. Inspect
the VM before selecting a provider or claiming hardware parity.

Run model, evaluation, export, and browser jobs serially through the existing
guard. Preserve the 4 GiB reserve, no swap, two-core quota, and all caps:

| Job | Hard cap | Required available RAM at launch |
|---|---:|---:|
| Full browser/model | 3072 MiB | 6144 MiB |
| Small component | 1280 MiB | 5376 MiB |
| Adapter-only component | 1024 MiB | 5120 MiB |
| Tiny component/build | 640 MiB | 4736 MiB |

Do not raise caps because the VM has more RAM. Stop below the 4096 MiB reserve.
Do not retry an oversized working set under a larger cap. Reduce it first.
`--backend single-process` supports guarded Python-only work with denied child
processes. It cannot run Chrome. If systemd is unavailable, establishing an
equivalent enforceable process-tree/cgroup guard is an infrastructure prerequisite
for browser work. Do not run browser jobs unguarded. If hardware WebGPU is
absent, report that boundary; software results are a separate experiment.

## First commands

Run from the repository root. Install Node dependencies with
`npm --prefix browser_tts ci`. Use `docs/ENVIRONMENT.md` only when Python model
export or evaluation is needed. The baseline browser runner and downloader use
the Python standard library; the included state avoids an immediate ML install.

Restore the three runtime graphs and tokenizer/config assets:

```bash
python3 scripts/nano_lab/bounded_job.py --max-memory-mib 640 -- \
  python3 browser_tts/scripts/download-model.py --runtime-only --stage-runtime
```

The source is `owensong/chatterbox-nano-ONNX`, pinned revision
`4a66d7dab72a9e98f24b515d49a1d7a81632df2e`. Preserve all hashes.

Run component tests, then build. These are distinct serial jobs:

```bash
python3 scripts/nano_lab/bounded_job.py --max-memory-mib 640 -- \
  npm --prefix browser_tts test
python3 scripts/nano_lab/bounded_job.py --max-memory-mib 640 -- \
  npm --prefix browser_tts run build:low-copy
```

Inspect the build output and served asset hashes. The low-copy script is a new
option, not a measured optimization. Ordinary `npm run build` remains available.
Do not run training to verify the publication or documentation.

Check the port registry if one exists on the VM. Reuse a healthy project server
whose process directory matches the checkout. Otherwise start a production
preview in a retained terminal:

```bash
npm --prefix browser_tts run preview
```

The explicit host/port are `127.0.0.1:4187` with `--strictPort`. The preview
serves `dist`. Only stop PIDs owned by this task. Do not prepare a model in an
unrelated browser to bypass the guard.

First full FP16 measurement, after checking hardware capability:

```bash
python3 scripts/nano_lab/bounded_job.py -- \
  python3 browser_tts/scripts/measure-browser.py \
  --hardware-webgpu --power-preference low-power \
  --model-base /models/chatterbox-nano-browser/ \
  --text "Take a slow breath in, and let your shoulders relax." --seed 1337 \
  --output-dir artifacts/nano_lab/browser_measurements/cloud_fp16_first
```

Select an installed normal Chrome binary with `--chrome` if needed. Headless
Shell is permitted only for static UI and adapter-only components. Optional
`--nvidia-smi` adds compute-process rows; empty rows do not prove zero graphics
memory. See `MEASUREMENT.md` for instrumentation overhead and missing coverage.

## Work order and acceptance

1. Verify the latest build, tests, real page, and installed voice-state hash.
2. Make a full FP16 load and short first/warm synthesis complete. Fix failures
   from logs and traces. Save both WAVs and generated speech tokens. Check all
   tensors/audio for invalid values. Preserve strict numerical gates.
3. Independently transcribe the actual WAVs. Check expected words. Inspect
   waveform integrity and listen in the delivered player. Compare controlled
   quality measurements and record uncertainty.
4. Rebuild Q4 through the guarded commands in `EMBEDDING_Q4.md`. Rebuild the
   preview afterward. Repeat the same text/seed/provider in a separate run with
   `--embedding-manifest /experiments/embedding_q4_candidate/manifest.json`.
   Compare tokens, words, audio, memory, TTFA, warm latency, and listening.
   Do not promote Q4 from file size or CPU errors alone.
5. Verify actual ready, reading, stop, restart, failure, retry, and save controls
   in the rendered page. Inspect screenshots and relevant interactions.
6. Extend the runner for a full Federalist reading. Its current short-run WAV
   API limits base64 export to 60 seconds. Use the real streamed Save path or a
   bounded streaming export for long audio. Do not collect the whole paper as
   a base64 string. Measure playback gaps, queue bounds, host memory over time,
   and saved-audio completeness. Listen at passage boundaries.
7. If speed or memory misses the target, identify the measured bottleneck and
   compare the same workload before/after any change. Preserve speech behavior.
   Only then investigate decoder precision, scratch reuse, or a smaller
   fixed-speaker model. Do not call Nano impossible from partial measurements.

Unfinished: full browser inference, correct browser-generated words, matched
listening, WebGPU equivalence, final voice realism, Q4 downstream acceptance,
full-paper continuity, stop/restart/save integration, and measured runtime
target. Local available RAM was about 2.6 GiB at the last status check. No model
job was left running. The handoff does not relax the native acoustic CUDA gate.
