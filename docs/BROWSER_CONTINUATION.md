# Fixed-voice browser continuation

## Objective and current boundary

Continue the selected Chatterbox Nano ASMR reader. The required result is correct,
clean, realistic speech and smooth Federalist reading with measured low memory.
The strict runtime target remains below 500 MiB. Report absolute browser RSS,
absolute PSS, PSS above the clean-browser baseline, and GPU allocation evidence
separately. PSS distributes shared host pages among processes. Requested WebGPU
buffer sizes are not physical VRAM residency.

The current code and CPU voice-state export do not prove this outcome. The
selected reference is a generated clip, not an accepted voice-quality result.
The Harvey research track and its strict CPU/CUDA gates remain in
[EXPERIMENT_HANDOFF.md](EXPERIMENT_HANDOFF.md).

## Implemented changes

- The worker loads three pinned graphs. It does not load the speech encoder.
- The offline exporter saves the encoder's four outputs for the exact selected
  reference. The browser checks provenance, tensor contracts, and binary hash.
- ONNX Runtime Web 1.30 uses its JSPI entry point and Blob external weights.
  Graph and weight hashes are checked before session creation.
- Recurrent language-model key/value outputs remain in WebGPU buffers.
- Two playback credits bound unplayed passage audio. PCM audio is saved in
  IndexedDB. Chrome can stream the completed WAV to disk.
- Playback and saved audio receive the same short edge fade. Scheduled gaps,
  synthesis times, and speech tokens are available in `window.voiceStudy`.
- A passage that reaches the speech-token limit fails explicitly.
- The default embedding remains FP16. Q4 requires an explicit local manifest.

## Verified component evidence

The selected WAV is mono PCM-24 at 24 kHz, lasting 3.52 seconds. Its SHA-256 is
`67f94a868976b22a47bce8fd00a873d4c5b7085ba6eedd698f8a898a26ce76c0`.

The guarded CPU export used ONNX Runtime 1.29 with CPUExecutionProvider. It saved
331,200 bytes. All outputs passed exact serialization round-trip checks:

| Output | Type | Shape |
|---|---|---|
| Audio features | float32 | `[1, 89, 768]` |
| Audio tokens | int64 | `[1, 88]` |
| Speaker embedding | float32 | `[1, 192]` |
| Speaker features | float32 | `[1, 176, 80]` |

The binary SHA-256 is
`6f56c0dd844ecc038e5b60debff22d1e16b44c9d13386d0f7a91156bfca1b38e`.
The export service took 9.900 seconds and had a 423.7 MiB cgroup peak. This is
not a browser process RSS measurement. The public base encoder produced this
state. No local fitted adapter was applied. WebGPU encoder parity and final
speech equivalence remain unverified.

At an earlier source checkpoint, the production build and all 21 component
tests passed under the resource guard. The tests cover WAV encoding, finite
state values, tensor/hash contracts, JSPI requirements, the actual exported Q4
manifest, and FP16 conversion.
The corrected FP16 converter handles
rounding across exponent boundaries. These checks do not exercise real model
inference or long reading.
Later low-copy, adapter-preference, and logit-check source changes were not
included in a successful production build. The low-copy source passed a syntax
check. The latest normal build was interrupted under the 4 GiB desktop reserve.

The pinned three-graph runtime and tokenizer/config files were restored and
SHA-256 verified. The stage command places ten runtime assets under
`browser_tts/public/models/chatterbox-nano-browser/`. It omits the encoder.
The reader discovers this local installation by its revision manifest. All
graph and weight hashes are still checked before session creation. An explicit
same-origin `modelBaseUrl` is also available for matched diagnostics.

## Q4 component result

The Q4 embedding export completed under the 1280 MiB guard. Its service took
19.211 seconds and had a 472 MiB cgroup peak. External weights fell from
87,304,704 to 22,678,761 bytes, a 74.0% reduction. Both tables were compared in
full on CPU. The measured maximum absolute errors were 0.0552063 for text and
0.1845703 for speech. Mean absolute errors were 0.0110672 and 0.0510274.
These errors are descriptive. There is no numerical acceptance gate.
The candidate is unpromoted. Its provenance, hashes, and detailed measurements
are retained in [the candidate report](../artifacts/nano_lab/browser_embedding_q4_20261001/manifest.json).

## Browser UI result

The corrected static UI check completed with Headless Shell under the 640 MiB
guard. All 16 checks passed. Read stayed disabled. The fixed-state failure,
retry button, hidden audio and reading controls, and hidden load progress were
checked. The mobile layout had `documentScrollWidth` 390 at a 390-pixel viewport.

The probe requested `/voice/__measurement_missing_state__.json`. Vite returned
HTTP 200 with an 8,599-byte `text/html` body. The worker rejected it with
`Could not parse the fixed voice state manifest: Unexpected token '<'...`.
This confirms the controlled invalid-manifest error path. It does not indicate
that the installed voice-state asset is missing. No Hugging Face or runtime
asset request was observed.

The [measurement report](../artifacts/nano_lab/browser_measurements/ui_fixed_voice_mobile_fixed/measurement.json)
records a sampled peak of 405.988 MiB RSS and 209.442 MiB PSS across the owned
Chrome tree. The [desktop screenshot](../artifacts/nano_lab/browser_measurements/ui_fixed_voice_mobile_fixed/browser-ui.png)
is 1265 by 1257 pixels. The [mobile screenshot](../artifacts/nano_lab/browser_measurements/ui_fixed_voice_mobile_fixed/browser-ui-mobile-390x844.png)
has a 390 by 844 viewport and a 390-pixel document width. These are static UI
results. They do not measure model load, inference, audio, or full-run memory.

Earlier 640 MiB attempts remain as historical failures. Normal Chrome exceeded
the aggregate RSS cap before the page check. A single-process reduction crashed
with exit code -11. A reduced multiprocess configuration also exceeded the cap
at clean startup. Its sampled Chrome peak was 1,068.199 MiB RSS and 384.806 MiB
PSS. Do not retry these same layouts under a larger cap.

The first Headless Shell UI run rendered the page but failed because its worker
target did not support CDP `Fetch.enable`. It fetched the real local state and
then failed because GPU was disabled. The screenshot exposed CSS-hidden
controls. The CSS was corrected, and the later `ui_fixed_voice_mobile_fixed`
report passes. Two paused-worker wrapper attempts also timed out; the runner
now uses the same-origin manifest override without pausing workers. Retained
reports include `ui_fixed_voice_headless_shell/`,
`ui_fixed_voice_fetch_wrapper/`, and `ui_fixed_voice_sync_wrapper/`.

## Adapter-only component checks

The adapter-only checks ran with the cached Chrome 154 Headless Shell under the
existing 1024 MiB guard. They did not prepare a model or synthesize speech.
The default/unspecified Intel selection passed the browser's non-fallback and
`shader-f16` checks. The explicit `high-performance` selection failed before
model preparation:

| Selection | Adapter | `shader-f16` | Result | Page-stage peak RSS/PSS |
|---|---|---:|---|---:|
| `default/unspecified` | Intel `gen-12lp` | Yes | Adapter check passed | 525.340/274.717 MiB |
| `high-performance` | NVIDIA `lovelace` | No | Stopped at feature check | 536.965/272.411 MiB |

The reports are
[default/unspecified Intel](../artifacts/nano_lab/browser_measurements/hardware_adapter_component/measurement.json)
and
[high-performance](../artifacts/nano_lab/browser_measurements/hardware_adapter_high_performance/measurement.json).
The Intel report predates the explicit power-preference option. It used
`requestAdapter()` without a preference, so it shows the browser's
default/unspecified selection. It does not prove that an explicit low-power
request selects Intel on Chrome 152. The app and runner now default to
`low-power`; a new adapter check is needed to confirm that request. The NVIDIA
report used explicit `high-performance`. It does not show that the model runs
correctly or faster on either adapter. No Dawn NVIDIA f16 toggle was forced.
The pinned [Dawn Vulkan source](https://dawn.googlesource.com/dawn/+/c294f092edae33d036dee4b8640f5c3560fc5da2/src/dawn/native/vulkan/PhysicalDeviceVk.cpp)
records unresolved NVIDIA f16 CTS failures and the feature rejection when the
toggle is disabled.

Run a current explicit low-power component check with the 1024 MiB cap and
4 GiB desktop reserve:

```bash
python scripts/nano_lab/bounded_job.py --max-memory-mib 1024 -- \
  python browser_tts/scripts/measure-browser.py --adapter-only --hardware-webgpu \
  --power-preference low-power \
  --chrome /home/terryd/.cache/puppeteer/chrome-headless-shell/linux-154.0.8037.57/chrome-headless-shell-linux64/chrome-headless-shell
```

The guard requires 5120 MiB available at launch. This adapter-only result is
separate from model inference and the strict runtime-memory target.

## Full model and audio status

No full-model hardware load, synthesis, or browser WAV has completed. The first
normal-Chrome FP16 attempt reached the embedding and language-model sessions,
then failed while fetching the conditional decoder. It selected Google's
SwiftShader software adapter. Clean-browser mean RSS/PSS were
1,141.117/398.174 MiB. Sampled load peaks were 1,837.613/1,003.231 MiB. Peak
PSS above the clean baseline was 605.057 MiB. These are incomplete software
load measurements. They do not measure completed inference, hardware GPU
residency, or voice quality. GPU buffer instrumentation was unavailable in
that run. The full report is
`artifacts/nano_lab/browser_measurements/fixed_voice_fp16_first/measurement.json`.

## Restore and reproduce

Run each workload alone through the existing guard. The default cap is
3072 MiB, with 6144 MiB required at launch. Small jobs need 5376 MiB for a
1280 MiB cap, 5120 MiB for a 1024 MiB cap, and 4736 MiB for a 640 MiB cap.
Every job preserves the 4096 MiB desktop reserve. Never raise caps or bypass a
refusal.

### Optional voice-state export reproduction

The fixed voice-state manifest and 331,200-byte binary are included at
`browser_tts/public/voice/asmr-state.json` and
`browser_tts/public/voice/asmr-state.bin`. Normal app use does not require the
exporter. Restore the pinned encoder and embedding source files only to
reproduce or regenerate that state:

```bash
python scripts/nano_lab/bounded_job.py --max-memory-mib 640 -- \
  python browser_tts/scripts/download-model.py --encoder-only
python scripts/nano_lab/bounded_job.py --max-memory-mib 640 -- \
  python browser_tts/scripts/download-model.py --embedding-only
```

Both downloads completed locally and passed pinned SHA-256 checks. A fresh
checkout needs the excluded encoder files only for this optional reproduction.
Export the voice state:

```bash
python scripts/nano_lab/bounded_job.py --max-memory-mib 640 -- \
  .venv-nano-cpu/bin/python browser_tts/scripts/export-voice-state.py \
  --model-dir models/chatterbox-nano-browser \
  --reference browser_tts/public/voice/asmr_t3_seed47_fit.wav \
  --output-dir browser_tts/public/voice
```

Restore and stage the runtime assets separately:

```bash
python scripts/nano_lab/bounded_job.py --max-memory-mib 640 -- \
  python browser_tts/scripts/download-model.py --runtime-only --stage-runtime
```

The staging code uses hard links when supported, with a file-copy fallback.
It verifies every staged hash. The completed local staging service took
3.724 seconds and had a 396.8 MiB cgroup peak. This includes file-cache charges;
it is not browser RSS. The fixed voice-state manifest and its 331,200-byte
binary are included in the repository publication at
`browser_tts/public/voice/asmr-state.json` and
`browser_tts/public/voice/asmr-state.bin`. The binary SHA-256 is
`6f56c0dd844ecc038e5b60debff22d1e16b44c9d13386d0f7a91156bfca1b38e`.
Restoring the encoder and rerunning the exporter is an optional reproducibility
check. The large encoder, pinned runtime model files, and Q4 weights remain
excluded. Restore the encoder and runtime files with the commands above.
Restore or regenerate Q4 weights with the
[Q4 export instructions](../browser_tts/EMBEDDING_Q4.md).

An earlier build with staged runtime files passed under the 640 MiB guard, with
a 512.7 MiB cgroup peak. Later source changes were not included in that build.
The latest normal build did not complete under the available 4 GiB reserve.

Build before measuring a production preview. Reuse the healthy project server
on `127.0.0.1:4187`; check its process directory. The preview serves `dist`.
New local voice/candidate files need a new build before preview measurement.

```bash
python scripts/nano_lab/bounded_job.py --max-memory-mib 640 -- \
  npm --prefix browser_tts run build
```

See [measurement commands and scope](../browser_tts/MEASUREMENT.md) and
[Q4 export and comparison](../browser_tts/EMBEDDING_Q4.md).

## Remaining acceptance work

1. Run the full FP16 model measurement in normal Chrome with the low-power
   adapter preference. Verify successful model load and worker adapter
   features. Measure matched first and warm reads. Save both WAVs. Full model
   inference and audio remain unverified.
2. Inspect the retained all-row Q4 CPU errors. Verify its browser operator
   support and downstream effects. These errors are descriptive, not a quality
   acceptance gate.
3. Measure FP16 and Q4 in normal Chrome with matched text and seed. Record the
   selected adapter identity, load/idle/inference peaks, first-audio time, warm
   latency, output speech tokens, and both WAVs.
   The seed controls the worker's speech-token sampler. It does not prove
   byte-identical final audio. Inspect the returned tokens and waveform hashes.
4. Verify spoken words independently. Compare controlled quality measurements
   and matched listening WAVs. Do not promote Q4 from file size alone.
5. Read Federalist No. 10. Check passage boundaries, actual audible continuity,
   stop/restart/save behavior, and memory over the full reading. A short WAV
   cannot prove long-text behavior.
6. If measured memory or speed misses the target, identify the measured
   bottleneck before changing the decoder or selecting a smaller voice model.

No browser voice or runtime profile is promoted.
