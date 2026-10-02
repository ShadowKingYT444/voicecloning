# Browser measurement

`scripts/measure-browser.py` measures the browser reader in a clean Chrome
profile. It records a clean-browser baseline, app startup, model load, idle,
first synthesis, warm synthesis, and WAV export. The first and warm synthesis
calls use the same short text and seed. The runner saves the first and warm
WAVs with SHA-256 hashes. The JSON report has timestamps, stage samples,
process IDs, text, seed, result chunks, and app metrics. A JSONL file keeps a
sample record on disk during the run.

The October 2 continuation adds bounded long-WAV transfer, `--full-paper`,
opt-in `--lossless-embedding-manifest`, and separate page/worker V8 heap samples.
Use `--require-hardware-webgpu` to require an identified non-fallback FP16
adapter without changing backend flags. Ordinary launches do not force unsafe
WebGPU. Read [the tested cloud handoff](../docs/BROWSER_MVP_20261002.md) for
evidence, exact asset recovery and unresolved speech/watermark gates.

The runner does not start or stop Vite. Keep the existing app server running on
`http://127.0.0.1:4187/`. The runner rejects another host or port. It launches
Chrome with a new temporary profile, then removes that profile and only the
Chrome process tree that it owns.

## Full browser measurement

Run from the repository root. The app server must already be healthy. Use the
existing resource guard for the browser and the model workload:

```bash
python3 scripts/nano_lab/bounded_job.py -- python3 browser_tts/scripts/measure-browser.py
```

The default guard keeps the 3072 MiB hard cap, 2500 MiB high limit, no swap,
two CPU cores, and the 4 GiB desktop reserve. The guard requires 6 GiB
available before launch. Do not raise these limits. Do not use the
single-process guard backend because it blocks Chrome child processes.

The full run has a 30 minute hard timeout. The model load timeout is 15
minutes. Each synthesis call has a five minute timeout. The default test text
is short. To select another short matched passage, set `--text` and optionally
`--seed`:

```bash
python3 scripts/nano_lab/bounded_job.py -- python3 browser_tts/scripts/measure-browser.py \
  --text "Take a quiet moment and let your shoulders relax." --seed 1337
```

The runner records output under
`artifacts/nano_lab/browser_measurements/run-<timestamp>/`. It writes
`measurement.json`, `measurement-samples.jsonl`, `first-reading.wav`,
`warm-reading.wav`, and `chrome.stderr.log`. Use
`--output-dir` to select another output directory. Use `--nvidia-smi` to add
optional per-process compute-app samples when `nvidia-smi` is available.

For an explicit Q4 embedding candidate, pass its local URL path. The browser
app validates the manifest and its asset hashes before it creates the graph:

```bash
python3 scripts/nano_lab/bounded_job.py -- python3 browser_tts/scripts/measure-browser.py \
  --embedding-manifest /experiments/embedding_q4_candidate/manifest.json
```

The runner passes `{embeddingManifestUrl: ...}` to `prepareVoice()` only for a
full measurement. Without this flag, the app uses its pinned default embedding.
The Q4 candidate run remains an explicit diagnostic. It does not establish a
general improvement.

To load model assets from a local same-origin directory, pass `--model-base`:

```bash
python3 scripts/nano_lab/bounded_job.py -- python3 browser_tts/scripts/measure-browser.py \
  --model-base /models/chatterbox-nano-browser/
```

The runner checks that the URL uses the app's origin and has no credentials,
query, or fragment. It passes `{modelBaseUrl: ...}` to `prepareVoice()` and
records both the requested and resolved URL. Without this option, the app
discovers a pinned local installation from its revision manifest and otherwise
uses the pinned remote model base. The ready snapshot records the actual base.

The fixed voice-state manifest and its 331,200-byte binary are included in the
repository publication at `public/voice/asmr-state.json` and
`public/voice/asmr-state.bin`. The binary SHA-256 is
`6f56c0dd844ecc038e5b60debff22d1e16b44c9d13386d0f7a91156bfca1b38e`.
Restoring the encoder and rerunning the exporter is an optional reproducibility
check. The large model files and Q4 weights remain excluded and can be
restored or rebuilt through the [model restore](../docs/BROWSER_CONTINUATION.md#restore-and-reproduce)
and [Q4 export](EMBEDDING_Q4.md) instructions.

For Linux headless Chrome, `--hardware-webgpu` adds Chrome's Vulkan WebGPU
flags. The full measurement checks the page adapter before loading and the
model worker adapter after loading. The `--adapter-only` mode runs the same
page adapter check without model preparation or synthesis. Both checks require
adapter identity, `isFallbackAdapter === false`, no known software-renderer
label, and the `shader-f16` adapter feature. The ordinary measurement keeps
its normal browser flags.

```bash
python3 scripts/nano_lab/bounded_job.py -- python3 browser_tts/scripts/measure-browser.py \
  --hardware-webgpu --model-base /models/chatterbox-nano-browser/ --nvidia-smi
```

The command adds `--use-angle=vulkan`, `--enable-features=Vulkan`, and
`--disable-vulkan-surface`. The runner already enables
`--enable-unsafe-webgpu`. It does not add `--no-sandbox` to a full measurement.
The check reads `GPUAdapterInfo.isFallbackAdapter` first. It uses the older
`GPUAdapter.isFallbackAdapter` property only when present. Chrome removed the
older property in Chrome 140. If neither property is available, the report
records the fallback status as unknown and the explicit hardware check fails.
The check identifies the adapter exposed by Chrome. It does not prove that the
driver uses a physical GPU. Verify the selected device with Chrome's
`chrome://gpu`, `vulkaninfo --summary`, and `nvidia-smi` when hardware identity
matters. Chrome's [Linux WebGPU guidance](https://developer.chrome.com/blog/supercharge-web-ai-testing)
and [WebGPU troubleshooting guide](https://developer.chrome.com/docs/web-platform/webgpu/troubleshooting-tips)
describe the Vulkan setup and software fallback checks. Chrome's [WebGPU 140
release notes](https://developer.chrome.com/blog/new-in-webgpu-140) document
removal of the legacy property.

### Adapter-only component check

Use `--adapter-only` to check adapter selection without loading weights or
creating inference sessions. The retained Intel report predates the explicit
power-preference option. It called `requestAdapter()` without a preference, so
it records the browser's default/unspecified selection. It does not prove that
an explicit low-power request selects Intel on Chrome 152. The app and runner
now default to `low-power`; use this command for a current explicit low-power
component check. It uses cached Headless Shell with one renderer and the
existing 1024 MiB guard:

```bash
python3 scripts/nano_lab/bounded_job.py --max-memory-mib 1024 -- \
  python3 browser_tts/scripts/measure-browser.py --adapter-only --hardware-webgpu \
  --power-preference low-power \
  --chrome /home/terryd/.cache/puppeteer/chrome-headless-shell/linux-154.0.8037.57/chrome-headless-shell-linux64/chrome-headless-shell
```

The guard requires 5120 MiB available at launch and keeps a 4 GiB desktop
reserve. The runner adds one-renderer flags in adapter-only mode. This mode
does not benchmark model load, inference, latency, or audio. The report is
`artifacts/nano_lab/browser_measurements/hardware_adapter_component/measurement.json`.
The historical default/unspecified request selected Intel `gen-12lp`, with
`shader-f16` and a non-fallback adapter. Its sampled page-stage peak was
525.340 MiB RSS and 274.717 MiB PSS, with 69.987 MiB PSS above the
clean-browser baseline.

The paired `high-performance` adapter check selected NVIDIA `lovelace` but did
not expose `shader-f16`. The explicit check stopped before model preparation.
It did not test the NVIDIA inference path. Its report is
`artifacts/nano_lab/browser_measurements/hardware_adapter_high_performance/measurement.json`.
The pinned [Dawn Vulkan source](https://dawn.googlesource.com/dawn/+/c294f092edae33d036dee4b8640f5c3560fc5da2/src/dawn/native/vulkan/PhysicalDeviceVk.cpp)
records unresolved NVIDIA f16 CTS failures and keeps this feature unsupported
unless a Dawn toggle enables it. Do not force that toggle. The current app and
runner source defaults to `low-power` so the browser can select an adapter that
advertises the required FP16 shader feature. The historical Intel report did
not test that explicit preference. Adapter-only evidence does not show that the
model runs correctly or quickly on that adapter.

## UI-only check

`--ui-only` does not load model weights or synthesize speech. It waits for the
page text and fonts to render. It checks that Read stays disabled before the
model is ready. Before calling `prepareVoice()`, it requests the unique
same-origin path `/voice/__measurement_missing_state__.json`. The report
records the actual response status, content type, and body kind. The runner
then passes that URL as `voiceStateManifestUrl`. The worker checks the small
voice-state manifest before it requests model assets. This is a controlled
invalid-path probe. It does not claim that the installed voice-state asset is
missing. UI-only mode does not pause workers or install DevTools fetch hooks.
It checks the fixed voice-state error, retry control, hidden audio and reading
controls, hidden load progress, and observed model-host requests. It saves
desktop and 390 by 844 screenshots. It checks that document content fits
within 390 CSS pixels. The completed check passed all 16 assertions. The test
manifest path returned HTTP 200 with an HTML body from the Vite fallback. The
worker rejected that body as invalid fixed voice-state JSON before it requested
model weights. No Hugging Face or runtime asset request was observed.

The report is
`artifacts/nano_lab/browser_measurements/ui_fixed_voice_mobile_fixed/measurement.json`.
The [desktop screenshot](../artifacts/nano_lab/browser_measurements/ui_fixed_voice_mobile_fixed/browser-ui.png)
is 1265 by 1257 pixels. The [390-pixel screenshot](../artifacts/nano_lab/browser_measurements/ui_fixed_voice_mobile_fixed/browser-ui-mobile-390x844.png)
was captured at an 844-pixel viewport height. Its document scroll width was
390 pixels. The UI-only sampled peak was 405.988 MiB RSS and 209.442 MiB PSS.
This is page and error-state memory. It is not model memory or a full runtime
result.

The UI-only command still launches Chrome. Use the small-job guard only when
the 640 MiB cap and 4 GiB desktop reserve are available:

```bash
python3 scripts/nano_lab/bounded_job.py --max-memory-mib 640 -- \
  python3 browser_tts/scripts/measure-browser.py --ui-only
```

The guard requires 4736 MiB available at launch and stops below 4 GiB. A guard
refusal means the check did not run. Do not bypass the guard.

To reduce the UI-only process count, this mode uses `--no-zygote`,
`--no-sandbox`, `--renderer-process-limit=1`, `--disable-gpu`, and
`--disable-software-rasterizer`. It reuses Chrome's startup page. These flags
apply only to the static UI check. The check does not measure WebGPU
compatibility. The full measurement uses the normal Chrome process layout and
GPU settings. Do not compare UI-only memory with full-run model memory. The
controlled error path prevents the saved voice from starting a model download
during this static check.

The normal Chrome UI-only attempt exceeded the 640 MiB aggregate RSS cap during
`clean_baseline`. It had 10 Chrome processes, 1068.199 MiB summed RSS, and
384.806 MiB summed PSS. The UI page did not load. The local Puppeteer cache has
a `chrome-headless-shell` binary. Its file size is 197,979,464 bytes. The
cached regular Chrome binary is 294,067,520 bytes. Chrome describes Headless
Shell as a smaller automation and screenshot option with fewer dependencies
than full Chrome. The binary size does not prove lower resident memory.

Use Headless Shell only for UI-only and adapter-only component checks. This
cached build is Chrome 154; the installed regular Chrome is Chrome 152.
Headless Shell also differs from regular Chrome. The UI screenshot is a static
check, and the adapter check reports only this shell's adapter selection. They
are not exact rendering, runtime, or inference matches for the full browser.
The runner rejects Headless Shell in full model mode. Its `--chrome` option
accepts the path and selects the shell's `--headless` flag.

```bash
python3 scripts/nano_lab/bounded_job.py --max-memory-mib 640 -- \
  python3 browser_tts/scripts/measure-browser.py --ui-only \
  --chrome /home/terryd/.cache/puppeteer/chrome-headless-shell/linux-154.0.8037.57/chrome-headless-shell-linux64/chrome-headless-shell
```

The corrected UI check completed with Headless Shell under the existing 640
MiB guard. Do not increase either cap to make another check start. See the
[Chrome Headless Shell documentation](https://developer.chrome.com/docs/automation-and-testing/headless-chrome-shell) for its intended use and feature differences.

## Memory fields

Every sample sums `Rss` and `Pss` from `/proc/<pid>/smaps_rollup` for the
Chrome root PID and its current `/proc` child-process tree. The report includes
the clean `about:blank` browser baseline, each stage's sampled peak RSS and
PSS, and the sampled peak PSS increase over the clean baseline. Each stage
reports the configured and observed sample interval. Slow DevTools reads can
increase that interval. GPU counter reads can wait up to four seconds per
worker. This can delay a sample and perturb worker timing. A sampled maximum
can miss a shorter peak. This is host process memory for the owned browser
profile. It excludes the Vite server, other desktop processes, and GPU device
memory. Baseline samples keep incremental PSS null. Later stages use the mean
clean-browser PSS as their reference.

The project keeps the strict runtime-memory target below 500 MiB. The report
shows incremental PSS because it helps separate Chrome's clean startup cost
from the app and model cost. It also shows absolute RSS and PSS. Incremental
PSS is a diagnostic value. It does not redefine or pass the strict target by
itself. Compare the measured scope with the target before making a claim.

Before each full-run worker starts its module code, the runner pauses it through
Chrome DevTools Protocol. It wraps `navigator.gpu.requestAdapter()` first. When
the app receives an adapter, the runner wraps that adapter's
`requestDevice()`. It then wraps `createBuffer()` on each returned device and
`destroy()` on each buffer that it creates. These hooks install lazily because
some workers do not expose `GPUDevice` and `GPUBuffer` constructors at startup.
The report distinguishes `installationPending`, device-hook installation,
active buffer instrumentation, partial coverage, and unavailable hooks. It
records adapter and device features, hook failures, creates, and explicit
destroys. The counters sum requested `GPUBuffer` descriptor sizes. They do not
measure physical VRAM residency. The current byte estimate can stay high when
the app relies on garbage collection. Browser, ONNX Runtime, and driver
allocations outside these JavaScript methods may be absent. The wrappers and
DevTools reads also add measurement cost.

Host PSS and WebGPU requested-buffer bytes are separate measures. Do not add
them and call the result exact resident memory. The optional `nvidia-smi`
query reports compute-app rows for owned Chrome PIDs. A WebGPU Vulkan graphics
allocation may not appear in that query. An empty result does not prove zero
GPU memory use.

The first and warm stage wall times include synthesis, browser storage writes,
and scheduled playback completion. The app's synthesis-time metrics exclude
playback. The WAV exports run in separate stages. The report hashes both WAVs
and records whether they are byte-identical. Hash equality proves byte identity
only. A hash difference needs waveform checks and listening.

The runner verifies that the app returned audio chunks, speech-token metadata,
and valid WAV containers. It does not run ASR or listen to the audio. These
checks do not prove correct spoken words, realistic voice quality, clean sound,
continuous playback, or a production-ready memory result. Inspect both WAVs
and compare quality and latency separately.

## Current run boundary

Do not run Chrome or a model when available host memory is below the existing
guard threshold. When this runner was prepared, reported host memory was about
3.2 GiB available. That is below both launch thresholds. The guard must decide
whether a later run can start. A refusal is not a browser result.
