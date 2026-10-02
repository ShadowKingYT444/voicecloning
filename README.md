# Voice cloning research workspace

This project aims to read long text, including Federalist No. 10, in a selected ASMR voice with a small, low-memory TTS model. The target is less than 500 MiB of runtime memory, with streaming and short batches so playback starts quickly and continues smoothly. The reader can use a precomputed voice state. Zero-shot cloning in the browser is not a requirement. The research also tests whether a model can run in the browser, but a local inference service is acceptable when it meets the memory and playback goals. Chatterbox Nano, Pocket TTS, Kokoro, Whisper, and Voicebox remain research candidates.

Local browser inference now completes with the selected GPU device. The native-donor voice state passes the short independent transcript check in first, warm and restarted reads. See [local verification](docs/LOCAL_BROWSER_VERIFICATION_20261002.md). Browser execution is only one deployment option. The current measured CPU ONNX profile takes 118.07 seconds to produce 6.96 seconds of audio and uses 861.75 MiB of process-tree RSS. The existing Pocket TTS ONNX benchmark uses 798 MiB after model load and 1,921 MiB after voice conditioning. Neither meets the under-500-MiB target. Quantized Pocket model files total 559 MiB on disk, before runtime memory. Streaming and a precomputed voice state can reduce startup delay and repeated conditioning work. They do not by themselves reduce model memory below the target. These native measurements do not establish the memory limit of a fixed-voice browser runtime. The latest browser trace does not meet the strict runtime memory target. See [latest matched voice results](artifacts/voice-experiments-20260930/README.md) and the [browser runtime documentation](https://onnxruntime.ai/docs/tutorials/web/).

**The requested voice realism is not achieved.** The user rejected the earlier comparison: “Neither sounds convincing yet.” The newer candidates remain experimental. Speaker-specific fitted adapters are not evidence of general zero-shot improvement.

## Start here, future agent

For the current browser task, start with [local verification](docs/LOCAL_BROWSER_VERIFICATION_20261002.md), then [the cloud handoff](docs/CLOUD_BROWSER_HANDOFF.md).
The [October 2 tested continuation](docs/BROWSER_MVP_20261002.md) includes
guarded browser controls, lossless embedding probes and a long-export harness.
Real browser Stop/restart and WAV export pass. The native-donor candidate speaks the short sentence correctly.
Full-paper speech, accepted fitted voices, watermark parity and the memory target
remain unfinished; the lossless reader is opt-in.

1. Read [the experiment handoff](docs/EXPERIMENT_HANDOFF.md), then [current status](artifacts/nano_lab/WORK_STATUS.md).
2. Read [resource constraints](scripts/nano_lab/AGENTS.md). The original machine suffered memory pressure. Run one model workload at a time through `bounded_job.py`.
3. Restore the local assets described in [asset transfer](docs/ASSETS.md). A Git clone alone cannot synthesize the fitted voices or resume fitting.
4. Rebuild the environments using [environment notes](docs/ENVIRONMENT.md). Do not copy the old virtual environments.
5. Finish the pending runtime checks, then run the [stricter T3 dataset experiment](artifacts/nano_lab/t3_clip_consensus/NEXT_EXPERIMENT.md). Preserve the current controls and evaluate new text.

The primary reader target is convincing, clean speech in the selected ASMR voice, with a strict under-500-MiB runtime target and streamed Federalist playback. Keep Harvey experiments as a separate research track. Deliver matched listening samples and measured memory/latency reports. Automated similarity, transcript, and noise scores do not establish perceptual success.

## Current evidence

| Result | Evidence and limitation |
|---|---|
| ASMR acoustic attention fit | Rank 2 across 224 projections; 344,064 trained values. Validation reconstruction loss falls 15.32%. This is not a realism score. |
| Prompt plus acoustic attention | Six native clips pass independent Whisper-small word checks. New-text similarity improves slightly; the cleanliness proxy falls. Pitch and timing still differ from the source. |
| New CPU ONNX profile | 861.75 MiB sampled peak process-tree RSS. 118.07 seconds cold runtime for 6.96 seconds of audio. Its independent word audit remains pending. |
| New CUDA acoustic stage | Fails strict numerical comparison at four of five lengths. Inference remains blocked for this provider. CPU verification does not authorize CUDA. |
| Earlier persistent CUDA runtime | About 1.9 GiB process-tree RSS; measured warm requests faster than playback. These results use older profiles. |
| Harvey fit | Only two training clips and one validation clip. Six comparison word checks pass. Automatic gains are small; realism is unaccepted. |
| Latest CPU VM comparison | The September 30 ASMR fit lowers validation loss but lowers mean speaker similarity on six new-text samples. It is not promoted. Harvey temperature 0.6 passes six small content checks, with slightly lower mean speaker similarity. Human listening is pending. |

Read the [quality report](artifacts/nano_lab/QUALITY_REPORT.md) and [RSS report](artifacts/nano_lab/RSS_REPORT.md). The [listening page](artifacts/nano_lab/delivery/index.html) needs the excluded local WAV files before its players work.

## Browser speech reader

For ordinary browsers without WebGPU, use the [local CPU reader](docs/LOCAL_CPU_BROWSER_20261002.md).
The same-origin CPU service runs the pinned model and serves the site on port
4187. Chromium and Firefox short speech, word checks and WAV export pass.
Chromium Stop/restart also passes. No browser GPU or JSPI support is required
for this path. It remains slow and does not establish the complete app's memory
target. Safari and Edge have not been tested.

The `browser_tts/` app continues the Chatterbox Nano browser experiment. The current code loads a precomputed selected-voice state and three WebGPU inference graphs. It removes the reference encoder from the reading session. The three published graph/weight pairs total 374.14 MiB. Tokenizer and configuration files bring the model assets to 377.54 MiB. Voice state, runtime, activations, and browser overhead are additional. Download size does not prove the under-500-MiB runtime target.

The loader uses ONNX Runtime Web 1.30 JSPI with Blob-backed external weights. The worker keeps recurrent KV tensors on WebGPU. The playback queue holds at most two passages. Generated audio is saved to IndexedDB, and Chrome can stream the completed WAV to disk. Device-bound Intel hardware inference completes. The native-donor first, warm and restart exports have zero word errors on one short sentence. Warm generation remains slower than playback, and measured Chrome RSS exceeds the memory target. See [browser continuation](docs/BROWSER_CONTINUATION.md) for setup, evidence, and remaining gates.

The selected reference remains `asmr_t3_seed47_fit.wav`. It is a generated 3.52-second clip. Its fixed voice-state binary and manifest are included for cloud continuation. All four tensors passed exact byte serialization round-trip checks. The export does not fold the local fitted adapters into the community ONNX graphs. The default reader now uses a separate, hash-pinned native ASMR T3 conditioning prefix with one speech-start row. The other three decoder-conditioning tensors remain from the selected reference state. This fixes the short word check without applying a fitted adapter. Perceptual realism, full-paper continuity and WebGPU numerical equivalence remain unverified. A fresh checkout must restore the excluded model weights before inference.

The dev server uses an explicit port:

```bash
cd browser_tts
npm ci
npm run dev
```

Model export, browser inference, and browser verification must run serially through the existing resource guard. Do not click Prepare in an unrelated desktop Chrome session to bypass the guard. Use the isolated measurement runner described in [measurement instructions](browser_tts/MEASUREMENT.md).

## Repository map

- `scripts/nano_lab/`: optimized native loader, fitting, dataset checks, evaluation, mastering, ONNX export/runtime, and tests.
- `voices/nano/`: opt-in voice profiles and [usage notes](voices/nano/README.md).
- `nano-clone`, `nano-clone-onnx`: guarded native and staged ONNX launchers.
- `artifacts/nano_lab/`: retained text reports, configurations, manifests, and experiment helper code. Binary outputs are excluded.
- `artifacts/nano_lab/vm_20260930/`: latest CPU VM run reports and audit data. `artifacts/voice-experiments-20260930/` contains its selected matched listening WAVs and results.
- `vendor/chatterbox/`: vendored upstream source with local lazy-import changes. Preserve these changes.
- `vendor/dnsmos/`: scoring source and its license; the ONNX scoring weights are excluded.
- `Pocket_package/`, `Whisper_fast_package/`, `kokoro_package/`, `scripts/`: earlier baseline tools.
- `voicebox/`: upstream application source snapshot. Nano experiments run through the lab launchers, not the app server.

## Run after asset and environment restoration

Run from the repository root on Linux with a working systemd user manager and cgroup v2. The native path needs 6 GiB available to start. The staged ONNX launcher uses a 1280 MiB cap and needs 5376 MiB available.

```bash
./nano-clone --list
./nano-clone --voice asmr_soft --text "Take a quiet moment and let yourself relax." --output samples/asmr_soft.wav
./nano-clone --voice harvey --text "Check the details and prepare your next move." --output samples/harvey.wav
```

The new acoustic profile's measured CPU path is:

```bash
./nano-clone-onnx --voice asmr_acoustic_attention_experimental --model-dir artifacts/nano_lab/onnx_staged_adapter_aligned_all_attn_decoder_attention --ort-provider cpu --experimental-t3 --text "Let's see who's the lucky person who gets to kiss me. Wait a minute. Do I know you?" --seed 31 --output samples/asmr_acoustic_attention_cpu.wav
```

Keep `--experimental-t3`: a separate long-context T3 cache mismatch remains unresolved. Equal seeds across Torch and ONNX do not imply equal waveforms because their random samplers differ.

## Verification and continuation rules

- Measure the same workload before and after a performance change. Report RSS scope, startup time, audio duration, and warm latency separately.
- Keep source audio, references, training clips, validation clips, and evaluation passages distinct. Preserve source intervals, hashes, and transcript admission checks.
- Do not promote the rejected MLP adapter or silently rebuild a fitted conditioning cache from another reference.
- Keep the watermark and controlled mastering. Do not hide voice errors with aggressive denoising or a fixed pitch shift.
- Do not raise resource limits or run concurrent models to force a test through. A stronger host can run the existing guarded queue first.
- Verify the actual user-facing WAVs. A lower training loss, passing build, or proxy score is insufficient.

The final local stop was caused by low host memory, not a successful completion. Pending work and exact commands are in the handoff. No training job was left running.

## Source and asset boundaries

This is a source snapshot, not a self-contained model distribution. Git excludes model weights, trained binary checkpoints, generated audio, tensor caches, ONNX graphs, virtual environments, credentials, and large execution traces. The shared source recordings and curated reference clips are tracked so cloud agents can reproduce the reference preparation steps. Historical reports retain original paths and measurements. See [asset transfer](docs/ASSETS.md) before moving to another directory or machine.

Third-party licenses remain with their respective sources. See [provenance](docs/THIRD_PARTY.md). No new blanket license overrides those licenses.
