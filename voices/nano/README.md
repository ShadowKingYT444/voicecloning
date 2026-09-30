# Nano voice candidates

These profiles are research candidates. The user listened to the ASMR samples
and reported that they do not sound convincing. All output is synthetic and
retains the Chatterbox watermark. Zero-shot and fitted profiles are listed below.

From the project directory:

```bash
./nano-clone --voice asmr_soft --text "Take a quiet moment and let yourself relax." --output samples/asmr_soft.wav
./nano-clone --voice harvey --text "Check the details and prepare your next move." --output samples/harvey.wav
```

The launcher enforces one job, a 3 GiB memory limit, no job swap use, and a
two-core CPU quota. It requires 6 GiB available to start. It stops if system headroom falls below 4 GiB. Do not bypass
the launcher during desktop use. CUDA is the default device.

Zero-shot profiles: `asmr_conversational`, `asmr_soft`, `asmr_intimate`, and `harvey`.
The optional `asmr_conversational_adapted` profile uses a learned speaker
adjustment at 75% strength. It is speaker-specific adaptation, not zero-shot,
and is now disabled because its training cache has transcript/audio mismatches.
Its prior samples remain available for historical comparison.

The opt-in `asmr_fitted_experimental` profile combines the audited all-attention
adapter with the bounded mel-envelope diagnostic. It is speaker-specific fitted
audio, not zero-shot, and its status is `awaiting_listening_validation`.
The mel report is diagnostic only and does not establish human-level realism.
Run the matched robustness question with the native CLI:

```bash
./nano-clone --voice asmr_fitted_experimental --text "Did you remember the blue folder? I thought we agreed to meet at half past nine." --seed 79 --output artifacts/nano_lab/native_asmr_fitted_question_79.wav
```

The WAV sidecar records the calibration report hash, fitted-delta hash,
conditioning and model hashes, and the applied strength.

The reference-feature cache avoids loading the reference encoders during normal
synthesis. A missing ordinary reference cache is reconstructed from the profile's source references. Fitted decoder profiles require the exact learned cache and fail before model loading if the cache is missing or its SHA-256 differs.

The WAV sidecar records generation time, host RSS, GPU allocations, and output
processing. Output mastering applies a 45 Hz high-pass, bounded loudness gain,
a true-peak ceiling estimated at 4x oversampling, and short endpoint fades.
It uses no noise gate. Use `--raw` to retain the unmastered waveform.

Full precision is the default. In the matched ASMR test, T3 half precision used
more host RSS and ran slower. The acoustic model always uses full precision.
Do not interpret a successful generation or a speaker-similarity score as proof
of human-level realism.

Review the [sample package](../../artifacts/nano_lab/delivery/index.html) and
[RSS report](../../artifacts/nano_lab/RSS_REPORT.md). The experimental fitted
conversational sample is included for comparison. The original zero-shot
profiles remain unchanged. See `robustness_review.json` in the sample parent
directory for the adapted profile's measured quality/content tradeoffs.

Long text is grouped at sentence boundaries, normally up to 32 words per
segment. A segment that reaches the token limit is split into smaller pieces
and retried. A failed segment of 12 words or fewer stops generation. The
sidecar records these retries. A failed request preserves any previous complete
output file and does not publish partial audio.

An experimental CPU ONNX launcher is available for short text:

```bash
./nano-clone-onnx --voice asmr_soft --experimental-t3 --text "Take a quiet moment and let yourself relax." --output samples/asmr_onnx.wav
```

This launcher uses a 1280 MiB memory/RSS cap and requires 5376 MiB available.
It loads model components in separate processes. The measured CPU runtime is
slow: approximately 11 to 13 seconds per second of output audio, including
startup. It does not replace the faster GPU launcher. The strict long-context
T3 cache check has a small unresolved mismatch. Output uses a different random
sampler from Torch, so equal seeds do not produce equal WAVs. Run artifacts
record stage timings, RSS, and the limitation. This path currently accepts one
short passage; use the GPU launcher for automatic long-text segmentation.

`harvey_fitted_experimental` uses the new rank-2 adapter with the isolated reference. This speaker-specific experiment has only two training clips and one validation clip. It is not zero-shot and has not passed listening acceptance. Compare the [Harvey samples](../../artifacts/nano_lab/harvey_fitted_comparison/index.html).

```bash
./nano-clone --voice harvey_fitted_experimental --text "We have a clear plan. Check the details, prepare your next move, and walk into that meeting with confidence." --seed 31 --output samples/harvey_fitted.wav
```

Matching fitted ONNX exports now pass short-context CPU and CUDA checks. The base export still rejects fitted profiles. Select the matching graph explicitly:

```bash
./nano-clone-onnx --voice asmr_fitted_experimental --model-dir artifacts/nano_lab/onnx_staged_adapter_aligned_all_attn --ort-provider cuda --cuda-kv-resident --experimental-t3 --text "Did you remember the blue folder? I thought we agreed to meet at half past nine." --seed 79 --output samples/asmr_fitted_onnx.wav
./nano-clone-onnx --voice harvey_fitted_experimental --model-dir artifacts/nano_lab/onnx_staged_adapter_harvey_rank2 --ort-provider cuda --cuda-kv-resident --experimental-t3 --text "Check the details and prepare your next move." --output samples/harvey_fitted_onnx.wav
```

This serial-stage mode used 1004 MiB process-tree RSS on the measured ASMR passage. A persistent batch trades about 1.9 GiB RSS for faster warm requests. It exits after processing its manifest:

```bash
python scripts/nano_lab/bounded_job.py -- .venv-nano-ort/bin/python scripts/nano_lab/onnx_batch.py run --manifest artifacts/nano_lab/onnx_batch_asmr_fitted_manifest.json --batch-dir artifacts/nano_lab/my_asmr_batch --model-dir artifacts/nano_lab/onnx_staged_adapter_aligned_all_attn --ort-provider cuda --cuda-kv-resident --experimental-t3 --finish-in-worker
```

Use a new batch directory for each run. For Harvey, substitute `onnx_batch_harvey_fitted_manifest.json` and `onnx_staged_adapter_harvey_rank2`. CUDA TF32 is disabled to preserve the full-precision reference. The strict context-400 cache mismatch is still unresolved; short-context verification does not establish production readiness.

## New acoustic decoder profiles

`asmr_decoder_fitted_experimental` adds the constrained 192-value acoustic speaker fit. `asmr_decoder_slow_experimental` uses the same fitted cache with repetition penalty 1.0. Both are speaker-specific experiments awaiting listening acceptance. Their required cache is SHA-256 bound; it cannot be silently rebuilt from the reference.

The standard decoder fit improves mean similarity and DNSMOS on three matched passages. The slower variant raises identity similarity further but reduces DNSMOS. See the [same-text comparison and source](../../artifacts/nano_lab/decoder_cadence_combined/index.html) before choosing. No realism claim follows from these scores.

```bash
./nano-clone --voice asmr_decoder_slow_experimental --text "Did you remember the blue folder? I thought we agreed to meet at half past nine." --seed 79 --output samples/asmr_decoder_slow.wav
./nano-clone-onnx --voice asmr_decoder_fitted_experimental --model-dir artifacts/nano_lab/onnx_staged_adapter_aligned_all_attn --ort-provider cuda --cuda-kv-resident --experimental-t3 --text "Did you remember the blue folder? I thought we agreed to meet at half past nine." --seed 79 --output samples/asmr_decoder_onnx.wav
```

The native slower profile was verified byte-identical to its comparison master and used 2392 MiB peak process RSS. The staged ONNX standard profile passed an independent word check and used 1046 MiB peak process-tree RSS. Its cold run took 50.87 seconds for 5.48 seconds of audio. The existing long-context ONNX limitation still applies. These two runtime measurements have different sampling settings and output durations.

## Prompt and acoustic attention candidates

`asmr_prompt_experimental` changes the T3 reference prompt and timing setting. `asmr_acoustic_attention_experimental` adds the fitted rank-2 acoustic attention adapter. Both use the same pinned acoustic speaker cache and mel correction. They remain speaker-specific candidates awaiting listening acceptance. The [six-sample comparison](../../artifacts/nano_lab/decoder_attention_prompt_comparison_v2/index.html) shows mixed identity and cleanliness scores; pitch and timing still differ from the source.

The acoustic profile completed this CPU ONNX command:

```bash
./nano-clone-onnx --voice asmr_acoustic_attention_experimental --model-dir artifacts/nano_lab/onnx_staged_adapter_aligned_all_attn_decoder_attention --ort-provider cpu --experimental-t3 --text "Let's see who's the lucky person who gets to kiss me. Wait a minute. Do I know you?" --seed 31 --output samples/asmr_acoustic_attention_cpu.wav
```

Measured peak process-tree RSS is 861.75 MiB, with 118.07 seconds cold pipeline time for 6.96 seconds of output. Independent word auditing of this new ONNX WAV is pending due desktop memory pressure. CPU estimator checks pass at five lengths. CUDA estimator checks fail at four lengths, so this acoustic graph currently rejects CUDA inference. The earlier T3 long-context limitation still applies.

The native CLI implements the same optional profile. Its standalone verification is pending because the memory guard refused to start below 6 GiB available. The native comparison samples were generated through the verified sweep path. No default profile changes.
