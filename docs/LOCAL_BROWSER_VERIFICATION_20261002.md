# Local browser verification, 2026-10-02

The cloud branch is merged into local `main` at `2ed423b`. This continuation
uses the existing desktop resource guard. No caps or desktop reserve were
changed. The native-donor candidate passes the short browser word check.
Full Federalist playback, perceptual realism and target memory remain
unverified or unmet.

## Verified changes

- The worker passes one explicitly selected Intel GPU device to every WebGPU
  session. Setting `ort.env.webgpu.adapter` alone did not bind the native
  C++ WebGPU execution provider to that adapter. The earlier load selected a
  second NVIDIA adapter without `shader-f16` and failed hardware verification.
- The ready-state label now says `Ready to read.`. The actual rendered page
  was inspected after the change.
- The measurement runner captures ready and completed screenshots. It can
  exercise the real Stop control after the first generated passage, restart
  the reader, and export the restart WAV.
- Optional `--trace-inference` records four initial logit vectors per passage,
  text IDs, and input metadata. It is disabled for normal reads and cannot be
  combined with the full-paper measurement.
- Lossless embedding verification now checks the manifest digest and complete
  table counts. All 43,652,352 Float32 values match the pinned CPU embedding
  graph bit for bit. The verifier's cgroup peak was 216.5 MiB.

The JavaScript suite has 40 passes and one existing skip. The Python measurement
suite has 12 passes. The latest low-copy build completed with a 451.9 MiB cgroup
peak. These checks do not establish voice quality.

## Real inference evidence

The device-bound browser run is in
`artifacts/nano_lab/browser_measurements/local_lossless_device_bound_20261002/`.
It uses the selected generated reference, serialized CPU voice state, pinned
community graphs, lossless embeddings, seed 1337, and this sentence:

> Take a slow breath in, and let your shoulders relax.

Both first and warm reads generate 35 speech tokens and 1.52 seconds of audio.
Independent Whisper Small CPU int8 transcription returns `E, E, E.` for both.
Word error rate is 1.0. Successful WAV export is not a successful speech result.

| Measurement | First read | Warm read |
| --- | ---: | ---: |
| Synthesis seconds | 19.766 | 4.372 |
| Scheduled first-audio seconds | 20.143 | 4.438 |
| Real-time factor | 13.004 | 2.876 |
| Peak Chrome process-tree RSS, MiB | 1682.754 | 1608.922 |
| Peak process-tree PSS, MiB | 859.023 | 874.328 |
| Incremental PSS over clean baseline, MiB | 387.164 | 402.469 |
| Requested GPU buffers, MiB | 356.343 | 357.595 |

PSS divides shared memory among processes. It is not the strict RSS target.
Requested GPU buffers are a separate allocation counter, not GPU resident
memory. The Chrome baseline was 1196.305 MiB RSS and 471.859 MiB PSS. These
numbers do not demonstrate the under-500-MiB runtime goal. Cold load took
114.555 seconds. This run's cgroup peak was about 1.1 GiB.

The later diagnostic run is in
`artifacts/nano_lab/browser_measurements/local_gpu_trace_20261002/`. It verifies
Stop during a six-passage read after the first chunk, idle/ready state, an empty
playback queue, and a completed restart with WAV export. Stop acknowledgement
took 0.564 seconds. The whole guarded job peaked at about 1.3 GiB.

## CPU controls and rejected hypothesis

`browser_tts/scripts/probe-cpu-pipeline.py` loads the same public embedding and
language-model graphs and serialized state on CPU. It uses the worker's LCG
sampler. Decoder execution is optional and follows release of the other sessions.
The control produces 28 speech tokens and 1.24 seconds. Its transcript is
`epistext.`, with word error rate 1.0. Sampled process RSS peaks at 419.0 MiB;
cgroup peak is 614.6 MiB. This result does not pass the content gate.

CPU and GPU tokenizer IDs are identical. Their first ten raw logit IDs have the
same order. The first-vector maximum absolute difference is 0.10785 and RMSE is
0.02797. These are descriptive measurements, not a passed numerical gate.

Native Nano tokenization supplies 12 raw text IDs and one speech-start embedding.
The public conversion's tokenizer supplies two end-of-text IDs, which its hybrid
embedding graph maps to two speech-start embeddings. A CPU diagnostic removed
one duplicate embedding. It still failed: 26 speech tokens, transcript
`Epis Diffmer`, word error rate 1.0. That diagnostic is not applied to the reader.

The later public revision `19265a0f25b51c723b71e2098f766c95fbb94b29` has identical
graph, weight, tokenizer, and runner hashes to our pinned revision. Its additional
single-fixture quality report does not repair this local failure. No model pin
was changed.

## Native conditioning diagnosis

The original native Nano language model also fails with the short public voice
conditioning. It generates 19 speech tokens; the decoded transcript is `Ipoof.`,
with word error rate 1.0. The selected reference WAV itself has zero word errors.
Its 3.52-second duration is below the native API's supported reference minimum.

Removing one speech-start row, restoring exact prompt embeddings, using only
the speaker row, and using native conditioning from that same short reference
do not repair the short sentence. The unpenalized greedy diagnostic reaches the
256-token limit without an end token. These are failed controls.

The saved native ASMR conditioning provides a working CPU control. It has a
334-frame T3 prefix, including 333 prompt tokens. It uses the same public model
graphs, sampler, text and seed, with one speech-start row. The three decoder
conditioning tensors remain from the original public state. It generates 78
speech tokens and 3.24 seconds of audio. Independent Whisper Small returns the
complete requested sentence with word error rate 0.0. Sampled process peak RSS
is 477.1 MiB. This is a speaker-specific control; no fitted adapter is applied.
It does not prove perceptual realism or smooth long reading.

The separate candidate is `browser_tts/public/voice/native-asmr-donor/`.
`export-native-donor-state.py` re-derives the native prefix from hashed source
assets, verifies it byte for byte against the measured control, and preserves
the three public decoder tensors byte for byte. Candidate binary SHA-256 is
`77a2955a6b50122c581cf211f86d7c624381b7dfa8bb8ce61af3fb6359e16496`.
The loader requires that exact binary digest. The worker now selects this state
by default. The original generated-reference artifact remains available for
controlled failure reproduction. This changes the functional reader default;
it does not promote perceptual quality or numerical equivalence.

### Native-donor browser result

`artifacts/nano_lab/browser_measurements/local_native_donor_browser_20261002/`
contains the actual hardware run. First, warm and restarted WAVs all have zero
word errors in independent Whisper Small transcription. Each generates 83
speech tokens and 3.44 seconds of audio. Real Stop/restart passes; Stop
acknowledgement takes 1.136 seconds. The guarded job completes in 212.168 seconds
with a cgroup peak of about 1.5 GiB. The ready and completed UI were inspected.

| Measurement | First read | Warm read | Restart |
| --- | ---: | ---: | ---: |
| Synthesis seconds | 17.755 | 11.233 | 9.125 |
| Scheduled first-audio seconds | 18.113 | 11.309 | 9.204 |
| Real-time factor | 5.161 | 3.265 | 2.653 |
| Peak Chrome process-tree RSS, MiB | 1745.215 | 1683.344 | 1751.773 |
| Peak process-tree PSS, MiB | 924.516 | 944.718 | 1004.188 |
| Incremental PSS over clean baseline, MiB | 447.308 | 467.510 | 526.980 |
| Requested GPU buffers, MiB | 440.345 | 443.179 | 456.929 |

Cold model preparation takes 110.006 seconds. These results establish correct
words for one sentence and working controls. Generation is slower than
playback. They do not establish smooth long reading, perceptual success,
watermark parity, WebGPU numerical equivalence or the under-500-MiB target.

## Next check

Measure synthesis stages and compare the FP16 embedding session against the
lossless lookup with the same native-donor workload. Then check the full
Federalist reading. Preserve the selected reference, failed artifacts,
CPU-only acoustic ONNX gate, and memory limits.

## Matched embedding comparison

The default FP16 embedding session is measured in
`artifacts/nano_lab/browser_measurements/local_native_donor_fp16_20261002/`.
Its first and warm WAVs also pass the independent short word check. Both GPU
modes generate identical speech-token sequences for this text and seed.
The reusable native-donor CPU probe also passes the new WAV word check.

| Warm measurement | FP16 session | Lossless shards |
| --- | ---: | ---: |
| Synthesis seconds | 12.044 | 11.233 |
| Peak Chrome process-tree RSS, MiB | 1879.305 | 1683.344 |
| Peak PSS, MiB | 1020.809 | 944.718 |
| Incremental PSS, MiB | 545.138 | 467.510 |
| Requested GPU buffers, MiB | 526.471 | 443.179 |

This is one run per mode. Cold process and driver variation affect the result.
The first-read RSS is not lower in the lossless run, so these numbers do not
establish a universal RSS reduction. The warm saving is 195.961 MiB RSS.
The strict under-500-MiB target remains unmet.

Warm FP16 synthesis spends 8.193 seconds in autoregressive generation,
3.602 seconds in decoding, 0.217 seconds in prefill and 0.033 seconds in initial
conditioning. Autoregressive generation includes sampling and embedding calls.
These measurements identify the language-model loop as the largest stage.

The matched listening page is
`browser_tts/public/experiments/native-donor-listening/index.html`. The actual
desktop and 390-pixel mobile page were inspected. All four audio players load
and finish playback; there is no mobile horizontal overflow. This verifies
the listening artifact's controls, not perceptual quality.

## Full reading in progress

The current guarded full-paper job uses the native donor, lossless embeddings,
seed 1337 and 18-word passages. Its directory is
`artifacts/nano_lab/browser_measurements/local_native_donor_full_paper_20261002/`.
The latest saved `live-progress.json` records partial input coverage and
synthesis stages. It is not a completed reading or independent word audit.
An early 18-word passage takes 28.005 seconds to generate 7 seconds of audio.
It spends 15.383 seconds in autoregressive generation and 11.691 seconds in
decoding. Scheduled playback gaps exceed 25 seconds. Smooth playback already
fails on this hardware; completion and full-text transcription remain pending.

## GPU limitation on this Chrome build

The saved adapter report is from Chrome 152.0.7977.64. It records a real NVIDIA
Lovelace adapter without `shader-f16`, and an Intel gen-12lp adapter with it.
Adapter features are capabilities. Device features are only the features
requested and enabled on a created device; they are not interchangeable.

Chrome's [152 branch dependency record](https://chromium.googlesource.com/chromium/src.git/+/refs/tags/152.0.7977.42/chrome/chrome_branch_deps.json)
pins Dawn to branch 7977. That branch's [Vulkan device implementation](https://dawn.googlesource.com/dawn/+/refs/heads/chromium/7977/src/dawn/native/vulkan/PhysicalDeviceVk.cpp)
blocks NVIDIA `ShaderF16` because of unresolved conformance-test failures. The
local RTX 4060 Vulkan driver reports the basic float16 capabilities. That does
not override Chrome's feature policy. No internal override is recommended or
applied. The current FP16 WebGPU model cannot use this NVIDIA adapter safely.

The [WebGPU adapter options](https://gpuweb.github.io/types/interfaces/GPURequestAdapterOptions.html)
define power preference as a hint. The actual adapter must expose `shader-f16`,
and the requested device must enable it. The worker passes that exact device
to the [ONNX Runtime WebGPU provider](https://onnxruntime.ai/docs/api/js/interfaces/InferenceSession.WebGpuExecutionProviderOption.html).
A native GPU backend would require its own numerical and resource checks.
The acoustic CPU-only verification does not authorize CUDA.
