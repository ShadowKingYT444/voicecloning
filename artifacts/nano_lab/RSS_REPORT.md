# Latest prompt and acoustic attention profile

The CPU ONNX end-to-end run completes with the new folded acoustic adapter and pinned T3 prompt donor. It uses **861.75 MiB sampled peak process-tree RSS**. Cold pipeline time is **118.07 seconds for 6.96 seconds of audio**, or 16.96 times playback duration. The service takes 118.66 seconds and reports about 1 GiB cgroup peak with no swap. Process RSS and cgroup memory account for different memory categories.

This is a measured low-memory path, but it is slow. Independent word auditing is pending because desktop memory fell below the smaller job's launch threshold. The new CUDA acoustic stage is blocked after strict numerical verification failed. The older CUDA timings below do not describe these new weights.

Evidence: [run report](onnx_asmr_acoustic_attention_cpu/run.json), [generated WAV](onnx_asmr_acoustic_attention_cpu.wav), [service log](onnx_asmr_acoustic_attention_cpu.log). Sampler differences mean the native and ONNX outputs are not identical workloads even with equal text and seed.

The native six-case prompt/attention sweep peaked at 2496.49 MiB process RSS. A standalone native CLI check was refused before loading because system available memory was below 6144 MiB. No memory cap or reserve was changed.

## Earlier acoustic-decoder profile measurements

RSS is resident host memory. GPU memory is separate. All model jobs ran serially under the existing cgroup guard, with zero job swap and a 4 GiB desktop-headroom stop.

| Workload | Peak RSS | Scope | Measured time |
|---|---:|---|---|
| New standard decoder profile, staged CUDA ONNX, question seed 79 | 1045.57 MiB | Sampled process tree | 50.87 s cold pipeline time for 5.48 s audio |
| New slower decoder profile, native CUDA, question seed 79 | 2392.39 MiB | Process high-water mark | 35.28 s guarded service time for 6.32 s audio |
| Decoder embedding fit, 15 train / 2 validation clips | 1712.12 MiB | Process high-water mark | 299.19 s guarded service time |

The ONNX launch had a 1280 MiB hard memory/RSS cap. Native generation and fitting used the existing 3072 MiB cap. No cap was raised. The staged path prioritizes low host memory and is not real-time at cold startup. The native profile timer reports 12.07 seconds after imports; the 35.28-second service measurement includes those imports. Fit-body time is 293.90 seconds, also excluding earlier imports.

The native slower output is byte-identical to its comparison master (`native_decoder_slow_parity.json`); its underlying raw comparison clip passed Small ASR. The new standard ONNX WAV itself passes Small ASR with zero normalized word error (`onnx_decoder_fit_small_audit.json`). Cache hashes are required for these learned decoder profiles. Missing or changed caches fail before model loading.

Evidence: `onnx_asmr_decoder_fitted/run.json`, `native_asmr_decoder_slow_question_79.wav.json`, `decoder_embedding_fit/fit_report.json`, and their guard logs. These are separate cold runs, not a matched benchmark of the embedding fit's overhead. The previous faster persistent-mode measurements below use earlier profiles and remain separate.

Listen: [latest native source comparison](decoder_cadence_combined/index.html), [new ONNX sample](onnx_asmr_decoder_fitted.wav). Voice realism remains unaccepted. The experimental long-context ONNX limitation described below is unchanged.

# Earlier fitted text-adapter measurements

These results use the fitted ASMR and Harvey adapters with CUDA ONNX Runtime in full precision (`use_tf32=0`). ASMR also applies its fitted mel correction. All jobs ran serially, with no job swap use and a 4 GiB system-headroom stop. RSS is resident host memory; GPU memory is separate.

| Mode and workload | Peak sampled process-tree RSS | Timing |
|---|---:|---|
| Staged fitted ASMR, question, seed 79 | 1004.27 MiB | 42.42 s cold for 5.48 s audio |
| Persistent fitted ASMR, three requests | 1936.96 MiB | 49.14 s whole batch; warm 3.24 s / 5.48 s audio and 3.87 s / 6.52 s audio |
| Persistent fitted Harvey, three requests | 1939.57 MiB | 46.84 s whole batch; warm 3.16 s / 6.54 s audio and 2.71 s / 4.10 s audio |

Warm request timing includes token generation, acoustic decoding, intermediate file I/O, Perth watermarking, mastering, and WAV writing. It excludes initial reference preparation and process/session startup. Cold worker requests took 27.81 s for ASMR and 25.87 s for Harvey. These are short batches, not sustained server latency measurements. The worker exits after the batch.

The staged launcher enforces a 1280 MiB cap. It uses less memory but reloads components. The persistent benchmark uses the existing 3072 MiB cap. Its warm requests were faster than playback in these four tests. Each ORT CUDA session has its own 2048 MiB arena limit; this is not a combined GPU-memory limit.

Before caching verified graph hashes, the same ASMR warm requests took 5.58 and 6.21 s; Harvey took 4.55 and 4.27 s. After caching, speech tokens and mel features are exactly equal in all six requests. Final waveform maximum absolute differences are at most 3.58e-6. The cache stores only file hashes, is bounded to 64 entries, and invalidates on file identity, size, modification-time, or change-time updates. It does not retain model weights.

Evidence: `onnx_asmr_fitted_staged_fp32/run.json`, `onnx_batch_asmr_fitted_cached/{batch_report,worker_report,cache_parity}.json`, and `onnx_batch_harvey_fitted_cached/{batch_report,worker_report,cache_parity}.json`. Process trees were sampled every 50 ms for batches and 100 ms for the staged run. Small ASR matched all six pre-cache FP32 generated files. Those audited WAVs are the [ONNX listening samples](fitted_onnx_samples/index.html).

The ASMR and Harvey merged T3 graphs passed L32/L128 prefill and one-token decode checks against adapted Torch on CPU and CUDA at the preset 3e-4 absolute/relative gate. CUDA maximum absolute differences were 4.01e-5 and 4.20e-5. The first CUDA pass failed with default TF32; disabling it fixed those short checks. The separate base context-400 cache mismatch remains unresolved. `--experimental-t3` remains required. See [ORT precision documentation](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#use_tf32).

The older base-voice CUDA timings below used default TF32. They are historical measurements and do not describe the current full-precision runtime. Native fitted measurements below remain valid. ONNX and Torch use different random samplers, so equal seeds are not identical generated workloads.

## Earlier measurements

# Host RAM and inference measurements

Measured on the RTX 4060 laptop GPU and i7-12650H. The desktop remained active. Each new benchmark ran alone under a 3 GiB cgroup cap, no job swap, two CPU cores, and reduced scheduling priority. These are local measurements, not cross-machine claims.

| Profile / precision | Loaded RSS (MiB) | Final RSS (MiB) | Observed peak RSS (MiB) | First RTF | Warm mean RTF | GPU resident (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| asmr soft fp32 | 1449 | 2426 | 2426 | 1.006 | 0.483 | 1878 |
| asmr soft fp16 | 1421 | 2621 | 2621 | 1.143 | 0.550 | 1620 |
| harvey fp32 | 1452 | 2375 | 2375 | 0.894 | 0.409 | 1877 |

RTF is synthesis time divided by output duration. Values below 1 are faster than playback. Each profile used one fixed 24-word sentence, seed 31, acoustic seed 10031, one first request, and two warm repeats. The acoustic stage stays fp32 in every successful test. The fp16 row reduces T3 precision only. It changed two speech tokens, increased host RSS, and was slower, so fp32 remains the default.

Peak RSS includes loading and the first request. The reported peak is the maximum of observed RSS snapshots and the process high-water mark. GPU memory is separate from host RSS. The cgroup memory peak also includes charged cache and is not interchangeable with RSS.

The first requests include kernel initialization. Python/framework import time is included in service runtime but excluded from model-load timing. See the raw JSON and service logs for both values.

Raw reports:
- [benchmark_asmr_soft_fp32/report.json](benchmark_asmr_soft_fp32/report.json)
- [benchmark_asmr_soft_fp16/report.json](benchmark_asmr_soft_fp16/report.json)
- [benchmark_harvey_fp32/report.json](benchmark_harvey_fp32/report.json)

## Earlier matched CPU check

Before the desktop-memory constraint was added, stock and optimized fp32 CPU runs with the built-in cached voice produced the same WAV SHA-256. Sampled peak RSS fell from 5359.85 MiB to 3238.77 MiB, about 39.6%. Those runs had concurrent background work, so their latency is exploratory. They used a different reference from the new target-voice tests. Do not combine them into a target-voice speedup claim.

## Current interpretation

Cached reference features, omission of reference encoders, and direct GPU tensor loading are deployed in the candidate CLI. Full fp32 is the best measured precision choice here. A complete ONNX voice runtime has not been validated. These measurements do not establish the theoretical minimum RSS.

Stage profiling attributes roughly 490 MiB of first-use growth to token generation, 390 MiB to acoustic decoding, and 60 MiB to watermarking. Returning free glibc heap pages saved about 40 MiB and was not adopted as a major optimization.

## User-facing command checks

The actual `nano-clone` launcher produced byte-identical delivery WAVs to the
selected sweep outputs. These checks include mastering, file writing, and the
new 700-token generation guard.

| Command profile | Audio length | Generation + mastering | Peak RSS | Full service time |
|---|---:|---:|---:|---:|
| ASMR soft | 8.36 s | 7.45 s | 2436.8 MiB | 43.37 s |
| Harvey | 5.54 s | 5.94 s | 2386.0 MiB | 44.36 s |

The service time includes Python/framework startup. These are separate first
requests, not warm throughput measurements. Details are in `delivery/*.wav.json`
and `clone_soft_test.log` / `clone_harvey_test.log`.

The memory guard stopped the first staged T3 export when available RAM fell
below 4 GiB. Removing an unnecessary full graph copy allowed the retry to finish
with 2.4 GiB cgroup peak. This is export memory, not inference RSS. Further tests
must wait when available RAM is below the 6 GiB start threshold.

## Lightweight verification budget

Lightweight component checks can use a stricter 1280 MiB hard memory/RSS cap,
1024 MiB high threshold, and no swap. They require 5376 MiB at launch, which
reserves the complete job budget plus 4 GiB for the desktop. The runtime stop
below 4 GiB available remains in force. Synthesis and training limits are unchanged.

The initial shared T3 ONNX parity check completed with 977.8 MiB cgroup peak.
Its maximum absolute error was 1.72e-5 across one prefill and one decode example.
This is a component verification result, not complete audio inference RSS.
The spectral helper tests used 499.9 MiB cgroup peak; pure ORT spectral checks
used 53.3 MiB cgroup peak. Extended T3 checks and longer, voice-derived vocoder checks remain pending.

## Subsequent code changes awaiting full voice remeasurement

A component-only import benchmark measured vocoder module import at 846.0 MiB
RSS before lazy package imports and 547.8 MiB after. Import time fell from
29.61 to 9.37 seconds in these separate local runs. Public API identity checks
pass. This is an import benchmark, not complete synthesis RSS.

Identical read-only causal masks now share storage across the twelve GPT2
layers. A regression test verifies the values and shared storage. The earlier
full voice benchmarks predate this change and the lazy imports; complete voice
RSS and output hashes must be rechecked before claiming a new improvement.

The vocoder ONNX check passed with maximum absolute waveform error 4.14e-5
against a saved native Torch reference on the test input. Its cgroup peak was
400 MiB. Flow encoder export exceeded the 1280 MiB RSS limit both with and
without export-time constant folding; it remains unavailable. Extended T3
reference generation was stopped when desktop headroom fell below 4 GiB.
These stopped jobs are failures, not verified results.

Tiny component modes now permit stricter 640, 768, or 1024 MiB caps. Each
reserves its full budget plus 4096 MiB available at launch. All modes keep the
4 GiB runtime stop, no swap, and one-job lock. Guard tests verify these caps and
launch thresholds. At about 4.4 GiB available, even the 640 MiB export smoke
test was refused before any model process started.

## Full voice revalidation after memory changes

Both selected short CLI outputs are byte-identical to their previous delivery
WAVs after lazy package imports, shared causal-mask buffers, and atomic output
publication. These fresh first-request checks use the same text and seeds.

| Profile | Earlier reported peak RSS | New reported peak RSS | New max observed RSS | CUDA peak before / after |
|---|---:|---:|---:|---:|
| ASMR soft | 2436.8 MiB | 2125.1 MiB | 2126.4 MiB | 2376.3 / 1664.6 MiB |
| Harvey | 2386.0 MiB | 2134.6 MiB | 2136.2 MiB | 2376.3 / 1664.6 MiB |

Reported peak is the process resource high-water mark. Max observed also
includes the sampled final/ready RSS, which is slightly higher on this system.
Use the conservative observed value for capacity planning. RSS fell about
10–13 percent in these local checks; CUDA peak allocation fell about 712 MiB.
First-request generation plus mastering took 8.66 s for 8.36 s soft audio and
6.15 s for 5.54 s Harvey audio. These runs do not prove a latency improvement.
Full service startup took 55.52 s and 42.97 s, respectively.

Raw logs, sidecars, and hash comparison are in `revalidation/`. The tested
runtime SHA-256 is recorded in `revalidation/comparison.json`. Longer input
and new fitted-voice generalization remain separate validation gates.

## Additional ONNX components and CPU verification environment

External-parameter export avoids serializing a second copy of every parameter
in the Torch graph. Flow encoder export now completes within the 1280 MiB
cap. Independent ORT checks at 12, 120, and 600 tokens pass with 2.623e-6 maximum
absolute error and 709.7 MiB cgroup peak. The meanflow estimator passes at mel
lengths 8, 64, and 256 with 4.244e-5 maximum error and 610.9 MiB cgroup peak.
These peaks describe component checks, not a complete voice runtime.

A separate `.venv-nano-cpu` uses PyTorch and torchaudio 2.11.0 CPU wheels from
[PyTorch's CPU package index](https://download.pytorch.org/whl/cpu/torch/).
Other Python dependencies are reused through a read-only path. The original
GPU environment is unchanged. Under the same 1280 MiB cap, upstream T3 reference
generation now completes at 1081.1 MiB process peak RSS; the CUDA-linked build
exceeded that cap even when the requested operation used CPU.

The longer ORT T3 check remains a strict numerical failure: in the 400-position
prefill, 176 of 7,379,363 output values exceed atol/rtol 3e-4. Maximum absolute
cache error is 0.000989 and maximum relative L2 error is 1.436e-5. Logits pass
(max error 1.19e-6 in that case). Shorter cases and two cache-growth steps pass.
Disabling graph optimization did not resolve the failure. No tolerance was
relaxed and no complete ONNX deployment is claimed.

## Long-input CLI check

The original 55-word chunk policy hit the token limit on an 85-word passage.
The revised policy groups up to 32 words at sentence boundaries and splits
a length-limited chunk further, with a terminal failure at 12 words or fewer.
The actual long request now produces 29.14 seconds across
4 segments. The observed peak RSS is 2473.3 MiB,
higher than the short-sample measurements. Generation plus mastering totals
13.60 seconds; this excludes model startup.
No adaptive retry was needed in that completed request.

Independent Whisper-small auditing of all four actual waveform segments found
correct content, with one retained scoring difference: “ten” versus “10.”
Sidecar text and frame counts verify the complete text order and 0.1-second
segment joins. A separate test with a deliberately low 180-token limit
exercised one split retry and published a complete two-segment output.
Retry time is recorded separately and included in total request time.

## Experimental complete staged ONNX run

The CPU-only staged pipeline now produces complete audio while loading only one
model component at a time. The same 24-word morning passage produced 7.36 seconds
of audio in 81.27 seconds including all process starts and watermark/mastering.
The total real-time factor is 11.04, so this CPU implementation is not smooth
real-time synthesis. It is an experimental memory option, not the default.

| Stage | Peak process RSS (MiB) | Stage time (s) |
|---|---:|---:|
| Conditioning preparation | 224.9 | 2.89 |
| Text and T3 tokens | 833.2 | 25.36 |
| Flow encoder | 459.7 | 4.97 |
| Meanflow estimator | 591.7 | 23.91 |
| Vocoder | 360.0 | 11.97 |
| Watermark and mastering | 511.5 | 7.95 |

These are child-process RSS peaks, not summed process-tree peaks. The small
parent remains resident. The full process tree stayed within the enforced
1280 MiB summed-RSS and cgroup cap, with no swap. systemd rounded its cgroup
peak to 1G. Cgroup memory also includes charged file cache and is not RSS.
A prior acoustic-only run using known PyTorch speech tokens peaked at 1022.7 MiB
cgroup memory and took 64.22 seconds for 8.36 seconds of audio.

Artifacts: onnx_pipeline/soft_complete/run.json and soft_acoustic/run.json.
Perth remains enabled. NumPy sampling and acoustic random numbers differ from
Torch, so these WAVs are not expected to match the GPU WAV hash. Independent
Tiny and Small ASR checks both report zero word errors on these two files.
This does not establish voice realism. The strict long-context T3 cache mismatch
remains unresolved; --experimental-t3 is mandatory.

Harvey's complete staged run also passed after preserving the upstream handling
of an odd reference mel length (355 frames versus 177 speech tokens). It produced
5.38 seconds of audio in 67.71 seconds. The sampled peak summed
process-tree RSS was 807.9 MiB, sampled every 100 ms.
The 1280 MiB cap remained active; this CPU run is not real-time. An independent
regression test covers the odd reference length. Tiny ASR reports zero word
errors on the Harvey output. Independent Small audit remains pending.

## Repaired-data experiment

Single guarded jobs only. Feature preparation peaked at 2.4 GiB cgroup memory. The 256-parameter speaker fit also peaked at 2.4 GiB cgroup memory. The rank-4 adapter fit reported 2,015.3 MiB process peak RSS and 1.5 GiB cgroup memory. These accounting methods differ because shared pages can contribute to RSS without being charged to this cgroup.

The 15-case synthesis comparison peaked at 2,422.3 MiB process RSS. Its CPU evaluation ran under the 1,280 MiB cap and peaked at 805 MiB cgroup memory. Source-transcript auditing peaked at 946.9 MiB cgroup memory under the same cap. No job used swap. These are experiment costs, not per-request runtime measurements.

## Experimental CUDA ONNX pipeline

The first CUDA attempt could not locate shared native libraries. The isolated ORT environment now links the existing NVIDIA packages. A second attempt reached the explicit 2 GiB CUDA arena limit while growing the token cache. Disabling shape-specific memory patterns and releasing unused CUDA arena allocations after each run fixed that failure without raising either GPU or host limits.

The full profiled CUDA run completed in 60.17 seconds for 7.36 seconds of audio, with 1186.5 MiB sampled process-tree peak RSS and 1.7 GiB cgroup memory. All four model stages recorded CUDA kernels. Speech tokens exactly match the earlier CPU run. The CPU run took 81.27 seconds on the same text and reference, but used a different ORT version and had no profiling. This is an experimental comparison, not a controlled speedup claim. CUDA remains too slow for deployment in this staged form.

ORT allocator behavior follows the [official memory arena documentation](https://onnxruntime.ai/docs/get-started/with-c.html) and [run-option definition](https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/session/onnxruntime_run_options_config_keys.h).

## CUDA cache held on the GPU

The matched unprofiled CUDA runs used the same voice, text, seed, model exports, provider settings, and 7.36-second output duration. Keeping the growing token cache in CUDA memory reduced total cold request time from 44.95 to 37.76 seconds (16.0%). The token stage fell from 16.86 to 10.17 seconds. Sampled process-tree peak RSS fell from 1044.5 to 1023.1 MiB. The latter run used 901.7 MiB peak cgroup memory, with no swap. The token loop itself took 3.02 seconds; process startup and session creation still add substantial latency.

Speech-token arrays and generated mel arrays were exactly equal. Vocoder float output differed by at most 8.20e-8; this is not byte-identical WAV parity. The final real-time factor is 5.13. This experimental staged path remains too slow for smooth real-time use. The unresolved long-context T3 cache tolerance gate still applies.

Evidence: `onnx_pipeline/soft_cuda_unprofiled/run.json` and `onnx_pipeline/soft_cuda_resident/run.json`. RSS was sampled every 100 ms, so a shorter transient can be missed. This comparison measures one passage, not a latency distribution.

## Persistent ORT batch prototype

Three requests shared four CUDA ORT sessions in one worker. The first model-generation pass took 21.06 seconds, including 13.01 seconds of session creation. The same passage repeated warm in 4.21 seconds for 7.36 seconds of audio. A different passage took 4.13 seconds warm. The peak sampled process-tree RSS was 1504.79 MiB; the hard limit remained 3072 MiB and no swap was used.

These warm times exclude separate conditioning preparation and Perth/output children. The complete three-request batch took 64.96 seconds, including 10.19 seconds of preparation and 24.12 seconds of finishing. It is not yet a measured real-time complete-request engine. An integrated finishing experiment is next. Tokens and mels for both repeated passages exactly matched the fresh-process baseline. Float vocoder differences were below 8.20e-8.

Evidence: `onnx_batch_cuda/batch_report.json`, `onnx_batch_cuda/worker_report.json`, and `onnx_batch_cuda/parity.json`. All sessions were released when the batch worker exited.

## Complete warm requests with integrated watermarking

The next batch kept one CPU Perth watermarker in the same worker as the four CUDA ORT sessions. There was still only one model process. This complete worker request includes tokens, acoustic decoding, watermarking, mastering, intermediate artifact I/O, and final WAV publication.

| Case | Audio duration | Complete worker request | Real-time factor |
|---|---:|---:|---:|
| Cold morning passage | 7.36 s | 30.161 s | 4.098 |
| Same passage, warm | 7.36 s | 4.323 s | 0.587 |
| New passage, warm | 6.96 s | 3.844 s | 0.552 |

Peak sampled simultaneous process-tree RSS was **1942.91 MiB**. Worker peak RSS was 1811.75 MiB. The three-request batch took 52.40 seconds including 10.86 seconds of separate preparation and all startup. Systemd reported 1.4 GiB cgroup memory and no swap. Shared-memory charging makes cgroup memory different from summed RSS. These are three short requests, not a sustained throughput or tail-latency benchmark. Cold startup is not real-time.

All speech-token and mel arrays exactly matched the preceding ORT-only batch. Final PCM WAV differences were at most 3.58e-7 after integrating Perth under inference mode. The original batch passed Small ASR on all three outputs; the integrated outputs also passed Small ASR on all three complete transcripts.

Evidence: `onnx_batch_cuda_finish/batch_report.json` and `onnx_batch_cuda_finish/parity.json`. The unresolved long-context T3 tolerance check keeps this runtime experimental. Fitted adapter profiles currently use the native CLI; the base ONNX path rejects them instead of silently ignoring their adapter.

The native experimental ASMR profile was also tested: 2373.29 MiB peak RSS, and a byte-identical WAV match to the previously audited combined-adapter sweep case. Evidence: `native_fitted_parity.json` and `native_asmr_fitted_question_79.wav.json`.

The Harvey native fitted CLI also matched its previously audited sweep WAV byte for byte. It peaked at 2400.19 MiB process RSS and took 27.01 seconds guarded command wall time. `native_harvey_fitted_parity.json` records the exact match.

## Latest acoustic experiments

The twelve-case native decoder reference comparison completed at 2395.42 MiB maximum sampled process RSS, with a 3072 MiB hard cgroup cap and zero job swap. The source-token diagnostic previously peaked at 2665.20 MiB process RSS. Independent Whisper-small content checking peaked at 858.80 MiB process RSS, under a 1280 MiB cap. Praat diagnostics peaked at 127.86 MiB process RSS under a 640 MiB cap. These are experiment costs, not the optimized delivery runtime.

The fitted staged ONNX run's peak was confirmed to occur in the vocoder stage: 1004.27 MiB sampled process-tree RSS. Preparation peaked at 266.87 MiB. Stage counters reset for each fresh process. Avoiding Torch in preparation would not, by itself, lower this measured end-to-end maximum. No speculative memory reduction is claimed.

During this pass, the launch guard refused work while available system memory was below its threshold. Jobs resumed after headroom recovered. No unrelated desktop process was stopped and no system cache was flushed. Host swap was already full; every lab cgroup still had zero permitted swap.

## Latest decoder experiments

Harvey decoder embedding fit: peak process RSS 1699.92 MiB; guarded service 82.212 seconds, no job swap. Three-passage paired generation: maximum sampled process RSS 2373.13 MiB; guarded service 51.603 seconds for six clips. This is batch experimental generation, not a cold or warm production latency benchmark. ASMR T3 reference matrix: maximum sampled process RSS 2492.96 MiB for nine clips; service 60.342 seconds, no job swap. All use the existing hard 3072 MiB limit and 4 GiB desktop reserve. The projection adapter has no measured runtime result yet.

The decoder projection one-step test peaked at 1729.10 MiB process RSS. Its later 40-step run was stopped after 13 completed steps by the desktop reserve guard, rather than by its 3072 MiB job cap. The service recorded a 1.1 GiB cgroup peak and no swap. Cgroup memory and process RSS are different measures. A subsequent 640 MiB inspection job was refused at launch because available system memory was below 4736 MiB. No limits were raised and no unrelated processes were stopped.

## Partial decoder projection adapter comparison

The interrupted fit completed 13 steps and preserved its best checkpoint at step 11. The guard stopped the fit when desktop headroom fell below 4 GiB. No memory limit was increased. After headroom recovered, a serial twelve-sample native comparison completed successfully. It peaked at **2310.36 MiB process RSS** and took **77.487 seconds guarded service time**. The cgroup reported 2.4 GiB peak including its other accounted memory and zero swap. These accounting scopes are different.

Independent Small ASR took 82.535 seconds and reported a 1.0 GiB cgroup peak. Speaker and DNSMOS evaluation took 53.090 seconds and reported a 716.1 MiB cgroup peak. These are evaluation costs, not synthesis costs. The adapter folds 672 learned values into an existing decoder projection weight. The generated quality measurements are too small to justify promoting it. ONNX performance figures above refer to the earlier decoder profile, not this new projection experiment.

## Wider speaker-vector fit and comparison

The 60-step, 17-clip fit used **1710.74 MiB peak process RSS** and took **440.676 seconds guarded service time** (434.982 seconds inside the fitter). Eight native generated samples used **2393.07 MiB peak process RSS** and took **52.279 seconds service time**. Both cgroups reported 2.4 GiB peak and zero swap.

Independent Small ASR took 56.050 seconds with a 1.0 GiB cgroup peak. Matched speaker/DNSMOS evaluation took 37.417 seconds with a 685.2 MiB cgroup peak. These are separate evaluation costs. The new speaker vector has no extra inference modules, but these runs do not establish an inference-speed improvement. The ONNX measurements above still use the previous fitted vector.

The new acoustic-only tokenizer runs separately from the flow decoder. The successful 30-clip extraction took 25.220 seconds, with a 978.6 MiB cgroup peak and zero swap. All resource limits remain unchanged.

The 30-row flow preparation used 1483.16 MiB peak process RSS and took 52.291 seconds service time. The strict fit-consumer parity check used 1482.60 MiB and took 35.955 seconds. These separate stages passed under the unchanged guard with zero swap.

## Attention adapter smoke

The strict-FP32 one-step attention fit used 1825.25 MiB peak process RSS and took 85.348 seconds service time. Its 224 low-rank factor pairs contain 344,064 learned values. Full fit and generated-speech runtime measurements remain pending. The folding path retains no dense base-weight or factor copies after application.

The isolated listening-page browser check used a separate 640 MiB guard. Its successful metadata/render run reported a 224.4 MiB cgroup peak and finished in 4.797 seconds. This browser was closed after the check.


Acoustic attention fit: 30/30 steps complete, best30. Held-out reconstruction total .78627586 -> .66581574 (15.32% lower), rank2 adapters on224 decoder attention projections,344064 trained values. Peak process RSS1830.0MiB; service643.211s,cgroup1.4G/no swap. StrictFP32 factorized-versus-folded model check passes unchanged1e-4 max/1e-5 RMS gate on3 actual source-token flow reconstructions; observedmax6.20e-6,repeatfold+restoreexact. This is reconstruction evidence, not new-text perceptual acceptance. Fourteen matched synthesis cases running.


Completed attention synthesis14cases peakprocess2495.461MiB (corrects earlier provisional2488.82), service74.586s/cgroup2.4G/no swap. Smallcontentaudit88.222s/cgroup1G; identity+DNS56.239s/cgroup542.6M. No parallelmodeljobs. These nativeexperimentsareseparatefromearlierONNXprofileRSSfigures.


Prompt+attention6casegeneration peakprocess2496.488MiB, service49.089s/cgroup1.9G/no swap. ASRSmall40.826s/cgroup1G. Speaker+DNS34.602s/cgroup490.3M under1280cap. Alljobsserial. Firstattemptfailedinputvalidation(noaudio): donorhelper250tokenlimitconflictedwithbothloaders375override; code+boundarytestsupdated, v2passed.


T3MLPfit peak2009.145MiB/service28.898s/cgroup1.6G. Nineclipnativegeneration peak2405.605MiB/service51.118s/cgroup2.1G. SmallASR61.870s/cgroup1G. Speaker+DNS43.801s/cgroup551.8M under1280. Allnoswap/serial.
