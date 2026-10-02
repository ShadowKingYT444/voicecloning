# CPU voice experiments — 2026-09-30

36 synthesis trials completed. The realism target is not achieved and no voice is promoted. Human listening is pending.

## Reference comparison

Means across two fixed seeds, one new passage per voice. Same text, sampling and acoustic seeds within each voice.

| Reference | Speaker cosine | DNSMOS overall | Exact word audits |
|---|---:|---:|---:|
| asmr_raw | 0.8401 | 1.969 | 2/2 |
| asmr_natural | 0.8650 | 2.696 | 2/2 |
| asmr_clean | 0.8195 | 3.014 | 2/2 |
| harvey_raw | 0.6809 | 3.318 | 1/2 |
| harvey_natural | 0.6944 | 3.130 | 1/2 |
| harvey_clean | 0.6551 | 3.267 | 1/2 |

The natural reference has the highest mean identity proxy in both voices. The gated ASMR reference improves the cleanliness proxy while reducing identity. Neither score establishes perceptual realism.

## Strict ASMR T3 fit

Twelve training clips and two validation clips retain their original audited audio/text hashes and protected source intervals. Rank-4 attention LoRA trains 221,184 values. Eight epochs / 96 optimizer steps complete with base-logit KL coefficient 1.0 and patience 2. Epoch 6 is best: validation token loss 4.909708 → 4.376112 (10.87% lower).

Three new-text passages × two fixed seeds compare fresh base and fitted models. Acoustic weights, natural reference, sampling and acoustic seeds remain fixed. The old fitted acoustic comparator is absent and was not used.

| Condition | Speaker cosine | DNSMOS overall | Exact word audits |
|---|---:|---:|---:|
| base | 0.8587 | 2.771 | 6/6 |
| fit | 0.8394 | 2.929 | 6/6 |

**Decision: do not promote this adapter.** Identity cosine drops in five of six paired cases despite lower validation loss and higher mean DNSMOS. It remains archived for review. F0 rises in five cases and durations change; breathy/whispered pitch estimates are unreliable and do not prove improved prosody.

## Harvey sampling experiment

Three passages × two fixed seeds compare temperature 0.8 and 0.6 with the same natural reference and decoder.

| Temperature | Speaker cosine | DNSMOS overall | Content-exact audits¹ |
|---|---:|---:|---:|
| 0.8 | 0.7180 | 3.212 | 5/6 |
| 0.6 | 0.7081 | 3.259 | 6/6 |

¹ Raw ASR/WER reports are unchanged. Whisper writes “9” for expected “nine” in all four meeting cases; a separate, explicit number-equivalence annotation excludes this formatting mismatch. Temperature 0.8 still changes “with preparation” to “at preparation” in one case. At 0.6 that flagged mismatch disappears. This is six short development cases, not a general word-error guarantee.

**Decision: retain 0.6 as an experimental content candidate.** Mean cleanliness rises slightly, mean speaker cosine falls slightly, and listening remains necessary. No Harvey profile is promoted.

## Runtime and verification

All model jobs run serially on two CPU cores with no swap, the unchanged 3072 MiB hard budget, 2500 MiB RSS stop and 4096 MiB desktop reserve. Peak RSS across successful synthesis/training jobs is 2056.24 MiB (2.008 GiB). This is not a 500 MB pipeline. Per-trial latency is in SUMMARY.json; it excludes model loading. RSS is a process high-water mark shared across cases, not an independent per-case memory measurement.

The streamed reader matches all 2,662 tensors in the three pinned checkpoints exactly. Deferred shared causal masks pass full and cached GPT-2 numerical parity. Four matched ASMR/Harvey controls match the earlier generated WAV hashes exactly. Four live guard tests pass; a 640 MiB job also exits 125 at its 512 MiB RSS threshold. Fourteen focused regressions plus three subtests pass.

Initial mmap/pread opens fail under the address-space cap. A full reference-plus-synthesis load then fails during reference encoding. Tensor streaming and separate native reference encoding reduce the working set without increasing limits. Early watchdog PID-based RSS readings are invalid; only in_process_getrusage reports support memory claims. Combined speaker/DNSMOS scoring exceeds the small address-space budget; separate serial evaluators succeed. PyAV 19 rejects the Whisper API call; pinned 16.1 completes the audits. A first prosody manifest has wrong relative reference paths; the corrected audit completes all 30 original rows with zero errors. These failures are retained.

## Listening and continuation

Open listening/index.html after extracting the delivery archive. All 42 listening files (36 trials and six references) use constant gain to a shared -27 LUFS target, preserve dynamics, and pass a 4× oversampled -1 dBTP ceiling. There is no delivery denoiser, compressor or noise gate. Raw watermarked generations are also included. Proxy scores use raw generations; matched copies are for listening only.

Speaker cosine uses the existing development held-outs (ASMR 02/03, Harvey 01); protected final-audit intervals remain unused for selection. DNSMOS repeats short clips to its 9.01-second window and ASMR is outside its usual domain. The passages, seeds and source recordings are small samples. No claim of general zero-shot improvement or ElevenLabs-level quality is justified.

The next useful quality work is acoustic/prosody comparison against the retained natural baseline, with matched new-text words and listening judgments. Repeating this T3 fit or promoting the highest DNSMOS clip would ignore the identity regression. Historical fitted acoustic/ONNX artifacts must be restored before using their profiles as comparators.

The initial 2026-09-30 publication attempt failed with HTTP 403, so the source changes were archived in a patch. Those source changes and this report are now included in the repository update. VM absolute paths in archival manifests must be rebased for a different checkout while preserving the recorded hashes.
