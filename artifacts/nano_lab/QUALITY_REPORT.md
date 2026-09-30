# Current voice quality status

The user rejected the first six samples: “Neither sounds convincing yet.” The requested realistic cloning quality is not achieved or verified. No new fitted profile has been promoted. Generated samples are synthetic.

## Current prompt and attention candidates

[Compare the current source and samples](delivery/index.html). The 30-step decoder attention fit reduces held-out reconstruction loss by 15.32%. On two new passages with the prompt donor and mel correction held constant, adding attention fitting increases mean speaker cosine from 0.88641 to 0.89360 but reduces DNSMOS from 3.1664 to 3.1094. All six comparison outputs pass the independent Small word check. These mixed proxy results do not establish realism.

The same source phrase lasts 10 seconds. Both generated candidates last 7.4 seconds. Median voiced pitch is about 197 Hz versus 259 Hz in the source; breathy speech makes this estimate uncertain. The last-four-layer MLP experiment is not selected because new-text similarity falls and three samples have word errors.

The folded acoustic ONNX estimator passes CPU checks at five lengths. CUDA fails four lengths at the unchanged tolerance. Its GPU inference remains blocked. The CPU end-to-end sample completes; independent word auditing is pending due desktop headroom. This sample is an implementation check, not a quality improvement claim.

## Earlier fitted decoder and cadence results

A new constrained fit changes only the 192-value acoustic speaker embedding. All flow weights stay frozen. The implementation first reproduces all 17 native cached reconstructions exactly. Native meanflow draws target noise before full-sequence noise; the fit helper now reproduces both draws and the target-suffix replacement. Fifteen clips train the embedding and two separate clips select the checkpoint. Validation objective falls 24.3%, from 1.18127 to 0.89404, with cosine constrained to at least 0.98 of the initial embedding. This objective is not a perceptual score.

The twelve-clip decoder/spectral comparison has identical tokens across variants and Small ASR exact on all clips. In that three-passage set, mean similarity changes from 0.8686 to 0.8895 with the decoder fit and existing spectral correction; DNSMOS changes from 3.1773 to 3.1671.

A separate matched set contains the source excerpt text plus question and narrative passages. It tests the decoder fit together with a lower repetition penalty.

| Variant | Mean similarity | Mean DNSMOS | Small exact |
|---|---:|---:|---:|
| Previous fitted candidate, penalty 1.2 | 0.8785 | 2.9171 | 3/3 |
| Previous candidate, penalty 1.0 | 0.8912 | 2.9782 | 3/3 |
| New decoder fit, penalty 1.2 | 0.8959 | 2.9961 | 3/3 |
| New decoder fit, penalty 1.0 | 0.9025 | 2.8012 | 3/3 |

All scores use -27 LUFS copies and the same development reference subset. Penalty 1.1 was rejected after a word error. The slower fitted variant trades additional identity similarity for lower cleanliness scores. The new standard and slower profiles remain experimental. They do not establish general zero-shot improvement.

Listen to the [earlier decoder/timing comparison](decoder_cadence_combined/index.html). Both launchers were exercised: native slower output equals its comparison master byte-for-byte, and the new standard ONNX output passed an independent Small word check. Runtime memory is recorded in the RSS report. Realism is still unaccepted.

## Decoder reference investigation

The source-token reconstruction probe separates decoder conditioning from token generation. With the same genuine target tokens and acoustic seeds, changing only the decoder speaker embedding moved estimated pitch toward the target; changing the acoustic prompt also affected transcript accuracy. Self-reference variants deliberately use the target recording for diagnosis and are not zero-shot quality evidence. See `acoustic_conditioning_probe/README.md`.

A separate comparison uses an independent expressive source reference for new text. All twelve outputs pass independent Whisper-small transcription. Saved speech tokens are identical across four decoder variants per passage. At -27 LUFS, the current fitted baseline has mean similarity 0.8686 and DNSMOS 3.1773. Half donor embedding gives 0.8807/3.1217; full donor embedding gives 0.8828/3.0014. Replacing all acoustic conditions gives 0.8422/3.1785. The full acoustic swap raises average per-clip median voiced pitch from 201.2 to 220.2 Hz but reduces similarity. No new default was selected.

The conversational reference itself has lower estimated pitch than the expressive reference. Thus source-vs-generated pitch differences are partly confounded by reference speaking style. Pitch alone must not select a voice or justify a fixed pitch shift.

Listen: [decoder reference comparison](decoder_embedding_comparison/index.html). Exact results: `decoder_embedding_comparison/review.json`. The comparison has only three texts. Metrics are proxies; realism remains unaccepted.

## Corrections to the training pipeline

The first adaptation dataset had incorrect transcripts. Full-source Tiny proposals and independent clip-level Small transcripts admitted 17 replacement clips by exact normalized agreement: 15 for fitting and 2 for validation. Source, clip, and label hashes are checked before fitting. This automated agreement is not human transcription verification.

A second fault affected reference preparation. Training used a different loudness and speaker-encoder path from inference. The old reference differed by 37 of 333 prompt tokens. Training now calls the same reference function as inference. All 17 reference feature rows exactly match the saved inference tensors. Old unverified caches are rejected. Earlier fitted results cannot support adoption.

## New matched comparisons

These are development metrics on generated WAV files. Speaker cosine estimates identity similarity. DNSMOS estimates speech quality. Neither metric proves realistic voice character, richness, or absence of audible artifacts. Small ASR is an independent transcript audit.

| Comparison | Mean speaker cosine | Mean DNSMOS | Small exact transcripts |
|---|---:|---:|---:|
| Three passages, base | 0.8817 | 2.615 | 3/3 |
| Three passages, aligned speaker fit | 0.8644 | 3.104 | 2/3 |
| Three passages, last-four-layer adapter | 0.8660 | 3.292 | 3/3 |
| Three passages, all-layer adapter | 0.8897 | 3.044 | 3/3 |
| Three passages, spectral correction only | 0.8840 | 2.747 | 3/3 |
| Four new text/seed cases, base | 0.8668 | 2.979 | 4/4 |
| Four new text/seed cases, spectral correction only | 0.8702 | 3.040 | 4/4 |

The aligned speaker fit and last-four-layer adapter reduce average identity similarity. They are not accepted improvements. The all-layer adapter and spectral correction show small average gains, with variation across passages. On four additional text/seed cases, the all-layer adapter plus spectral correction changed mean speaker cosine from 0.8668 to 0.8745 and DNSMOS from 2.979 to 3.239. Small ASR matched all four base, all four adapter, and all four combined clips. Tiny disagreed on one combined clip. This is still an experimental candidate awaiting listening acceptance. See `all_attn_robustness/review.json`.

The spectral correction uses 80 smooth, time-independent mel-band offsets fitted only on training audio. It changes the relative frequency envelope before the vocoder. It does not explicitly transform pitch or timing; indirect waveform changes remain possible. Its maximum offset is 0.155 natural-log units. Two validation clips show lower spectral-envelope error, but this is a small validation set.

Listen: [all-layer adapter comparison](all_attn_comparison/index.html), [spectral correction comparison](mel_comparison/index.html). Measurements: the respective review.json, evaluation.json, small_audit.json, and manifest.json files.

## Loudness-controlled check

All new ASMR comparisons also have constant-gain copies at -27 LUFS. No compression, limiter, filter, or denoiser was added. The maximum measured level error across 36 files is below 0.000001 LU. Playback pages now use these copies, including the source reference.

At this matched level, three-passage base DNSMOS averages 2.482 and the all-layer adapter averages 3.067. On the four additional text/seed cases, base averages 2.877, adapter 3.187, and adapter plus spectral correction 3.235. The gains therefore persist under this level control. DNSMOS is still an automatic estimate, and ASMR is outside its typical denoising evaluation domain. Evidence: `level_matched/manifest.json` and `level_matched/dnsmos.json`.

## Harvey

The source is a 40.82-second compilation with background music. A voice-isolated reference remains the zero-shot baseline. Raw-source transcript consensus admitted no training clips. Independent clip-level rechecking admits three isolated-source clips: two for fitting and one for validation. Reference feature parity is exact against inference. A rank-2 attention adapter and a bounded speaker-vector fit have been trained; all 12 generated comparison files passed Small ASR. On three passages, full-adapter speaker cosine averaged 0.7347 versus 0.7221 for the zero-shot baseline against the original mixed held-out recording. Against the same excerpt after voice isolation, the values were 0.8187 and 0.7989. Both reference views retain a small average gain. The half-strength adapter reduced similarity against the mixed reference. [Listen to all Harvey variants](harvey_fitted_comparison/index.html). At matched -27 LUFS, full-adapter DNSMOS is 3.379 versus baseline 3.399, so it does not retain a cleanliness gain under level control. The speaker-vector fit averages 3.453 and has a smaller identity gain. These are tradeoffs, not a universal improvement. This tiny dataset cannot establish generalization. No unverified Harvey labels enter fitting.

## Runtime and cleanliness

All model jobs run serially with a hard memory cap, no swap, two CPU cores, and at least 4 GiB reserved system headroom. See [RSS measurements](RSS_REPORT.md). The low-memory staged CUDA option produced 7.36 seconds of speech in 37.76 seconds at 1023.1 MiB sampled peak process-tree RSS. A new persistent worker completed warm requests, including watermarking and WAV writing, in 4.32 seconds for 7.36 seconds of audio and 3.84 seconds for 6.96 seconds of audio. Peak sampled process-tree RSS was 1942.91 MiB. Cold startup remains slow. The runtime remains experimental because a strict long-context numerical cache check is unresolved. Those base-profile timings used the previous TF32 setting and are historical. Current fitted ONNX results are listed below.

Perth watermarking remains enabled. Output mastering uses a 45 Hz high-pass, bounded loudness gain, a -1 dBTP ceiling estimate, and brief endpoint fades. No noise gate or denoiser is enabled by default. Earlier denoising tests could remove voice detail. No claim of professional cleanliness or ElevenLabs parity is supported.

[Historical experiment notes](QUALITY_REPORT_history.md) contain superseded results and must be read with the corrections above.

Both new fitted profiles are available through `nano-clone`: `asmr_fitted_experimental` and `harvey_fitted_experimental`. Each CLI smoke test reproduced its audited sweep WAV byte for byte. Peak RSS was 2373.29 MiB for ASMR and 2400.19 MiB for Harvey. These names do not imply listening acceptance.

## Fitted ONNX and remaining voice differences

Both fitted adapters now run through ONNX with exact adapter and graph hash checks. The ASMR mel correction is applied immediately before the vocoder. Full-precision CPU and CUDA short-context comparisons pass. Default CUDA TF32 failed the numerical check and is now disabled. The separate context-400 cache mismatch remains unresolved, so the runtime remains explicitly experimental.

[Listen to the fitted ONNX samples](fitted_onnx_samples/index.html). All six FP32 batch WAVs passed the independent Small transcript check. The listening page uses four unique generated passages plus two source references, all matched to -27 LUFS using constant gain. The optimized warm worker uses about 1.9 GiB process-tree RSS. Staged fitted ASMR uses 1004.27 MiB and is slower. See the current table at the top of the RSS report.

A same-text acoustic diagnostic helps describe the remaining mismatch. The held-out ASMR source is 10.0 s long; the base output is 5.96 s and the fitted output is 6.8 s. Praat estimated median voiced pitch at 258.9 Hz in the source and about 194.0 Hz in the fitted output. Voiced coverage is only 32% in the source and 41% in the fitted output. These estimates can be unreliable for breathy or whispered speech. They do not justify an automatic pitch shift.

In a separate six-second reconstruction diagnostic, source-mel vocoding retained estimated pitch (253.1 to 252.6 Hz), while native source-token acoustic reconstruction measured 221.1 Hz. This points toward token-to-mel conditioning and prosody as investigation targets. It does not isolate tokenization from reference-conditioning effects. Evidence: `asmr_same_text/prosody_audit.json` and `codec_diagnostics/prosody_audit.json`.

The data-coverage review found no verified ASMR expansion ready to use. The current 17-row fit uses full-source Tiny proposals checked against independent clip-level Small transcripts. A separate, stricter clip-Tiny rebase admits 15 rows. These are distinct protocols; the 17 rows must not be described as exact clip-Tiny/clip-Small consensus. Protected source intervals and confidence gates remain enforced.

The requested convincing realism is still not established. No profile was promoted based on automatic measurements.

## T3 reference input comparison

Nine clips changed the speech prompt, speaker vector, or both while holding the fitted decoder fixed. Same three texts/seeds, RP1.0, -27 LUFS, and development references. The prompt-only change moved mean speaker cosine from 0.90254 to 0.89458 and DNSMOS from 2.801 to 3.164. All three transcripts were exact. Speaker-only also passed all three, but scored 0.89670 / 2.882. Changing both inputs introduced word errors in two of three passages and is rejected as a default. No variant is promoted. The prompt-only same-source phrase remains 7.4 seconds with estimated median voiced F0 197.1 Hz, versus 10 seconds / 258.9 Hz for the source. Breathiness limits pitch reliability. [Listen with controls](t3_reference_matrix/index.html). Generation peak sampled process RSS was 2492.96 MiB; all jobs used the resource guard.

## Harvey decoder embedding fit

The decoder fit uses the same two training clips and one validation clip. The updated fitting helper supports native odd reference lengths and exact native noise placement. All three prepared native outputs match exactly before fitting. Validation objective decreased from 1.34567 to 1.25322 (6.9%) after 40 steps. The fitted vector preserves its norm and has cosine 0.98 to its initial value.

On three matched texts/seeds, isolated-reference speaker cosine changed from 0.83044 to 0.83405. DNSMOS changed from 3.37335 to 3.37453, effectively unchanged. All six word checks passed, and saved text-model token payloads match within each pair. Audio uses -27 LUFS. These are small development gains, with no listening acceptance or generalization claim. [Listen to the new comparison](harvey_decoder_comparison/index.html). No new profile is promoted.

## Decoder projection adapter comparison

Implemented a rank-2 adapter on the decoder final projection. It has 672 trainable values and merges into an existing weight for inference. The one-step test reproduced all 17 native prepared outputs exactly, preserved the zero-adapter output exactly, and passed gradient checks. Validation loss moved from 0.8938364 to 0.8937674. Peak process RSS was 1729.10 MiB.

The longer fit stopped after 13 completed steps when available desktop memory fell below 4 GiB. The best checkpoint is step 11, with validation objective 0.8882467. The latest step 13 objective was 0.8892207. This is an interrupted run, not a completed 40-step fit. Native sweep integration is now verified. CLI and ONNX deployment of this adapter are not implemented. The first smoke test hit a GPU allocation failure because validation retained gradient graphs. That defect is fixed, and a regression test verifies one no-gradient evaluation per clip. Separate guarded test runs passed: 9 fitter tests and 5 folding-runtime tests.

Twelve new samples compare the saved adapter with an exact baseline, across three texts and with the mel correction on or off. All twelve independent Small transcript checks pass. All six baseline raw WAVs exactly match the earlier fitted-decoder sweep. Without mel correction, mean speaker cosine changes from 0.884832 to 0.884969 and DNSMOS from 3.14625 to 3.15575. With mel correction, cosine changes from 0.889507 to 0.890468 and DNSMOS from 3.16707 to 3.16430. These changes are small and do not establish a useful perceptual gain. No default is changed. [Adapter comparison](decoder_projection_comparison/index.html). All listening copies use -27 LUFS.

The acoustic-only dataset now contains 30 genuine clips, 28 training and two validation, totaling 159.26 seconds. It excludes all declared reference and evaluation intervals with a two-second buffer. Waveform checks found no clipped samples. This does not establish perceptual cleanliness. Token and flow preparation remain pending. This acoustic-only path uses no text labels and does not weaken the separate transcript requirements for text-model training.

## Wider decoder speaker-vector constraint

On the same 15 training and two validation clips, the wider cosine floor of 0.95 reached validation objective 0.8472352 at step 40. The previous 0.98-constrained 40-step run reached 0.8940363. The new run continued to step 60 and reached 0.8381018. All 60 steps completed. All 17 native preparation comparisons were exact. This is a speaker-specific fit, not a general zero-shot model change.

Eight new speech samples pass Small ASR with zero normalized word error. Speech-token archive payloads match the prior fit for all three general passages and the same-source phrase. Across the three general passages, with mel correction, speaker cosine rises from 0.889507 to 0.897474 while DNSMOS falls from 3.16707 to 3.09296. Without mel correction, cosine rises from 0.884832 to 0.892532 while DNSMOS falls from 3.14625 to 3.08959. This is a metric tradeoff, not a demonstrated perceptual improvement. DNSMOS also rates several genuine ASMR excerpts lower than generated speech; it is not an acceptance rule for breathiness or realism.

A matched Praat check using the 60–500 Hz range estimates the source phrase at 258.87 Hz and 10 seconds. Both fitted generated phrases remain 6.6 seconds and approximately 165.8 Hz. Whispered pitch estimates are unreliable, but the wider vector has not corrected this diagnostic timing gap. [Listen to the wider-vector comparison](decoder_embedding_cap095_comparison/index.html). No profile is promoted.

The 30-row acoustic-only token cache is ready in `acoustic_only_asmr/cache_v2`. All 14 overlapping speech-token sequences match the earlier audited cache exactly. The first extraction completed all rows but failed its final metadata check because the overlap helper omitted the required status field. The helper and regression test were fixed; the second extraction passed the strict final validator. Failed artifacts remain preserved. Native flow preparation is complete. Its original 17 reconstruction checks match exactly; all 14 overlapping source mels pass. A separate strict fit-consumer run reproduces all 30 new reconstructions exactly. See `acoustic_only_asmr/readiness.json`. A legacy hash-algorithm mismatch initially stopped flow preparation before model inference; the check now uses the original producer’s hash algorithm.

## Decoder attention adapter: earlier smoke verification

The larger acoustic dataset supports a rank-2 adapter on all 224 meanflow self-attention projections, with 344,064 trainable factor values. Base weights and the existing 192-value speaker vector remain frozen. The strict-FP32 one-step test reproduces all 30 native prepared reconstructions before adaptation and all 30 zero-adapter anchors exactly. Initial up-factor gradients are nonzero, so reconstruction error reaches the adapter. The validation objective changes from 0.7862759 to 0.7818824. This validation split differs from the earlier 17-clip experiment; those loss values are not directly comparable.

A factorized-versus-folded check initially failed at maximum error 0.003812 with cuDNN TF32 enabled. Disabling TF32 reduced the maximum difference to 0.00001192 across three source-token flow reconstructions. The original 0.0001 maximum and 0.00001 RMS limits were preserved. Repeated folding and base restoration are exact. New attention training and sweep comparisons require strict FP32. Native preparation replay retains its recorded original precision before switching to the fit precision.

At this smoke checkpoint, the full fit and speech comparisons were still pending. Their completed results follow below. The native folding path streams original weights and retains only references, shapes, and hashes. CLI and ONNX deployment were pending at that checkpoint. Current deployment evidence is at the top of this report. Nine fitter unit tests and three runtime tests pass under separate 640 MiB guards. Runtime tests include interrupted-fold recovery, strength switching, restoration, and changed-base rejection.

The central listening page now shows the current ASMR and Harvey comparisons first. Its rendered output was inspected in an isolated headless browser. All six visible WAV players load valid durations. This browser check does not constitute auditory quality assessment.


Acoustic attention fit: 30/30 steps complete, best30. Held-out reconstruction total .78627586 -> .66581574 (15.32% lower), rank2 adapters on224 decoder attention projections,344064 trained values. Peak process RSS1830.0MiB; service643.211s,cgroup1.4G/no swap. StrictFP32 factorized-versus-folded model check passes unchanged1e-4 max/1e-5 RMS gate on3 actual source-token flow reconstructions; observedmax6.20e-6,repeatfold+restoreexact. This is reconstruction evidence, not new-text perceptual acceptance. The fourteen matched synthesis results follow below.


Acoustic attention speech comparison COMPLETE:14clips, allSmallWER0, exactspeech-tokenpayloadwithinall4textgroups, strictFP32matchedcontrols, all-27LUFS. Three-new-text means: controlraw cosine.885287/DNS3.131835; attentionraw .892158/3.125945; controlmel .889821/3.153682; attentionmel .896536/3.164181. Small identity gain; no realism acceptance. Same-source phrase source10s/258.87Hz vs control6.6s/174.42Hz vs adapter6.6s/165.08Hz. Pitch measured60-500Hz, voicedcoverage32/35/36%, breathinesslimitsreliability. Pitch/pacinggapremains. The six prompt-donor comparison results follow below.


Prompt plus attention comparison COMPLETE (v2):6clips, allSmallWER0, matchedtokenswithin3pairs, -27LUFS. Two NEW-TEXT means: prompt-only cosine.8864105/DNS3.1664034 vs prompt+attention .8935955/3.1093683. Heldout samephrase prompt-only cosine.910653/DNS3.2047088 vs combined .915628/3.1781804. Thus attention improvesidentityproxybutreducesDNSproxy in this configuration. Samephrase both7.4s; F0197.045Hz(prompt)/196.639Hz(combined) versus10s/258.874Hzsource (60-500HzPraat,voicedcoverage36%vs32%,breathinesscaveat). No defaultpromotionorrealismacceptance. Centraldeliveryupdatedwithsource/A/B andcurrentnativeRSS; full14and6controlpageslinked.


T3 MLP rank2 test NOTSELECTED. Last4MLP8projections61440params,strict15/2aligneddata,6epochs90stepsKL.05. CE5.254716->4.698818. Ninegenerationcases(scale0/.5/1),3Smallworderrors(questionhalf.0625,heldouthalf/full.05). Two-new-text meanidentity/DNS:base .8913535/3.157036;half .883760/3.016633;full .857902/3.367350. Samephrase fullidentity.918918/DNS3.348436,pitch212.27Hz/7.4s vsbase192.11Hz/6.6s;source258.87Hz/10s. Sourcephraseimprovesbutnewtextidentityregresses,so no promotion. Samplesandfailuremetricsin t3_mlp_comparison/index.html, linkedcollapsedhistory.
