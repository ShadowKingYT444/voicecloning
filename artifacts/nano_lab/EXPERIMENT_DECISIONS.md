# Experiment decisions, 2026-09-29

## Active constraints

The user reported desktop crashes from RAM pressure. All subsequent model work
uses scripts/nano_lab/bounded_job.py: one process, 3 GiB cgroup maximum,
2500 MiB high threshold, zero job swap, two CPU cores, priority nice 10.
Require 6 GiB available to start; stop below 4 GiB available. Do not raise these
limits. Separate cgroup memory (includes charged cache) from process RSS.
No model jobs or model agents run concurrently.

## Evidence so far

- Better reference selection improved ASMR speaker cosine from 0.749 for the
  original early reference to 0.890 for a conversational reference on the same
  comfort sentence. This is a development metric, not human listening evidence.
- Rank-4 and rank-8 full-strength LoRA adapters reduced validation token loss but
  introduced pronunciation errors. They are rejected as default profiles.
  Weaker strengths also have mixed results across sentences.
- Meanflow decoder steps 1/2/4/8 do not show monotonic quality improvement.
  The larger original decoder at 10/20 steps did not consistently improve the
  combined identity, content, and DNSMOS results. Keep it experimental.
- Gentle output denoising can improve DNSMOS background/overall estimates, but
  its benefit depends on the reference. It can reduce speaker similarity.
  No universal denoiser is enabled in the delivery profiles.
- Harvey reference 02 leads the initial speaker-similarity comparison. Reference
  isolation improves source background scores, but generated quality is mixed.
  Separate T3 and decoder references and combined excerpts are under evaluation.
- The optimized loader uses meta construction, tensor streaming, removes unused
  T3 weights, and skips reference encoders when cached conditionals are supplied.
  A CPU built-in-voice comparison produced exactly matching WAV hashes and about
  40 percent lower peak RSS. Its concurrent-load timing is exploratory.
- Two T3 ONNX graphs were exported with numerical parity on representative inputs.
  They are not a complete voice runtime. Growing-cache validation, a shared graph,
  and acoustic/vocoder export are not complete. Do not claim ONNX deployment.
- Upstream meanflow ignores inference_cfg_rate, and its temperature parameter is
  unused. acoustic_controls.py explicitly scales initial noise for the new
  experiments. Unity noise scale calls the original forward unchanged.

## Latest comparison and release decisions

Round 2 completed 36 cases and two independent recognizer audits. Round 3
completed 39 of 40 cases. Raw audio, delivery masters, transcripts, and failures
remain in the sweep directories. The recognizers disagree on some ASMR source
transcripts. Training labels are therefore uncertain; lower training loss alone
cannot select a release model.

A constrained 256-dimensional learned speaker residual reduced validation token
cross-entropy from 5.5531 to 5.2173. On six new conversational ASMR clips, the
speaker cosine mean changed from 0.8682 to 0.8676, DNSMOS overall from 2.716 to
3.199, and background from 3.654 to 3.928. All six passed the Whisper-small exact
normalized transcript check. This fitted variant advances to more text/seed
checks. It is speaker-specific adaptation, not zero-shot cloning.

The same residual is rejected for the soft profile. One of six clips hit the
700-token limit, and the five completed clips lost 0.0177 mean speaker cosine.
Initial acoustic-noise scaling did not consistently improve both voices, so the
stock scale remains the default.

Harvey uses the isolated second excerpt. Mixed natural/isolated conditioning
sometimes improves similarity, but its content audits include missing initial
words. Six meeting-clip transcript differences are simply `9` versus `nine`.
The existing WER files retain these literal differences for auditability.

The user-facing launcher reproduced the selected soft ASMR and Harvey masters
byte for byte. Measured peak RSS was 2436.8 MiB and 2386.0 MiB respectively.
Matched warm inference benchmarks favor fp32: ASMR mean RTF 0.483 versus 0.550
with T3 fp16, while fp16 also increased host RSS. Acoustic weights stay fp32.
The deterministic-buffer restoration regression and resource-guard tests pass.

A single 368 MiB T3 ONNX graph was exported under the memory guard. The first
attempt was stopped below 4 GiB desktop headroom. Removing a redundant in-process
ONNX protobuf copy let export complete. Initial prefill/decode numerical validation passed with maximum absolute error
1.72e-5 and 977.8 MiB cgroup peak. Longer growing-cache checks against upstream
GPT2 were stopped by the desktop headroom guard. They remain pending, together
with acoustic exports and complete runtime measurement. The graph is not deployed.

## Current artifacts and next gates

- `delivery/index.html` contains five synthetic samples with source excerpts.
- `RSS_REPORT.md` contains measured runtime data and its limits.
- `round3_review.json` contains paired comparisons and selection decisions.
- `sweep_round4_config.json` defines 18 new text/seed tests, using cached
  conditionals to avoid loading reference encoders.
- `onnx_t3_reference.py` implements separate upstream-PyTorch and pure-ORT
  verification stages; these still require execution.
- Atomic publication failure/success checks and text-chunking tests pass.
  Long-input CLI synthesis and fitted-profile deployment remain to be verified.
- ONNX sampling now applies temperature, top-k, top-p, then repetition penalty,
  matching upstream processor order. Three pure-NumPy regression checks pass.
- The experimental voiced-excitation control and its 18-case comparison are
  prepared but have not run.

No human listening assessment has been performed. DNSMOS, transcription, and
speaker embeddings do not establish realistic richness or ElevenLabs equivalence.
The full goal remains active and unverified.

## Latest memory-limited work

Vocoder ONNX export and separate CPU verification passed on one short input,
with 4.14e-5 maximum waveform error and 400 MiB cgroup peak. This does not
validate complete voice generation. The flow encoder exceeded its 1280 MiB cap
both with and without constant folding. An external-parameter export helper is
prepared but unverified; its tiny smoke test cannot start below 4736 MiB available.

Lazy imports reduced measured vocoder-only import RSS from 846.0 to 547.8 MiB.
Shared read-only GPT2 causal masks pass storage/value tests. Full voice RSS and
deterministic audio must be remeasured after these changes. Prior synthesis
numbers must not be presented as measurements of this latest revision.

No model service was active during the latest resource check. Desktop headroom
was about 4.4 GiB. Unrelated processes remain untouched. Four resource-guard,
three atomic-audio/chunking, and three ONNX sampling regression checks pass.

## Resumed validation results

The RAM blocker temporarily cleared. Latest soft and Harvey CLI outputs match
their delivery WAVs exactly. Conservative observed peak RSS is 2126.4 and
2136.2 MiB; GPU peak allocation is 1664.6 MiB. The previous GPU peak was
2376.3 MiB. See revalidation/comparison.json for the tested runtime hash.

Round 4 completed 17/18 cases. Full-strength fitted conversational conditioning
keeps a quality-score advantage but fails one of six independent content
checks. Do not promote it. A 20-case sweep tests reduced fitted strengths and
recovery settings for Harvey's failed expressive passage.

External-parameter ONNX export passed a tiny independent ORT check. Applying it
to the flow encoder reduced export memory enough to fit the 1280 MiB cap.
Disabling Torch ONNX shape inference emitted an invalid ConcatFromSequence
for constant Pad arguments; enabling shape inference fixed the graph. The
flow encoder passed separate ORT checks at token lengths 12, 120 (117 valid),
and 600. Maximum absolute error was 2.623e-6; verification cgroup peak was
709.7 MiB. This is a verified component, not full audio inference.

The estimator exporter now selects external parameters and prepares reference
lengths 8, 64, and 256. That code is unexecuted. Larger jobs were refused again
when unrelated desktop memory use increased. Do not raise the resource limits.

## Current acceptance and next diagnostic

The user listened and reported that neither ASMR candidate sounds convincing.
Current candidates fail perceptual acceptance regardless of proxy score gains.
Do not promote the fitted vector or declare realism achieved.

The 75 percent residual and 20-case robustness sweep are complete. Voiced
excitation-noise scales produced identical audio and are not an improvement.
Meanflow ONNX export and numerical component checks are complete. Complete
staged CPU audio generation now runs under the 1280 MiB cap but is slow and
experimental. T3's strict long-prefill cache check still fails.

Next diagnostic: compare original audio, vocoder-only reconstruction, acoustic
decoding from genuine source speech tokens, and text-generated speech. This
separates decoder limitations from token-generation/style limitations. These
reconstructions must never be presented as successful new-text cloning.

## Training-label failure found by reconstruction audit

Two independent recognizers agree that source01's actual words differ from its
cached transcript. Additional rows contain truncated or extra phrases. Small
ASR exactly matches only3/10 old cache labels. The previous adapter and speaker
residual training therefore used flawed supervision. Their prior losses and
proxy scores cannot justify selecting a fitted voice. The adapted profile is
disabled; old files remain available as historical artifacts.

Preparation and both training paths now require an accepted transcript audit
with matching SHA-256 for source audio and transcript. Dirty old cache training
is rejected before model loading. Two tests cover missing audits and changed
text/audio. A new word-aligned dataset is being prepared; no new adapter is
trained until it passes the independent transcript admission check.
