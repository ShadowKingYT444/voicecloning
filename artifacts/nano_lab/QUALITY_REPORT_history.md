## Current aligned-data result

Training reference preparation now uses the exact inference function. The prior independent encoder path differed by 37/333 prompt tokens and speaker cosine 0.9653. All 17 aligned cache rows now match the saved inference prompt and speaker tensors exactly. Target speech/text tokens remained unchanged. Old reference caches are rejected before fitting.

On three matched new sentences, aligned speaker adaptation changed mean speaker cosine from 0.8817 to 0.8644 and DNSMOS overall from 2.615 to 3.104. It changed one word according to both recognizers. The aligned rank-4 adapter changed cosine to 0.8660 and DNSMOS to 3.292. Small ASR matched all three adapter texts; Tiny reported two discrepancies. Neither fit is promoted. These metrics do not establish convincing realism.

Comparison: `aligned_comparison/index.html`. Raw results: `aligned_comparison/review.json`. Peak synthesis RSS: 2399.9 MiB. New acoustic spectral-calibration experiment is pending.

## Repaired dataset experiment

The user rejected the prior ASMR candidates as unconvincing. A new source audit found incorrect transcripts in the old adaptation cache. Earlier fitted candidates are archived and cannot support a quality claim.

The replacement dataset contains 17 source clips admitted by exact agreement between Tiny and Small ASR after text normalization, plus matching audio and text hashes. It has 15 training clips and 2 validation clips. Agreement is an automatic check, not human verification.

A 256-parameter speaker adjustment reduced validation token loss from 4.9583 to 4.7207. A 73,728-parameter rank-4 attention adapter reduced it to 4.6243. These losses do not measure realism. Fifteen matched synthesis cases are being evaluated before any adoption.

Raw results: `repaired_training_summary.json`, `conditioning_repaired_report.json`, `adapter_repaired_rank4_report.json`.

# Voice quality report

**Training-data audit in progress:** one cached training transcript does not
match its source excerpt. Existing adapter and speaker-residual fitting results
must not justify promotion. The zero-shot baseline checkpoints are unchanged.

The package contains three zero-shot ASMR variants, two fitted conversational
ASMR experiments, and one Harvey variant. All generated clips are synthetic.
Use [the sample comparison page](delivery/index.html) to compare them with the
actual source excerpts. The user reviewed the ASMR candidates and reported:
“Neither sounds convincing yet.” These candidates do not meet acceptance.
Automated scores are diagnostic evidence, not a reason to override this result.

## What changed

Reference selection produced the largest reliable identity improvement in early
ASMR tests. A selected conversational excerpt increased the held-out speaker cosine
from 0.749 to 0.890 on the same comfort text. Different excerpts preserve
different delivery characteristics, so the conversational, soft, and intimate
profiles remain separate.

The ASMR speaker residual fits only 256 values with the main model frozen. Its
norm is constrained to 0.1. This provides a small speaker-specific adjustment
without an additional runtime network. The frozen reference cache can store the
result. It is fitted adaptation; the other installed profiles are zero-shot.

| Conversational ASMR, six paired clips | Base | Fitted residual |
|---|---:|---:|
| Mean held-out speaker cosine | 0.8682 | 0.8676 |
| Mean DNSMOS overall estimate | 2.716 | 3.199 |
| Mean DNSMOS background estimate | 3.654 | 3.928 |
| Whisper-small exact normalized transcript | 5/6 | 6/6 |

The table uses the same three texts and two seeds. It is a development result,
not a blinded listening trial. Subsequent text/seed tests are reported below.
The fitted residual is not a universal improvement: it failed one soft-voice
case and reduced the soft voice's similarity on the completed clips.

Harvey uses a music-isolated excerpt. Other combinations can score higher on
speaker similarity but have more initial-word recognition failures. The selected
isolated reference was more consistent in the tested passages.

## Rejected and experimental changes

- Full-strength rank-4 and rank-8 attention adapters improved validation loss
  but introduced pronunciation errors. They are not enabled in release profiles.
- Lower adapter strengths had mixed identity and content results.
- Increasing meanflow steps from two to four or eight did not reliably improve
  all quality measures. Two steps remain the default.
- The larger original acoustic decoder did not consistently beat meanflow.
- DeepFilterNet output denoising can reduce background estimates but can also
  remove speaker detail. It is not enabled by default.
- Acoustic initial-noise scaling had no consistent benefit across the two ASMR
  profiles. The original scale remains enabled.
- Reduced voiced vocoder excitation noise is implemented only as a research
  option. Its completed 18-case comparison produced identical WAVs at all
  tested scales. It is not an improvement.

## Output and content checks

The soft ASMR and Harvey commands reproduce their selected sweep masters
byte for byte. The sample package contains 24 kHz mono PCM-24 WAV files.
Mastering uses a 45 Hz high-pass, bounded loudness gain, a -1 dBTP ceiling
estimated at four-times oversampling, and brief endpoint fades. No noise gate
is applied. Raw outputs remain in the sweep folders.

The generation loop rejects empty output, excessively long text, non-finite
samples, and generation that reaches its token limit. Long CLI input is split
at sentence boundaries. The new output writer publishes a WAV only after all
segments finish. Failure-publication and text-chunking tests pass without loading the model.
The actual 85-word CLI and a forced retry passed end-to-end synthesis and
independent content checks. See the RSS report for memory and timing.

## Limits of the evidence

DNSMOS is an estimated score. It does not prove naturalness, lack of audible
static, or preservation of ASMR breath. Speaker embeddings do not measure all
vocal detail. The two speech recognizers disagree on several ASMR source labels;
these labels were not manually corrected. The user listening result rejects the current ASMR quality.
No claim of ElevenLabs-equivalent quality is supported by the current evidence.

The full ONNX audio runtime is not deployed. See [the RSS report](RSS_REPORT.md)
for actual PyTorch memory and speed measurements, and [experiment decisions](EXPERIMENT_DECISIONS.md)
for the remaining validation gates.

## Additional held-out text checks (round 4)

The 18-case sweep completed 17 clips. Harvey's expressive passage with seed 63
hit the 700-token limit; this is a failed case, not a delivered voice sample.
All three soft ASMR clips and the two completed Harvey clips passed the
independent Whisper-small normalized transcript check.

Across six paired conversational clips, baseline versus fitted mean speaker
cosine was 0.87809 versus 0.87495. DNSMOS overall was 2.90875 versus 3.22109,
and background was 3.68902 versus 3.94347. The baseline passed 6/6 independent
transcript checks; the full-strength fitted vector passed 5/6. In the remaining
expressive clip, Whisper-small transcribed “very beginning” as “fairy beginning”;
the smaller recognizer also reported a discrepancy. Human listening has not
confirmed the exact sound.

The fitted vector remains experimental. Lower strengths of 0.5 and 0.75 are
prepared for comparison on all six cases. A focused Harvey comparison varies
sampling temperature, repetition penalty, and prompt length for the failing
expressive passage. The combined 20-case sweep is complete. Results follow below. Zero-shot
profile defaults remain unchanged.

Raw artifacts: `sweep_round4/manifest.json`, `eval.json`, `asr_small.json`,
and `round4_review.json`. These proxy measurements do not establish subjective
realism, richness, or parity with a commercial voice service.

## Reduced adjustment and delivered-file comparison

The 50% speaker adjustment passed 6/6 strict Whisper-small checks, with mean
cosine 0.87949 and DNSMOS overall 2.97966 on the six new cases. At 75%, those
means were 0.87576 and 3.13090. The 75% setting passed 5/6 strict checks; its
remaining transcript is “you’d forgotten,” which correctly contracts “you had
forgotten.” The scorer always expands “you’d” to “you would.” This scoring
artifact is annotated in robustness_review.json; original scores are preserved.

The 75% setting is available as the experimental
`asmr_conversational_adapted` profile. Its actual launcher sample passed the
smaller recognizer’s word check. Six delivered, mastered WAVs were then scored
together under identical evaluation settings:

| Delivered sample | Speaker cosine | DNSMOS overall | Tiny normalized WER |
|---|---:|---:|---:|
| asmr_soft | 0.8504 | 2.746 | 0.0000 |
| asmr_conversational | 0.8605 | 2.107 | 0.0417 |
| asmr_intimate | 0.8498 | 2.093 | 0.0833 |
| asmr_conversational_fitted | 0.8302 | 2.796 | 0.0000 |
| harvey | 0.7880 | 3.240 | 0.0000 |
| asmr_conversational_adapted | 0.8376 | 3.016 | 0.0000 |

These scores apply to the delivered files, whereas earlier sweep tables score
raw generated audio. Do not combine those two evaluation scopes. The adapted
75% sample improves quality and similarity over the full-strength fitted sample
on this passage, but the unadapted conversational sample retains higher speaker
similarity. No listening assessment has established realistic richness.

Harvey temperature 0.9 / repetition penalty 1.1 recovered the previously failed
expressive passage with exact independent transcription, cosine 0.8135, and
DNSMOS overall 3.560. It still needs broader-text validation before becoming
the default. Four other recovery settings still hit the generation limit.

## Voiced excitation noise experiment

All 18 cases completed. Six voice/text groups used scales 0, 0.5, and 1.0.
Speech token payloads matched within every group. Raw and mastered waveform
hashes also matched within every group. This setting produced no measurable
change. It is not a noise-removal improvement and is not promoted.
See sweep_vocoder/review.json, token_identity.json, and evaluation.json.

The forced length-limit retry sample passed independent Whisper Small ASR
with zero word errors. Tiny ASR's have/had discrepancy is retained in its raw
report. Neither automated result establishes perceptual realism.

## Source reconstruction diagnostics

The diagnostic page is codec_diagnostics/index.html. Its TTS clip uses a
cached transcript now known to mismatch the source. It is not a same-sentence
comparison until that transcript and sample are corrected. It compares
loudness-matched source, vocoder reconstruction from source mel features,
acoustic reconstruction from genuine source tokens, and text-generated speech.
The first three are not demonstrations of new-text cloning. The synthesized
outputs retain Perth. The new diagnostics await listening review.

A full 7.36-second native/ORT vocoder comparison passes a combined 3e-4
absolute/relative tolerance. Maximum waveform difference is 1.049e-5 and
relative L2 difference is 7.723e-6. The checked model pitch output stays below
the voiced threshold on both one generated and one source-feature case. This
is an internal predictor output, not measured physical pitch of the recording.
Voiced-only noise adjustment is therefore inactive on those cases. Scaling
all explicit source noise does change the waveform: half scale changes its
relative L2 norm by 0.00568, zero scale by 0.01137. This alone is not evidence
of better perceived cleanliness. No noise setting has been promoted.
