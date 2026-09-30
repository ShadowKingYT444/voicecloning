# Acoustic-only Nano cache

`prepare_acoustic_data.py` builds a source-audio cache for the Nano S3Gen
decoder.  It does not use transcripts.  It does not change the audited T3
adaptation cache.  A cache row contains the recorded clip identity, the exact
25 Hz S3 speech-token stream, and a normalized source log-mel array.

The command has two stages.  Run them in separate processes:

```bash
python scripts/nano_lab/bounded_job.py -- \
  .venv-nano/bin/python scripts/nano_lab/prepare_acoustic_data.py tokens \
  --source-manifest artifacts/nano_lab/acoustic_only_asmr/manifest.json \
  --output-dir artifacts/nano_lab/acoustic_only_asmr/cache \
  --device cuda

python scripts/nano_lab/bounded_job.py -- \
  .venv-nano/bin/python scripts/nano_lab/prepare_acoustic_data.py flow \
  --token-manifest artifacts/nano_lab/acoustic_only_asmr/cache/manifest.json \
  --output-dir artifacts/nano_lab/acoustic_only_asmr/flow \
  --conditionals-path artifacts/nano_lab/sweep_round2/asmr_conversational_morning_31.conds.pt \
  --device cuda
```

The `tokens` stage loads only `S3Tokenizer` from
`s3gen_meanflow.safetensors`.  It uses an unnormalized 16 kHz waveform for
token extraction, exactly as `adaptation.extract_target_tokens`.  It separately
loads a 24 kHz waveform, applies the upstream `norm_loudness` operation at
-27 LUFS, and extracts the vendor S3Gen mel spectrogram.  The optional aligned
cache check requires exact speech-token equality for every overlapping clip.
Each stage sets Torch intra-op and inter-op worker limits to two.

The `flow` stage loads only the S3Gen flow encoder and two-step meanflow
estimator.  It first replays the existing `mel_calibration_asmr` rows through
the shared streamed preparation and parity helper.  It then checks normalized
source-mel equality on overlapping clips.  If either check fails, the stage
does not admit new reconstructions.  New rows are diagnostic source-token
reconstructions.  They are not text generation and do not prove zero-shot
voice-cloning quality.

Both manifests are written atomically.  Only `status: ready` is accepted by
`validate_acoustic_cache`.  A `partial` or `failed` cache cannot enter the
decoder fit.  Resume accepts an existing token directory only when the source
manifest, source contract, and tokenizer checkpoint hashes match.  Every row
also records the source WAV/MP3 hashes and interval.  Protected reference and
held-out intervals use the declared two-second buffer.

The flow stage records each completed row and writes a partial manifest after
every reconstruction.  It deliberately restarts from the beginning because
the reference parity gate must run in the same process before a new row is
admitted.  Deterministic per-row seeds make that restart reproducible.  A
partial flow manifest is never accepted by the fitter.

The flow manifest uses the existing `nano_mel_calibration_prepare_v1` shape so
the decoder fitting helper can consume it.  It points to the acoustic token
manifest and has `acoustic_only: true`, `transcript_labels_used: false`, and
no transcript fields.  The fixed ASMR conditionals cache is accepted only with
SHA-256
`92e644466f79befe5ef937f819c921729ba00f29790efb81684cc3e88ef18c52`.

Pure contract tests are available in
`test_prepare_acoustic_data.py`.  They create tiny files and arrays.  They do
not load Torch, decode audio, or start a model job.
