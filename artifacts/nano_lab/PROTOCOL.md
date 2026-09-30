# Nano voice cloning experiment

Started 2026-09-29. All generated speech in this directory is synthetic.

## Scope

Improve Chatterbox Nano on the supplied ASMR 7 Minutes recording first, then
the supplied Harvey Specter recording. Preserve original MP3s. Generate neutral
test sentences. Retain the built-in Perth watermark. Keep zero-shot conditioning
results separate from per-speaker adapted results.

## Hardware and source

- NVIDIA RTX 4060 laptop GPU, 8 GiB VRAM.
- Intel Core i7-12650H; 16 GiB host RAM shared with desktop applications.
- Upstream source: https://github.com/resemble-ai/chatterbox
- Source commit: 5de7a54aa4e5e2baadb0182dde554908b48b85c2.
- Checkpoints: pre-existing models/chatterbox-nano.
- Local environment: .venv-nano uses the existing Voicebox dependency installation
  via a .pth file and an editable local Chatterbox install. Versions are recorded
  in environment.json. No changes to the original Voicebox environment.

## Quality evaluation

Reference selection and reference cleaning are tested separately from token
sampling, speaker-conditioning changes, and speech decoding. Hold out source
intervals from the conditioning set for speaker-similarity evaluation. Test new
sentences, with several random seeds in final confirmation. Save raw generated
audio as well as a delivery master. Loudness matching must not conceal distortion.

Metrics are diagnostic signals, not proof of realistic speech:

- ASR word error rate checks spoken content. Recognition can fail on ASMR.
- Resemblyzer speaker cosine compares identity with held-out source excerpts.
  Whispered speech and music can bias this measure.
- DNSMOS P.835 estimates speech quality and background cleanliness. It is a
  learned proxy from denoising research, not a listening panel. ASMR may be
  outside its training distribution. Short clips are repeated to 9.01 seconds,
  as in the upstream evaluation protocol.
- Clipping, peak/RMS level, spectral statistics, and quiet-frame energy detect
  technical issues. Quiet-frame energy is not a calibrated noise-floor estimate.

DNSMOS source/model: https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS
The downloaded model and upstream script are in vendor/dnsmos with its license.

The delivery master uses a 45 Hz high-pass, up to 12 dB gain toward -19 LUFS,
a -1 dBTP limit estimated with 4x oversampling, and 5 ms endpoint fades. It uses no noise gate.

## Performance evaluation

Use separate processes for configurations. Measure cold load, first synthesis,
and warmed synthesis. Peak RSS includes loading and generation; resident RSS
after generation is separate. Record CUDA allocated/reserved memory separately
from host RSS. Use matching texts and seeds. Correct real-time factor is elapsed
generation seconds divided by output-audio seconds; less than 1 is faster than
real time. Exclude evaluator models from inference-only process measurements.

More decoder steps, adapters, quantization, and ONNX are experiments. Adopt a
configuration only after checking the resulting audio and relevant metrics.
Do not claim a full ONNX runtime if only one component was exported.
