# Kokoro voice package

This is the minimal deployment bundle for giving a local software agent the
enrolled Mommy voice. It contains only the Kokoro base model, the custom voice
pack, a small runtime API/CLI, and optional enrollment assets.

Contents:

- `kokoro_voice.py` — application-facing `KokoroVoice` class and CLI.
- `models/kokoro/` — Kokoro config and 82M model weights.
- `voices/af_asmr7.pt` — default/main ASMR7 voice, promoted from refined candidate 1.
- `voices/af_asmr7_secondary.pt` — secondary ASMR7 voice, promoted from refined candidate 2.
- `voices/af_mommy.pt` — original enrolled `[510, 1, 256]` custom voice pack.
- `voices/af_asmr7_v1.pt`, `af_asmr7_v2.pt`, `af_asmr7_v3.pt` — three packs enrolled from the new 7 Minutes in Heaven recording.
- `voices/af_asmr7_refined_v1.pt`, `af_asmr7_refined_v2.pt`, `af_asmr7_refined_v3.pt` — longer-reference/pitch-tested candidates from the same recording.
- `models/inno/model.safetensors` — optional Inno tuner weights for future enrollment.
- `reference/mommy_reference.wav` — the cleaned source reference used for enrollment.
- `requirements.txt` — runtime dependencies.
- `requirements-cloning.txt` — optional dependencies for making more voice packs.

## Application integration

```python
from kokoro_voice import KokoroVoice

voice = KokoroVoice()  # CUDA when available; CPU fallback otherwise
audio = voice.synthesize("Hello from my local agent.")  # 24 kHz float32 NumPy array
voice.save("Hello from my local agent.", "reply.wav")
```

The runtime splits long text at sentence/word boundaries before synthesis, so
the app does not lose text when a request exceeds Kokoro's per-segment limit.

## CLI

```bash
python kokoro_voice.py "Hello from my local agent." -o reply.wav
```

Install dependencies into the app's environment with `requirements.txt`. The
`torch==2.11.0+cu128` line matches the source machine; use the corresponding
PyTorch wheel for another CUDA, ROCm, Apple Silicon, or CPU target.

The optional cloning files are not required for normal agent inference. They
are included only to reproduce or replace the packaged voice pack.
