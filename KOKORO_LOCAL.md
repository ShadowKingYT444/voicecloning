# Local Kokoro setup

This workspace contains a local Kokoro-82M install and the `inno-kokoro` zero-shot
voice tuner. The runtime is the existing Python 3.12 environment used by
`voicebox/backend/venv`, with CUDA-enabled PyTorch.

Installed components:

- `kokoro==0.9.4`
- `inno-kokoro==0.2.0`
- `torch==2.11.0+cu128`
- local base model: `models/kokoro/`
- local tuner weights: `models/inno/model.safetensors`

The generated voice pack is `voices/af_mommy.pt`, enrolled from the cleaned
24 kHz mono reference derived from the root `mommy-asmr.mp3`. The original MP3
is not modified.

## Run

```bash
voicebox/backend/venv/bin/python scripts/kokoro_local.py
```

The script uses the RTX GPU, enrolls the reference, creates
`outputs/af_mommy_test.wav`, and writes the measured benchmark to
`outputs/kokoro_benchmark.json`. It performs a warmup before timing and reports
both phoneme tokens/second and generated audio seconds/second.

For a CPU run, use `--device cpu --precision fp32`. For a repeat benchmark that
reuses the saved pack, use `--skip-enroll`.
