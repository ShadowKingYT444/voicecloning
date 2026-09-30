# Mommy ASMR local voice clone

## Result

The selected model is **Qwen3-TTS 1.7B Base**, run locally through the repository's Voicebox backend on an RTX 4060 8 GB GPU. The best dry take is `outputs/qwen_natural_a/seed31.wav`; `final/best_qwen_seed31.wav` is the same take with only +1.3 dB gain.

Test text:

> You have done enough for today. Come closer, breathe slowly, and let yourself relax.

The final was transcribed back exactly by Whisper Small (punctuation aside), is 5.92 seconds at 24 kHz mono PCM16, peaks at -1.5 dBFS, and has no clipped samples.

## Why Qwen3-TTS 1.7B

Voicebox recommends Qwen3-TTS 1.7B for best overall zero-shot cloning quality. Qwen's Base model performs transcript-conditioned in-context cloning and officially supports clips as short as three seconds. The supplied 14.6-second clip is within Voicebox's recommended 10-30-second range. On this machine the loaded model used about 4.9 GB VRAM.

Primary sources:

- https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base
- https://github.com/QwenLM/Qwen3-TTS
- https://github.com/jamiepine/voicebox
- https://github.com/ysharma3501/LuxTTS
- https://github.com/resemble-ai/chatterbox

LuxTTS was the runner-up research choice for smooth 48 kHz output and low VRAM use. It was not selected for the final sweep because Qwen 1.7B fit the GPU, produced strong identity scores, and Voicebox's LuxTTS integration uses only the first five seconds of the reference prompt.

## Reference preparation

The source is centered stereo, so it was safely downmixed to mono. The primary reference trims only leading/trailing silence and preserves the high-frequency breath/noise bed. Aggressive denoising was avoided because it removes part of the ASMR character. See `references/analysis.md` and `references/spectrogram.png`.

Whisper transcript used for Qwen conditioning:

> Mmm, cute, cute, cute, cute. You're so cute, handsome. I love you so much, I love you. I mean, seriously, I can't get enough of you. I could just...

Voicebox profiles:

- Natural: `aefe696c-b18a-438c-aced-feaf0f2693a2`
- Band-limited: `b113807a-adcb-478b-9cc7-1d2ab0e106e5`

## Candidate ranking

Speaker similarity uses local Resemblyzer cosine similarity against the trimmed reference. Whispered speech is outside the encoder's ideal domain, so this is a ranking signal rather than a definitive perceptual score.

| Rank | Candidate | Cosine | Duration | Peak | Notes |
|---:|---|---:|---:|---:|---|
| 1 | Natural seed 31 | 0.8297 | 5.92 s | -2.77 dBFS | Best identity; high-frequency ratio nearly identical to reference |
| 2 | Band-limited seed 17 | 0.7999 | 5.12 s | -4.59 dBFS | Clean and concise |
| 3 | Natural seed 43 | 0.7814 | 6.72 s | -9.03 dBFS | Softest and smoothest, but less identity-faithful |
| 4 | Natural seed 17 | 0.7688 | 5.36 s | -7.38 dBFS | Initial smoke take |
| 5 | Paced seed 101 | 0.7348 | 6.40 s | -7.03 dBFS | Strong deliberate pauses; lowest identity score |

## Reproduce a take

Start Voicebox from the repository root:

```bash
backend/venv/bin/python -m backend.main \
  --host 127.0.0.1 --port 17493 \
  --data-dir /home/terryd/gooning/voicecloning/voicebox/data
```

Then generate from the natural profile:

```bash
curl --fail-with-body -X POST http://127.0.0.1:17493/generate/stream \
  -H 'Content-Type: application/json' \
  -d '{
    "profile_id":"aefe696c-b18a-438c-aced-feaf0f2693a2",
    "text":"You have done enough for today. Come closer, breathe slowly, and let yourself relax.",
    "language":"en",
    "engine":"qwen",
    "model_size":"1.7B",
    "seed":31,
    "normalize":false
  }' \
  -o take.wav
```

Voicebox's cloned Qwen backend does not expose or honor delivery instructions; meaningful variations come from the exact reference/transcript, seed, and punctuation.
