# Mommy ASMR reference inspection

Source: `/home/terryd/Downloads/mommy-asmr.mp3` (original preserved; not edited in place).

## File and timing

- MP3, 44,100 Hz, stereo, ~151 kb/s, 14.627 s (starts at an MP3 delay of ~25 ms).
- Silence detection (`silencedetect`, -45 dB, 150 ms) finds leading silence 0.000–0.736 s and trailing silence 14.228–14.499 s. A practical reference region is **0.75–14.25 s** (13.50 s).
- At a conservative -20 dB threshold there is also a quiet gap around 9.84–10.48 s and the ending becomes quiet after ~13.49 s. This suggests an uninterrupted main vocal take with a brief pause, not multiple clean isolated utterances.

## Level / channel observations

- Integrated loudness: **-13.5 LUFS**, LRA 7.5 LU; peak about **-1.3 dBFS**, RMS about **-17.4 dBFS**.
- L/R are effectively the same centered recording: L-R difference RMS is ~-53 dBFS versus mono RMS ~-17.35 dBFS. Downmixing to mono is safe and avoids presenting a model with redundant stereo channels.

## Spectral / acoustic observations

- `spectrogram.png` shows strong low-frequency voiced/formant structure plus a persistent broadband high-frequency bed extending roughly through 15–16 kHz. This is consistent with an intimate/whispered ASMR recording and audible room/mic noise; do **not** aggressively low-pass or denoise if preserving the original breath/noise character is desired.
- The voiced energy is non-stationary, with pauses and consonant/breath transients; no reliable single F0 can be reported from this whisper-like material. Pitch/prosody should be matched by conditioning on the full cleaned reference, rather than attempting pitch shifting.
- The MP3 encode is already lossy. Generated references are PCM WAV to avoid another codec stage.

## Prepared candidates

All are mono PCM WAV at 24 kHz, which matches Voicebox/Qwen's expected reference format. The original MP3 remains untouched.

| File | Use |
|---|---|
| `mommy_raw_mono_24k.wav` | Full source, only stereo downmix + resample; retains all silence/context. |
| `mommy_trimmed_mono_24k.wav` | Recommended first reference: 0.75–14.25 s, no leading/trailing silence. |
| `mommy_trimmed_bandlimited_mono_24k.wav` | Same region with gentle 60 Hz high-pass and 18 kHz low-pass to remove subsonic/ultrasonic junk. |
| `mommy_trimmed_denoised_mono_24k.wav` | Experimental only: same gentle band limits plus FFmpeg `afftdn` (nr=6); compare by ear because denoising can erase ASMR breath texture. |
| `mommy_core_0.75-9.80_mono_24k.wav` | Main continuous section before the quiet 9.84 s pause; useful when a model benefits from a single uninterrupted prompt. |

## Transcript status

No local ASR runtime/model was available during this inspection (only FFmpeg was present), so I did not invent a transcript. Voicebox's Whisper endpoint or its bundled Whisper model should transcribe the recommended trimmed file; pass the exact transcript alongside the reference for Qwen's “ultimate” cloning path. The 13.5 s cleaned file is within Voicebox's documented ~5–15 s reference window.

## Recommendation

Start with `mommy_trimmed_mono_24k.wav`; retain the natural high-frequency breath/noise bed. Use the bandlimited file as an A/B candidate, and treat the denoised file as a fallback rather than the default. Keep generated TTS dry during model comparison; add ASMR effects only after selecting the closest clone.
