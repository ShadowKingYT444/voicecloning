# Whisper-fast + VAD package

This is the low-latency STT half of the local voice stack. It ships the
75 MB `faster-whisper-tiny.en` CTranslate2 model and a CPU-only WebRTC VAD.
The default path is English speech, one CPU worker, INT8 CTranslate2 compute,
greedy decoding, and VAD-gated inference.

## Quick start

```bash
python transcribe.py ../artifacts/references/mommy_trimmed_mono_24k.wav
python transcribe.py recording.wav --json --timestamps
```

Python API:

```python
from whisper_fast_vad import WhisperFastVAD

stt = WhisperFastVAD(compute_type="int8", cpu_threads=1)
text = stt.transcribe_file("recording.wav")

# For microphone integration, feed 16 kHz float32 chunks as they arrive.
for result in stt.stream_transcribe(microphone_chunks, sample_rate=16_000):
    print(result.text)
```

## Latency design

- WebRTC VAD removes silence before Whisper sees it and merges short gaps with
  240 ms padding, which preserves whispered/soft speech better than a hard
  threshold.
- CTranslate2's `tiny.en` INT8 model is loaded once and reused.
- `beam_size=1`, `best_of=1`, `temperature=0`, and
  `condition_on_previous_text=False` minimize decoder work and cross-chunk
  prompt state.
- No timestamps or word timestamps are computed on the fast path.
- Audio is resampled once to 16 kHz mono and long speech is bounded to 30-second
  windows.
- `cpu_threads` is exposed because one thread is generally best for a single
  realtime stream, while 2–4 can help batch transcription.

The previous Transformers Whisper path remains useful as a larger accuracy
fallback. Do not enable eager PyTorch dynamic quantization here: it produced
repeated-token output in validation, so this package uses the tested
CTranslate2 INT8 implementation instead.

## Verification

Run the reproducible benchmark from the workspace root:

```bash
python scripts/benchmark_whisper_fast_vad.py
```

The benchmark records model load, VAD segmentation, transcription latency,
audio/speech duration, real-time factor, RSS, and the transcript. Results are
stored under `benchmarks/`.

Measured locally on the 13.5-second Mommy reference:

| Path | Model footprint | RSS after load | Transcribe time | RTF |
| --- | ---: | ---: | ---: | ---: |
| This package | 75 MB, tiny.en INT8 | 307 MB | 2.003 s with VAD | 0.100 |
| Existing Transformers path | 923 MB, Whisper Small FP32 | 1,773 MB | 13.346 s | 0.989 |

The tiny path produced a coherent transcript and was approximately 6.7× faster
with 5.8× lower RSS in this run. On a 73.5-second test containing 60 seconds
of silence, VAD reduced decode time from 3.968 s to 2.003 s (1.98×), while
keeping the speech segment plus padding.

The ASMR7 smoke transcription is saved in
`benchmarks/asmr7_transcription.txt` (3.161 s for 30 seconds of audio, RTF
0.105).

The streaming API was also smoke-tested with 100 ms chunks; it emitted one
finalized transcription event after 1.588 s on the 13.5-second reference.

## Files

- `whisper_fast_vad.py`: reusable batch and streaming API
- `transcribe.py`: command-line wrapper
- `models/tiny.en/`: offline CTranslate2 model files
- `benchmarks/`: measured evidence
- `requirements.txt`: runtime dependencies
