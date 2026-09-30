# Pocket TTS ONNX package

This folder is a self-contained CPU-oriented export of the local Pocket TTS
English voice-cloning checkpoint. The default runtime uses ONNX Runtime with
INT8 MatMul-weight quantization, sequential execution, full graph fusion, a
single intra-op thread, explicit streaming KV/cache state, and a cached voice
conditioning state. FP32 graphs remain beside the INT8 graphs for fallback and
quality comparisons.

The promoted reference is `voices/asmr7_30s_loud_reference.wav`; the preferred
render made with `temperature=0.2` is `voices/asmr7_promoted_temp02.wav`.

## Use from Python

```python
from pathlib import Path
from pocket_tts_onnx import PocketTTSOnnx

root = Path(__file__).resolve().parent
tts = PocketTTSOnnx(temperature=0.2, precision="int8", intra_op_num_threads=1)
audio = tts.generate(
    "Hello from the realtime Pocket TTS package.",
    voice=root / "voices/asmr7_30s_loud_reference.wav",
)
tts.save_audio(audio, root / "example.wav")
```

The first call for a new reference performs Mimi encoding and conditioning.
Subsequent calls reuse the in-process voice state. The included
`voices/asmr7_30s_loud_state.npz` is already conditioned for the promoted
reference; pass that file as `voice` to avoid reference conditioning on startup.
For another voice, persist it once with
`tts.save_voice_state(reference, "voice_state.npz")` and pass the NPZ on future
starts.

CLI:

```bash
python generate.py "Hello from Pocket TTS." \
  voices/asmr7_30s_loud_reference.wav output.wav --temperature 0.2 --threads 1
```

`stream()` yields decoded PCM chunks as soon as the first latent frames are
available, which is the intended integration point for realtime playback.

## Bundle layout

- `onnx/english_2026-04/flow_lm_main*.onnx`: stateful transformer backbone
- `onnx/english_2026-04/flow_lm_flow*.onnx`: stateless flow step
- `onnx/english_2026-04/mimi_encoder*.onnx`: reference-audio encoder
- `onnx/english_2026-04/mimi_decoder*.onnx`: streaming neural codec decoder
- `onnx/english_2026-04/text_conditioner*.onnx`: SentencePiece text embedding
- `bundle.json`, `tokenizer.model`, `bos_before_voice.npy`: runtime metadata

## Performance choices

The export keeps the model split into small graphs so flow steps and codec
frames can run incrementally. The runtime reuses sessions and conditioning,
uses ORT's `ORT_ENABLE_ALL` graph optimization level, sequential execution,
CPU memory arenas/memory patterns, and one thread by default to reduce
tail-latency contention. `--threads 0` delegates thread selection to ORT; tune
this per host and keep the selected value in the benchmark record.

The INT8 files are dynamic MatMul quantizations and are the default on CPU.
The quantizer reduced the five graph files from 419.1 MB to 139.3 MB
(66.7% on-disk reduction for this export). The full package also contains
the FP32 fallback graphs, so the distributable directory is larger than the
runtime's default working set.

## Verification and benchmark

The exporter compared every Mimi encoder/decoder state and FlowLM state output
against PyTorch within tolerance before quantization. Run the local comparison
from the workspace root:

```bash
python scripts/benchmark_pocket_onnx.py
```

The resulting JSON records model-load, reference-conditioning, warm synthesis,
audio duration, real-time factor, and resident-set memory for Pocket and the
existing `kokoro_package` on the same fixed passage.

Latest local evidence (16 logical CPU threads available; both rows below were
measured with one inference thread and the same 113-character passage):

| Runtime | Provider / precision | Warm mean | Audio produced | Audio/sec | RTF | RSS after load |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Pocket package | ONNX Runtime CPU / INT8 | 5.742 s | 4.32 s | 0.752× | 1.329 | 798 MB |
| Kokoro package | PyTorch CPU / FP32 | 12.632 s | 7.625 s | 0.604× | 1.657 | 1,781 MB |

On this CPU measurement Pocket has 1.245× higher audio throughput and 0.455×
the warm latency. Its one-time 30-second reference-conditioning pass was
15.217 s and reached 1,921 MB RSS after chunking (the included NPZ state avoids
that pass on startup). The existing Kokoro GPU benchmark is kept separately in
`benchmarks/kokoro_existing_cuda.json`; it reports 89.54× audio throughput on
CUDA/autocast FP16, so it is not a like-for-like provider comparison with the
CPU rows above.

An INT8-vs-FP32 deterministic smoke check is recorded in
`benchmarks/pocket_int8_vs_fp32_smoke.json`; both paths execute successfully,
with INT8 taking 3.128 s versus 4.968 s for that short utterance on one CPU
thread.

The streaming smoke record in `benchmarks/pocket_stream_smoke.json` yielded its
first 160 ms PCM chunk after 1.112 s with the precomputed ASMR7 state.

## Attribution

The ONNX graph split and runtime structure are derived from KevinAHM's
`pocket-tts-onnx-export` / `pocket-tts-onnx` implementation. See
`LICENSES.md` for upstream links and model terms.
