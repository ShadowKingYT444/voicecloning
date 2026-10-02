# Experimental Q4 embedding candidate

This note describes one offline experiment. It converts the two constant token tables in the pinned public Nano ONNX graph to Q4. It does not change the default browser model. It does not promote a voice or a runtime profile.

The source revision is [`4a66d7dab72a9e98f24b515d49a1d7a81632df2e`](https://huggingface.co/owensong/chatterbox-nano-ONNX/tree/4a66d7dab72a9e98f24b515d49a1d7a81632df2e) from `owensong/chatterbox-nano-ONNX`. The graph header has two FP16 tables:

| Table | Shape | Source Gather axis |
|---|---:|---:|
| `text_emb.weight` | `[50276, 768]` | `0` |
| `speech_emb.weight` | `[6563, 768]` | `0` |

The graph file is 1,520 bytes. Its external data file is 87,304,704 bytes. The exporter pins these SHA-256 values:

| File | SHA-256 |
|---|---|
| `SHA256SUMS` | `ecd0b0ee1fde0f9342564fda72fe1ebb4907baa6c9d013c09052bcad5220dcf4` |
| `onnx/embed_tokens_fp16.onnx` | `019d257243774091d78c2ad91c2c0f61e4e442740cb7b3b00b5a89109417b18d` |
| `onnx/embed_tokens_fp16.onnx_data` | `bcd7b35ae4f206932e2491cb60b42ebb80f6d8facfdb53ba7d7449ad00a3237b` |

It checks the `SHA256SUMS` file and both model file hashes before it reads the graph weights. It checks the source hashes again after the export and CPU comparison.

## Quantizer settings

The tool uses the installed ONNX Runtime `MatMulNBitsQuantizer` with the default round-to-nearest algorithm, a QOperator Gather, 4-bit asymmetric weights, and block size 128. It uses `quant_axes={"Gather": 1}`. The source `Gather.axis=0` selects a token row. The quantizer axis is a separate setting. Axis 1 is the 768-value embedding width. Its size divides evenly into six 128-value blocks per token.

The installed ONNX Runtime source describes constant FP16 or FP32 weights for this quantizer. It restricts Gather weight-only quantization to 4 bits and QOperator format. ONNX Runtime's quantization guide uses `Gather:1`, upgrades models below opset 21 for 4-bit types, and requires ONNX Runtime 1.20 or newer to run `GatherBlockQuantized`. The installed CPU environment exposes ONNX Runtime 1.29.0. The exporter checks the source dtype and reports a clear error if the quantizer API is unavailable.

- [ONNX Runtime Int4/UInt4 quantization guide](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
- [ONNX Runtime MatMulNBitsQuantizer source](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/quantization/matmul_nbits_quantizer.py)
- [ONNX Runtime WebGPU provider guide](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html)

The source graph uses a hybrid embedding input. It sends all but the last two IDs to the text table. It sends the last two IDs to the speech table. The sentinel ID `50256` maps to start-speech ID `6561`. The CPU check uses this input shape when it scans each table. It checks all text and speech rows in batches of 32. It also records fixed boundary and seeded random token probes, plus the `[50256, 50256]` initial tail.

The CPU check reports maximum absolute error, mean absolute error, and root mean square error. It holds one model session at a time and streams reference outputs through a temporary file. It compares exactly `50276 × 768` text values and `6563 × 768` speech values. The report is descriptive. It does not set an acceptance threshold. It does not test WebGPU support. The browser package uses ONNX Runtime Web 1.30.0. A successful CPU check does not establish support or correctness on WebGPU.

## Restore the pinned source files

Download only the checksum file, graph, and external data. The exporter checks every downloaded byte against the pinned hashes.

```bash
python3 scripts/nano_lab/bounded_job.py --max-memory-mib 640 -- \
  python3 browser_tts/scripts/download-model.py --embedding-only
```

## Build the local candidate

Run the export through the existing resource guard. `--small-job` caps the job at 1,280 MiB and requires at least 5,376 MiB available at launch. The guard preserves the 4 GiB desktop reserve. Run this job alone. Do not lower its cap or bypass the guard.

```bash
python3 scripts/nano_lab/bounded_job.py --small-job -- \
  .venv-nano-cpu/bin/python browser_tts/scripts/quantize-embedding.py \
  --source-dir models/chatterbox-nano-browser \
  --output-dir browser_tts/public/experiments/embedding_q4_candidate
```

The exporter refuses an existing output directory. Use a new output path for another candidate. It writes:

- `embed_tokens_gather_q4.onnx`
- `embed_tokens_gather_q4.onnx.data`
- `manifest.json`

The manifest schema is `browser_tts.embedding-q4-candidate/v1`. It records source and output hashes, file sizes, quantization settings, CPU component errors, and the status `experimental_unpromoted`. A completed CPU check means only that the measurement ran. It is not a browser or quality gate. If the CPU check cannot run, the exporter keeps the candidate and records the failure in the manifest. The command returns a non-zero status in that case.

The local browser candidate URL is:

```text
/experiments/embedding_q4_candidate/manifest.json
```

## Matched browser comparison

The browser experiment needs an existing Vite server on the project's pinned port `4187`. Check the port registry first. Reuse a healthy server for this project. If no server is active, start `npm run dev` in `browser_tts`; its Vite config uses `127.0.0.1:4187` with `--strictPort`.

Run the FP16 baseline first. Then run the Q4 candidate. Use the same text and seed in both runs. Run one measurement at a time through the normal 3 GiB guard. It requires 6 GiB available at launch and stops below 4 GiB available. `--nvidia-smi` records compute-process GPU memory when that tool is available. It does not measure every graphics allocation.

```bash
python3 scripts/nano_lab/bounded_job.py -- \
  python3 browser_tts/scripts/measure-browser.py \
  --hardware-webgpu --power-preference low-power \
  --model-base /models/chatterbox-nano-browser/ \
  --text "Take a slow breath in, and let your shoulders relax." \
  --seed 1337 --nvidia-smi

python3 scripts/nano_lab/bounded_job.py -- \
  python3 browser_tts/scripts/measure-browser.py \
  --hardware-webgpu --power-preference low-power \
  --model-base /models/chatterbox-nano-browser/ \
  --embedding-manifest /experiments/embedding_q4_candidate/manifest.json \
  --text "Take a slow breath in, and let your shoulders relax." \
  --seed 1337 --nvidia-smi
```

The runner saves JSON reports and WAV files. Compare generated speech token sequences, process RSS/PSS, load time, synthesis latency, and GPU samples across the two reports. Use an independent transcript check to verify the words in both WAVs. Listen to the matched WAVs. Record failed loads or changed words. Do not infer quality from a completed run, lower file size, or an error metric alone. Q4 remains experimental until browser compatibility, matched content, measured memory and latency, and listening checks support promotion.

## Local export result

The guarded export completed. The source was `models/chatterbox-nano-browser`.
The service took 19.211 seconds and had a 472 MiB cgroup peak. External data
fell from 87,304,704 bytes to 22,678,761 bytes. The graph is 2,324 bytes.
The [retained manifest](../artifacts/nano_lab/browser_embedding_q4_20261001/manifest.json)
records the hashes and measurements for every row:

| Table | Rows | Maximum absolute error | Mean absolute error | RMS error |
|---|---:|---:|---:|---:|
| Text | 50,276 | 0.0552063 | 0.0110672 | 0.0132516 |
| Speech | 6,563 | 0.1845703 | 0.0510274 | 0.0593740 |

These are CPU component results. Browser inference, generated-word accuracy,
final audio, matched latency, and listening are pending. The candidate remains
unpromoted. Recheck available memory before each job. Full browser measurement
still requires 6 GiB available.
