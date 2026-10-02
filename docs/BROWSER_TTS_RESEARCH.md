# Browser TTS experiment

## Result from research

Browser TTS is feasible, but browser-side zero-shot cloning is not a project requirement. The current objective is a selected ASMR voice loaded from a precomputed voice state, with streamed, short-batch reading of Federalist No. 10 and runtime memory below 500 MiB. Inference may run in a local service while the browser handles text, playback, and controls. The fixed-voice Nano browser path remains an experiment. Its three-session architecture has not yet been measured against the memory target. See [continuation evidence and pending gates](BROWSER_CONTINUATION.md).

Resemble AI describes Chatterbox Nano as a 110 million parameter English model. Its model card reports about 3x real-time on an eight-core CPU. That is an upstream claim for its supported PyTorch runtime. It does not predict browser speed. [Nano model card](https://huggingface.co/ResembleAI/chatterbox-nano)

Resemble AI's official browser demo uses Transformers.js with a larger Chatterbox ONNX model. It runs inference in a Web Worker, encodes a voice reference once, and uses a quantized language-model graph. This proves that the Chatterbox family can run in a browser. It does not show that the Nano conversion or this repository's fitted profile has the same quality or speed. [Official browser demo](https://github.com/resemble-ai/transformersjs-chatterbox-demo)

Kyutai Pocket TTS has an incremental audio generator and accepts saved voice states. The repository's ONNX implementation can load the state at startup and stream decoded audio chunks. Existing local measurements do not meet the memory target: the int8 ONNX model files total 559 MiB on disk; a benchmark measured 798.1 MiB RSS after load and 1,921.0 MiB after conditioning with a 30-second ASMR reference. Its saved voice state avoids re-encoding that reference, but does not avoid loading the model. A smoke test measured 1.11 seconds to its first 0.16-second audio chunk. This is useful streaming evidence, not proof of low-memory or smooth full-text playback. The measured Pocket state uses `asmr7_30s_loud_reference.wav`, not the browser prototype's `asmr_t3_seed47_fit.wav`; voice equivalence is unverified. Upstream Pocket documentation describes a 100 million parameter model and reports a 200 ms first chunk on its supported Python path. That upstream figure is not reproduced by the local ONNX benchmark. The project lists community browser ports, but does not provide official browser support. [Pocket TTS upstream project](https://github.com/kyutai-labs/pocket-tts)

The selected Nano export is an independent conversion. Its maintainer publishes four graphs with mixed precision: FP16 token embeddings, Q4F16 speech encoder, Q4F16 autoregressive language model with a KV cache, and a Q4 conditional decoder. The four graph files and external weight files total about 543 MiB. Tokenizer and configuration files bring the full first download to about 547 MiB. The app pins revision `4a66d7dab72a9e98f24b515d49a1d7a81632df2e`; the repository also publishes SHA-256 digests for its files. Its report describes a successful WebGPU smoke run on one Windows computer with an RTX 3060. It also says that broad quality, compatibility, reliability, and speed have not been established. The package is not a ready-made browser app or a generic Transformers.js pipeline. [Pinned Nano ONNX conversion and limitations](https://huggingface.co/owensong/chatterbox-nano-ONNX/tree/4a66d7dab72a9e98f24b515d49a1d7a81632df2e)

## Inference path

The published conversion includes four ONNX graphs. The updated browser reader loads only the last three after an offline voice export:

1. Offline, the speech encoder converts the selected reference WAV into audio features, speech tokens, and speaker features. These exact tensor bytes are serialized as the fixed voice state. The reader does not load this encoder.
2. The tokenizer converts each text chunk into GPT-2 token IDs. The embedding graph maps those IDs to vectors.
3. The autoregressive language model emits speech tokens. Each decode step updates 24 key/value cache tensors across 12 layers.
4. The conditional decoder converts the prompt tokens and the complete generated speech-token sequence into mono 24 kHz audio.

I parsed all four published ONNX graph headers at the pinned revision. The language model accepts 768-wide embeddings, an int64 attention mask and position IDs, and 24 FP16 cache tensors with shape `[batch, 12, sequence, 64]`. It returns 6,563 logits per token plus the 24 updated cache tensors. The encoder returns 768-wide audio features, audio tokens, 192-wide speaker embeddings, and speaker features. The decoder accepts the reference audio tokens followed by generated speech tokens and returns a float waveform. The external-weight location embedded in each graph is the weight filename alone; ONNX Runtime Web also needs the corresponding remote URL. This distinction is enforced in the browser runner.

The model's acoustic decoder returns a complete waveform. It does not expose an incremental waveform stream. This site therefore divides the passage at clause boundaries, with a 24-word maximum. It sends the next chunk while the current chunk plays. Playback can begin after the first chunk finishes. The first chunk is not audible while its speech tokens are still being generated. Independent chunks can add pauses or prosody changes. The site marks each boundary so these effects can be measured and heard.

The offline voice-state exporter uses the exact `asmr_t3_seed47_fit.wav` artifact as its speaker reference. Its SHA-256 is `67f94a868976b22a47bce8fd00a873d4c5b7085ba6eedd698f8a898a26ce76c0`, matching the source artifact. It does not load the local fitted adapter. The browser export is the community's base Nano conversion. Voice similarity from a generated reference is not equivalent to using the fitted model weights, and it has not been measured for this browser path.

## Browser runtime and optimization

ONNX Runtime Web supports WebAssembly on CPU and WebGPU on compatible browsers. WebGPU supports only a subset of ONNX operators. A session can assign supported subgraphs to WebGPU and use another provider for remaining work. This experiment requests WebGPU and records whether its three runtime sessions load. It does not treat a successful session load as proof that all operations ran on the GPU. [ONNX Runtime Web overview](https://onnxruntime.ai/docs/tutorials/web/)

Autoregressive TTS reuses the KV cache at each speech-token step. ONNX Runtime Web can keep tensor outputs in GPU buffers and use them as later inputs. The worker keeps those cache tensors on GPU and reads logits on the CPU for sampling. This avoids copying the full cache after each token. The saved conditioning, small embedding outputs, masks, token IDs, and final waveform still cross the CPU/GPU boundary. [WebGPU I/O binding](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html)

ONNX external weight files need explicit URLs in browser session options. The site loads the graph and its matching external data from one pinned Hugging Face revision. This preserves each graph's weight offsets. The reader fetches each external weight file as a Blob and passes it to the JSPI build. ORT 1.30 local source (`lib/wasm/wasm-core-impl.ts`) bypasses `loadFile()` for Blob external data when JSPI is enabled. Its API documentation describes range loading for this case. The code change can avoid a full JS ArrayBuffer weight copy during initialization. It does not establish total browser memory savings. The three graph/weight pairs total 374.14 MiB. Tokenizer and configuration files bring the model assets to 377.54 MiB. The browser may cache these files for later visits. Browser cache retention depends on available storage. [ONNX Runtime Web external data](https://onnxruntime.ai/docs/tutorials/web/large-models.html)

The site moves inference to a Web Worker so model loading and synthesis do not block the page. It reports model-load duration, estimated time until scheduled playback, the latest passage synthesis time, total generated audio duration, and mean real-time factor. The page records scheduled playback gaps. The new isolated [measurement runner](../browser_tts/MEASUREMENT.md) records browser process-tree RSS/PSS and requested GPUBuffer sizes. The runner still needs a guarded browser run. Requested buffer size does not prove physical VRAM residency. It does not claim a two-second first-audio time until this browser has measured it. The reader keeps at most two unplayed passages and persists PCM audio in IndexedDB. Chrome can stream the final mono PCM WAV to a file without retaining all Float32 waveforms in RAM. It applies a short fade at each passage edge to reduce clicks; sentence and prosody discontinuities remain.

## Local measurements and limits

The repository's native CUDA batch report confirms the figures in the supplied brief for one fitted Nano experiment: warm repeat generation took 2.95 seconds for 5.48 seconds of audio. Cold generation took 20.84 seconds. These are native ONNX Runtime CUDA measurements, not browser measurements. The warm report used the repository's speaker-specific ONNX profile and is not a result for the public Q4/Q4F16 browser conversion. [Local batch report](../artifacts/nano_lab/onnx_batch_asmr_fitted_cached/batch_report.json)

The selected reference is a 3.52-second, 24 kHz mono generated clip. The September 30 fit report says the ASMR T3 adapter reduced validation loss but lowered speaker similarity on six matched new-text samples. The adapter was not promoted. Human listening remains necessary. [Selected sample and experiment decisions](../artifacts/voice-experiments-20260930/README.md)

The host uses Chrome 152, an RTX 4060 Laptop GPU, and NVIDIA driver 595.91.07. ONNX Runtime WebGPU support on this Linux driver stack must be tested directly. A model file tag or a successful browser adapter query is not an inference result.

The existing resource guard requires at least 6 GiB available before a full model job and preserves 4 GiB for the desktop. Model load, synthesis, and speech-quality evaluation must run one at a time through `scripts/nano_lab/bounded_job.py`. Smaller component jobs preserve the same reserve. The guarded CPU voice-state export completed under a 640 MiB cap. Its measured cgroup peak was 423.7 MiB. That measurement is not browser RSS or final speech-quality evidence. Full browser inference still requires sufficient host headroom. See the continuation note for completed checks and failed browser startups.

## References

- [Chatterbox Nano upstream model card](https://huggingface.co/ResembleAI/chatterbox-nano)
- [Independent quantized Nano ONNX conversion](https://huggingface.co/owensong/chatterbox-nano-ONNX)
- [Resemble AI Transformers.js browser demo](https://github.com/resemble-ai/transformersjs-chatterbox-demo)
- [ONNX Runtime Web deployment guide](https://onnxruntime.ai/docs/tutorials/web/)
- [ONNX Runtime Web WebGPU provider and GPU-buffer I/O](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html)
- [ONNX Runtime Web external model data](https://onnxruntime.ai/docs/tutorials/web/large-models.html)
