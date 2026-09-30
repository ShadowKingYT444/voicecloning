# Acoustic CUDA discrepancy

CPU ORT 1.29 matches strict native Torch at five mel lengths (8, 64, 256, 512, 768), maximum error 4.94e-5. CUDA ORT 1.26 with use_tf32=0 matches length 8 but differs by approximately 0.02 at the other four lengths. The unchanged tolerance is atol=rtol=3e-4. These failed results cannot authorize CUDA inference.

A separate process with NVIDIA_TF32_OVERRIDE=0 failed before numerical comparison. cuDNN 9.19 reported HEURISTIC_QUERY_FAILED at the first convolution. The failing padded input was [1,320,258,1]. No JSON result was published by that older verifier; the log is the evidence.

The installed-release [ORT v1.26 convolution source](https://github.com/microsoft/onnxruntime/blob/v1.26.0/onnxruntime/core/providers/cuda/nn/conv.cc) first builds cuDNN frontend plans, then filters tensor-core numeric notes when use_tf32 is false. If support or plan building fails, its fallback retries without fusion and with use_tf32=true. This provides a concrete explanation to test for reduced precision despite the requested setting; it does not prove that this branch caused the observed error.

The [ORT CUDA provider documentation](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html) describes alternative convolution algorithm search and one-dimensional padding layout settings. The diagnostic script tests DEFAULT plan selection and the optional NC1D layout in isolated guarded processes. It never publishes a production verification gate. Global environment and default runtime options remain unchanged until actual reference checks pass.

Next test: DEFAULT with pad0 and NVIDIA_TF32_OVERRIDE unset. If needed, compare pad1 and then an explicit per-process TF32 override with DEFAULT. Do not relax tolerances. If a configuration passes, integrate that exact configuration and reverify through the production stage before enabling it.
