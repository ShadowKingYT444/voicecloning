# Long-context T3 numerical mismatch

Read-only investigation. No model execution in this investigation.

Evidence: `onnx_staged/t3/extended_verification.json`, ORT 1.29 CPU float32, sequential, 1 intra/inter-op thread, optimization all. L32/L128 prefill/decode and L400 single-token decode pass. L400 prefill fails 176 of 7,372,800 cache values at atol=rtol=3e-4. First combined-gate failure output11 (layer5 key), max .000813961. Maximum output19 (layer9 key) .000988960, relative L2 7.63e-6. Logits max error 1.19e-6. Argmax coordinates were not saved.

Manual export attention (`onnx_t3_core.py`) uses dynamic causal Range/LessOrEqual, float32 QK, softmax and AV. Native Transformers reference uses default SDPA. Wrapper/native L400 drift already grows from .000138 layer3K to .000370 layer9K, within the original gate. ORT adds drift. Optimization disabled also fails (max .001031876). No fixed context cap found. Leading hypothesis is accumulation of floating-point kernel/reduction-order differences, not a proven cause.

Next discriminating diagnostics: native eager attention reference; expose per-block hidden/K/V with query/head/channel error coordinates; compare the SAME saved L400 NPZ on CUDA ORT with use_tf32=0 and CPU. Preserve current gate. Sequential token prefill is only an unverified fallback and adds many session calls. Keep --experimental-t3 required.
