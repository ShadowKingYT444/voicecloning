# Static Nano T3 adapter patch

`patch_onnx_adapter.py` merges a `nano_t3_lora_checkpoint_v1` checkpoint into
the existing staged Nano T3 ONNX graph. The merge targets the inline FP32
`TensorProto.raw_data` payloads for GPT-2 `Conv1D` attention projections. The
script copies the graph as a byte stream, changes only the selected payloads,
and writes a new model directory. The baseline directory is never changed.

For a GPT-2 projection, the checkpoint stores `lora_A` as `[rank, in]` and
`lora_B` as `[out, rank]`. HF `Conv1D` stores its base weight as `[in, out]`.
At evaluation time the merged matrix is:

```text
W_merged = W + (alpha / rank) * scale * (lora_A.T @ lora_B.T)
```

Dropout is disabled by evaluation mode. Bias initializers are unchanged. The
mapping is exact:

```text
h.7.attn.c_attn -> t3.tfmr.h.7.attn.c_attn.weight  [768, 2304]
h.7.attn.c_proj -> t3.tfmr.h.7.attn.c_proj.weight  [768, 768]
```

The layer index and projection name vary for each checkpoint module. The
patcher supports attention modules (`c_attn`, `c_proj`) from the staged 12
layer graph, including 24 rank-4 ASMR projections and 24 rank-2 Harvey
projections across 12 layers. MLP adapters and other graph layouts fail closed.

Inspect graph metadata without reading tensor payloads:

```bash
.venv-nano-cpu/bin/python scripts/nano_lab/patch_onnx_adapter.py \
  --base-model-dir artifacts/nano_lab/onnx_staged --inspect
```

Create a patched graph directory:

```bash
.venv-nano-cpu/bin/python scripts/nano_lab/bounded_job.py --small-job \
  .venv-nano-cpu/bin/python scripts/nano_lab/patch_onnx_adapter.py \
    --base-model-dir artifacts/nano_lab/onnx_staged \
    --adapter artifacts/nano_lab/adapter_aligned_all_attn.pt \
    --scale 1 \
    --output-model-dir artifacts/nano_lab/onnx_staged_adapter_aligned_all_attn
```

The output T3 manifest records the adapter hash, base and patched graph hashes,
effective scale, module names, byte offsets, before/after payload hashes, and
the patcher source hash. External embedding tables and non-T3 stages are
read-only relative symlinks to the baseline. The patched T3 verification field
is `not_run`; a fresh ORT run must compare logits and KV outputs against a
Torch LoRA reference before the patched profile is enabled.

The profile loader must reject a nonzero adapter unless the selected T3
manifest contains `adapter_patch` with matching adapter SHA256, `scale`, base
graph SHA256, and patched graph SHA256. Mel calibration is separate. It stays
in the profile as `mel_calibration`, `mel_calibration_strength`, and its
existing `nano_mel_calibration_fit_v1` report; this patcher does not apply
or embed the mel delta.
