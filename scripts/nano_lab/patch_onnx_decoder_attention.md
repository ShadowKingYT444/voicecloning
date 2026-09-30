# Static ONNX decoder-attention folding

`patch_onnx_decoder_attention.py` folds the fitted
`nano_decoder_attention_lora_v1` factors into the external weight sidecar for
the staged `meanflow_estimator` graph. It does not load an ONNX graph with
`onnx`, construct a PyTorch model, or add an inference module.

The patcher requires the canonical 224 targets, rank 2, alpha 2, checkpoint
scale 1, the matching S3Gen checkpoint SHA-256, the matching inventory
SHA-256, and every per-target base-weight SHA-256. The native target name
`flow.decoder.estimator.<...>` maps to the exported state name
`estimator.<...>`. The exporter writes each tensor in its original PyTorch
`[out,in]` shape, so the patcher writes `base + strength * (up @ down)` without
a transpose. The graph bytes remain unchanged.

Run the patch into a new model root:

```bash
.venv-nano-cpu/bin/python scripts/nano_lab/patch_onnx_decoder_attention.py \
  --base-model-dir artifacts/nano_lab/onnx_staged \
  --adapter artifacts/nano_lab/decoder_attention_fit/best_attention.pt \
  --strength 1 \
  --output-model-dir artifacts/nano_lab/onnx_staged_decoder_attention
```

The output copies the meanflow stage and streams the 293 MB external sidecar
one span at a time. Unchanged stages can be relative symlinks. The input model
root and its sidecar are never modified. An existing meanflow
`adapter_patch` or `decoder_attention_verification` field is rejected, which
prevents cumulative folding.

The meanflow stage manifest contains this patch contract:

- `adapter_patch.format`: `nano_decoder_attention_merged_v1`
- `adapter_sha256`, `inventory_sha256`, `model_checkpoint_sha256`, and
  `initial_conditionals_sha256`
- `strength`, `rank`, `alpha`, `target_count`, and `target_parameter_count`
- `base_weights_sha256` and `patched_weights_sha256`
- stage-relative `weights_file` and `external_report_file`, with
  `external_report_sha256`
- unchanged `graph_sha256`, changed spans, and source patcher hash

The original numerical result is moved to `base_verification`. The ordinary
`verification` field and the new `decoder_attention_verification` field are
set to `pending`. A separate Torch-reference versus ORT verifier must update
`decoder_attention_verification` to `verified` before normal runtime loading.
That verifier must compare the adapted estimator outputs, check the graph and
sidecar hashes again, and record the exact adapter SHA-256 and strength.

The pure tests use tiny synthetic protobuf metadata and sidecars. They verify
external locations, offsets, shapes, hashes, factor arithmetic, graph-byte
identity, non-target-byte identity, atomic failure, and cumulative-patch
rejection. They do not load Nano or ONNX Runtime.
