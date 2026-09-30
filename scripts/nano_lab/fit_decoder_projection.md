# Decoder projection adapter experiment

`fit_decoder_projection.py` is a bounded research helper. It loads only the
S3Gen flow encoder and meanflow estimator through the existing streamed loader.
It freezes every base parameter and the 192-value speaker embedding. It trains
a rank-two zero-initialised LoRA on
`flow.decoder.estimator.final_proj`, a `Conv1d(256, 80, 1)`. The adapter has
`2*(256+80)=672` trainable values. Its merged update can be folded into the
native or ONNX `final_proj.weight`, so deployment adds no module or model
memory.

The helper reuses the validated `mel_calibration` preparation directory and
`fit_decoder_embedding` alignment. It checks the original prepared conditionals
against an optional `--initial-conditionals` file. Every tensor and value must
match except `gen.embedding`. Native parity always uses the original prepared
embedding. If an initial fitted embedding is supplied, the adapter starts from
that embedding and distils toward its pre-adapter output; the report records
both embeddings and both conditionals hashes.

The objective is source-token reconstruction only:

``raw_mel_mse + 0.25 * band_envelope_mse + 0.5 * pre_adapter_distill_mse + 1e-4 * adapter_l2``

The source rows remain split by the preparation manifest. Epoch zero is evaluated
before any optimizer step and is always saved as `best_projection.pt`. Reports
include native parity, zero-adapter equality, finite nonzero LoRA gradients,
frozen-parameter checks, train/valid IDs, output delta RMS/max, hashes, and RSS.
The output is experimental. It cannot correct T3 token errors, pitch timing, or
cadence by itself.

Run model work under the repository guard. For a one-step smoke check:

```bash
python scripts/nano_lab/bounded_job.py -- \
  .venv-nano/bin/python scripts/nano_lab/fit_decoder_projection.py \
  --device cuda --max-steps 1 \
  --output-dir artifacts/nano_lab/decoder_projection_fit_smoke
```

The checkpoint schema is `nano_decoder_projection_lora_v1`. The top-level
tensors are `down_weight [2,256,1]`, `up_weight [80,2,1]`,
`merged_delta_weight [80,256,1]`, and `base_target_weight [80,256,1]`. The
target key is `flow.decoder.estimator.final_proj.weight`. The runtime must
verify the target shape, base weight hash, S3Gen checkpoint hash, conditionals
hash, rank, alpha, and delta before folding once.
