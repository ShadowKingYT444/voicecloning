# Nano decoder attention LoRA experiment

`fit_decoder_attention.py` fits a speaker-specific rank-two LoRA on the 224
self-attention projections in the Nano S3Gen meanflow decoder.  The targets
are every `attn1.to_q`, `to_k`, `to_v`, and `to_out.0` weight in the down,
mid, and up paths.  The inventory contains 168 `[512,256]` and 56 `[256,512]`
weights.  The factors contain 344,064 trainable values.

The fitter consumes a strict `nano_mel_calibration_prepare_v1` flow directory.
It replays the prepared rows with the native target-noise then full-noise draw
order before installing any adapter.  It stops on parity failure.  The 192-D
speaker embedding and every base decoder parameter remain frozen.  The fitted
loss is source-mel MSE, a band-envelope term, a baseline-output distillation
term, and small factor L2 regularization.  Validation runs under `no_grad`.
Training uses 30 steps at most, patience three, and learning rate `1e-4`.

Native preparation replay retains its original GPU precision. After that
check passes, fitting disables both CUDA matmul TF32 and cuDNN TF32. Zero
adapter outputs are compared against anchors collected before installation
in this strict FP32 mode. Deployment requires the same mode. In the smoke
test, cuDNN TF32 caused a factorized-versus-folded maximum mel difference of
0.003812. Disabling TF32 reduced that difference to 0.00001192, below the
unchanged 0.0001 maximum and 0.00001 RMS gates. Repeated folding and restoration
were exact. This verifies three acoustic reconstructions, not perceptual
speech quality.

`quality_sweep.py` requires `strict_fp32: true` for attention comparisons,
including their unchanged controls. It records both precision flags and does
not resume artifacts from a different precision mode. Runtime folding streams
one base weight at a time. Only module references, shapes, and hashes remain
in its state; neither base copies nor factor tensors stay resident.

Run through the bounded job wrapper after the parent agent grants the model
slot:

```bash
python scripts/nano_lab/bounded_job.py -- \
  .venv-nano/bin/python scripts/nano_lab/fit_decoder_attention.py \
  --prepare-dir artifacts/nano_lab/acoustic_only_asmr/flow_v2 \
  --initial-conditionals artifacts/nano_lab/decoder_embedding_fit/conditionals.pt \
  --output-dir artifacts/nano_lab/decoder_attention_fit \
  --device cuda
```

The checkpoint is `best_attention.pt` with format
`nano_decoder_attention_lora_v1`.  Its `targets` mapping uses the full
inventory name as the key.  Each value stores `shape`, CPU fp32 `down_weight`
and `up_weight` factors, and a SHA-256 hash of the matching base weight.  The
top level stores rank, alpha, factor count, inventory SHA-256, S3Gen checkpoint
SHA-256, initial and prepared conditionals SHA-256, cache SHA-256, validation
metrics, and the best step.  It stores no base tensors and no merged dense
matrices.  A runtime loader must stream the matching base checkpoint, verify
all hashes and shapes, then fold `(alpha/rank) * up @ down` into each weight.

The result is an experiment.  A lower reconstruction loss does not establish
better zero-shot realism, timing, pitch, or clean vocoder output.  Pure tests
in `test_fit_decoder_attention.py` cover target inventory, factor arithmetic,
zero-adapter parity, checkpoint rejection, finite metadata, and no-grad
validation.  They do not load Nano checkpoints.
