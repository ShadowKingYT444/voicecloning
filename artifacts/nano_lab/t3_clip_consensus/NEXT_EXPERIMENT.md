# Next quality experiment

Status: dataset metadata verified; features and model fitting not run.

The separate manifest contains 14 existing source clips: 12 training clips (55.80 seconds) and two validation clips (18.22 seconds). All have exact normalized Tiny/Small consensus on the actual clip, with unchanged audited transcript and audio hashes. Source windows avoid all protected reference/evaluation intervals by two seconds. Clip 011 was removed for the extra buffer. New validation clip 030 was never in the earlier T3 training cache; clip 022 was already used for validation. Both clips occur in acoustic-model fitting data, which limits full-pipeline generalization claims.

Four clips absent from the previous T3 dataset contribute 20.12 seconds across both splits. Six earlier rows fail the stricter clip-level consensus. Therefore this is a smaller, differently selected dataset, not a 20-second union with the old set. Preserve both source manifests.

Run serially after the 6144 MiB launch threshold is available:

```bash
python scripts/nano_lab/bounded_job.py -- .venv-nano/bin/python scripts/nano_lab/adaptation.py prepare --manifest artifacts/nano_lab/t3_clip_consensus/manifest.json --cache artifacts/nano_lab/t3_clip_consensus/cache.json --device cuda --threads 2
python scripts/nano_lab/bounded_job.py -- .venv-nano/bin/python scripts/nano_lab/adaptation.py train --cache artifacts/nano_lab/t3_clip_consensus/cache.json --checkpoint artifacts/nano_lab/adapter_clip_consensus_all_attn.pt --speaker-id asmr7 --device cuda --threads 2 --rank 4 --alpha 8 --layers all --target-modules attn --lr 0.0002 --kl-coef 1.0 --epochs 8 --patience 2 --max-steps 100
```

Use the current all-attention adapter as a matched synthesis comparator. Keep acoustic cache, acoustic attention fit, prompt donor, mel correction, text, seeds, and sampling fixed. Compare three passages, including two new texts. Run independent Small word auditing and level-matched speaker/cleanliness measurements. Do not compare absolute validation loss across the different datasets. Do not promote on the training loss or a single source phrase.

The separate pending runtime checks remain first: one 640 MiB verifier fault-injection test; the CPU ONNX output's independent Small word audit; DEFAULT/pad0 CUDA convolution diagnosis, then pad1 if needed. These require the existing 4736/5376 MiB launch thresholds. Native CLI parity and training require 6144 MiB. No model is currently running.
