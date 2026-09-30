# Decoder speaker embedding experiment

This fits one 192-value S3Gen speaker embedding. It does not train a general zero-shot model. All flow weights stay frozen. The output is an ordinary native conditionals cache with only `gen.embedding` changed. Do not select a voice from reconstruction loss alone.

Use the existing resource guard and run one model job at a time:

```sh
python scripts/nano_lab/bounded_job.py -- .venv-nano/bin/python scripts/nano_lab/fit_decoder_embedding.py --device cuda --max-steps 40 --patience 3 --output-dir artifacts/nano_lab/decoder_embedding_fit
```

The input is the audited `mel_calibration_asmr` preparation. It has 15 training clips and 2 validation clips. The script checks source, cache, and conditioning hashes, finite arrays, split assignments, and exact token/frame geometry. It loads only the flow encoder, speaker projection, and meanflow estimator. It omits T3, the tokenizer, reference encoders, and HiFT.

The unchanged embedding must reproduce every saved native reconstruction within the preset tolerance before optimization begins. Native meanflow noise requires two draws: target-only noise first, then full prompt-plus-target noise. The second draw's target suffix is replaced with the first draw. Omitting this preliminary draw causes a large mismatch despite an otherwise exact flow implementation. The corrected 17-row smoke matched all saved reconstructions exactly.

Source targets contain 2N or 2N-1 frames for N genuine tokens. The decoder receives N+3 tokens after appending three silence IDs. The loss excludes the six silence frames and at most one framing tail. It never warps time. The objective combines raw log-mel MSE, band-envelope MSE, and a cosine regularizer. Each optimizer step preserves the original embedding norm and stays within cosine 0.98 of the original embedding.

Validation rows never receive optimizer updates. The script keeps the best validation checkpoint, including epoch zero, and records a finite-gradient probe and frozen-weight checks. `fit_report.json` preserves progress and failures. `--parity-only` stops before training. A one-step smoke uses `--max-steps 1` in a separate output directory.

After fitting, generate new text with the resulting cache. Check transcripts, equal-loudness identity and cleanliness metrics, and actual listening quality. Keep the reference-only and fitted comparisons separate. The existing T3 adapter and mel correction must be held constant when evaluating the embedding change.

Native odd-length reference handling: upstream can retain `2*prompt_tokens+1` mel frames. The generated suffix then has `2*(target_tokens+3)-1` frames. The fitter measures this offset, requires it to be 0 or 1, checks the exact encoder output length, and preserves the full native parity gate. Source cropping accounts for that offset; no time resampling is used.

Noise generation also follows token count. Native meanflow draws `2*target_token_count` frames and replaces that many frames at the end of the full noise tensor. With an odd reference, this replacement includes the last prompt frame. Cropping the initial noise draw to the returned target length changes the seed trajectory and fails native parity.

Acoustic-only preparation uses a separate `nano_acoustic_cache_v1` contract. The fitter dispatches that format to `prepare_acoustic_data.validate_acoustic_cache`, which verifies recorded-source hashes, splits, protected intervals, and token artifacts. The existing text-model adaptation cache and transcript audit requirements remain separate. This path needs end-to-end validation before training on the larger dataset.
