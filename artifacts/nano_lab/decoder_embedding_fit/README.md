# Constrained decoder embedding fit

The fit is complete. It is experimental and has no listening acceptance. It adjusts 192 decoder speaker values for the ASMR source. It is not a general zero-shot model improvement.

All 17 unchanged native reconstructions matched exactly before training. Fifteen source clips were used for fitting and two separate clips for validation. All model weights stayed frozen. The first gradient was finite and nonzero, and no frozen parameter received a gradient.

The best validation objective changed from 1.181268 to 0.894036, a 24.32% reduction. The best checkpoint was step 37 of 40. The embedding norm is 13.843838, and its cosine to the original embedding is 0.98000002, at the imposed 0.98 boundary.

Measured peak process RSS: 1712.12 MiB. Fit-body elapsed time: 293.90 seconds. This timer excludes module imports before the fit entry point; see the guard log for complete service time. The cgroup had a 3072 MiB hard cap, no swap, and two CPU cores.

`conditionals.pt` is the output cache. `best_embedding.pt` holds the selected embedding. `fit_report.json` records hashes, source splits, parity, objective history, and constraints. New-text audio, transcript, equal-loudness identity/noise, and listening checks are required before choosing a runtime profile.
