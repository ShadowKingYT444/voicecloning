# Voice research continuation

Read `README.md`, `docs/EXPERIMENT_HANDOFF.md`, and `scripts/nano_lab/AGENTS.md` before model work.

- Preserve the full goal: realistic and clean ASMR and Harvey speech, matched listening samples, and low-memory smooth inference. Current results do not satisfy the realism target.
- Run model, evaluation, export, and browser workloads serially through the existing resource guard. The user reported desktop crashes. Do not raise caps or remove the 4 GiB desktop reserve without a new instruction.
- Restore excluded assets before trying fitted profiles. Preserve hash checks and the separation of source, training, validation, and evaluation intervals.
- The new acoustic ONNX graph has CPU verification only. Do not use that result to authorize CUDA or relax numerical gates.
- Fitted profiles are speaker-specific. Do not call them general zero-shot improvements.
- Verify generated words, controlled quality metrics, the actual listening artifact, and matched RSS/latency. Report uncertainty and unsuccessful experiments.
- Speak in short, direct sentences. Keep explanations precise. Do not declare completion from an intermediate artifact or automatic quality score.
- Do not run training simply to verify a documentation or repository-publication change.
