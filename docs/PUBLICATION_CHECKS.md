# Publication checks

The source snapshot was prepared on 2026-09-29. Publication does not constitute a new model-quality result.

- All 181 lab, vendor, earlier baseline, and experiment-helper code files selected from the workspace were included with identical bytes.
- All 681 tracked files from the unchanged Voicebox upstream checkout were included. Nested Git metadata was excluded. No Git submodule placeholders remain.
- Python syntax parsing passed for 303 files. This check does not import frameworks or execute model tests.
- A credential-pattern scan found no matching private keys, common GitHub/Hugging Face/AWS/OpenAI token forms, or URLs containing passwords. This is a limited pattern scan, not proof against every secret format.
- No model-weight, checkpoint, tensor-cache, source-audio, or generated-audio files were staged. Public upstream application static assets were preserved.
- The main README and handoff-document local links resolve in the source snapshot. Audio players need the excluded assets.
- The asset checker passed against all 2305 inventoried local assets on the original machine. It correctly reports those assets missing from the source-only clone. It checks sizes and symlink targets; experiment hashes remain authoritative.
- New handoff documents pass Git whitespace checks.
- No new speech, training, or CUDA verification run was performed during publication. Fresh dependency installation was not tested.

Preserved experimental evidence and unresolved checks are described in `EXPERIMENT_HANDOFF.md`.
