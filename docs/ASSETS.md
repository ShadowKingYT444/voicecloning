# Restore local assets before continuing

Git contains code, text evidence, and the shared voice-reference bundle. The original workspace also contains approximately 9.0 GiB of excluded model, cache, and generated assets. [The inventory](local-assets.json) records paths, sizes, and symlink targets. These sizes are an inventory, not cryptographic verification. Existing experiment manifests contain the authoritative hashes.

The tracked reference bundle includes the source MP3 recordings, curated ASMR and Harvey reference variants, audited repair clips, and the Pocket/Kokoro reference files. The browser's selected reference WAV and 331,200-byte fixed voice-state binary are also tracked. Most generated voices, model weights, fitting caches, and ONNX graphs remain local. The listening HTML pages still need generated WAV files. Fitted profile JSON files also need their exact conditioning caches and adapters. See [the cloud browser handoff](CLOUD_BROWSER_HANDOFF.md) for the public model restore path.

## Transfer the saved state

Clone the repository on the stronger machine. Prefer the original absolute path:

```bash
git clone https://github.com/ShadowKingYT444/voicecloning.git /home/terryd/gooning/voicecloning
cd /home/terryd/gooning/voicecloning
```

Use your actual original-machine SSH host in place of `SOURCE_HOST` below. The command copies only the inventoried assets. It preserves symlinks and does not delete destination files or copy virtual environments.

```bash
rsync -a --info=progress2 --files-from=docs/local-assets.paths SOURCE_HOST:/home/terryd/gooning/voicecloning/ ./
```

The tracked source and reference bundle is available in GitHub. Transfer of excluded assets still needs the original machine or a separate asset store. The repository does not provide trained adapters, experiment caches, or most generated listening WAVs. Public base-model downloads do not recreate those files.

The optional checker reports missing assets and size differences without importing a model:

```bash
python tools/check_local_assets.py
```

Preserve the same absolute workspace path when possible. Many archived JSON files, tensor caches, and ONNX manifests contain that path. A mount at the old path also works if all referenced files resolve there. If a different path is required, regenerate path-bearing caches and derived verification records through their producer scripts. Update dependency hashes in order. Do not globally replace paths inside hash-bound records and then disable the checks.

## Required groups

| Group | Purpose |
|---|---|
| `models/chatterbox-nano/` | Native model weights, tokenizer files, and configuration |
| Root source MP3 files | Original ASMR and Harvey material for rebuilding references and clips; tracked in the repository |
| `artifacts/references/` | Mommy ASMR reference variants and preparation notes; tracked in the repository |
| `artifacts/nano_lab/references/` | Selected ASMR and Harvey references, manifests, and protected evaluation excerpts; tracked in the repository |
| `dataset_repair_complete/clips/`, other source-clip directories | Actual audio behind the audited training manifests; tracked in the repository |
| `*.pt`, `*.npy`, `*.npz` under the experiment directories | Adapters, pinned conditioning caches, source tokens, prepared flow data, and numerical references |
| `onnx_staged*` directories and symlinks | Graphs, external weights, and stage layout for low-memory inference |
| `models/faster-whisper-small/` and `Whisper_fast_package/models/` | Independent Small and Tiny transcript checks |
| `vendor/dnsmos/sig_bak_ovr.onnx` | Audio-quality proxy model |

The inventory also includes earlier baseline assets and optional generated samples. The shared reference bundle is already in GitHub. Restoring the remaining entries is still required for fitted-model continuation and historical comparisons. A smaller transfer must satisfy every input and SHA-256 binding for the chosen experiment.

## Reacquire base models

The native Nano source declares `ResembleAI/chatterbox-nano`. The saved model download metadata records revision `71ccd1d0081b430592cea481f4307e764e07bc64`. After installing `huggingface-hub`, this command restores the base model snapshot, not the fitted local state:

```bash
hf download ResembleAI/chatterbox-nano --revision 71ccd1d0081b430592cea481f4307e764e07bc64 --local-dir models/chatterbox-nano
hf download Systran/faster-whisper-small --revision 536b0662742c02347bc0e980a01041f333bce120 --local-dir models/faster-whisper-small
```

The second revision comes from the saved `download_manifest.json`. Downloading files does not recreate that local provenance manifest, source recordings, trained adapters, reference caches, or numerical verification outputs. Prefer transfer for exact continuation. Rebuild missing derived assets with the corresponding producer scripts and record fresh checks.
