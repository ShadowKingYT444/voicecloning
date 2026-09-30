# Third-party source provenance

The repository includes source snapshots so a fresh clone does not contain empty nested Git repositories.

| Directory | Upstream | Snapshot |
|---|---|---|
| `vendor/chatterbox` | https://github.com/resemble-ai/chatterbox | `5de7a54aa4e5e2baadb0182dde554908b48b85c2` |
| `voicebox` | https://github.com/jamiepine/voicebox | `51f49dea198384b4eb6087b72c17057c6eb1c1cd` |

Chatterbox has local changes in `src/chatterbox/__init__.py` and `src/chatterbox/models/s3gen/__init__.py`. These lazy imports reduce unnecessary framework loading. The published snapshot includes those changes. The Voicebox tracked source was unchanged at publication.

Preserve `vendor/chatterbox/LICENSE`, `voicebox/LICENSE`, `vendor/dnsmos/LICENSE`, and the `LICENSES.md` files in the earlier model packages. Model and source-audio licensing is separate from the code snapshot. The original recordings and model weights are not distributed in Git.
