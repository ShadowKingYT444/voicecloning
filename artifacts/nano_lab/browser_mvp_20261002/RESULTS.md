# Browser MVP continuation result, 2026-10-02

Branch: `codex/browser-streaming-mvp-20261002`; base `09ba969b28b8856c87d74e4c83aafad8843990e1`.
Draft PR: https://github.com/ShadowKingYT444/voicecloning/pull/1.
Runtime/control fixes were verified at `63634ea0f182c0245bf13fefc2234b39cb1b6c5e`.
Final recovered source publication: `d9d5cbbdc5b95c59d7fd56a9136342b95413bd24`.
Detailed commands and provenance: [BROWSER_MVP_20261002.md](../../../docs/BROWSER_MVP_20261002.md).

## Delivered and verified

- Opt-in lossless FP16 embedding shards, independently pinned manifest and per-shard hashes, maximum 4 MiB retained shard ArrayBuffers. Default Prepare still uses the full FP16 ONNX session.
- Bounded WAV export and a full-paper measurement harness. Visible Save, short export and programmatic export protect the reading lifecycle; cancellation does not unlock another exporter.
- Web Locks protect live tabs while first-reading startup removes abandoned app-owned UUID/passage records. Unknown keys remain untouched; no-Web-Locks cleanup is skipped and reported.
- Real model-free capability page, serial guarded Chromium checks, screenshots, retry/stop/restart tests, long synthetic playback, disk-backpressure concurrency checks and active-tab cleanup checks.
- Pinned conversion LICENSE/NOTICE verified and packaged. Research-only output labels and metadata keep watermark parity explicitly unverified.
- Browser RSS/PSS, page/worker V8 heap and requested GPU buffers remain separate. `--require-hardware-webgpu` checks hardware without forcing backend flags.

CPU ONNX versus lossless reader matched **43,652,352 Float32 values bitwise**.
Real Chromium 153.0.8010.12 CPU/WASM embedding probes matched **5,376 values bitwise** across three hybrid inputs.
No full Nano browser speech was generated.

The verified CI run passed 35 JS checks (one single-process-specific guard check skipped under systemd and separately passed locally), 5 Python checks, build and 7 browser checks:
https://github.com/ShadowKingYT444/voicecloning/actions/runs/36981314189.
The final hardware-only assertion option passed a sixth guarded Python check locally before the executor disconnected. Automatic CI will check the recovered publication; consult the PR checks for its final result.

The synthetic full-paper fixture covered 215 passages / 86 seconds with at most two queued passages.
Its streamed WAV was 4,128,044 bytes, SHA-256
`d0e9e065c7874602d036d9641e2719eb5ebd5f2838d65589981843b06a1328e4`.
This is a control/export check using synthetic tones, not TTS speed or spoken-word/voice validation.

## Distinct measurement scopes

| Quantity | Evidence | Interpretation |
| --- | --- | --- |
| Ten model runtime files | 395,880,313 bytes / 377.54 MiB | Package bytes, not RAM |
| License / notice | 1,087 + 246 bytes | Additional distribution notices |
| Lossless FP16 source tables | 87,304,704 bytes | Unchanged source package bytes, plus 129,432-byte manifest |
| Shard ArrayBuffer cache | 4 MiB maximum | Requested cache budget, not total JS/browser memory |
| Python all-row comparison | 148.73 MiB peak process RSS | Includes CPU ONNX session; not the historical streamed-only comparison |
| UI-only Chromium rendered-mobile run at e3dc959 | 1,275.652 MiB summed RSS; 503.879 MiB PSS; 1.571 MiB observed V8 used heap | Reduced UI-only process layout, GPU disabled, no model inference; separate measures |
| Test-service cgroup peak at 63634ea | 754.6M in systemd job log | Entire test service, not isolated browser/model target |
| Real speech cold load / first and warm TTFA / full playback | Unmeasured | No speech performance claim |

The previous task's 158.53 → 45.77 MiB Python result was not reproduced here.
`artifacts/nano_lab/browser_cpu_components_20261002/RESULTS.md` is absent.

## Asset recovery and missing state

Public conversion revision: `4a66d7dab72a9e98f24b515d49a1d7a81632df2e`.
All ten model runtime files were restored and hash-verified in the disconnected executor; they remain excluded from Git and must be restored on the next host. A fresh checkout does not contain them.
Reference SHA-256: `67f94a868976b22a47bce8fd00a873d4c5b7085ba6eedd698f8a898a26ce76c0`.
Fixed-state SHA-256: `6f56c0dd844ecc038e5b60debff22d1e16b44c9d13386d0f7a91156bfca1b38e` (331,200 bytes).
These exact inputs are preserved; the community runtime does not contain a folded local fitted adapter.

Historical fitted checkpoints, conditioning/donor caches, stage graphs, listening WAVs and Q4 binaries are missing.
See [fitted-assets.json](fitted-assets.json). Terry's positive October 2 voice feedback remains recorded, but the exact praised sample/profile/seed/adapter is unidentified. No replacement fit was run.

## Remaining minimum external tests and release gates

1. A permitted cloud browser with enforceable process-tree guard, JSPI, an identified non-fallback FP16 WebGPU adapter, or a separately validated alternate inference provider. Current Chromium returned no adapter; the separate parent Work browser also lacked usable WebGPU. The executor has no GPU devices and cannot enforce the required multiprocess guard locally.
2. Actual fixed-state speech: matched first/warm requests and saved WAV/tokens, cold load, owned browser RSS/PSS, page/worker heap, requested GPU buffers and physical GPU evidence kept separate.
3. FP16 versus lossless downstream browser logits/tokens/audio parity under matched text/seed/provider. Measure cache/loading tradeoffs before promoting streaming as the default.
4. Independent words and controlled listening on actual speech, including recovery/identification of the praised fitted voice.
5. Actual full-paper playback/joins, complete saved audio, stop/restart and memory drift.
6. Verify watermark preservation within the community graph and end-to-end output. There is no browser application-layer Perth step. Research WAVs are not release-ready native equivalents; no stripping or license violation is inferred from missing verification.

## Transfer evidence and interruption

Latest verified CI artifact: **11216170110**, run **36981314189**, 1,234,139-byte ZIP.
SHA-256: `c1c49b659acd7c798ab1655f41b8ecca9afbefc4357db78873fa6ab8ce9e93e7`.
It includes JSON, screenshots, synthetic WAV and UI-only harness traces. It expires October 3; the parent downloaded it.
Checked-in small evidence survives under `ci_first/` and `ci_wasm_failure/`.
Additional successful WASM/UI evidence artifact: **11215287164**, run **36980252819**, ZIP SHA-256 `fa59197beeea3abcbcb30eb837e351262ec55206458555cc71bfb7be55277466`.
No Library object, hosted Site, build deployment or credential was created.

The execution environment disconnected at 08:04 UTC. Shell Git authentication had also stopped working.
Local unpublished commit `5a7d4c6da68be757e486f74987418123bcb6c851` was reconstructed using checked remote files and the existing connected GitHub app. A later checkout should use the published branch, not assume the local clone or ignored assets transferred.
No training, paid compute, public deployment, guard bypass, raised caps or unrelated process termination occurred.
