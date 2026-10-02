# Voice cloning experiment handoff

2026-09-30 continuation: source/reference audio is now tracked and verified.
A guarded CPU VM run generated matched ASMR/Harvey baselines and completed the
strict T3 clip-consensus fit. The fit is **rejected for promotion** because
new-text speaker similarity regresses despite lower validation loss. Read
[VM results](../artifacts/nano_lab/vm_20260930/RESULTS.md) and
[CPU VM instructions](CPU_VM_EXPERIMENTS.md). Historical fitted checkpoints and
ONNX graphs remain absent; the original CUDA gates and limits are unchanged.

This repository contains an active Chatterbox Nano 110M investigation. The requested quality target is not achieved. The user rejected the earlier samples as unconvincing. No voice profile is promoted, and no result proves general zero-shot improvement or ElevenLabs-level realism.

Use this file with the current reports:

- [work status](../artifacts/nano_lab/WORK_STATUS.md)
- [resume log](../artifacts/nano_lab/RESUME.md)
- [quality report](../artifacts/nano_lab/QUALITY_REPORT.md)
- [RSS report](../artifacts/nano_lab/RSS_REPORT.md)
- [current listening page](../artifacts/nano_lab/delivery/index.html)
- [next strict T3 experiment](../artifacts/nano_lab/t3_clip_consensus/NEXT_EXPERIMENT.md)

## State at handoff

No model job is active in the latest recorded state. The service is inactive. The last documented desktop memory check was about 3.2 GiB available. Recheck `/proc/meminfo` before every workload. Do not start a job below its guard threshold.

The host has 16 GiB RAM and an RTX 4060 Laptop GPU. The user reported desktop crashes from high RSS. Run one model or test job at a time through `scripts/nano_lab/bounded_job.py`. Preserve the existing limits:

- Default jobs: 3072 MiB hard limit, 2500 MiB high limit, no swap, two CPU cores, 6144 MiB required at launch.
- Small jobs: `--small-job`, 1280 MiB hard limit, 1024 MiB high limit, 5376 MiB required at launch.
- Tiny checks: `--max-memory-mib 640`, `768`, or `1024`. Launch requires cap plus 4096 MiB.
- Stop any job if available desktop memory falls below 4096 MiB. Do not raise caps, bypass the wrapper, kill unrelated processes, or flush caches.

## What is verified

The acoustic attention fit is complete. It uses rank-2 LoRA factors on 224 meanflow attention projections, with 344,064 trained values. The 30-clip acoustic dataset has 28 training clips and two validation clips, totaling 159.26 seconds. It excludes every protected reference and held-out interval with a two-second buffer. Token, flow, and strict fit-consumer caches are ready. The fit lowers held-out reconstruction objective from `0.7862759` to `0.6658157` (15.32%). Strict FP32 factorized-versus-folded checks pass on three source-token reconstructions, with maximum error `6.20e-6`. These are reconstruction checks. They do not prove new-text realism.

The folded acoustic adapter is in `artifacts/nano_lab/onnx_staged_adapter_aligned_all_attn_decoder_attention`. CPU ONNX estimator parity passes at mel lengths 8, 64, 256, 512, and 768. Maximum CPU error is `4.9353e-5` at the unchanged `3e-4` tolerance. The provider gate records CPU as verified.

The CPU ONNX end-to-end acoustic-attention profile completed. It produced 6.96 seconds of audio in 118.07 seconds, with 861.75 MiB sampled process-tree RSS and about 1 GiB cgroup peak. The output is `artifacts/nano_lab/onnx_asmr_acoustic_attention_cpu.wav`. Its independent Small ASR audit is still pending.

The 14-case acoustic-attention comparison has zero Small WER in every case and exact speech-token payloads within each text group. Three new-text means are:

| Condition | Speaker cosine | DNSMOS |
|---|---:|---:|
| control, raw | 0.885287 | 3.131835 |
| attention, raw | 0.892158 | 3.125945 |
| control, mel correction | 0.889821 | 3.153682 |
| attention, mel correction | 0.896536 | 3.164181 |

The six prompt-donor cases also pass Small WER. On two new texts, prompt plus attention changes cosine/DNSMOS from `0.8864105 / 3.1664034` to `0.8935955 / 3.1093683`. The identity proxy improves while the cleanliness proxy falls. The same-source phrase remains shorter and lower-pitched than the source. Source is about 10.0 seconds and 258.9 Hz; generated variants are about 6.6 to 7.4 seconds and about 165 to 197 Hz. Breathiness makes pitch estimates uncertain, but the timing gap is real.

The strict level matcher uses constant gain to -27 LUFS. It does not use a denoiser, compressor, limiter, or noise gate. Playback pages and metadata render correctly. No auditory quality assessment has been performed by the tooling.

## Candidates and decisions

All of these profiles are experimental, speaker-specific, `zero_shot: false`, and awaiting listening validation:

- `voices/nano/asmr_prompt_experimental.json`: pinned prompt donor, fitted decoder cache, and mel correction.
- `voices/nano/asmr_acoustic_attention_experimental.json`: the prompt candidate plus the new acoustic attention fit.
- `voices/nano/asmr_decoder_fitted_experimental.json`: earlier fitted decoder and standard repetition penalty.
- `voices/nano/asmr_decoder_slow_experimental.json`: earlier fitted decoder with slower sampling.
- `voices/nano/harvey_fitted_experimental.json`: Harvey rank-2 attention adapter.

The Harvey decoder fit uses two training clips and one validation clip. Native parity is exact. Validation objective changes from `1.3456695` to `1.2532183`. On three matched passages, isolated-reference cosine changes from `0.8304447` to `0.8340537`; DNSMOS changes from `3.3733524` to `3.3745267`. All six generated clips pass Small WER. The dataset is too small for a generalization claim.

The T3 MLP experiment is rejected. It has word errors on three of nine cases and lower new-text identity similarity. The wider 0.95 speaker-vector fit improves cosine but reduces DNSMOS. The decoder projection fit stopped after 13 steps because of the memory guard; its checkpoint is not deployed. Do not promote any of these variants.

## ONNX CUDA failure

CUDA ORT 1.26 with `use_tf32=0` passes mel length 8 but differs from strict CPU/native results by about `0.02` at lengths 64, 256, 512, and 768. The unchanged tolerance is `atol=rtol=3e-4`, so CUDA is blocked. A separate process with `NVIDIA_TF32_OVERRIDE=0` fails earlier in cuDNN 9.19 with `HEURISTIC_QUERY_FAILED` on the first convolution. The log is `artifacts/nano_lab/decoder_attention_onnx_reference/verify_cuda_tf32off.log`.

The concrete diagnosis is in [CUDA_DIAGNOSIS.md](../artifacts/nano_lab/decoder_attention_onnx_reference/CUDA_DIAGNOSIS.md). ORT 1.26 may retry a cuDNN plan with TF32 after strict support or plan construction fails. This is a hypothesis to test, not a confirmed cause. Do not relax tolerances or change global environment variables. The diagnostic script never publishes a production gate.

The older staged ONNX profile measured about 1046 MiB RSS, but it uses an earlier adapter and is not evidence for the new acoustic profile. Do not combine those timings with the new CPU result.

## Main entry points

- `./nano-clone` runs the native guarded CLI.
- `./nano-clone-onnx` runs the CPU ONNX CLI by default. Pass `--ort-provider cuda` only for an explicit experimental CUDA run.
- `scripts/nano_lab/quality_sweep.py` runs controlled comparisons. Keep text, seed, prompt donor, acoustic cache, mel correction, and sampling fixed across variants.
- `scripts/nano_lab/adaptation.py` prepares and trains the T3 LoRA adapter.
- `scripts/nano_lab/fit_decoder_attention.py` trains the acoustic attention fit.
- `scripts/nano_lab/patch_onnx_decoder_attention.py` folds the attention factors into external ONNX weights.
- `scripts/nano_lab/verify_decoder_attention_onnx.py` performs strict Torch-reference and ORT provider checks.
- `scripts/nano_lab/diagnose_decoder_attention_cuda.py` tests isolated CUDA convolution options.

## Pending guarded commands

Run these only after the required launch headroom is available. Run them sequentially. Each command is an experiment, not a reason to weaken a gate.

1. Run the provider-failure regression test. It must preserve an existing CPU gate after a later runtime failure:

```bash
python scripts/nano_lab/bounded_job.py --max-memory-mib 640 -- .venv-nano-cpu/bin/python -m pytest -q scripts/nano_lab/test_verify_decoder_attention_onnx.py
```

2. Audit the new CPU ONNX WAV with Whisper-small. This checks content only:

```bash
python scripts/nano_lab/bounded_job.py --small-job -- .venv-nano-cpu/bin/python scripts/nano_lab/audit_asr_ct2.py artifacts/nano_lab/onnx_asmr_acoustic_attention_cpu_asr_inputs.json --out artifacts/nano_lab/onnx_asmr_acoustic_attention_cpu_asr.json
```

3. Test the CUDA convolution plan with the default algorithm and pad-0 layout. Keep the environment override unset:

```bash
python scripts/nano_lab/bounded_job.py --small-job -- env -u NVIDIA_TF32_OVERRIDE PYTHONPATH=scripts/nano_lab .venv-nano-ort/bin/python scripts/nano_lab/diagnose_decoder_attention_cuda.py --model-dir artifacts/nano_lab/onnx_staged_adapter_aligned_all_attn_decoder_attention --reference-dir artifacts/nano_lab/decoder_attention_onnx_reference --algo DEFAULT --pad-nc1d 0 --out artifacts/nano_lab/decoder_attention_onnx_reference/diagnostic_default_pad0.json
```

If pad-0 fails, run the same command with `--pad-nc1d 1` and output `diagnostic_default_pad1.json`. Test `EXHAUSTIVE` or disabled graph optimization only as separate diagnostics. A passing diagnostic must be rechecked by the production verifier before any CUDA profile is enabled.

4. Test native profile parity after the desktop has at least 6144 MiB available:

```bash
./nano-clone --voice asmr_acoustic_attention_experimental --text "Let's see who's the lucky person who gets to kiss me. Wait a minute. Do I know you?" --seed 31 --output artifacts/nano_lab/native_asmr_acoustic_attention.wav
```

Record RSS, service time, output duration, and an independent Small audit. Compare the native WAV byte-for-byte with `artifacts/nano_lab/decoder_attention_prompt_comparison_v2/heldout01_prompt_attention1_mel.master.wav`. Investigate any difference before claiming parity. The native CLI does not save the sweep's token artifact.

5. Prepare and fit the stricter T3 dataset on a stronger CUDA machine. The two commands must run one at a time:

```bash
python scripts/nano_lab/bounded_job.py -- .venv-nano/bin/python scripts/nano_lab/adaptation.py prepare --manifest artifacts/nano_lab/t3_clip_consensus/manifest.json --cache artifacts/nano_lab/t3_clip_consensus/cache.json --device cuda --threads 2
python scripts/nano_lab/bounded_job.py -- .venv-nano/bin/python scripts/nano_lab/adaptation.py train --cache artifacts/nano_lab/t3_clip_consensus/cache.json --checkpoint artifacts/nano_lab/adapter_clip_consensus_all_attn.pt --speaker-id asmr7 --device cuda --threads 2 --rank 4 --alpha 8 --layers all --target-modules attn --lr 0.0002 --kl-coef 1.0 --epochs 8 --patience 2 --max-steps 100
```

The manifest has 12 training and two validation clips, 74.02 seconds total. Four rows are new to the earlier T3 fit. Six prior rows fail the stricter clip-level consensus. This is a smaller, different dataset, not a blind union. Both validation clips also occur in acoustic-model data, so disclose that exposure.

## Data and publication limits

The repository tracks the original source recordings and curated reference clips needed to rebuild the datasets. It still excludes trained checkpoints, model weights, generated audio, binary caches, and local virtual environments. Restore those assets from private storage or rebuild them on the target machine. The reports and JSON manifests contain absolute paths such as `/home/terryd/gooning/voicecloning`. Prefer the same workspace path or a mount at that path. Do not blindly rewrite manifests: their hashes are part of downstream provenance checks. See [asset transfer](ASSETS.md) for relocation requirements. A fresh clone without the model, fitted caches, and local dependencies cannot run the samples or reproduce the fitted reports.

The acoustic-only cache contains source speech tokens and flow data. It has no text labels and does not relax transcript requirements for the separate T3 text-model fit. Small ASR is a content check, not a perceptual test. Automatic speaker cosine and DNSMOS are selection proxies. They do not establish timbre, richness, timing, background-noise removal, or professional cleanliness. Human listening on held-out new text remains required.

The remaining quality work is to reduce the source-to-output prosody and timing mismatch, preserve breath and timbre without static or background noise, and show consistent held-out gains for both ASMR and Harvey. Keep every candidate experimental until those checks pass.
