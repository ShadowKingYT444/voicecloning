# CPU VM experiments

The 2026-09-30 VM has CPU inference only, no running systemd, and no writable
cgroup hierarchy. The desktop resource limits still apply. Explicitly select
the stricter backend; the default desktop systemd path remains unchanged:

```bash
NANO_SAFETENSORS_BACKEND=streamed HF_HUB_OFFLINE=1 \
python3 scripts/nano_lab/bounded_job.py --backend single-process \
  --guard-report artifacts/nano_lab/vm_20260930/job_guard.json \
  -- .venv-nano-cpu/bin/python scripts/nano_lab/vm_reference_trial.py \
  --config artifacts/nano_lab/vm_20260930/paired_config.json \
  --out artifacts/nano_lab/vm_20260930/new_paired_run
```

The backend retains the job lock, 6144 MiB launch threshold, 4096 MiB runtime
reserve, no swap, two CPUs, and nice 10. Its 3072 MiB **address-space** cap is
stricter than an RSS cap. It requires a Python workload and libseccomp; it
refuses hosts with enabled swap. Kernel filters deny child processes, raising
limits, and changing CPU affinity while allowing inference threads. A watchdog
inside the actual Python process reports `getrusage` peak RSS and stops at
2500 MiB. Small/tiny modes preserve their lower existing budgets. This backend
does not authorize GPU jobs or a higher memory cap.

Container PID namespaces made `/proc/<parent-observed-pid>/status` report the
bootstrap process rather than the executed workload. Early import/smoke guard
reports consequently have invalid peak RSS readings. Use only reports marked
`measurement_source: in_process_getrusage` for measured memory claims.

The official mmap and pread readers failed under the address-space cap. The
explicit CPU `streamed` path reads a validated safetensors header and one tensor
buffer at a time. It leaves the official mmap default unchanged. GPT-2 causal
masks are deferred before `to_empty`, then rebuilt once and shared across
layers, preserving the existing exact read-only values.

Prepare conditioning separately with `prepare_reference_only.py`. It omits
T3/flow/vocoder weights and calls upstream `prepare_conditionals` and `embed_ref`
unchanged. Pass the saved conditionals into the optimized runtime; it removes
reference-only modules before allocating weights. For adaptation feature
preparation, `adaptation.py prepare --reference-only --device cpu` uses the same
reference stage, target tokenizer, and text tokenizer without synthesis models.
All transcript/audio hash checks and split checks remain mandatory.

Use unique run directories: the VM trial runner refuses an existing report and
does not resume by filename. It records reference, conditioning, adapter,
configuration, implementation and output hashes. Evaluate synthesis separately:
Whisper Small for words, Resemblyzer for development speaker proxies, DNSMOS for
cleanliness proxies, and constant-gain matched copies for listening. Run each
model serially; keep protected final-audit intervals out of development selection.

The pinned model is `ResembleAI/chatterbox-nano` revision
`71ccd1d0081b430592cea481f4307e764e07bc64`. Whisper Small is
`Systran/faster-whisper-small` revision
`536b0662742c02347bc0e980a01041f333bce120`. The VM uses Torch/Torchaudio
2.11.0+cpu and the common requirements plus the evaluation additions. PyAV
19.0.0 failed the Whisper decoder's `metadata_errors` call; 16.1.0 completed
the audit. The VAD wheel replacement is recorded and historical cosine scores
must not be treated as a matched control for these short new passages.

Read [the experiment results](../artifacts/nano_lab/vm_20260930/RESULTS.md) for
decisions and limitations. No voice is promoted from a proxy or training loss.
