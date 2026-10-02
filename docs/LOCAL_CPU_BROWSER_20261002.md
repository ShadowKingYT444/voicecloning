# Local CPU reader, 2026-10-02

The reader can now use a same-origin local CPU service. This removes the
browser WebGPU, FP16 shader and JSPI requirements from this path. The page uses
the local service first when it is available. A static preview still needs the
existing WebGPU runtime. GPU measurements explicitly select that runtime.

## Start the reader

Restore the pinned assets and CPU environment first. The production files must
already exist in `browser_tts/dist`. Stop the owned Vite preview before using
the same port. Run from the repository root:

```sh
python3 scripts/nano_lab/bounded_job.py --max-memory-mib 1024 \
  --guard-report artifacts/nano_lab/browser_measurements/local_cpu_reader_live_guard.json \
  -- env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  .venv-nano-cpu/bin/python browser_tts/scripts/local-reader-server.py
```

Open `http://127.0.0.1:4187/`, reload the page, and click **Prepare voice**.
Execution should show **Local CPU**. Then click **Read Federalist No. 10**.
Session measurements show first audio, synthesis time, playback gaps and
current/peak service RSS. Browser memory is separate. Preparation verifies
assets and the voice; model loading occurs inside each passage's first-audio
latency. Model sessions are released between the language-model and decoder
stages and after each passage. CPU service overhead remains included in RSS.

The service accepts one synthesis request at a time. Stop cancels playback
immediately and waits for the active native model call to return. Another
tab gets a busy error instead of running a second model. Keep all model work
serial. Stop this owned guarded service before launching other lab jobs.
The existing desktop reserve and hard caps remain mandatory.

## Actual verification

Artifacts are in
`artifacts/nano_lab/browser_measurements/local_cpu_service_browser_20261002/`.
The input is `Take a slow breath in, and let your shoulders relax.`, seed 1337.
The pinned native ASMR donor is used without a fitted adapter.

| Result | Chromium 151 headless client | Firefox 157 headless |
| --- | ---: | ---: |
| Audio duration, seconds | 3.24 | 3.24 |
| First-audio time, seconds | 40.283 | 41.970 |
| Synthesis time, seconds | 40.093 | 41.590 |
| Real-time factor | 12.375 | 12.836 |
| Independent Whisper Small word error rate | 0.0 | 0.0 |

Chromium first, warm and restarted audio exports are byte-identical. All three
independent word checks pass. Real Stop/restart passes; acknowledgement takes
6.287 seconds. Firefox selects Local CPU with `navigator.gpu` absent and exports
a complete WAV. Its independent word check also passes. Rendered screenshots
were inspected. The Chromium check uses a smaller headless-shell process
layout; its browser RAM is not a normal desktop Chrome comparison.

Observed CPU-service high-water RSS is 403.242 MiB across the Chromium control,
including its Stop test. The fresh Firefox service peaks at 362.902 MiB. These
values exclude browser processes. They do not establish the complete app's
under-500-MiB target. No matched before/after speedup or memory reduction is
claimed. The saved earlier CPU control has identical speech tokens but a
different waveform; the new WAVs received fresh independent word audits.

The final JavaScript suite has 47 passes and one existing skip. Measurement
runner controls have 15 passes. CPU service controls have five passes. The
site build passes. The first integration attempt found a native `fetch`
receiver error; it was fixed and covered by a regression test. A small-budget
normal Chrome attempt stopped at the aggregate RSS cap. The later small-budget
Chromium run completed, but its following Firefox attempt hit the same cap.
Firefox then passed as a separate fresh workload under the original standard
guard. Failed reports and guard records are preserved.

## Limits

This is a local CPU service, not verified browser WASM speech. It needs the
excluded model assets and native CPU runtime on the host. Browser code targets
ES2022 and requires audio playback and IndexedDB. Actual checks cover Chromium
and Firefox on Linux. Safari and Edge are not tested. Do not claim every browser
or every operating system has been verified. A hosted website also needs a
hosted inference service; a static page alone cannot supply this CPU path.

Generation remains much slower than playback. Full Federalist completion,
smooth playback, perceptual realism, native watermark parity and the full-app
memory target remain unfinished. Short word checks do not establish them.
