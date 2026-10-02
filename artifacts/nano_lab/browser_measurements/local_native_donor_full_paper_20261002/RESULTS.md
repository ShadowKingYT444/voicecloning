# Native donor full-paper browser run

## Outcome

The run did not finish the Federalist No. 10 reading. The bounded job stopped with `headroom_below_4096_mib`. The browser report records `interrupted` and `Received signal 15.` No `full-reading.wav` was exported. No full-reading ASR ran.

The guard kept its configured limits: 3,072 MiB maximum, 2,500 MiB high limit, no swap, two CPU quota units, and a 4,096 MiB desktop reserve. The guard sampled a process-tree RSS peak of 1,751.012 MiB. It returned code 1 after 2,627.576 seconds.

The latest saved app progress was `reading`, `busy: true`, with 29 completed passages and no app error. The input has 3,004 words and 186 passages at the configured 18-word chunk size. The 29 completed passage texts match the checked-in Federalist source prefix. They cover 464 input words, or 15.45%. This is partial text progress. It is not full-text coverage.

The checked-in input file is `browser_tts/public/federalist-no-10.txt`, SHA-256 `0bdde6323482b7b1be92ab0041f3c82d65dfc64d6c4621a2e439a66e72e679d6`. The measurement report ended before it could save the full-reading input hash.

## Completed short readings and partial long reading

| Stage | Result | Synthesis / audio | RTF | Peak RSS / PSS / incremental PSS | Peak live requested WebGPU buffers |
| --- | --- | ---: | ---: | ---: | ---: |
| Clean Chrome baseline | Complete | N/A | N/A | 1,233.962 / 480.281 / reference | 0 MiB |
| First short reading | Complete | 17.168 / 3.440 s | 4.991 | 1,821.527 / 946.056 / 465.775 MiB | 440.345 MiB |
| Warm short reading | Complete | 11.165 / 3.440 s | 3.246 | 1,707.473 / 947.021 / 466.740 MiB | 443.179 MiB |
| Full reading | Interrupted | 632.955 / 168.360 s before stop | 3.760 | 1,976.707 / 1,179.312 / 699.031 MiB | 682.083 MiB |

The full-reading stage lasted 2,456.035 seconds before the stop. Its largest worker lifetime peak for requested buffer sizes was 682.539 MiB. The incremental PSS peak exceeded the 500 MiB target. The live scheduled playback gap reached 26.533 seconds. These gaps describe the scheduled AudioContext timeline. They do not prove physical speaker continuity.

The RSS and PSS values above are sampled Chrome process-tree values from the measurement report. The guard's 1,751.012 MiB RSS peak uses its separate cgroup process sampler. Do not compare these as if they were the same metric.

The WebGPU counters report JavaScript `GPUBuffer` descriptor sizes. They do not measure physical VRAM or total device residency. Per-process `nvidia-smi` sampling was disabled.

## Runtime and model identity

Chrome 152 ran headless with the normal multi-process layout. The worker reported an Intel `gen-12lp` adapter, `shader-f16`, and a non-fallback adapter. The app used the low-power preference. The native T3 donor state is speaker-specific. No fitted adapter was applied. The application did not add a watermark. Native Perth watermark parity remains unverified.

The short first and warm WAV files passed their report hash checks. Their hashes differ. This run did not audit either WAV with ASR. No human listening was performed for this run.

## Reproduction and retained evidence

The original guarded invocation used the full-paper mode with a 5,400-second overall timeout and a 5,100-second full-reading timeout. The run stopped before full-reading export. Do not treat the partial chunk records as a completed long-form sample.

- [Full result summary](full-result-summary.json)
- [Measurement report](measurement.json)
- [Latest saved app progress](live-progress.json)
- [Guard report](../local_native_donor_full_paper_guard.json)
- [Full-reading sample log](measurement-samples.jsonl)
