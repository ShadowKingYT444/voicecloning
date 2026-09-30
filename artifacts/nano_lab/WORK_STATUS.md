# Current voice work

The requested voice realism is not achieved. The user rejected the earlier samples. The newer samples remain experimental and need listening review.

- [Current listening page](delivery/index.html): source, prompt/timing candidate, and acoustic-attention candidate for ASMR; source and two fitted Harvey candidates.
- [ASMR prompt and attention comparison](decoder_attention_prompt_comparison_v2/index.html): six samples pass the independent word check. The attention adapter improves the speaker-similarity proxy but reduces the cleanliness proxy on the two new passages. Source pitch and timing still differ substantially.
- [Fourteen acoustic controls](decoder_attention_comparison/index.html): fixed speech tokens separate decoder effects from text-model sampling. All fourteen word checks pass.
- [MLP adapter experiment](t3_mlp_comparison/index.html): not selected. Three samples have word errors, and similarity falls on new text despite improvement on the source phrase.
- [Harvey decoder fit](harvey_decoder_comparison/index.html): six word checks pass. Automatic gains are small. The training dataset contains only two clips.
- [RSS report](RSS_REPORT.md): the new native ASMR comparison peaks at 2496 MiB process RSS. The earlier 1046 MiB staged ONNX result uses an older profile.

The 30-step acoustic attention fit is complete. It trains 344,064 values across 224 attention projections. Validation reconstruction loss falls 15.32%. Strict full-precision folding passes on three source-token reconstructions. This does not establish convincing speech on new text or general zero-shot improvement.

The new acoustic adapter is folded into an independent ONNX weight file. CPU estimator parity passes at five lengths. CUDA parity fails at four lengths. A CUDA precision override then causes a cuDNN runtime error. CUDA inference with these new weights remains blocked. Provider-specific checks prevent the CPU result from authorizing CUDA. All 31 integration tests pass.

Each model job runs alone with a hard memory cap, no swap, and 4 GiB reserved system headroom. The guard refused the latest native CLI check because available RAM was below 6 GiB. The smaller CPU ONNX end-to-end check completed at 861.75 MiB sampled peak process-tree RSS. It took 118.07 seconds for 6.96 seconds of audio. Its independent word audit remains pending because available memory fell below the 5376 MiB launch threshold. Even a 640 MiB unit-test job was later refused below 4736 MiB. No model job is active.

A separate [stricter T3 dataset](t3_clip_consensus/derivation_report.json) is now prepared at the metadata level. It contains 12 training clips and two validation clips, with a two-second buffer around every protected source interval. Four clips are new to the text-model fit; six previous clips fail the stricter clip-level transcript agreement. Feature preparation and fitting remain pending. [Exact next experiment](t3_clip_consensus/NEXT_EXPERIMENT.md).

Execution is blocked by desktop memory. The latest check shows 3311 MiB available and the model service inactive. This condition has persisted across three goal turns. The smallest pending guarded test requires 4736 MiB available; smaller model checks require 5376 MiB; fitting and native synthesis require 6144 MiB. All samples, reports, and pending commands are preserved. Voice realism and the new CUDA runtime remain unfinished.
