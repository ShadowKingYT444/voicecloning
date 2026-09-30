# Decoder reference comparison

Status: experimental. No realism acceptance or profile promotion.

Twelve generated clips compare three texts at fixed seeds. All variants use the same all-attention ASMR adapter and existing mel correction. The baseline uses the conversational reference. Other variants replace half or all of the decoder speaker embedding, or all decoder conditions, with conditions from the separate bully reference. The T3 conditions remain unchanged. No target recording supplies conditions.

`token_parity.json` verifies identical saved token tensor metadata and storage for all four variants of each text. The comparison isolates acoustic decoder conditioning after token generation.

The source-token probe in `../acoustic_conditioning_probe` was a separate reconstruction diagnostic. Its self-reference outputs are not zero-shot evidence. The current clips contain new text and use independent reference excerpts.

Synthesis completed with no errors under the 3072 MiB cgroup limit. Maximum sampled process RSS recorded in the manifest: 2395.42 MiB. The guarded command reported 2.4 GiB peak cgroup memory and zero job swap. All twelve raw clips pass independent Whisper-small transcription with zero normalized word error. Equal-loudness speaker/DNSMOS evaluation and pitch diagnostics are complete. The host-memory guard refused launch during temporary desktop memory pressure; checks resumed only after headroom recovered.

## Results

| Decoder conditions | Mean similarity cosine | Mean DNSMOS overall | Mean per-clip median voiced F0 |
|---|---:|---:|---:|
| Current fitted baseline | 0.8686 | 3.1773 | 201.2 Hz |
| Half donor embedding | 0.8807 | 3.1217 | 204.3 Hz |
| Full donor embedding | 0.8828 | 3.0014 | 208.7 Hz |
| Full donor acoustic conditions | 0.8422 | 3.1785 | 220.2 Hz |

Three matched texts and seeds. DNSMOS inputs were level-matched to -27 LUFS after the usual master processing. Similarity uses the same held-out development reference subset for every variant. Pitch is an autocorrelation estimate over voiced frames; it is less reliable for breathy ASMR. Source-reference speaking style differs, so pitch distance alone is not a success criterion.

No variant is promoted. Speaker embedding swaps improve the similarity proxy but reduce the cleanliness proxy. Full acoustic conditions raise pitch but reduce similarity. These results do not prove realism. Listen at `index.html`; detailed measurements and caveats are in `review.json`.
