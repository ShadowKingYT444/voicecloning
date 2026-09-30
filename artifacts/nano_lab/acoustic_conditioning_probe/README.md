# Acoustic reference probe

This experiment reuses speech tokens from the source recording. The self-reference variants deliberately use the target audio. They are diagnostic upper bounds, not zero-shot or new-text voice clones.

| Conditioning | Median voiced pitch, seed 10031 | Seed 10097 | Small word error rate |
|---|---:|---:|---:|
| Conversational reference | 235.2 Hz | 233.9 Hz | 10% / 10% |
| Source decoder embedding only | 264.0 Hz | 253.6 Hz | 10% / 10% |
| Source prompt pair only | 246.2 Hz | 245.8 Hz | 0% / 0% |
| Complete source reference | 268.4 Hz | 264.1 Hz | 0% / 0% |

The source median is 258.9 Hz. Source-mel vocoding measures 260.0 Hz. Both source and generated speech have about 30% detected voiced frames, so these pitch estimates are descriptive and can be unreliable in whispered or breathy speech.

All prompt feature tensors have shape [1,500,80]. The embedding-only variants preserve all T3 tensors and every other decoder tensor exactly. All variants use the same genuine source token IDs and two fixed noise seeds.

The speaker embedding changes pitch. It does not fix the baseline transcript difference: the recognizer hears "to get" instead of "who gets". The source-prompt variants pass the transcript check, but reference content leakage prevents treating that as a new-text accuracy improvement. The source-mel reconstruction has one recognizer substitution ("got" for "gets").

The selected conversational reference itself measures 204.8 Hz, while other reference excerpts measure about 219 to 251 Hz. This makes reference style a material confound. The next test uses an independent reference embedding for new text, with fixed T3 conditions and fitted weights.

Peak process RSS: 2665.20 MiB. The job completed under the 3072 MiB guard. No job swap was used.

Evidence: manifest.json, review.json, prosody.json, small_audit.json.
