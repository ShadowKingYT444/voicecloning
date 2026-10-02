"""Summarize completed VM experiments and build an offline listening index.

No models or waveform processing run here. Run level_match.py separately
through the resource guard before calling this artifact compiler.
"""
import argparse
import hashlib
import html
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-dir', type=Path, required=True)
    args = parser.parse_args()
    root = args.experiment_dir.resolve()
    def read(name):
        return json.loads((root/name).read_text())
    phases = [('paired', 'eval_inputs.json', 'quality.json', 'dnsmos.json', 'asr.json'),
              ('adaptation', 'adaptation_eval_inputs.json', 'adaptation_quality.json', 'adaptation_dnsmos.json', 'adaptation_asr.json'),
              ('harvey_sampling', 'harvey_eval_inputs.json', 'harvey_quality.json', 'harvey_dnsmos.json', 'harvey_asr.json')]
    rows = []
    for phase, inputs, quality, mos, asr in phases:
        q = {r['label']: r for r in read(quality)['inputs']}
        d = {r['id']: r for r in read(mos)}
        a = {r['id']: r for r in read(asr)['inputs']}
        for r in read(inputs):
            rows.append({**r, 'phase': phase,
                         'speaker_cosine': q[r['id']]['speaker_similarity']['mean_cosine'],
                         'dnsmos': d[r['id']]['dnsmos_overall'],
                         'wer': a[r['id']]['wer_contraction_normalized']['wer'],
                         'transcript': a[r['id']]['transcript']})
    numeric = {r['id']: r for r in read('harvey_asr_number_equivalence.json')['inputs']}
    for r in rows:
        if r['id'] in numeric:
            r['number_equivalent_wer'] = numeric[r['id']]['number_equivalent_wer']
    def mean(group, key):
        return statistics.mean(r[key] for r in group)
    reference_groups = {}
    for voice in ('asmr', 'harvey'):
        for variant in ('raw', 'natural', 'clean'):
            group = [r for r in rows if r['phase']=='paired' and r['reference_id']==voice+'_'+variant]
            reference_groups[voice+'_'+variant] = {'n': len(group), 'speaker_cosine': mean(group, 'speaker_cosine'),
                       'dnsmos': mean(group, 'dnsmos'), 'word_exact': sum(r['wer']==0 for r in group)}
    fit_groups = {}
    for label in ('base', 'fit'):
        group = [r for r in rows if r['phase']=='adaptation' and r['id'].endswith('_'+label)]
        fit_groups[label] = {'n': len(group), 'speaker_cosine': mean(group, 'speaker_cosine'),
                           'dnsmos': mean(group, 'dnsmos'), 'word_exact': sum(r['wer']==0 for r in group)}
    harvey_groups = {}
    for label in ('temp08', 'temp06'):
        group = [r for r in rows if r['phase']=='harvey_sampling' and r['id'].endswith('_'+label)]
        harvey_groups[label] = {'n': len(group), 'speaker_cosine': mean(group, 'speaker_cosine'),
            'dnsmos': mean(group, 'dnsmos'), 'raw_word_exact': sum(r['wer']==0 for r in group),
            'number_equivalent_word_exact': sum(r['number_equivalent_wer']==0 for r in group)}
    training = read('adapter_clip_consensus_all_attn_report.json')
    peak = max(read(name)['sampled_peak_rss_mib'] for name in
               ('paired_guard.json', 'adaptation_generation_guard.json', 'harvey_generation_guard.json', 'training_guard.json'))
    decisions = {'human_listening_completed': False, 'quality_target_achieved': False, 'promoted_profiles': [],
        'asmr_adapter': {'status': 'not_promoted_identity_proxy_regression',
                        'sha256': hashlib.sha256((root/'adapter_clip_consensus_all_attn.pt').read_bytes()).hexdigest()},
        'asmr_reference': 'natural retained as experimental baseline; clean trades identity for cleanliness',
        'harvey_temperature_06': 'experimental content candidate; slight speaker proxy regression; listening pending'}
    summary = {'synthesis_trials': len(rows), 'reference_groups': reference_groups,
               'adaptation_groups': fit_groups, 'harvey_sampling_groups': harvey_groups,
               'peak_model_job_rss_mib': peak, 'decisions': decisions, 'trials': rows}
    (root/'SUMMARY.json').write_text(json.dumps(summary, indent=2))
    (root/'DECISIONS.json').write_text(json.dumps(decisions, indent=2))
    text = ['# CPU voice experiments — 2026-09-30', '',
        f'{len(rows)} synthesis trials completed. The realism target is not achieved and no voice is promoted. Human listening is pending.', '',
        '## Reference comparison', '', 'Means across two fixed seeds, one new passage per voice. Same text, sampling and acoustic seeds within each voice.', '',
        '| Reference | Speaker cosine | DNSMOS overall | Exact word audits |',
        '|---|---:|---:|---:|']
    for label, r in reference_groups.items():
        text.append(f"| {label} | {r['speaker_cosine']:.4f} | {r['dnsmos']:.3f} | {r['word_exact']}/{r['n']} |")
    text += ['', 'The natural reference has the highest mean identity proxy in both voices. The gated ASMR reference improves the cleanliness proxy while reducing identity. Neither score establishes perceptual realism.', '',
        '## Strict ASMR T3 fit', '',
        f"Twelve training clips and two validation clips retain their original audited audio/text hashes and protected source intervals. Rank-4 attention LoRA trains {training['parameter_counts']['adapter_trainable']:,} values. Eight epochs / 96 optimizer steps complete with base-logit KL coefficient 1.0 and patience 2. Epoch {training['best_epoch']} is best: validation token loss {training['initial_valid']['loss']:.6f} → {training['best_valid_loss']:.6f} ({100*(1-training['best_valid_loss']/training['initial_valid']['loss']):.2f}% lower).", '',
        'Three new-text passages × two fixed seeds compare fresh base and fitted models. Acoustic weights, natural reference, sampling and acoustic seeds remain fixed. The old fitted acoustic comparator is absent and was not used.', '',
        '| Condition | Speaker cosine | DNSMOS overall | Exact word audits |', '|---|---:|---:|---:|']
    for label, r in fit_groups.items():
        text.append(f"| {label} | {r['speaker_cosine']:.4f} | {r['dnsmos']:.3f} | {r['word_exact']}/{r['n']} |")
    text += ['', '**Decision: do not promote this adapter.** Identity cosine drops in five of six paired cases despite lower validation loss and higher mean DNSMOS. It remains archived for review. F0 rises in five cases and durations change; breathy/whispered pitch estimates are unreliable and do not prove improved prosody.', '',
        '## Harvey sampling experiment', '',
        'Three passages × two fixed seeds compare temperature 0.8 and 0.6 with the same natural reference and decoder.', '',
        '| Temperature | Speaker cosine | DNSMOS overall | Content-exact audits¹ |', '|---|---:|---:|---:|']
    for label, r in harvey_groups.items():
        text.append(f"| {label[-2:][0]}.{label[-1]} | {r['speaker_cosine']:.4f} | {r['dnsmos']:.3f} | {r['number_equivalent_word_exact']}/{r['n']} |")
    text += ['', '¹ Raw ASR/WER reports are unchanged. Whisper writes “9” for expected “nine” in all four meeting cases; a separate, explicit number-equivalence annotation excludes this formatting mismatch. Temperature 0.8 still changes “with preparation” to “at preparation” in one case. At 0.6 that flagged mismatch disappears. This is six short development cases, not a general word-error guarantee.', '',
        '**Decision: retain 0.6 as an experimental content candidate.** Mean cleanliness rises slightly, mean speaker cosine falls slightly, and listening remains necessary. No Harvey profile is promoted.', '',
        '## Runtime and verification', '',
        f'All model jobs run serially on two CPU cores with no swap, the unchanged 3072 MiB hard budget, 2500 MiB RSS stop and 4096 MiB desktop reserve. Peak RSS across successful synthesis/training jobs is {peak:.2f} MiB ({peak/1024:.3f} GiB). This is not a 500 MB pipeline. Per-trial latency is in SUMMARY.json; it excludes model loading. RSS is a process high-water mark shared across cases, not an independent per-case memory measurement.', '',
        'The streamed reader matches all 2,662 tensors in the three pinned checkpoints exactly. Deferred shared causal masks pass full and cached GPT-2 numerical parity. Four matched ASMR/Harvey controls match the earlier generated WAV hashes exactly. Four live guard tests pass; a 640 MiB job also exits 125 at its 512 MiB RSS threshold. Fourteen focused regressions plus three subtests pass.', '',
        'Initial mmap/pread opens fail under the address-space cap. A full reference-plus-synthesis load then fails during reference encoding. Tensor streaming and separate native reference encoding reduce the working set without increasing limits. Early watchdog PID-based RSS readings are invalid; only in_process_getrusage reports support memory claims. Combined speaker/DNSMOS scoring exceeds the small address-space budget; separate serial evaluators succeed. PyAV 19 rejects the Whisper API call; pinned 16.1 completes the audits. A first prosody manifest has wrong relative reference paths; the corrected audit completes all 30 original rows with zero errors. These failures are retained.', '',
        '## Listening and continuation', '',
        'Open listening/index.html after extracting the delivery archive. All 42 listening files (36 trials and six references) use constant gain to a shared -27 LUFS target, preserve dynamics, and pass a 4× oversampled -1 dBTP ceiling. There is no delivery denoiser, compressor or noise gate. Raw watermarked generations are also included. Proxy scores use raw generations; matched copies are for listening only.', '',
        'Speaker cosine uses the existing development held-outs (ASMR 02/03, Harvey 01); protected final-audit intervals remain unused for selection. DNSMOS repeats short clips to its 9.01-second window and ASMR is outside its usual domain. The passages, seeds and source recordings are small samples. No claim of general zero-shot improvement or ElevenLabs-level quality is justified.', '',
        'The next useful quality work is acoustic/prosody comparison against the retained natural baseline, with matched new-text words and listening judgments. Repeating this T3 fit or promoting the highest DNSMOS clip would ignore the identity regression. Historical fitted acoustic/ONNX artifacts must be restored before using their profiles as comparators.', '',
        'GitHub branch creation fails with 403 “Resource not accessible by integration.” The changes are preserved as a local commit and an applyable patch in the delivery archive. No remote branch or PR was created. VM absolute paths in archival manifests must be rebased for a different checkout while preserving the recorded hashes.', '']
    (root/'RESULTS.md').write_text('\n'.join(text))
    listening = read('listening/manifest.json')
    by_id = {r['id']: r for r in listening}
    sections = [('ASMR: base versus fitted candidate', [r for r in rows if r['phase']=='adaptation']),
                ('Harvey: sampling temperature', [r for r in rows if r['phase']=='harvey_sampling']),
                ('Reference variants: generated speech', [r for r in rows if r['phase']=='paired']),
                ('Source references', [r for r in listening if r['id'].startswith('reference_')])]
    cards = []
    for heading, group in sections:
        cards.append('<section><h2>'+html.escape(heading)+'</h2><div class="grid">')
        for row in group:
            matched = by_id[row['id']]
            name = Path(matched['path']).name
            cards.append('<article><h3>'+html.escape(row['id'].replace('_', ' '))+'</h3><p>'+html.escape(row['text'])+
                '</p><audio controls preload="none" src="'+html.escape(name)+'"></audio><a download href="'+html.escape(name)+'">Download WAV</a></article>')
        cards.append('</div></section>')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Voice experiments — listening comparison</title><style>
body{margin:0;background:#f6f5f1;color:#172b2a;font:16px/1.5 system-ui,sans-serif}main{max-width:1080px;margin:auto;padding:32px 20px}h1{font-size:32px;line-height:1.15}h2{margin-top:36px}.note{padding:18px;border-left:4px solid #436e67;background:#e8eeea}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(285px,1fr));gap:14px}article{padding:18px;background:white;border:1px solid #d9dfd9;border-radius:10px}h3{font-size:16px;margin:0 0 10px}audio{width:100%;margin:12px 0}a{color:#175b54}p{margin:8px 0}.meta{color:#52625e;font-size:14px}</style>
<main><p class="meta">2026-09-30 · CPU Nano experiments · 36 synthesis trials</p><h1>Voice comparison</h1>
<div class="note">No voice is promoted. The ASMR fit lowers training loss but reduces average identity similarity. Harvey temperature 0.6 improves the content check in this small batch. Listen for identity, breathiness, noise and pacing. All clips are matched to −27 LUFS using constant gain. Only one clip plays at a time.</div>
<p><a href="../RESULTS.md">Read results and limitations</a></p>'''+''.join(cards)+'''</main>
<script>document.addEventListener('play',event=>{if(event.target.tagName==='AUDIO')document.querySelectorAll('audio').forEach(a=>{if(a!==event.target)a.pause()})},true)</script></html>'''
    (root/'listening/index.html').write_text(page)
    print(json.dumps({'synthesis_trials': len(rows), 'listening_files': len(listening), 'peak_model_job_rss_mib': peak,
                      'results': str(root/'RESULTS.md')}))


if __name__ == '__main__':
    main()
