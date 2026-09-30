"""Build a measured attention-adapter comparison from completed artifacts."""
import argparse
import hashlib
import html
import json
import os
import statistics
from pathlib import Path

from review_decoder_projection import _token_payload_signature

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/nano_lab/decoder_attention_comparison'


def read(name):
    return json.loads((OUT / name).read_text())


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prompt', action='store_true')
    args = parser.parse_args()
    if args.prompt:
        OUT = ROOT / 'artifacts/nano_lab/decoder_attention_prompt_comparison_v2'
    texts = ('question', 'narrative') if args.prompt else ('morning', 'question', 'narrative')
    variants = ('mel',) if args.prompt else ('raw', 'mel')
    prefix = 'prompt_' if args.prompt else ''
    manifest = read('manifest.json')
    levels = {r['id']: r for r in read('level_matched/manifest.json')}
    audits = {r['id']: r for r in read('small_audit.json')['inputs']}
    evaluation = {r['label']: r for r in read('evaluation_matched.json')['inputs']}
    expected = {f'{text}_{prefix}attention{strength}_{mel}' for text in texts
                for strength in (0, 1) for mel in variants}
    expected |= {f'heldout01_{prefix}attention{strength}_mel' for strength in (0, 1)}
    assert len(manifest) == len(expected) and {r['id'] for r in manifest} == expected
    rows = []
    adapter_hashes = set()
    for r in manifest:
        key = r['id']
        assert not r.get('error'), r
        assert r['precision'] == dict(matmul_tf32=False, cudnn_tf32=False)
        runtime = r['decoder_attention_runtime']
        if args.prompt:
            donor = r['t3_donor_runtime']
            assert donor['gen_untouched'] and donor['selected_fields'] == ['t3.cond_prompt_speech_tokens']
            assert r['repetition_penalty'] == 1.0
        enabled = r['decoder_attention_strength'] > 0
        assert runtime['enabled'] == enabled
        if enabled:
            assert runtime['strength'] == r['decoder_attention_strength']
            adapter_hashes.add(runtime['adapter_sha256'])
        level, audit, score = levels[key], audits[key], evaluation[key]
        assert hashlib.sha256(Path(r['path']).read_bytes()).hexdigest() == audit['audio_sha256']
        assert Path(level['path']).resolve() == Path(score['audio_path']).resolve()
        assert abs(level['achieved_lufs'] - level['target_lufs']) <= .01
        rows.append(dict(id=key, text=r['text'], path=level['path'],
                         level_audio_sha256_at_review=hashlib.sha256(Path(level['path']).read_bytes()).hexdigest(),
                         wer=audit['wer_contraction_normalized']['wer'],
                         identity_cosine=score['speaker_similarity']['mean_cosine'],
                         dnsmos_overall=score['dnsmos_overall'], seconds=r['seconds'],
                         generation_seconds=r['generation_seconds'], peak_rss_mib=r['peak_rss_mib']))
    assert len(adapter_hashes) == 1
    token_checks = {}
    for text in (*texts, 'heldout01'):
        signatures = {r['id']: _token_payload_signature(Path(r['path']).with_suffix('.tokens.pt'))
                      for r in manifest if r['id'].startswith(text + '_')}
        assert len(set(signatures.values())) == 1, signatures
        token_checks[text] = dict(exact_equal=True, signatures=signatures)
    groups = {}
    for strength in (0, 1):
        for mel in variants:
            key = f'attention{strength}_{mel}'
            cohort = [r for r in rows if r['id'].endswith(key) and not r['id'].startswith('heldout')]
            assert len(cohort) == len(texts)
            groups[key] = {metric: statistics.mean(r[metric] for r in cohort)
                           for metric in ('identity_cosine', 'dnsmos_overall', 'wer', 'seconds', 'generation_seconds')}
    fit = json.loads((ROOT / 'artifacts/nano_lab/decoder_attention_fit/fit_report.json').read_text())
    fold = json.loads((ROOT / 'artifacts/nano_lab/decoder_attention_fit/fold_verification.json').read_text())
    assert fold['status'] == 'passed' and fold['adapter_sha256'] in adapter_hashes
    report = dict(status='experimental_not_promoted', human_listening=False, prompt_donor=args.prompt, rows=rows, groups=groups,
                  token_checks=token_checks, all_wer_zero=all(r['wer'] == 0 for r in rows),
                  peak_process_rss_mib=max(r['peak_rss_mib'] for r in rows),
                  fit=dict(best_step=fit['best_step'], epoch0_valid=fit['epoch0']['valid']['mean']['total'],
                           best_valid=fit['best_valid_total'], peak_rss_bytes=fit['peak_rss_bytes']),
                  fold_verification=fold, prosody=read('prosody.json'))
    (OUT / 'review.json').write_text(json.dumps(report, indent=2))
    def player(path):
        return '<audio controls preload="metadata" src="' + html.escape(os.path.relpath(path, OUT), quote=True) + '"></audio>'
    source = ROOT / 'artifacts/nano_lab/cadence_penalty/level_matched/source_heldout01.wav'
    body = ['<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Acoustic attention comparison</title>',
            '<style>body{max-width:1000px;margin:40px auto;padding:0 24px;font:16px/1.5 system-ui;color:#20252c;background:#faf9f6}audio{width:100%}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:20px}article{background:white;padding:18px;border:1px solid #ddd;border-radius:12px}table{border-collapse:collapse}td,th{padding:9px;border-bottom:1px solid #ddd}</style>',
            '<h1>Acoustic attention comparison</h1><p>Experimental synthetic speech. Realism remains unaccepted. All listening copies use constant gain at the same loudness. Generated masters use the same 45 Hz high-pass filter. No noise gate or denoiser was added.</p>',
            ('<p>Both variants use the new reference prompt and repetition penalty 1.0. Only the acoustic adapter changes.</p>' if args.prompt else ''),
            '<h2>Same words as the source</h2><div class="grid"><article><h3>Source recording</h3>' + player(source) + '</article>']
    for r in rows:
        if r['id'].startswith('heldout'):
            body.append('<article><h3>' + ('Attention adapter' if 'attention1' in r['id'] else 'Control') + '</h3>' + player(r['path']) + f"<p>{r['seconds']:.2f} seconds</p></article>")
    change_description = 'Only the acoustic attention adapter changes.' if args.prompt else 'Only the acoustic attention adapter and mel correction change.'
    body += ['</div><h2>New text</h2><p>Each group has identical generated speech tokens. ' + change_description + '</p>']
    for text in texts:
        subset = [r for r in rows if r['id'].startswith(text + '_')]
        body += ['<h3>' + html.escape(subset[0]['text']) + '</h3><div class="grid">']
        for r in subset:
            body.append('<article><strong>' + html.escape(r['id'].split('_', 1)[1]) + '</strong>' + player(r['path']) + '</article>')
        body.append('</div>')
    body.append('<h2>Measured results</h2><p>These are automated proxies. They do not establish realism or clean ASMR texture.</p><table><tr><th>Variant</th><th>Speaker cosine</th><th>DNSMOS</th><th>WER</th></tr>')
    for key, values in groups.items():
        body.append(f"<tr><td>{key}</td><td>{values['identity_cosine']:.4f}</td><td>{values['dnsmos_overall']:.4f}</td><td>{values['wer']:.3f}</td></tr>")
    body.append(f"</table><p>Peak synthesis process RSS: {report['peak_process_rss_mib']:.1f} MiB. One serial model job, 3072 MiB hard limit, no swap.</p><p><a href='review.json'>Full evidence</a> | <a href='../delivery/index.html'>Main comparison</a></p>")
    (OUT / 'index.html').write_text('\n'.join(body))
    print(json.dumps({k: report[k] for k in ('groups', 'all_wer_zero', 'peak_process_rss_mib')}, indent=2))


if __name__ == '__main__':
    main()
