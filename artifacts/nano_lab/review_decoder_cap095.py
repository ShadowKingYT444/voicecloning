"""Artifact-only comparison of speaker-vector constraints, not a promotion gate."""
import html
import json
import os
import statistics
from pathlib import Path
from review_decoder_projection import _token_payload_signature

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'decoder_embedding_cap095_comparison'


def read(path):
    return json.loads(path.read_text())


def index(path, key='id'):
    rows = read(path)
    if isinstance(rows, dict):
        rows = rows['inputs']
    return {r[key]: r for r in rows}


def main():
    generated = index(OUT / 'manifest.json')
    assert len(generated) == 8 and all('error' not in r for r in generated.values())
    levels = index(OUT / 'level_matched/manifest.json')
    audit = index(OUT / 'small_audit.json')
    evaluated = index(OUT / 'evaluation_matched.json', 'label')
    base_dir = ROOT / 'decoder_projection_comparison'
    base_levels = index(base_dir / 'level_matched/manifest.json')
    base_eval = index(base_dir / 'evaluation_matched.json', 'label')
    base_audit = index(base_dir / 'small_audit.json')
    groups, parity = {}, {}
    for cap in ('098', '095'):
        for mel in ('raw', 'mel'):
            rows = []
            for text in ('morning', 'question', 'narrative'):
                name = f'{text}_cap095_{mel}' if cap == '095' else f'{text}_projection0_{mel}'
                es, ls, audits = (evaluated, levels, audit) if cap == '095' else (base_eval, base_levels, base_audit)
                e, l, a = es[name], ls[name], audits[name]
                rows.append(dict(id=name, speaker_cosine=e['speaker_similarity']['mean_cosine'],
                                 dnsmos_overall=e['dnsmos_overall'], small_wer=a['wer_contraction_normalized']['wer'],
                                 listening_path=l['path']))
                if cap == '095':
                    parity[name] = _token_payload_signature(OUT / f'{name}.tokens.pt') == _token_payload_signature(base_dir / f'{text}_projection0_{mel}.tokens.pt')
            groups[f'cap{cap}_{mel}'] = dict(rows=rows, mean={k: statistics.mean(r[k] for r in rows) for k in ('speaker_cosine', 'dnsmos_overall', 'small_wer')})
    heldout_base = ROOT / 'decoder_cadence_combined'
    parity['heldout01_cap095_mel'] = _token_payload_signature(OUT / 'heldout01_cap095_mel.tokens.pt') == _token_payload_signature(heldout_base / 'fit_heldout01_rp1p2.tokens.pt')
    parity['heldout01_cap095_raw'] = _token_payload_signature(OUT / 'heldout01_cap095_raw.tokens.pt') == _token_payload_signature(OUT / 'heldout01_cap095_mel.tokens.pt')
    assert all(parity.values()), parity
    report = dict(status='experimental_not_promoted', human_realism_verified=False, general_zero_shot_improvement=False,
                  groups=groups, token_payload_parity=parity, peak_process_rss_mib=max(r['peak_rss_mib'] for r in generated.values()),
                  all_eight_exact_transcripts=all(a['wer_contraction_normalized']['wer'] == 0 for a in audit.values()),
                  fit=read(ROOT / 'decoder_embedding_fit_cap095/fit_report.json')['best_valid_total'],
                  controls='Same text, seeds, T3 adapter, optional mel correction, and reference cohort. Listening copies at -27 LUFS. Earlier baseline artifacts retained.')
    prosody = read(OUT / 'prosody.json')
    assert prosody['error_count'] == 0
    report['same_source_phrase'] = {r['id']: dict(duration_s=r['duration_s'], median_f0_hz=r['pitch']['f0_hz']['median'],
        voiced_fraction=r['pitch']['voiced_frame_fraction']) for r in prosody['rows'] if 'heldout01' in r['id']}
    report['pitch_method'] = prosody['pitch']
    (OUT / 'review.json').write_text(json.dumps(report, indent=2) + '\n')
    style = 'body{font:17px system-ui;max-width:940px;margin:40px auto;padding:0 22px;background:#12151b;color:#eef1f7}p{line-height:1.55}a{color:#a8cbff}article{background:#202633;border-radius:12px;padding:18px;margin:12px 0}audio{display:block;width:100%;margin-top:12px}td,th{padding:10px;text-align:left}'
    parts = [f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ASMR speaker-vector comparison</title><style>{style}</style><h1>ASMR speaker-vector comparison</h1><p>These are synthetic, speaker-specific experiments. Realism remains unaccepted. The new fit permits a larger change in the decoder speaker vector. Both fits use the same 15 training and two validation clips. Every player uses -27 LUFS.</p>']
    def player(label, path):
        path = Path(path).resolve()
        assert path.exists(), path
        parts.append(f'<article><strong>{html.escape(label)}</strong><audio controls preload="none" src="{html.escape(os.path.relpath(path, OUT))}"></audio></article>')
    parts.append('<h2>Same words as the source</h2><p>' + html.escape(generated['heldout01_cap095_mel']['text']) + '</p>')
    player('Excluded source excerpt', ROOT / 'cadence_penalty/level_matched/source_heldout01.wav')
    player('Previous fit, cosine limit 0.98, mel correction', heldout_base / 'level_matched/fit_heldout01_rp1p2.wav')
    player('New fit, cosine limit 0.95, mel correction', levels['heldout01_cap095_mel']['path'])
    player('New fit, cosine limit 0.95, no mel correction', levels['heldout01_cap095_raw']['path'])
    for text in ('morning', 'question', 'narrative'):
        parts.append('<details><summary>' + html.escape(text.title()) + '</summary><p>' + html.escape(generated[f'{text}_cap095_mel']['text']) + '</p>')
        for cap in ('098', '095'):
            for mel in ('raw', 'mel'):
                row = next(r for r in groups[f'cap{cap}_{mel}']['rows'] if r['id'].startswith(text + '_'))
                player(f'Cosine limit 0.{cap[1:]}, {mel}', row['listening_path'])
        parts.append('</details>')
    parts.append('<h2>Three-passage automatic measurements</h2><p>These measurements do not establish realism or professional audio quality.</p><table><tr><th>Variant</th><th>Similarity</th><th>DNSMOS</th><th>Exact words</th></tr>')
    for name, group in groups.items():
        mean = group['mean']
        parts.append(f'<tr><td>{name}</td><td>{mean["speaker_cosine"]:.4f}</td><td>{mean["dnsmos_overall"]:.3f}</td><td>{sum(r["small_wer"] == 0 for r in group["rows"])}/3</td></tr>')
    parts.append('</table><p><a href="review.json">Measurements</a> · <a href="../RSS_REPORT.md">RSS report</a> · <a href="../delivery/index.html">Main voice page</a></p></html>')
    (OUT / 'index.html').write_text(''.join(parts))
    print(json.dumps({k: v['mean'] for k, v in groups.items()}, indent=2))


if __name__ == '__main__':
    main()
