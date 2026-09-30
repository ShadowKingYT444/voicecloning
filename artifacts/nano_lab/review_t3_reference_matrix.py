import json,statistics,html,os
from pathlib import Path
root=Path('artifacts/nano_lab'); out=root/'t3_reference_matrix'
evals={r['label']:r for r in json.loads((out/'evaluation_matched.json').read_text())['inputs']}
asrs={r['id']:r for r in json.loads((out/'small_audit.json').read_text())['inputs']}
levels={r['id']:r for r in json.loads((out/'level_matched/manifest.json').read_text())}
base=json.loads((root/'decoder_cadence_combined/review.json').read_text())['groups']['decoder_slower']
groups={'unchanged_t3':base}
for mode in ('prompt','speaker','all_t3'):
 rows=[]
 for passage in ('heldout01','question','narrative'):
  name=f'{passage}_{mode}';e=evals[name];a=asrs[name];l=levels[name]
  rows.append(dict(id=name,speaker_cosine=e['speaker_similarity']['mean_cosine'],dnsmos_overall=e['dnsmos_overall'],small_wer=a['wer_contraction_normalized']['wer'],seconds=e['measurements']['duration_s'],listening_path=l['path']))
 groups[mode]={'rows':rows,'mean':{k:statistics.mean(r[k] for r in rows) for k in ('speaker_cosine','dnsmos_overall','small_wer','seconds')}}
report=dict(kind='t3_reference_factorial_fixed_fitted_decoder',promoted=False,human_listening_accepted=False,general_zero_shot_improvement=False,groups=groups,decision='All-T3 rejected for two word errors. Prompt-only retained as listening candidate; improves quality proxy but loses average identity against unchanged T3.',controls='Same three texts/seeds, all-attention adapter, fitted decoder, spectral correction, RP1.0, -27 LUFS, same independent development references.')
(out/'review.json').write_text(json.dumps(report,indent=2)+'\n')
style='body{font:17px system-ui;max-width:940px;margin:40px auto;padding:0 22px;background:#12151b;color:#eef1f7}p{line-height:1.55}a{color:#a8cbff}article{background:#202633;border-radius:12px;padding:18px;margin:12px 0}audio{display:block;width:100%;margin-top:12px}td,th{padding:10px;text-align:left}summary{cursor:pointer}'
parts=[f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ASMR reference comparison</title><style>{style}</style><h1>ASMR reference comparison</h1><p>Generated clips are synthetic speaker-specific experiments. Realism remains unaccepted. All players use -27 LUFS. The prompt-only candidate changes the reference speech tokens supplied to the text model. The fitted acoustic decoder is unchanged.</p>']
def player(label,path):
 path=Path(path).resolve();assert path.exists(),path
 return f'<article><strong>{html.escape(label)}</strong><audio controls preload="none" src="{html.escape(os.path.relpath(path,out.resolve()))}"></audio></article>'
parts+=['<h2>Same words as the source</h2>',player('Original source excerpt',root/'cadence_penalty/level_matched/source_heldout01.wav')]
for i,passage in enumerate(('heldout01','question','narrative')):
 if i:parts.append(f'<details><summary>Additional passage {i}</summary>')
 parts.append('<p>'+html.escape(levels[f'{passage}_prompt']['text'])+'</p>')
 parts.append(player('Fitted decoder, unchanged text-model reference',base['rows'][i]['listening_path']))
 parts.append(player('Prompt-only reference change',levels[f'{passage}_prompt']['path']))
 parts.append('<details><summary>Other diagnostic variants</summary>')
 for mode,label in [('speaker','Speaker-vector reference change'),('all_t3','Both reference changes (word errors in two passages)')]:parts.append(player(label,levels[f'{passage}_{mode}']['path']))
 parts.append('</details>')
 if i:parts.append('</details>')
parts.append('<h2>Matched checks across three passages</h2><p>These are automatic proxies. Higher scores do not establish realism. The full reference change fails the word check in two passages.</p><table><tr><th>Variant</th><th>Similarity</th><th>DNSMOS</th><th>Exact transcripts</th></tr>')
for name,g in groups.items():
 m=g['mean'];n=sum(r['small_wer']==0 for r in g['rows']);parts.append(f'<tr><td>{html.escape(name)}</td><td>{m["speaker_cosine"]:.3f}</td><td>{m["dnsmos_overall"]:.3f}</td><td>{n}/3</td></tr>')
parts.append('</table><p><a href="review.json">Measurements</a> · <a href="../decoder_cadence_combined/index.html">Decoder comparison</a> · <a href="../RSS_REPORT.md">RSS report</a> · <a href="../delivery/index.html">Main voice page</a></p></html>')
(out/'index.html').write_text(''.join(parts))
for name,g in groups.items():print(name,g['mean'])
