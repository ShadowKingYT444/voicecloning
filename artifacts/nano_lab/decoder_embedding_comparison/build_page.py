from pathlib import Path
import html,json
p=Path(__file__).resolve().parent
rows=json.loads((p/'level_matched/manifest.json').read_text())
by_id={r['id']:r for r in rows}
def player(id,label):
 r=by_id[id]
 return '<article><strong>'+html.escape(label)+'</strong><audio controls preload="none" src="level_matched/'+html.escape(id)+'.wav"></audio></article>'
parts=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Decoder voice comparisons</title><style>body{font:17px system-ui;max-width:960px;margin:40px auto;padding:0 22px;background:#12151b;color:#eef1f7}p{line-height:1.55;color:#ccd4e3}a{color:#a8cbff}article{background:#202633;border-radius:12px;padding:18px;margin:12px 0}audio{display:block;width:100%;margin-top:14px}h2{margin-top:36px}</style><h1>Decoder voice comparisons</h1><p>All generated clips are synthetic experiments. Realism remains unverified. The same fitted text model produces identical speech tokens for all four options in each passage. Only the acoustic reference conditions change.</p><p>Every player uses equal measured loudness ('+str(round(rows[0]['target_lufs'],2))+' LUFS). Listening copies use constant gain, with no additional filter, gate, or limiter.</p><h2>Source references</h2>',player('source_conversational_reference','Conversational source reference'),player('source_bully_reference','More expressive source reference')]
labels={'baseline':'Current fitted voice','embedding_half':'Half of the expressive speaker embedding','embedding_full':'Expressive speaker embedding','full_gen':'Expressive speaker embedding and acoustic prompt'}
for passage in ('morning','question','narrative'):
 parts += ['<h2>'+passage.title()+'</h2><p>'+html.escape(by_id[passage+'_baseline']['text'])+'</p>']
 for mode,label in labels.items():parts.append(player(passage+'_'+mode,label))
parts.append('<p>The source references are separate from the generated text. This is a speaker-specific fitted comparison, not evidence of general zero-shot improvement. Automated results are available in <a href="review.json">the review report</a>. <a href="../delivery/index.html">Return to the main samples and RSS report</a>.</p></html>')
(p/'index.html').write_text(''.join(parts))
