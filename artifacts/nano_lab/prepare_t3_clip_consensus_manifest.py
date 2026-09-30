"""Derive a separate dataset from existing clip-level consensus, without new labels."""
import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/nano_lab'))
from dataset_contract import require_audited_rows


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


source = ROOT / 'artifacts/nano_lab/dataset_repair_clip/adaptation_repaired.json'
proposal = ROOT / 'artifacts/nano_lab/dataset_repair_clip/proposal_rebased.json'
original = ROOT / 'artifacts/nano_lab/dataset_repair_complete/adaptation_repaired.json'
output = ROOT / 'artifacts/nano_lab/t3_clip_consensus'
if output.exists():
    raise FileExistsError('Preserve prior artifacts: ' + str(output))
data = json.loads(source.read_text())
protected = json.loads(proposal.read_text())['protected_intervals']
prior = json.loads(original.read_text())
old_ids = {r['id'] for r in prior['rows']}
old_train = {r['id'] for r in prior['rows'] if r['split'] == 'train'}
valid_ids = {'asmr7_repair_022', 'asmr7_repair_030'}
if old_train & valid_ids:
    raise ValueError('New validation must not include a previous T3 training row')
rows, excluded, evidence = [], [], {}
for row in data['rows']:
    conflicts = [r['id'] for r in protected
                 if row['start_s'] < r['end_s'] + 2 and row['end_s'] > r['start_s'] - 2]
    if conflicts:
        excluded.append(dict(id=row['id'], reason='protected_interval_2s_buffer', intervals=conflicts))
        continue
    if not row['clip_tiny_small_consensus']['exact'] or row['rebase']['quality_reasons']:
        raise ValueError('Clip consensus gate missing: ' + row['id'])
    if row.get('synthetic_tts', False):
        raise ValueError('Synthetic source rejected')
    for entry in (row['rebase']['tiny_audit'], row['rebase']['small_audit']):
        path = entry['report']
        if path not in evidence:
            evidence[path] = sha(path)
        if evidence[path] != entry['report_sha256']:
            raise ValueError('Changed ASR report: ' + path)
    row = copy.deepcopy(row)
    row['split'] = 'valid' if row['id'] in valid_ids else 'train'
    rows.append(row)
require_audited_rows(rows)
if {r['id'] for r in rows if r['split'] == 'valid'} != valid_ids:
    raise ValueError('Missing validation row')
ordered = sorted(rows, key=lambda r: r['start_s'])
if any(a['end_s'] > b['start_s'] for a, b in zip(ordered, ordered[1:])):
    raise ValueError('Overlapping target rows')
counts = {k: sum(r['split'] == k for r in rows) for k in ('train', 'valid')}
counts['all'] = len(rows)
duration = {k: round(sum(r['duration_s'] for r in rows if r['split'] == k), 3) for k in ('train', 'valid')}
data.update(rows=rows, counts=counts, generated_by=str(Path(__file__).resolve()),
    production_ready=False, human_verified=False,
    provenance=dict(derivation='subset_of_existing_clip_consensus_manifest',
        parent_manifest=str(source), parent_manifest_sha256=sha(source),
        previous_manifest=str(original), previous_manifest_sha256=sha(original),
        proposal_sha256=sha(proposal), protected_interval_buffer_s=2,
        protected_intervals=protected, excluded=excluded, evidence_hashes=evidence,
        valid_ids=sorted(valid_ids), duration_s=duration,
        new_ids=sorted({r['id'] for r in rows}-old_ids),
        removed_prior_ids=sorted(old_ids-{r['id'] for r in rows}),
        limitation='Validation 022 was used previously; 030 was not used for T3 training. Both occur in acoustic decoder fitting data. No general unseen-voice claim.'))
output.mkdir()
(output/'manifest.json').write_text(json.dumps(data, indent=2))
(output/'derivation_report.json').write_text(json.dumps(dict(status='metadata_verified', counts=counts,
    duration_s=duration, provenance=data['provenance'], prepared_features=False, trained=False), indent=2))
print(json.dumps(dict(counts=counts, duration_s=duration, excluded=excluded,
    new_ids=data['provenance']['new_ids']), indent=2))
