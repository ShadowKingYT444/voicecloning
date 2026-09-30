import copy,hashlib,json
from pathlib import Path
import torch
root=Path('artifacts/nano_lab');out=root/'t3_reference_matrix_caches';out.mkdir(exist_ok=True)
a=root/'decoder_embedding_fit/conditionals.pt';b=root/'sweep_round2/asmr_intimate_morning_31.conds.pt'
base=torch.load(a,map_location='cpu',weights_only=True);donor=torch.load(b,map_location='cpu',weights_only=True)
def hashes(value):
 if isinstance(value,dict):return {k:hashes(v) for k,v in value.items()}
 if value is None:return None
 return {'shape':list(value.shape),'dtype':str(value.dtype),'sha256':hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()}
report={'base':str(a.resolve()),'base_sha256':hashlib.sha256(a.read_bytes()).hexdigest(),'donor':str(b.resolve()),'donor_sha256':hashlib.sha256(b.read_bytes()).hexdigest(),'donor_reference':'asmr7_bully_01__natural.wav','donor_interval_s':[117.7,129.0],'heldout01_interval_s':[92.1,102.1],'variants':{}}
for mode in ('prompt','speaker','all_t3'):
 p=copy.deepcopy(base)
 if mode=='all_t3':p['t3']=copy.deepcopy(donor['t3'])
 elif mode=='speaker':p['t3']['speaker_emb']=donor['t3']['speaker_emb'].clone()
 else:p['t3']['cond_prompt_speech_tokens']=donor['t3']['cond_prompt_speech_tokens'].clone()
 p['t3']['cond_prompt_speech_emb']=None
 assert hashes(p['gen'])==hashes(base['gen'])
 path=out/(mode+'.conds.pt')
 if path.exists():raise FileExistsError(path)
 torch.save(p,path);report['variants'][mode]={'path':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'decoder_gen_exact':True,'t3':hashes(p['t3'])}
(out/'report.json').write_text(json.dumps(report,indent=2));print('Created three reference variants; fitted decoder unchanged.')
