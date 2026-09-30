import json,sys,hashlib,types
from pathlib import Path
root=Path('/home/terryd/gooning/voicecloning');sys.path.insert(0,str(root/'scripts/nano_lab'))
import torch
from safetensors import safe_open
from decoder_projection import configure_decoder_projection
torch.set_num_threads(2)
p=root/'artifacts/nano_lab/decoder_projection_fit/best_projection.pt'
weights=root/'models/chatterbox-nano/s3gen_meanflow.safetensors'
cache=root/'artifacts/nano_lab/decoder_embedding_fit/conditionals.pt'
payload=torch.load(p,map_location='cpu',weights_only=True)
with safe_open(weights,framework='pt',device='cpu') as f:
 base=f.get_tensor('flow.decoder.estimator.final_proj.weight').clone()
 bias=f.get_tensor('flow.decoder.estimator.final_proj.bias').clone()
projection=torch.nn.Conv1d(256,80,1)
with torch.no_grad():projection.weight.copy_(base);projection.bias.copy_(bias)
model=types.SimpleNamespace(s3gen=types.SimpleNamespace(flow=types.SimpleNamespace(decoder=types.SimpleNamespace(estimator=types.SimpleNamespace(final_proj=projection)))))
torch.manual_seed(71);hidden=torch.randn(1,256,19)
with torch.no_grad():
 baseline=projection(hidden)
 residual=torch.nn.functional.conv1d(torch.nn.functional.conv1d(hidden,payload['down_weight']),payload['up_weight'])*(payload['alpha']/payload['rank'])
metadata=configure_decoder_projection(model,p,conditionals_path=cache,model_checkpoint_path=weights)
with torch.no_grad():actual=projection(hidden);error=float((actual-(baseline+residual)).abs().max())
assert error<1e-5,error
first=projection.weight.detach().clone();configure_decoder_projection(model,p,conditionals_path=cache,model_checkpoint_path=weights)
assert torch.equal(first,projection.weight)
configure_decoder_projection(model)
assert torch.equal(base,projection.weight)
report={'status':'passed','scope':'actual saved adapter folded into actual source final-projection weight; isolated layer only, not full decoder inference or perceptual acceptance','best_step':payload['step'],'valid':payload['valid'],'factorized_vs_folded_max_abs':error,'repeat_application_exact':True,'restore_exact':True,'metadata':metadata}
(root/'artifacts/nano_lab/decoder_projection_fit/artifact_verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='metadata'}))
