import sys,json
from pathlib import Path
sys.path.insert(0,str(Path('scripts/nano_lab').resolve()))
import torch,numpy as np
import fit_decoder_embedding as f
root=Path('artifacts/nano_lab');m,rows,cp,cache,mp=f._validate_prepare_inputs(root/'mel_calibration_asmr');row=rows[0]
payload=f._load_conditionals_payload(torch,cp);enc,est,loader=f._stream_s3gen_modules(torch,Path('models/chatterbox-nano').resolve(),torch.device('cuda'))
state=f._prepare_flow_row(torch,enc,payload,row,'cuda');noise=f._fixed_noise(torch,state,seed=row['seed']);embedding=payload['gen']['embedding'].cuda()
with torch.no_grad():manual=f._basic_euler(torch,est,state,embedding,noise,steps=2)[:,:,state['prompt_len']:].cpu().numpy()
from chatterbox.models.s3gen.flow import CausalMaskedDiffWithXvec
from chatterbox.models.s3gen.flow_matching import CausalConditionalCFM
with torch.device('meta'):
 decoder=CausalConditionalCFM(spk_emb_dim=80,estimator=est.estimator)
 flow=CausalMaskedDiffWithXvec(encoder=enc.encoder,decoder=decoder)
flow.input_embedding=enc.input_embedding;flow.encoder_proj=enc.encoder_proj;flow.spk_embed_affine_layer=est.spk_embed_affine_layer;flow.eval()
captured={};original=est.estimator.forward
def hook(x,*args,**kwargs):
 if not captured:
  captured['x']=x.detach().clone()
  for k in ('mu','mask','cond','spks'):captured[k]=kwargs[k].detach().clone()
 return original(x,*args,**kwargs)
est.estimator.forward=hook
gen={k:(v.cuda() if torch.is_tensor(v) else v) for k,v in payload['gen'].items()}
tokens=torch.from_numpy(f._append_silence_tokens(row['tokens'])).cuda().view(1,-1)
torch.manual_seed(row['seed']);torch.cuda.manual_seed_all(row['seed'])
with torch.inference_mode(): native,_=flow.inference(token=tokens,token_len=torch.tensor([tokens.shape[1]],device='cuda'),finalize=True,n_timesteps=2,meanflow=True,**gen)
est.estimator.forward=original
native=native.cpu().numpy();expected=row['prepared_reconstructed_mel'][None]
def diff(a,b):
 a=np.asarray(a);b=np.asarray(b);return {'shape_a':list(a.shape),'shape_b':list(b.shape),'max_abs':float(np.max(np.abs(a-b))),'rmse':float(np.sqrt(np.mean((a-b)**2)))}
report={'manual_native':diff(manual,native),'native_saved':diff(native,expected),'manual_saved':diff(manual,expected),'noise':diff(noise.cpu().numpy(),captured['x'].cpu().numpy())}
for k in ('mu','mask','cond'):report[k]=diff(state[k].cpu().numpy(),captured[k].cpu().numpy())
report['seed']=row['seed'];report['peak_rss_mib']=f._peak_rss_bytes()/1048576
(root/'decoder_flow_parity_probe.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
