"""Benchmark one cached voice profile in a fresh, externally bounded process."""
import argparse,hashlib,json,resource,time
from pathlib import Path
import numpy as np
import psutil
import soundfile as sf
import torch
from runtime import NanoEngine
from acoustic_controls import configure_acoustics

ROOT=Path(__file__).resolve().parents[2]
TEXT='The rain has finally stopped. I opened the curtains and made a fresh cup of coffee. We have the whole morning ahead of us.'

def memory():
    return dict(rss_mib=psutil.Process().memory_info().rss/2**20,
                process_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                gpu_allocated_mib=torch.cuda.memory_allocated()/2**20,
                gpu_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--voice',required=True)
    p.add_argument('--dtype',choices=['fp32','fp16','bf16'],default='fp32')
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2)
    profile=json.loads((ROOT/'voices/nano'/f'{a.voice}.json').read_text())
    cache=ROOT/profile['conditioning_cache']
    if not cache.exists():raise FileNotFoundError(cache)
    a.out.mkdir(parents=True,exist_ok=True)
    report=dict(voice=a.voice,dtype=a.dtype,device='cuda',cpu_threads=2,text=TEXT,seed=31,
                acoustic_seed=10031,profile=profile,conditioning_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),
                before_load=memory(),runs=[],resource_policy='3 GiB cgroup maximum, no swap, CPUQuota 200%, nice 10')
    start=time.perf_counter()
    engine=NanoEngine.from_pretrained(ROOT/'models/chatterbox-nano',device='cuda',dtype=a.dtype,optimized=True,
        decoder=profile.get('decoder','meanflow'),conditionals_path=cache,cpu_threads=2)
    configure_acoustics(engine.model,noise_scale=profile.get('acoustic_temperature',1.),guidance=profile.get('acoustic_cfg'))
    if profile.get('adapter'):
        from adaptation import load_adapter
        for module in load_adapter(engine.model.t3,ROOT/profile['adapter']).values():module.scaling*=profile.get('adapter_scale',1.)
    torch.cuda.synchronize();report.update(load_seconds=time.perf_counter()-start,loaded=memory(),runtime_load_report=engine.load_report)
    for i in range(3):
        torch.cuda.synchronize();start=time.perf_counter()
        audio,tokens=engine.generate(TEXT,seed=31,acoustic_seed=10031,n_cfm_steps=profile.get('steps',2),return_tokens=True,**profile['sampling'])
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        x=audio.squeeze().numpy();assert np.isfinite(x).all()
        path=a.out/f'{a.voice}_{a.dtype}_{i}.wav';sf.write(path,x,24000,subtype='PCM_24')
        torch.save(tokens,a.out/f'{a.voice}_{a.dtype}_{i}.tokens.pt')
        row=dict(id=path.stem,path=str(path.resolve()),voice='harvey' if a.voice=='harvey' else 'asmr7',text=TEXT,
                 phase='first' if i==0 else 'warm',generation_seconds=elapsed,seconds=len(x)/24000,
                 rtf=elapsed/(len(x)/24000),token_count=int(tokens.numel()),
                 audio_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),**memory())
        report['runs'].append(row)
        report['warm_mean_rtf']=float(np.mean([r['rtf'] for r in report['runs'][1:]])) if i else None
        (a.out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(row),flush=True)

if __name__=='__main__':main()
