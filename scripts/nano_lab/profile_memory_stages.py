"""Locate host-memory growth across token, decoder, and watermark stages."""
import argparse,ctypes,json,time
from pathlib import Path
import psutil
import soundfile as sf
import torch
from runtime import NanoEngine
from benchmark_profile import ROOT,TEXT

def snapshot():
    torch.cuda.synchronize()
    result={'rss_mib':psutil.Process().memory_info().rss/2**20,'gpu_allocated_mib':torch.cuda.memory_allocated()/2**20}
    for line in Path('/proc/self/smaps_rollup').read_text().splitlines():
        if ':' in line:
            key,value=line.split(':',1)
            if key in ['Rss','Pss','Pss_Anon','Pss_File','Private_Dirty','Shared_Clean']:
                result[key+'_mib']=int(value.split()[0])/1024
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--watermark-device',choices=['cpu','cuda'],default='cpu')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    profile=json.loads((ROOT/'voices/nano/asmr_soft.json').read_text());rows=[]
    engine=NanoEngine.from_pretrained(ROOT/'models/chatterbox-nano',device='cuda',dtype='fp32',optimized=True,
        conditionals_path=ROOT/profile['conditioning_cache'],cpu_threads=2)
    if a.watermark_device=='cuda':engine.model.watermarker.perth_net.to('cuda').eval()
    rows.append({'stage':'loaded',**snapshot()})
    def instrument(owner,name,label):
        original=getattr(owner,name)
        def call(*args,**kwargs):
            rows.append({'stage':label+'_before',**snapshot()});start=time.perf_counter()
            result=original(*args,**kwargs)
            rows.append({'stage':label+'_after','seconds':time.perf_counter()-start,**snapshot()})
            return result
        setattr(owner,name,call)
    instrument(engine.model.t3,'inference_turbo','tokens')
    instrument(engine.model.s3gen,'inference','acoustic')
    instrument(engine.model.watermarker,'apply_watermark','watermark')
    for i in range(2):
        wav=engine.generate(TEXT,seed=31,acoustic_seed=10031,**profile['sampling'])
        sf.write(a.out/f'run{i}.wav',wav.squeeze().numpy(),24000,subtype='PCM_24')
        # Return unused glibc heap pages after the request; never touch model
        # tensors, CUDA allocations, OS caches, or other processes.
        ctypes.CDLL('libc.so.6').malloc_trim(0)
        rows.append({'stage':f'after_request_{i}_trim',**snapshot()})
        (a.out/'report.json').write_text(json.dumps({'watermark_device':a.watermark_device,'stages':rows},indent=2))
    for row in rows:print(json.dumps(row),flush=True)

if __name__=='__main__':main()
