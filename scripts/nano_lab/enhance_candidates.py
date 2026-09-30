"""A/B test bounded DeepFilterNet3 cleanup; never overwrite raw synthesis."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""  # Offline preprocessing uses CPU only.
os.environ.setdefault("OMP_NUM_THREADS","2")
import argparse,json,sys,types,time
from pathlib import Path
from dataclasses import make_dataclass
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import torch
import torchaudio
# DeepFilterNet 0.5.6 imports a removed torchaudio type for annotations. The
# actual I/O below uses soundfile. No torchaudio backend behavior is replaced.
try:
    from torchaudio.backend.common import AudioMetaData
except ModuleNotFoundError:
    backend=types.ModuleType("torchaudio.backend");common=types.ModuleType("torchaudio.backend.common")
    common.AudioMetaData=make_dataclass("AudioMetaData",[("sample_rate",int),("num_frames",int),("num_channels",int),("bits_per_sample",int),("encoding",str)])
    backend.common=common;sys.modules[backend.__name__]=backend;sys.modules[common.__name__]=common
from df.enhance import init_df,enhance

def main():
    p=argparse.ArgumentParser();p.add_argument("manifest",type=Path);p.add_argument("--out",type=Path,required=True)
    p.add_argument("--ids",nargs="+");p.add_argument("--limits",nargs="+",type=float,default=[3.,6.,12.])
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    rows=json.loads(a.manifest.read_text());rows=rows if isinstance(rows,list) else rows["runs"]
    model,state,_=init_df(log_level="WARNING");model=model.cpu().eval();assert state.sr()==48000
    output=[]
    for r in rows:
        if r.get("error") or "path" not in r or (a.ids and r["id"] not in a.ids):continue
        x,sr=sf.read(r["path"],dtype="float32");assert sr==24000 and x.ndim==1
        x48=resample_poly(x,2,1).astype(np.float32)
        for limit in a.limits:
            start=time.perf_counter()
            with torch.inference_mode():y=enhance(model,state,torch.from_numpy(x48[None]),pad=True,atten_lim_db=limit)
            y=resample_poly(y.squeeze().numpy(),1,2)[:len(x)].astype(np.float32)
            if not np.isfinite(y).all() or len(y)!=len(x):raise RuntimeError("Invalid enhanced waveform")
            name=r["id"]+f"_df{int(limit)}";path=a.out/f"{name}.wav";sf.write(path,y,sr,subtype="PCM_24")
            inherited={k:v for k,v in r.items() if k not in {"master_path","master_processing"}}
            output.append({**inherited,"id":name,"path":str(path.resolve()),"parent_audio":r["path"],
                "postprocessing":{"model":"DeepFilterNet3","attenuation_limit_db":limit,"seconds":time.perf_counter()-start,"device":"cpu"}})
            (a.out/"manifest.json").write_text(json.dumps(output,indent=2));print(name,flush=True)

if __name__=="__main__":main()
