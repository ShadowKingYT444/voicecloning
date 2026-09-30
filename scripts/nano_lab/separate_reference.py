"""Optional local music separation for the supplied Harvey reference.

This preprocessing is separate from deployment inference and is measured
separately. Keep source, isolated voice, and residual for inspection.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS","2")
import argparse, json, subprocess, time, resource
from pathlib import Path
import torch
import soundfile as sf
import numpy as np
from scipy.signal import resample_poly
from demucs.pretrained import get_model
from demucs.apply import apply_model

ROOT=Path(__file__).resolve().parents[2]
def main():
    p=argparse.ArgumentParser();p.add_argument("--source",type=Path,default=ROOT/"when Harvey speaks, success listens #Suits #HarveySpecter #Shorts.mp3")
    p.add_argument("--out",type=Path,default=ROOT/"artifacts/nano_lab/harvey_separation")
    p.add_argument("--device",default="cpu");a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.manual_seed(31);t=time.perf_counter()
    subprocess.run(["ffmpeg","-nostdin","-v","error","-y","-i",str(a.source),"-ar","44100","-ac","2",str(a.out/"source.wav")],check=True)
    x,sr=sf.read(a.out/"source.wav",dtype="float32");wav=torch.from_numpy(x.T)
    model=get_model("htdemucs").to(a.device).eval()
    mono=wav.mean(0);mean=mono.mean();std=mono.std().clamp_min(1e-8)
    with torch.inference_mode():
        separated=apply_model(model,((wav-mean)/std)[None],device=a.device,shifts=1,split=True,overlap=.25,progress=True,num_workers=0)[0]
    # Each source shares the normalization scale. Assign global DC only to
    # the residual so the summed signals still reproduce the original mix.
    voice=separated[model.sources.index("vocals")].cpu()*std
    voice=voice.numpy().T;residual=x-voice
    sf.write(a.out/"voice_44k.wav",voice,sr,subtype="PCM_24")
    sf.write(a.out/"residual_44k.wav",residual,sr,subtype="PCM_24")
    voice24=resample_poly(voice.mean(1),80,147).astype(np.float32)
    sf.write(a.out/"voice_24k.wav",voice24,24000,subtype="PCM_24")
    manifest=json.loads((ROOT/"artifacts/nano_lab/references/manifest.json").read_text());rows=[]
    for r in manifest["references"]+manifest["heldout"]:
        if r["family"]!="harvey":continue
        start,end=r["source"]["start_s"],r["source"]["end_s"]
        dest=a.out/f"{r['id']}_isolated.wav"
        sf.write(dest,voice24[round(start*24000):round(end*24000)],24000,subtype="PCM_24")
        rows.append({"id":r["id"]+"_isolated","path":str(dest),"voice":"harvey","start_s":start,"end_s":end,"source":str(a.source),"preprocessing":"Demucs htdemucs vocals"})
    report={"model":"facebookresearch/demucs htdemucs","device":a.device,"elapsed_seconds":time.perf_counter()-t,"peak_rss_mib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,"reference_rows":rows,"mix_rms":float(np.sqrt(np.mean(x*x))),"voice_rms":float(np.sqrt(np.mean(voice*voice))),"residual_rms":float(np.sqrt(np.mean(residual*residual)))}
    (a.out/"report.json").write_text(json.dumps(report,indent=2));(a.out/"manifest.json").write_text(json.dumps(rows,indent=2));print(json.dumps(report,indent=2))
if __name__=="__main__":main()
