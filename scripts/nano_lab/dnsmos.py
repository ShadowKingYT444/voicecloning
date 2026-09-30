"""Local DNSMOS P.835 proxy. Not a listening test or speaker similarity score.

Model/calibration: Microsoft DNS-Challenge/DNSMOS/dnsmos_local.py, MIT.
Short audio is repeated to 9.01 seconds, matching the upstream protocol.
ASMR is outside typical denoising evaluation data; report that limitation.
"""
import argparse, json, math
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import onnxruntime as ort
ROOT=Path(__file__).resolve().parents[2]

class DNSMOS:
    def __init__(self):
        opts=ort.SessionOptions(); opts.intra_op_num_threads=2; opts.inter_op_num_threads=1
        self.session=ort.InferenceSession(str(ROOT/"vendor/dnsmos/sig_bak_ovr.onnx"),sess_options=opts,providers=["CPUExecutionProvider"])
    def __call__(self,path):
        x,sr=sf.read(path,dtype="float32")
        if x.ndim>1: x=x.mean(1)
        if not len(x) or not np.isfinite(x).all(): raise ValueError("Invalid audio")
        g=math.gcd(sr,16000); x=resample_poly(x,16000//g,sr//g).astype(np.float32)
        original=len(x); required=144160
        while len(x)<required: x=np.concatenate([x,x])
        scores=[]
        for offset in range(0,len(x)-required+1,16000):
            raw=self.session.run(None,{"input_1":x[None,offset:offset+required]})[0][0]
            scores.append([np.polyval([-.08397278,1.22083953,.0052439],raw[0]),
                           np.polyval([-.13166888,1.60915514,-.39604546],raw[1]),
                           np.polyval([-.06766283,1.11546468,.04602535],raw[2])])
        sig,bak,ovr=np.mean(scores,axis=0)
        return {"dnsmos_signal":float(sig),"dnsmos_background":float(bak),"dnsmos_overall":float(ovr),"dnsmos_windows":len(scores),"dnsmos_repeated_short_clip":original<required}

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("manifest"); p.add_argument("--out",required=True)
    a=p.parse_args(); obj=json.loads(Path(a.manifest).read_text()); rows=obj if isinstance(obj,list) else obj["runs"]
    metric=DNSMOS(); output=[]
    for row in rows:
        if "path" not in row: continue
        try: result={**row,**metric(row["path"])}
        except Exception as e: result={**row,"dnsmos_error":repr(e)}
        output.append(result); Path(a.out).write_text(json.dumps(output,indent=2))
        print(json.dumps({k:v for k,v in result.items() if k.startswith("dnsmos") or k=="id"}),flush=True)
