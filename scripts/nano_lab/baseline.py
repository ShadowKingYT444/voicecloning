"""Reproducible, unmodified Nano baseline with process and GPU measurements."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import json, time, resource, platform, random
from pathlib import Path
import numpy as np
import psutil
import soundfile as sf
import torch
from chatterbox.tts_turbo import ChatterboxTurboTTS

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/nano_lab/baseline"
OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(4)
proc = psutil.Process()
def memory():
    return {"rss_mib": proc.memory_info().rss / 2**20,
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "cuda_allocated_mib": torch.cuda.memory_allocated()/2**20,
            "cuda_reserved_mib": torch.cuda.memory_reserved()/2**20,
            "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated()/2**20}
report = {"torch": torch.__version__, "python": platform.python_version(), "before_load": memory()}
t = time.perf_counter()
model = ChatterboxTurboTTS.from_local(ROOT / "models/chatterbox-nano", device="cuda", nano=True)
torch.cuda.synchronize()
report.update(load_seconds=time.perf_counter()-t, loaded=memory(), parameters={k:sum(p.numel() for p in getattr(model,k).parameters()) for k in ["t3","s3gen","ve"]})
t = time.perf_counter()
model.prepare_conditionals(str(ROOT / "artifacts/references/chunks/asmr7_chunk_36-46.wav"))
report["conditioning_seconds"] = time.perf_counter()-t
texts = ["Hey there. You are doing great. Take a slow breath and let your shoulders relax. I am right here with you.",
         "I left the blue notebook beside the window. Please bring it with you when we meet tomorrow.",
         "You have done enough for today. Take a quiet moment, breathe slowly, and let yourself relax."]
report["runs"] = []
for i, text in enumerate(texts):
    seed=31
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.cuda.synchronize(); t=time.perf_counter()
    with torch.inference_mode(): audio=model.generate(text)
    torch.cuda.synchronize(); elapsed=time.perf_counter()-t
    x=audio.squeeze().numpy(); path=OUT/f"asmr_baseline_{i}.wav"; sf.write(path,x,model.sr,subtype="PCM_24")
    row={"voice":"asmr","path":str(path),"text":text,"seed":seed,"seconds":len(x)/model.sr,"generation_seconds":elapsed,"rtf":elapsed/(len(x)/model.sr),**memory()}
    report["runs"].append(row)
    (OUT/"report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(row),flush=True)
