"""Generate synthetic speech using a measured Nano voice profile.

Run from any directory with this workspace's .venv-nano/bin/python.
Long input is split at sentence boundaries to bound autoregressive memory.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, os, resource, time
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS","2")
os.environ.setdefault("TOKENIZERS_PARALLELISM","false")

ROOT=Path(__file__).resolve().parents[2]
PROFILE_DIR=ROOT/"voices/nano"

from profile_conditioning import resolve_conditioning_cache


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _mel_calibration_metadata(path: Path, strength: float) -> dict:
    """Return auditable calibration provenance without loading a model."""

    path = path.resolve()
    metadata = {
        "requested": True,
        "applied": float(strength) > 0.0,
        "path": str(path),
        "report_sha256": _sha256_file(path),
        "strength": float(strength),
    }
    if path.suffix.lower() not in {".json", ".jsn"}:
        metadata["delta_path"] = str(path)
        metadata["delta_sha256"] = metadata["report_sha256"]
        metadata["format"] = "raw_delta_file"
        return metadata

    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"mel calibration report must be a JSON object: {path}")
    for key in (
        "format",
        "claim_status",
        "diagnostic_only",
        "speaker_id",
        "conditioning_sha256",
        "model_checkpoint_sha256",
        "prepare_manifest_sha256",
        "delta_mean",
        "delta_max_abs",
        "max_delta",
        "sigma_bands",
        "time_alignment",
    ):
        if key in payload:
            metadata[key] = payload[key]
    values = payload.get("delta")
    if isinstance(values, list):
        metadata["delta_values"] = len(values)
    delta_value = payload.get("delta_path")
    if delta_value:
        delta_path = Path(str(delta_value)).expanduser()
        if not delta_path.is_absolute():
            delta_path = ROOT / delta_path
        delta_path = delta_path.resolve()
        metadata["delta_path"] = str(delta_path)
        if delta_path.exists():
            metadata["delta_sha256"] = _sha256_file(delta_path)
            metadata["delta_bytes"] = delta_path.stat().st_size
        else:
            metadata["delta_missing"] = True
    return metadata

def prepare_profile(engine, profile):
    import torch
    def reference(path):
        return copy.deepcopy(engine.prepare_conditionals(ROOT/path,norm_loudness=profile.get("normalize_reference",True)))
    conds=reference(profile["reference"])
    if profile.get("decoder_reference"):
        conds.gen=reference(profile["decoder_reference"]).gen
    if profile.get("speaker_references"):
        vectors=[conds.t3.speaker_emb]+[reference(p).t3.speaker_emb for p in profile["speaker_references"]]
        conds.t3.speaker_emb=torch.nn.functional.normalize(torch.stack(vectors).mean(0),dim=-1)
    if profile.get("decoder_speaker_references"):
        vectors=[conds.gen["embedding"]]+[reference(p).gen["embedding"] for p in profile["decoder_speaker_references"]]
        conds.gen["embedding"]=torch.stack([torch.nn.functional.normalize(v,dim=-1) for v in vectors]).mean(0)
    conds.t3.speaker_emb=conds.t3.speaker_emb*profile.get("speaker_scale",1.)
    if profile.get("speaker_vector"):
        payload=json.loads((ROOT/profile["speaker_vector"]).read_text())
        mode=profile.get("speaker_vector_mode","absolute")
        vector=payload["delta" if mode=="delta" else "speaker_emb"]
        tensor=torch.tensor(vector,device=engine.device,dtype=engine.dtype).reshape(1,-1)
        if tensor.shape!=(1,256) or not torch.isfinite(tensor).all():raise ValueError("Invalid speaker vector")
        if mode=="delta":tensor=torch.nn.functional.normalize(conds.t3.speaker_emb+tensor*profile.get("speaker_vector_strength",1.),dim=-1)
        elif mode!="absolute":raise ValueError("Unknown speaker vector mode")
        conds.t3.speaker_emb=tensor
    if "prompt_frames" in profile:
        conds.t3.cond_prompt_speech_tokens=conds.t3.cond_prompt_speech_tokens[:,:profile["prompt_frames"]]
    conds.t3.cond_prompt_speech_emb=None
    engine.set_conditionals(conds)
    return conds

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--voice",help="Profile name in voices/nano, or absolute JSON path")
    p.add_argument("--list",action="store_true",help="List installed profiles")
    p.add_argument("--text"); p.add_argument("--text-file",type=Path)
    p.add_argument("--output",type=Path)
    p.add_argument("--device",choices=["cpu","cuda"],default="cuda")
    p.add_argument("--dtype",choices=["fp32","fp16","bf16"],default=None)
    p.add_argument("--seed",type=int,default=31)
    p.add_argument("--raw",action="store_true",help="Skip delivery high-pass/loudness mastering")
    p.add_argument("--prepare-only",action="store_true",help="Save reference conditionals without generating")
    a=p.parse_args()
    if a.list:
        for f in sorted(PROFILE_DIR.glob("*.json")):
            v=json.loads(f.read_text());print(f"{f.stem}: {v.get('label',f.stem)}")
        return
    if not a.voice: p.error("--voice is required")
    profile_path=Path(a.voice) if a.voice.endswith(".json") else PROFILE_DIR/f"{a.voice}.json"
    profile=json.loads(profile_path.read_text())
    if profile.get("status") == "rejected_training_label_mismatch":
        p.error(profile["rejection_reason"])
    text=a.text_file.read_text() if a.text_file else a.text
    if not a.prepare_only and (not text or not text.strip()): p.error("Provide --text or --text-file")
    if not a.prepare_only and not a.output: p.error("--output is required")
    # Validate strict fitted-profile caches before importing Torch or loading
    # any model.  Legacy profiles without either integrity field keep the
    # existing native auto-prepare fallback when their cache is absent.
    cache_info=resolve_conditioning_cache(profile,root=ROOT)
    cond_path=cache_info["path"]
    from profile_t3_donor import resolve_t3_donor
    donor_info=resolve_t3_donor(profile, ROOT)
    # Listing profiles and argument errors must not load a speech framework.
    import numpy as np
    import psutil
    import torch
    from runtime import NanoEngine, GenerationLimitError
    from quality_sweep import master
    from audio_output import chunks, completed_audio, split_failed_segment
    from collections import deque
    dtype=a.dtype or (profile.get("cuda_dtype","fp32") if a.device=="cuda" else "fp32")
    if profile.get('decoder_attention'):
        if dtype != 'fp32' or not profile.get('strict_fp32') or profile.get('decoder', 'meanflow') != 'meanflow':
            raise ValueError('Acoustic attention requires strict_fp32=true, fp32 dtype, and meanflow')
        if not cache_info['exists']:
            raise ValueError('Acoustic attention requires its existing fitted cache')
    if profile.get('strict_fp32'):
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    calibration_setting=profile.get("mel_calibration")
    calibration_path=None
    calibration_strength=0.0
    calibration_metadata={"requested":False,"applied":False,"strength":0.0}
    if calibration_setting:
        calibration_path=(ROOT/calibration_setting).resolve()
        if not calibration_path.exists():
            raise FileNotFoundError(f"mel calibration report does not exist: {calibration_path}")
        calibration_strength=float(profile.get("mel_calibration_strength",1.0))
    start=time.perf_counter()
    options=dict(device=a.device,dtype=dtype,optimized=True,decoder=profile.get("decoder","meanflow"),cpu_threads=2)
    if cache_info["exists"]: options["conditionals_path"]=cond_path
    engine=NanoEngine.from_pretrained(ROOT/"models/chatterbox-nano",**options)
    if not cache_info["exists"]:
        with torch.inference_mode():
            prepare_profile(engine,profile)
        cond_path.parent.mkdir(parents=True,exist_ok=True)
        engine.model.conds.save(cond_path)
    donor_metadata=None
    if donor_info:
        from chatterbox.tts_turbo import Conditionals
        from t3_conditioning_donor import copy_t3_conditioning
        donor=Conditionals.load(donor_info['path'], map_location='cpu')
        donor_metadata={**donor_info, **copy_t3_conditioning(donor, engine.model.conds, donor_info['mode'])}
        del donor
    if profile.get("adapter"):
        from adaptation import load_adapter
        adapters=load_adapter(engine.model.t3,ROOT/profile["adapter"])
        for module in adapters.values(): module.scaling*=profile.get("adapter_scale",1.)
    acoustic_metadata=None
    if profile.get('decoder_attention'):
        from decoder_attention import configure_decoder_attention
        acoustic_metadata=configure_decoder_attention(engine.model, ROOT/profile['decoder_attention'],
            strength=float(profile.get('decoder_attention_strength', 1.0)), conditionals_path=cond_path)
    if calibration_path is not None:
        # The hook wraps only HiFT's mel-to-waveform input. It is opt-in per
        # profile and leaves every existing zero-shot profile unchanged.
        from mel_calibration import configure_mel_calibration
        configure_mel_calibration(engine.model,calibration_path,strength=calibration_strength)
        calibration_metadata=_mel_calibration_metadata(calibration_path,calibration_strength)
    from acoustic_controls import configure_acoustics
    configure_acoustics(engine.model,noise_scale=profile.get("acoustic_temperature",1.),guidance=profile.get("acoustic_cfg"))
    # Persistent conditions let the synthesis process release all reference
    # encoders. A new voice requires loading another profile/engine.
    engine.unload_voice_encoder();engine.unload_decoder_reference_encoder();engine.unload_tokenizer()
    ready=time.perf_counter()
    report={"synthetic":True,"profile":str(profile_path),"device":a.device,"dtype":dtype,
            "seed":a.seed,"load_and_prepare_seconds":ready-start,"runtime_load_report":engine.load_report,
            "ready_rss_mib":psutil.Process().memory_info().rss/2**20,"segments":[],"length_limit_retries":[],
            "mel_calibration":calibration_metadata, "decoder_attention":acoustic_metadata,
            "t3_donor":donor_metadata,
            "precision":dict(matmul_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32)}
    report["profile_sha256"]=_sha256_file(profile_path)
    if profile.get("adapter"):
        adapter_path=(ROOT/profile["adapter"]).resolve()
        report["adapter"]={"path":str(adapter_path),"sha256":_sha256_file(adapter_path),
                           "scale":float(profile.get("adapter_scale",1.0))}
    if a.prepare_only:
        print(json.dumps({"conditioning_cache":str(cond_path),**report},indent=2));return
    a.output.parent.mkdir(parents=True,exist_ok=True)
    sampling=profile.get("sampling",{})
    samples=0
    pending=deque(chunks(text,max_words=profile.get("max_segment_words",32)))
    with completed_audio(a.output, engine.sr) as writer:
        while pending:
            part=pending.popleft()
            i=len(report["segments"])
            t=time.perf_counter()
            try:
                audio=engine.generate(part,seed=a.seed+i,acoustic_seed=a.seed+i+10000,n_cfm_steps=profile.get("steps",2),**sampling)
            except GenerationLimitError:
                smaller=split_failed_segment(part)
                if smaller is None:
                    raise
                report["length_limit_retries"].append({"text":part,"split_into":smaller,"failed_seconds":time.perf_counter()-t})
                pending.extendleft(reversed(smaller))
                print(f"Length limit: retrying as {len(smaller)} shorter segments",flush=True)
                continue
            x=audio.squeeze().cpu().float().numpy()
            if not np.isfinite(x).all(): raise RuntimeError("Generation produced non-finite audio")
            if not a.raw: x,processing=master(x,engine.sr)
            else: processing={}
            if i: writer.write(np.zeros(round(.10*engine.sr),dtype=np.float32));samples+=round(.10*engine.sr)
            writer.write(x);samples+=len(x)
            elapsed=time.perf_counter()-t
            report["segments"].append({"text":part,"seconds":len(x)/engine.sr,"generation_and_master_seconds":elapsed,"rtf":elapsed/(len(x)/engine.sr),"processing":processing})
            print(f"Segment {i+1}: {len(x)/engine.sr:.2f}s audio in {elapsed:.2f}s",flush=True)
    report.update(output=str(a.output.resolve()),audio_seconds=samples/engine.sr,
        total_seconds=time.perf_counter()-start,rss_mib=psutil.Process().memory_info().rss/2**20,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        cuda_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20 if a.device=="cuda" else None)
    sidecar=a.output.with_suffix(a.output.suffix+".json");sidecar.write_text(json.dumps(report,indent=2))
    print(json.dumps({"output":report["output"],"report":str(sidecar),"peak_rss_mib":report["peak_rss_mib"]}))

if __name__=="__main__":main()
