"""Controlled reference, conditioning and decoder experiments for Nano.

Every output is synthetic speech. Originals and raw generated candidates are
retained; the optional delivery master applies only high-pass and bounded gain.
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import argparse, copy, hashlib, json, random, time, resource
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import psutil
from scipy import signal
import pyloudnorm as ln
from chatterbox.tts_turbo import ChatterboxTurboTTS, Conditionals, punc_norm
from chatterbox.models.s3gen.const import S3GEN_SIL
from decoder_projection import configure_decoder_projection
from decoder_attention import configure_decoder_attention

ROOT = Path(__file__).resolve().parents[2]
TEXTS = {
    "comfort": "Hey there. You are doing great. Take a slow breath and let your shoulders relax. I am right here with you.",
    "neutral": "I left the blue notebook beside the window. Please bring it with you when we meet tomorrow.",
    "calm": "You have done enough for today. Take a quiet moment, breathe slowly, and let yourself relax.",
    "business": "We have a clear plan. Check the details, prepare your next move, and walk into that meeting with confidence.",
    "question": "Did you remember the blue folder? I thought we agreed to meet at half past nine.",
    "long": "There is no need to rush this decision. We can review the details together and choose a clear path forward. I have written down the questions that matter most. Take your time, and tell me what you think.",
}

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def sync(model):
    if str(model.device).startswith("cuda"): torch.cuda.synchronize()

def master(x, sr=24000):
    """No noise gate: preserve unvoiced consonants and soft breaths."""
    y=signal.sosfilt(signal.butter(2, 45, btype="highpass", fs=sr, output="sos"),x).astype(np.float32)
    loudness=ln.Meter(sr).integrated_loudness(y)
    gain=min(12., -19.-loudness) if np.isfinite(loudness) else 0.
    true_peak=float(np.max(np.abs(signal.resample_poly(y,4,1))))
    gain=min(gain, -1.-20*np.log10(max(true_peak,1e-9)))
    y *= 10**(gain/20)
    fade=min(int(.005*sr),len(y)//2)
    y[:fade]*=np.linspace(0,1,fade); y[-fade:]*=np.linspace(1,0,fade)
    return y, {"highpass_hz":45,"gain_db":float(gain),"target_lufs":-19,
               "achieved_lufs":float(ln.Meter(sr).integrated_loudness(y)),
               "true_peak_ceiling_dbtp":-1,"true_peak_oversampling":4,"noise_gate":False}

@torch.inference_mode()
def generate(model, text, seed=31, temperature=.8, top_p=.95, top_k=1000,
             repetition_penalty=1.2, steps=2, token_path=None, max_gen_len=700):
    seed_all(seed)
    sync(model); start=time.perf_counter()
    ids=model.tokenizer(punc_norm(text),return_tensors="pt").input_ids.to(model.device)
    if ids.shape[-1] > 350: raise ValueError("Text too long; split into sentences before generation")
    speech=model.t3.inference_turbo(t3_cond=model.conds.t3,text_tokens=ids,
        temperature=temperature,top_p=top_p,top_k=top_k,
        repetition_penalty=repetition_penalty,max_gen_len=max_gen_len)
    if speech.numel()>=max_gen_len: raise RuntimeError("Generation hit length limit")
    speech=speech[speech<6561].to(model.device)
    speech=torch.cat([speech,torch.full((3,),S3GEN_SIL,device=model.device,dtype=torch.long)])
    sync(model); token_seconds=time.perf_counter()-start
    if token_path: torch.save(speech.cpu(),token_path)
    # Decoding uses a separate fixed seed so token-sampler changes do not change
    # the acoustic noise sequence when comparing identical token sequences.
    seed_all(seed+10000)
    wav,_=model.s3gen.inference(speech_tokens=speech,ref_dict=model.conds.gen,n_cfm_timesteps=steps)
    x=wav.squeeze().detach().cpu().float().numpy()
    x=model.watermarker.apply_watermark(x,sample_rate=model.sr)
    sync(model)
    if not np.isfinite(x).all(): raise RuntimeError("Non-finite waveform")
    return x,{"generation_seconds":time.perf_counter()-start,"token_seconds":token_seconds,"token_count":len(speech)}

@torch.inference_mode()
def run(config_path):
    cfg=json.loads(Path(config_path).read_text())
    out=ROOT/cfg.get("output_dir","artifacts/nano_lab/sweep")
    out.mkdir(parents=True,exist_ok=True)
    attention_cases=[case for case in cfg.get("cases",[]) if case.get("decoder_attention")]
    if attention_cases and cfg.get("decoder","meanflow")!="meanflow":
        raise ValueError("decoder_attention requires decoder=meanflow")
    if attention_cases and not cfg.get('strict_fp32', False):
        raise ValueError('decoder_attention comparisons require strict_fp32=true for every case, including controls')
    if cfg.get('strict_fp32', False):
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    precision={'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,
               'cudnn_tf32':torch.backends.cudnn.allow_tf32}
    torch.set_num_threads(cfg.get("threads",4))
    if cfg.get("optimized_loader"):
        from runtime import NanoEngine
        cached_only=bool(cfg["cases"]) and all(case.get("input_conditioning_cache") for case in cfg["cases"])
        options={"conditionals_path":ROOT/cfg["cases"][0]["input_conditioning_cache"]} if cached_only else {}
        engine=NanoEngine.from_pretrained(ROOT/"models/chatterbox-nano",device=cfg.get("device","cuda"),dtype="fp32",optimized=True,decoder=cfg.get("decoder","meanflow"),cpu_threads=cfg.get("threads",2),**options)
        model=engine.model
    else:
        model=ChatterboxTurboTTS.from_local(ROOT/"models/chatterbox-nano",device=cfg.get("device","cuda"),nano=True)
    if cfg.get("decoder") == "original" and not cfg.get("optimized_loader"):
        from chatterbox.models.s3gen import S3Gen
        from safetensors.torch import load_file
        del model.s3gen
        torch.cuda.empty_cache()
        model.s3gen=S3Gen(meanflow=False)
        state=load_file(ROOT/"models/chatterbox-nano/s3gen.safetensors")
        # This older checkpoint predates persistence of the deterministic
        # 400-sample Hann analysis window. Keep the constructor's exact window.
        state.setdefault("tokenizer.window",model.s3gen.tokenizer.window)
        model.s3gen.load_state_dict(state,strict=True)
        del state
        model.s3gen.to(model.device).eval()
    adapters={}
    if cfg.get("adapter"):
        from adaptation import load_adapter
        adapters=load_adapter(model.t3,cfg["adapter"])
    adapter_scales={k:v.scaling for k,v in adapters.items()}
    cond_cache={}
    def get_cond(path,norm=True):
        key=(str(path),norm)
        if key not in cond_cache:
            previous=model.conds
            try:
                model.prepare_conditionals(str(ROOT/path),norm_loudness=norm)
                cond_cache[key]=copy.deepcopy(model.conds)
            finally:
                model.conds=previous
        return copy.deepcopy(cond_cache[key])
    rows=[]
    report_path=out/"manifest.json"
    previous_rows={r["id"]:r for r in json.loads(report_path.read_text())} if report_path.exists() else {}
    for case in cfg["cases"]:
        name=case["id"]
        if any(c in name for c in "/\\"): raise ValueError("Invalid case ID")
        raw_path=out/f"{name}.wav"
        prior=previous_rows.get(name,{})
        if (raw_path.exists() and (out/f"{name}.master.wav").exists()
            and prior.get("path") and not prior.get("error")
            and prior.get("decoder","meanflow")==cfg.get("decoder","meanflow")
            and prior.get("adapter")==cfg.get("adapter")
            and prior.get('precision', {'matmul_tf32':False,'cudnn_tf32':True})==precision
            and (not case.get('t3_donor_cache') or
                 prior.get('t3_donor_runtime', {}).get('sha256') == hashlib.sha256((ROOT / case['t3_donor_cache']).read_bytes()).hexdigest())
            and all(prior.get(k)==v for k,v in case.items()) and cfg.get("resume",True)):
            rows.append(prior)
            continue
        row={**case,"decoder":cfg.get("decoder","meanflow"),"synthetic":True,
             "runtime_loader":"optimized_fp32" if cfg.get("optimized_loader") else "stock_fp32",
             "adapter":cfg.get("adapter"),"adapter_scale":case.get("adapter_scale",cfg.get("adapter_scale",1.)) if adapters else None,
             "precision":dict(precision)}
        try:
            if case.get('t3_donor_mode') and not case.get('t3_donor_cache'):
                raise ValueError('t3_donor_mode requires t3_donor_cache')
            projection_value=case.get("decoder_projection")
            projection_strength=float(case.get("decoder_projection_strength",1.0)) if projection_value else 0.0
            if projection_value and (case.get("decoder_reference") or case.get("decoder_speaker_references")):
                raise ValueError("decoder_projection cannot be combined with decoder_reference or decoder_speaker_references")
            if projection_value and case.get("decoder_embedding_scale"):
                raise ValueError("decoder_projection cannot be combined with decoder_embedding_scale")
            if projection_value and not case.get("input_conditioning_cache"):
                raise ValueError("decoder_projection requires input_conditioning_cache for provenance validation")
            attention_value=case.get("decoder_attention")
            attention_strength=float(case.get("decoder_attention_strength",1.0)) if attention_value else 0.0
            if attention_value and (case.get("decoder_reference") or case.get("decoder_speaker_references")):
                raise ValueError("decoder_attention cannot be combined with decoder_reference or decoder_speaker_references")
            if attention_value and case.get("decoder_embedding_scale"):
                raise ValueError("decoder_attention cannot be combined with decoder_embedding_scale")
            if attention_value and case.get("decoder_projection"):
                raise ValueError("decoder_attention cannot be combined with decoder_projection")
            if attention_value and not case.get("input_conditioning_cache"):
                raise ValueError("decoder_attention requires input_conditioning_cache for provenance validation")
            if not attention_value and "decoder_attention_strength" in case and float(case["decoder_attention_strength"]) != 0.0:
                raise ValueError("decoder_attention_strength requires decoder_attention")
            if not projection_value and "decoder_projection_strength" in case and float(case["decoder_projection_strength"]) != 0.0:
                raise ValueError("decoder_projection_strength requires decoder_projection")
            from acoustic_controls import configure_acoustics
            configure_acoustics(model,noise_scale=case.get("acoustic_temperature",1.),guidance=case.get("acoustic_cfg"))
            from vocoder_controls import configure_vocoder
            row["vocoder_controls"]=configure_vocoder(model,voiced_noise_scale=case.get("vocoder_voiced_noise_scale",1.))
            from mel_calibration import configure_mel_calibration
            mel_path = ROOT / case["mel_calibration"] if case.get("mel_calibration") else None
            configure_mel_calibration(model, mel_path, case.get("mel_calibration_strength", 1.))
            row["mel_calibration_sha256"] = hashlib.sha256(mel_path.read_bytes()).hexdigest() if mel_path else None
            for k,v in adapters.items():v.scaling=adapter_scales[k]*row["adapter_scale"]
            if case.get("input_conditioning_cache"):
                model.conds=Conditionals.load(ROOT/case["input_conditioning_cache"],map_location="cpu").to(model.device)
            else:
                model.conds=get_cond(case["reference"],case.get("normalize_reference",True))
            if case.get('t3_donor_cache'):
                from t3_conditioning_donor import copy_t3_conditioning
                donor_path = ROOT / case['t3_donor_cache']
                donor = Conditionals.load(donor_path, map_location='cpu')
                row['t3_donor_runtime'] = copy_t3_conditioning(
                    donor, model.conds, case.get('t3_donor_mode', 'prompt'))
                row['t3_donor_runtime'].update(path=str(donor_path.resolve()),
                    sha256=hashlib.sha256(donor_path.read_bytes()).hexdigest())
                del donor
            if case.get("decoder_reference"):
                model.conds.gen=get_cond(case["decoder_reference"],case.get("normalize_reference",True)).gen
            if case.get("speaker_references"):
                others=[get_cond(p).t3.speaker_emb for p in case["speaker_references"]]
                emb=torch.stack([model.conds.t3.speaker_emb,*others]).mean(0)
                model.conds.t3.speaker_emb=torch.nn.functional.normalize(emb,dim=-1)
            if case.get("decoder_speaker_references"):
                embs=[model.conds.gen["embedding"]]+[get_cond(p).gen["embedding"] for p in case["decoder_speaker_references"]]
                model.conds.gen["embedding"]=torch.stack([torch.nn.functional.normalize(e,dim=-1) for e in embs]).mean(0)
            scale=case.get("speaker_scale",1.)
            model.conds.t3.speaker_emb *= scale
            if case.get("speaker_vector"):
                payload=json.loads((ROOT/case["speaker_vector"]).read_text())
                mode=case.get("speaker_vector_mode","absolute")
                vector=payload["delta" if mode=="delta" else "speaker_emb"]
                tensor=torch.tensor(vector,device=model.device,dtype=model.conds.t3.speaker_emb.dtype).reshape(1,-1)
                if tensor.shape!=(1,256) or not torch.isfinite(tensor).all():raise ValueError("Invalid speaker vector")
                if mode=="delta":tensor=torch.nn.functional.normalize(model.conds.t3.speaker_emb+tensor*case.get("speaker_vector_strength",1.),dim=-1)
                elif mode!="absolute":raise ValueError("Unknown speaker vector mode")
                model.conds.t3.speaker_emb=tensor
            if "prompt_frames" in case:
                model.conds.t3.cond_prompt_speech_tokens=model.conds.t3.cond_prompt_speech_tokens[:,:case["prompt_frames"]]
            model.conds.t3.cond_prompt_speech_emb=None
            if case.get("decoder_embedding_scale"):
                model.conds.gen["embedding"] *= case["decoder_embedding_scale"]
            projection_cache_value=case.get("input_conditioning_cache")
            projection_metadata=configure_decoder_projection(
                model,
                ROOT / projection_value if projection_value else None,
                strength=projection_strength,
                conditionals_path=(ROOT / projection_cache_value if projection_cache_value else None),
                model_checkpoint_path=ROOT / "models/chatterbox-nano/s3gen_meanflow.safetensors",
            )
            row["decoder_projection_runtime"]=projection_metadata
            attention_cache_value=case.get("input_conditioning_cache")
            attention_metadata=configure_decoder_attention(
                model,
                ROOT / attention_value if attention_value else None,
                strength=attention_strength,
                conditionals_path=(ROOT / attention_cache_value if attention_cache_value else None),
                model_checkpoint_path=ROOT / "models/chatterbox-nano/s3gen_meanflow.safetensors",
            )
            row["decoder_attention_runtime"]=attention_metadata
            if case.get("save_conditionals"):
                conditionals_path=out/f"{name}.conds.pt"
                model.conds.save(conditionals_path)
                row["conditioning_cache"]=str(conditionals_path)
            text=case.get("text",TEXTS[case.get("text_id","comfort")])
            kwargs={k:case[k] for k in ["seed","temperature","top_p","top_k","repetition_penalty","steps"] if k in case}
            x,stats=generate(model,text,token_path=out/f"{name}.tokens.pt",**kwargs)
            sf.write(raw_path,x,model.sr,subtype="PCM_24")
            y,processing=master(x,model.sr)
            sf.write(out/f"{name}.master.wav",y,model.sr,subtype="PCM_24")
            row.update(path=str(raw_path),master_path=str(out/f"{name}.master.wav"),text=text,
                seconds=len(x)/model.sr,master_processing=processing,**stats,
                rss_mib=psutil.Process().memory_info().rss/2**20,
                peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
            row["rtf"]=stats["generation_seconds"]/row["seconds"]
            print(json.dumps({k:row[k] for k in ["id","seconds","generation_seconds","rtf"]}),flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc(); row["error"]=repr(e)
        rows.append(row)
        report_path.write_text(json.dumps(rows,indent=2))
    return rows

if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("config")
    results = run(parser.parse_args().config)
    raise SystemExit(1 if any(row.get('error') for row in results) else 0)
