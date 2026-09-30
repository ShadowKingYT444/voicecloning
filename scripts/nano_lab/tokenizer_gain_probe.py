"""Reconstruct one source excerpt with raw versus normalized tokenizer input.

Research reconstruction, not new-text voice cloning. Run through bounded_job.py.
"""
import argparse
import json
from pathlib import Path
import resource

import librosa
import numpy as np
import soundfile as sf
import torch

from adaptation import load_nano, load_audio, extract_target_tokens, configure_runtime
from chatterbox.tts_turbo import Conditionals
from chatterbox.models.s3gen.const import S3GEN_SIL
from quality_sweep import master, seed_all


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--conditionals", type=Path, required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    configure_runtime(2)
    model = load_nano(Path("models/chatterbox-nano"), "cuda")
    model.conds = Conditionals.load(args.conditionals).to("cuda")
    raw = load_audio(args.source)
    wav24, _ = librosa.load(str(args.source), sr=24000)
    normalized = model.norm_loudness(wav24, 24000)
    normalized16 = librosa.resample(normalized, orig_sr=24000, target_sr=16000)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    baseline = None
    for label, wav in (("raw", raw), ("normalized", normalized16)):
        tokens = extract_target_tokens(model, wav, 400)
        if baseline is None:
            baseline = tokens
        speech = torch.tensor(tokens + [S3GEN_SIL] * 3, device="cuda", dtype=torch.long)
        seed_all(10031)
        output, _ = model.s3gen.inference(speech_tokens=speech, ref_dict=model.conds.gen, n_cfm_timesteps=2)
        x = output.squeeze().cpu().numpy()
        x = model.watermarker.apply_watermark(x, sample_rate=24000)
        x, processing = master(x)
        path = args.output_dir / f"source01_{label}.wav"
        sf.write(path, x, 24000, subtype="PCM_24")
        rows.append({"id": f"source01_{label}", "path": str(path.resolve()), "text": args.text,
                     "voice": "asmr7", "synthetic_tts": False, "kind": "source-token reconstruction",
                     "source": str(args.source.resolve()), "token_count": len(tokens), "tokens": tokens,
                     "different_tokens": sum(a != b for a, b in zip(tokens, baseline)) if len(tokens) == len(baseline) else None,
                     "master_processing": processing, "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024})
        (args.output_dir / "manifest.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(json.dumps({k: rows[-1][k] for k in ("id", "token_count", "different_tokens", "peak_rss_mib")}), flush=True)


if __name__ == "__main__":
    main()
