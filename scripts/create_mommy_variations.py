#!/usr/bin/env python3
"""Enroll three Mommy voice variants from separate reference blocks."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from inno_kokoro.enroll import Tuner, enroll, read

from kokoro_local import INNO_WEIGHTS, SAMPLE_RATE, load_pipeline, render


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "artifacts" / "references" / "mommy_trimmed_mono_24k.wav"
VOICE_DIR = ROOT / "voices"
OUTPUT_DIR = ROOT / "outputs"

TEXT = (
    "Olson begins with a short declarative sentence that echoes the familiar national slogan that emerged after 9/11. "
    "Because the phrase is so culturally recognizable, he can initially invoke the unity and determination Americans "
    "associate with the attacks without explaining them. More importantly, he sets up the phrase so he can later "
    "undermine it: the rest of the essay argues that America has in fact forgotten. This creates a structural irony "
    "between what Americans claim to believe and how Olson believes they actually behave."
)

# The middle block ends before the documented quiet gap; the final block starts
# after it. Each reference remains long enough for the tuner while emphasizing
# a different portion of the source performance.
BLOCKS = {
    "1": (0.0, 4.5),
    "2": (4.5, 9.5),
    "3": (10.5, 13.5),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    source, sr = read(str(REFERENCE))
    if sr != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE} Hz reference, got {sr} Hz")
    tuner = Tuner(str(INNO_WEIGHTS), device=args.device)
    pipeline = load_pipeline(args.device)
    VOICE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, (start_s, end_s) in BLOCKS.items():
        start = round(start_s * sr)
        end = round(end_s * sr)
        block = source[start:end]
        with torch.inference_mode():
            pack, stats = enroll(block.to(args.device), sr, tuner)
        voice_path = VOICE_DIR / f"af_mommy_v{name}.pt"
        torch.save(pack.detach().cpu(), voice_path)

        voice = torch.load(voice_path, map_location="cpu", weights_only=True)
        chunks = []
        for segment in TEXT.split(". "):
            segment = segment.strip()
            if not segment:
                continue
            if not segment.endswith("."):
                segment += "."
            audio, _ = render(
                pipeline,
                segment,
                voice,
                args.device,
                "autocast_fp16" if args.device == "cuda" else "fp32",
            )
            chunks.append(audio)
        audio = np.concatenate(chunks)
        output_path = OUTPUT_DIR / f"mommy_variation_{name}.wav"
        sf.write(output_path, audio, SAMPLE_RATE)
        print(
            f"variation={name} reference={start_s:.1f}-{end_s:.1f}s "
            f"voice={voice_path} output={output_path} duration={len(audio)/SAMPLE_RATE:.3f}s stats={stats}"
        )


if __name__ == "__main__":
    main()
