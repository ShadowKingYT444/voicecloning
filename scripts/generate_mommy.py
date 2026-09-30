#!/usr/bin/env python3
"""Generate speech with the locally enrolled Mommy Kokoro voice."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from kokoro_local import ROOT, SAMPLE_RATE, load_pipeline, render


DEFAULT_TEXT = (
    "Also, discard the idea in your review that a longer discovered harness would demonstrate successful recursion. "
    "More code can equally indicate duplication and unnecessary complexity. "
    "The relevant evidence is better improvements per unit of optimization compute, not additional lines or modules."
)
DEFAULT_VOICE = ROOT / "voices" / "af_mommy.pt"
DEFAULT_OUTPUT = ROOT / "outputs" / "mommy_review_statement.wav"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    voice = torch.load(DEFAULT_VOICE, map_location="cpu", weights_only=True)
    pipeline = load_pipeline(args.device)

    # Sentence boundaries keep each request well below Kokoro's 510-phoneme
    # limit while preserving the exact requested wording in the final WAV.
    segments = [
        "Also, discard the idea in your review that a longer discovered harness would demonstrate successful recursion.",
        "More code can equally indicate duplication and unnecessary complexity.",
        "The relevant evidence is better improvements per unit of optimization compute, not additional lines or modules.",
    ]
    rendered = []
    token_count = 0
    for segment in segments:
        audio, tokens = render(pipeline, segment, voice, args.device, "autocast_fp16" if args.device == "cuda" else "fp32")
        rendered.append(audio)
        token_count += tokens

    audio = np.concatenate(rendered)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, audio, SAMPLE_RATE)
    print(f"output={args.output}")
    print(f"segments={len(segments)} tokens={token_count} duration_seconds={len(audio) / SAMPLE_RATE:.3f}")


if __name__ == "__main__":
    main()
