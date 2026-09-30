#!/usr/bin/env python3
"""Create three Kokoro voices from the new 7 Minutes in Heaven recording."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from inno_kokoro.enroll import Tuner, enroll

from kokoro_local import INNO_WEIGHTS, SAMPLE_RATE, load_pipeline, render


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "ASMR  7 Minutes in Heaven... With Your Bully_  [Enemies to Lovers] [Tsundere] [Confession] [Kiss].mp3"
SOURCE_WAV = ROOT / "artifacts" / "references" / "asmr7_mono_24k.wav"
VOICE_DIR = ROOT / "voices"
OUTPUT_DIR = ROOT / "outputs"

TEXT = (
    "Olson begins with a short declarative sentence that echoes the familiar national slogan that emerged after 9/11. "
    "Because the phrase is so culturally recognizable, he can initially invoke the unity and determination Americans "
    "associate with the attacks without explaining them. More importantly, he sets up the phrase so he can later "
    "undermine it: the rest of the essay argues that America has in fact forgotten. This creates a structural irony "
    "between what Americans claim to believe and how Olson believes they actually behave."
)

# High-energy speech windows found by scanning the new 613.68-second source.
BLOCKS = {"1": 35.0, "2": 100.0, "3": 390.0}
BLOCK_SECONDS = 12.0


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    if not SOURCE_WAV.exists():
        raise FileNotFoundError(SOURCE_WAV)

    source, sample_rate = sf.read(SOURCE_WAV, dtype="float32")
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE} Hz, got {sample_rate} Hz")
    tuner = Tuner(str(INNO_WEIGHTS), device=device)
    pipeline = load_pipeline(device)
    VOICE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, start_s in BLOCKS.items():
        start = round(start_s * sample_rate)
        end = round((start_s + BLOCK_SECONDS) * sample_rate)
        block = torch.from_numpy(source[start:end]).to(device)
        with torch.inference_mode():
            pack, stats = enroll(block, sample_rate, tuner)
        voice_path = VOICE_DIR / f"af_asmr7_v{name}.pt"
        torch.save(pack.detach().cpu(), voice_path)
        voice = torch.load(voice_path, map_location="cpu", weights_only=True)

        audio_parts = []
        for segment in TEXT.split(". "):
            segment = segment.strip()
            if segment and not segment.endswith("."):
                segment += "."
            if segment:
                audio, _ = render(pipeline, segment, voice, device, "autocast_fp16" if device == "cuda" else "fp32")
                audio_parts.append(audio)
        audio = np.concatenate(audio_parts)
        output_path = OUTPUT_DIR / f"asmr7_variation_{name}.wav"
        sf.write(output_path, audio, SAMPLE_RATE)
        print(
            f"variation={name} source={SOURCE.name!r} block={start_s:.1f}-{start_s+BLOCK_SECONDS:.1f}s "
            f"voice={voice_path} output={output_path} duration={len(audio)/SAMPLE_RATE:.3f}s stats={stats}"
        )


if __name__ == "__main__":
    main()
