#!/usr/bin/env python3
"""Try higher-fidelity enrollments from longer clean windows of the new ASMR file."""

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from inno_kokoro.enroll import Tuner, enroll

from kokoro_local import INNO_WEIGHTS, SAMPLE_RATE, load_pipeline, render


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "ASMR  7 Minutes in Heaven... With Your Bully_  [Enemies to Lovers] [Tsundere] [Confession] [Kiss].mp3"
SOURCE_WAV = ROOT / "artifacts" / "references" / "asmr7_mono_24k.wav"
TEXT = (
    "Olson begins with a short declarative sentence that echoes the familiar national slogan that emerged after 9/11. "
    "Because the phrase is so culturally recognizable, he can initially invoke the unity and determination Americans "
    "associate with the attacks without explaining them. More importantly, he sets up the phrase so he can later "
    "undermine it: the rest of the essay argues that America has in fact forgotten. This creates a structural irony "
    "between what Americans claim to believe and how Olson believes they actually behave."
)

# Longer windows give the speaker encoder more identity evidence than the
# earlier 12-second clips. fmax candidates address pitch-tracking uncertainty
# in whispered/ASMR audio.
CANDIDATES = {
    "1": {"start": 35.0, "fmax": None},
    "2": {"start": 80.0, "fmax": 180.0},
    "3": {"start": 0.0, "fmax": 220.0},
}
WINDOW_SECONDS = 30.0


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not SOURCE.exists() or not SOURCE_WAV.exists():
        raise FileNotFoundError("New ASMR source or its local 24 kHz conversion is missing")
    source, sr = sf.read(SOURCE_WAV, dtype="float32")
    if sr != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE} Hz, got {sr} Hz")
    tuner = Tuner(str(INNO_WEIGHTS), device=device)
    pipeline = load_pipeline(device)
    (ROOT / "voices").mkdir(exist_ok=True)
    (ROOT / "outputs").mkdir(exist_ok=True)

    for name, config in CANDIDATES.items():
        start = round(config["start"] * sr)
        end = round((config["start"] + WINDOW_SECONDS) * sr)
        block = torch.from_numpy(source[start:end]).to(device)
        with torch.inference_mode():
            kwargs = {} if config["fmax"] is None else {"fmax": config["fmax"]}
            pack, stats = enroll(block, sr, tuner, **kwargs)
        voice_path = ROOT / "voices" / f"af_asmr7_refined_v{name}.pt"
        torch.save(pack.detach().cpu(), voice_path)
        voice = torch.load(voice_path, map_location="cpu", weights_only=True)

        parts = []
        for segment in TEXT.split(". "):
            segment = segment.strip()
            if segment and not segment.endswith("."):
                segment += "."
            if segment:
                audio, _ = render(pipeline, segment, voice, device, "autocast_fp16" if device == "cuda" else "fp32")
                parts.append(audio)
        audio = np.concatenate(parts)
        output = ROOT / "outputs" / f"asmr7_refined_{name}.wav"
        sf.write(output, audio, SAMPLE_RATE)
        print(
            f"variant={name} source={SOURCE.name!r} block={config['start']:.1f}-{config['start']+WINDOW_SECONDS:.1f}s "
            f"fmax={config['fmax']} output={output} duration={len(audio)/SAMPLE_RATE:.3f}s stats={stats}"
        )


if __name__ == "__main__":
    main()
