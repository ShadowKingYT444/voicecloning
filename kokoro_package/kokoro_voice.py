#!/usr/bin/env python3
"""Minimal local Kokoro voice runtime for application integration.

Usage as a library:

    from kokoro_voice import KokoroVoice
    voice = KokoroVoice()
    audio = voice.synthesize("Hello from my local agent.")

Usage as a CLI:

    python kokoro_voice.py "Hello from my local agent." -o reply.wav
"""

from __future__ import annotations

import argparse
import re
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from kokoro import KModel, KPipeline


PACKAGE_DIR = Path(__file__).resolve().parent
SAMPLE_RATE = 24_000


def _split_for_kokoro(text: str, max_chars: int = 320) -> list[str]:
    """Split at sentence/word boundaries to stay below Kokoro's token limit."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: list[str] = []
    for sentence in sentences:
        words = sentence.split()
        current: list[str] = []
        current_len = 0
        for word in words:
            added_len = len(word) + (1 if current else 0)
            if current and current_len + added_len > max_chars:
                chunks.append(" ".join(current))
                current = []
                current_len = 0
            current.append(word)
            current_len += added_len
        if current:
            chunks.append(" ".join(current))
    return [chunk for chunk in chunks if chunk]


class KokoroVoice:
    """Load the packaged Kokoro model and the packaged Mommy voice pack."""

    def __init__(
        self,
        device: str | None = None,
        precision: str = "autocast_fp16",
        voice_path: str | Path | None = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        if precision not in {"fp32", "autocast_fp16"}:
            raise ValueError("precision must be 'fp32' or 'autocast_fp16'")

        self.device = device
        self.precision = precision if device == "cuda" else "fp32"
        self.base_dir = PACKAGE_DIR / "models" / "kokoro"
        self.voice_path = Path(voice_path) if voice_path else PACKAGE_DIR / "voices" / "af_asmr7.pt"
        config_path = self.base_dir / "config.json"
        model_path = self.base_dir / "kokoro-v1_0.pth"
        for path in (config_path, model_path, self.voice_path):
            if not path.exists():
                raise FileNotFoundError(path)

        if device == "cuda":
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = False

        self.model = KModel(
            repo_id=str(self.base_dir),
            config=str(config_path),
            model=str(model_path),
        ).to(device).eval()
        self.pipeline = KPipeline(
            lang_code="a",
            repo_id=str(self.base_dir),
            model=self.model,
            device=device,
        )
        # Kokoro 0.9.x recognizes CPU FloatTensor voice packs and moves them
        # to the model device internally.
        self.voice_pack = torch.load(self.voice_path, map_location="cpu", weights_only=True)

    def synthesize(self, text: str) -> np.ndarray:
        """Return 24 kHz mono float32 audio for arbitrary-length text."""
        chunks: list[np.ndarray] = []
        autocast = self.precision == "autocast_fp16"
        context = torch.autocast(device_type="cuda", dtype=torch.float16) if autocast else nullcontext()
        with torch.inference_mode(), context:
            for segment in _split_for_kokoro(text):
                for result in self.pipeline(segment, voice=self.voice_pack):
                    if result.audio is not None:
                        chunks.append(result.audio.detach().float().cpu().numpy())
        if not chunks:
            raise ValueError("text produced no audio")
        return np.concatenate(chunks)

    def save(self, text: str, output_path: str | Path) -> Path:
        """Synthesize and save a WAV file, returning its path."""
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output, self.synthesize(text), SAMPLE_RATE)
        return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="?", help="Text to synthesize; stdin is used when omitted.")
    parser.add_argument("-o", "--output", type=Path, default=Path("reply.wav"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default=None)
    parser.add_argument("--fp32", action="store_true", help="Disable CUDA autocast.")
    args = parser.parse_args()
    text = args.text if args.text is not None else input()
    voice = KokoroVoice(device=args.device, precision="fp32" if args.fp32 else "autocast_fp16")
    output = voice.save(text, args.output)
    print(output)


if __name__ == "__main__":
    main()
