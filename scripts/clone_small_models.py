#!/usr/bin/env python3
"""Generate small-model voice-cloning comparison clips from local references.

This script assumes the Chatterbox-Nano checkpoint has been downloaded to
``models/chatterbox-nano`` and that the Pocket-TTS model is loaded from a local
config whose ``weights_path`` points at the non-gated OpenEnsemble mirror.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.io.wavfile
import soundfile as sf
import torch


ROOT = Path(__file__).resolve().parents[1]
TEXT = (
    "Hey there. You are doing great. Take a slow breath and let your shoulders "
    "relax. I am right here with you."
)
REFERENCES = {
    "asmr7": ROOT / "artifacts/references/chunks/asmr7_chunk_36-46.wav",
    "mommy": ROOT / "artifacts/references/chunks/mommy_chunk_0.75-9.80.wav",
}


def generate_chatterbox(out_dir: Path, device: str) -> dict[str, object]:
    from chatterbox.tts_turbo import ChatterboxTurboTTS

    model_dir = ROOT / "models/chatterbox-nano"
    required = [model_dir / name for name in ("ve.safetensors", "t3_nano_v1.safetensors", "s3gen_meanflow.safetensors")]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Chatterbox-Nano checkpoint is incomplete:\n" + "\n".join(missing))

    model = ChatterboxTurboTTS.from_local(model_dir, device=device, nano=True)
    result: dict[str, object] = {"model": "ResembleAI/chatterbox-nano", "device": device, "outputs": {}}
    for tag, reference in REFERENCES.items():
        with torch.inference_mode():
            audio = model.generate(TEXT, audio_prompt_path=str(reference), norm_loudness=True)
        output = out_dir / f"chatterbox_nano_{tag}.wav"
        array = audio.detach().cpu().numpy().squeeze(0).astype(np.float32)
        scipy.io.wavfile.write(output, model.sr, array)
        result["outputs"][tag] = {"path": str(output), "seconds": round(audio.shape[-1] / model.sr, 3)}
    return result


def generate_pocket(out_dir: Path, config_path: Path) -> dict[str, object]:
    from pocket_tts import TTSModel

    model = TTSModel.load_model(config=str(config_path))
    if not model.has_voice_cloning:
        raise RuntimeError("Pocket-TTS loaded without voice-cloning weights")
    result: dict[str, object] = {"model": "Pocket-TTS (OpenEnsemble mirror)", "sample_rate": model.sample_rate, "outputs": {}}
    for tag, reference in REFERENCES.items():
        state = model.get_state_for_audio_prompt(reference, truncate=True)
        audio = model.generate_audio(state, TEXT, frames_after_eos=2)
        array = audio.detach().cpu().numpy().astype(np.float32)
        output = out_dir / f"pocket_tts_{tag}.wav"
        scipy.io.wavfile.write(output, model.sample_rate, array)
        result["outputs"][tag] = {"path": str(output), "seconds": round(array.shape[-1] / model.sample_rate, 3)}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("chatterbox", "pocket", "both"), default="both")
    parser.add_argument("--pocket-config", type=Path, default=Path("/tmp/pocket-tts-openensemble.yaml"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in REFERENCES.values():
        if not path.exists():
            raise FileNotFoundError(path)

    results: dict[str, object] = {"text": TEXT, "references": {key: str(value) for key, value in REFERENCES.items()}}
    if args.model in ("chatterbox", "both"):
        results["chatterbox"] = generate_chatterbox(out_dir, args.device)
    if args.model in ("pocket", "both"):
        results["pocket"] = generate_pocket(out_dir, args.pocket_config)
    manifest = out_dir / "small_tts_clone_manifest.json"
    manifest.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
