#!/usr/bin/env python3
"""Benchmark Chatterbox-Nano/Pocket-TTS and try longer Pocket-TTS prompts."""

from __future__ import annotations

import argparse
import json
import time
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
POCKET_CONFIG = Path("/tmp/pocket-tts-openensemble.yaml")
POCKET_REFS = {
    "baseline_10s": ROOT / "artifacts/references/chunks/asmr7_chunk_36-46.wav",
    "tuned_30s": ROOT / "artifacts/references/chunks/asmr7_chunk_80-110.wav",
    "tuned_60s": ROOT / "artifacts/references/chunks/asmr7_chunk_50-110.wav",
}


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _summary(samples: list[float], audio_seconds: float) -> dict[str, object]:
    arr = np.asarray(samples, dtype=np.float64)
    median = float(np.median(arr))
    return {
        "runs_seconds": [round(float(x), 4) for x in arr],
        "mean_seconds": round(float(arr.mean()), 4),
        "median_seconds": round(median, 4),
        "min_seconds": round(float(arr.min()), 4),
        "max_seconds": round(float(arr.max()), 4),
        "output_audio_seconds": round(audio_seconds, 3),
        "median_real_time_factor": round(audio_seconds / median, 3),
        "median_audio_seconds_per_second": round(median / audio_seconds, 3),
    }


def benchmark_chatterbox(device: str, runs: int) -> dict[str, object]:
    from chatterbox.tts_turbo import ChatterboxTurboTTS

    model_dir = ROOT / "models/chatterbox-nano"
    ref = ROOT / "artifacts/references/chunks/asmr7_chunk_36-46.wav"
    _sync(device)
    started = time.perf_counter()
    model = ChatterboxTurboTTS.from_local(model_dir, device=device, nano=True)
    _sync(device)
    load_seconds = time.perf_counter() - started

    _sync(device)
    started = time.perf_counter()
    model.prepare_conditionals(str(ref), norm_loudness=True)
    _sync(device)
    conditioning_seconds = time.perf_counter() - started

    # Warm up CUDA kernels and tokenizer paths outside the timed runs.
    with torch.inference_mode():
        warm = model.generate(TEXT)
    _sync(device)

    timed: list[float] = []
    last = warm
    with torch.inference_mode():
        for _ in range(runs):
            _sync(device)
            started = time.perf_counter()
            last = model.generate(TEXT)
            _sync(device)
            timed.append(time.perf_counter() - started)

    end_to_end: list[float] = []
    with torch.inference_mode():
        for _ in range(max(1, min(2, runs))):
            _sync(device)
            started = time.perf_counter()
            last = model.generate(TEXT, audio_prompt_path=str(ref), norm_loudness=True)
            _sync(device)
            end_to_end.append(time.perf_counter() - started)

    audio_seconds = float(last.shape[-1] / model.sr)
    return {
        "checkpoint": "ResembleAI/chatterbox-nano",
        "device": device,
        "reference": str(ref),
        "model_load_seconds": round(load_seconds, 3),
        "reference_conditioning_seconds": round(conditioning_seconds, 3),
        "synthesis_only": _summary(timed, audio_seconds),
        "end_to_end": {
            **_summary(end_to_end, audio_seconds),
            "includes_reference_conditioning": True,
        },
    }


def benchmark_pocket(config: Path, runs: int, out_dir: Path) -> dict[str, object]:
    from pocket_tts import TTSModel

    started = time.perf_counter()
    model = TTSModel.load_model(config=str(config))
    load_seconds = time.perf_counter() - started
    result: dict[str, object] = {
        "checkpoint": "openensemble/pocket-tts (English voice-cloning mirror)",
        "device": str(model.device),
        "model_load_seconds": round(load_seconds, 3),
        "references": {},
    }

    for tag, ref in POCKET_REFS.items():
        started = time.perf_counter()
        state = model.get_state_for_audio_prompt(ref, truncate=False)
        conditioning_seconds = time.perf_counter() - started

        warm = model.generate_audio(state, TEXT, frames_after_eos=2)

        timed: list[float] = []
        last = warm
        for _ in range(runs):
            started = time.perf_counter()
            last = model.generate_audio(state, TEXT, frames_after_eos=2)
            timed.append(time.perf_counter() - started)

        # One request as an end-to-end measurement, including prompt encoding.
        started = time.perf_counter()
        e2e_state = model.get_state_for_audio_prompt(ref, truncate=False)
        e2e_audio = model.generate_audio(e2e_state, TEXT, frames_after_eos=2)
        end_to_end_seconds = time.perf_counter() - started

        audio_seconds = float(last.shape[-1] / model.sample_rate)
        output = out_dir / f"pocket_tts_asmr7_{tag}.wav"
        scipy.io.wavfile.write(output, model.sample_rate, last.detach().cpu().numpy().astype(np.float32))
        result["references"][tag] = {
            "reference": str(ref),
            "reference_seconds": round(float(sf.info(ref).duration), 3),
            "reference_conditioning_seconds": round(conditioning_seconds, 3),
            "synthesis_only": _summary(timed, audio_seconds),
            "end_to_end_seconds": round(end_to_end_seconds, 4),
            "end_to_end_real_time_factor": round(audio_seconds / end_to_end_seconds, 3),
            "output": str(output),
            "output_audio_seconds": round(float(e2e_audio.shape[-1] / model.sample_rate), 3),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-chatterbox", action="store_true")
    parser.add_argument("--skip-pocket", action="store_true")
    args = parser.parse_args()
    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, object] = {"text": TEXT, "runs": args.runs}
    if not args.skip_chatterbox:
        results["chatterbox_nano"] = benchmark_chatterbox(args.device, args.runs)
    if not args.skip_pocket:
        results["pocket_tts"] = benchmark_pocket(POCKET_CONFIG, args.runs, out_dir)
    output = out_dir / "latency_benchmark.json"
    if output.exists():
        previous = json.loads(output.read_text(encoding="utf-8"))
        previous.update(results)
        results = previous
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
