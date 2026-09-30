#!/usr/bin/env python3
"""Benchmark the packaged Pocket ONNX runtime against the packaged Kokoro runtime."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Pocket_package"))
sys.path.insert(0, str(ROOT))

BENCHMARK_TEXT = "Realtime speech needs low startup latency and steady streaming. This fixed passage measures warm voice synthesis."
POCKET_MAX_FRAMES = 54

POCKET_REFERENCE = ROOT / "Pocket_package" / "voices" / "asmr7_30s_loud_reference.wav"
DEFAULT_OUT = ROOT / "outputs" / "pocket_vs_kokoro_onnx.json"


def rss_mb() -> float:
    """Read the current process RSS without adding a dependency."""
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text().splitlines():
            if line.startswith("VmRSS:"):
                return round(float(line.split()[1]) / 1024.0, 1)
    return float("nan")


def timed(fn):
    started = time.perf_counter()
    value = fn()
    return value, time.perf_counter() - started


def benchmark_pocket(iterations: int, threads: int) -> dict[str, object]:
    from pocket_tts_onnx import PocketTTSOnnx

    started = time.perf_counter()
    tts = PocketTTSOnnx(
        models_dir=str(ROOT / "Pocket_package" / "onnx"),
        language="english_2026-04",
        precision="int8",
        device="cpu",
        temperature=0.2,
        lsd_steps=1,
        intra_op_num_threads=threads,
    )
    load_seconds = time.perf_counter() - started
    rss_after_load = rss_mb()

    _, conditioning_seconds = timed(lambda: tts.prepare_voice_state(POCKET_REFERENCE))
    rss_after_conditioning = rss_mb()

    # Warm up text/shape paths, then measure synthesis with the cached voice state.
    warm_audio = tts.generate(BENCHMARK_TEXT, voice=POCKET_REFERENCE, max_frames=POCKET_MAX_FRAMES, frames_after_eos=2)
    timings: list[float] = []
    audio_seconds: list[float] = []
    for _ in range(iterations):
        audio, elapsed = timed(lambda: tts.generate(BENCHMARK_TEXT, voice=POCKET_REFERENCE, max_frames=POCKET_MAX_FRAMES, frames_after_eos=2))
        timings.append(elapsed)
        audio_seconds.append(float(audio.shape[0] / tts.sample_rate))

    return {
        "model": "Pocket TTS local checkpoint",
        "runtime": "ONNX Runtime CPU",
        "precision": "INT8 dynamic MatMul",
        "device": tts.device,
        "threads": threads,
        "temperature": 0.2,
        "lsd_steps": 1,
        "reference": str(POCKET_REFERENCE),
        "reference_seconds": round(float(sf.info(POCKET_REFERENCE).duration), 3),
        "model_load_seconds": round(load_seconds, 6),
        "reference_conditioning_seconds": round(conditioning_seconds, 6),
        "rss_after_load_mb": rss_after_load,
        "rss_after_conditioning_mb": rss_after_conditioning,
        "iterations": iterations,
        "audio_seconds": round(float(np.mean(audio_seconds)), 4),
        "mean_seconds": round(float(np.mean(timings)), 6),
        "median_seconds": round(float(np.median(timings)), 6),
        "min_seconds": round(float(np.min(timings)), 6),
        "max_seconds": round(float(np.max(timings)), 6),
        "audio_seconds_per_second": round(float(np.mean(audio_seconds) / np.mean(timings)), 3),
        "real_time_factor": round(float(np.mean(timings) / np.mean(audio_seconds)), 4),
        "warm_output_seconds": round(float(warm_audio.shape[0] / tts.sample_rate), 4),
        "max_frames": POCKET_MAX_FRAMES,
    }


def benchmark_kokoro(iterations: int, device: str) -> dict[str, object]:
    from kokoro_package.kokoro_voice import KokoroVoice

    started = time.perf_counter()
    voice = KokoroVoice(device=device, precision="autocast_fp16" if device == "cuda" else "fp32")
    load_seconds = time.perf_counter() - started
    rss_after_load = rss_mb()
    voice.synthesize("Warm up the packaged Kokoro voice.")

    timings: list[float] = []
    audio_seconds: list[float] = []
    for _ in range(iterations):
        audio, elapsed = timed(lambda: voice.synthesize(BENCHMARK_TEXT))
        timings.append(elapsed)
        audio_seconds.append(float(audio.shape[0] / 24000.0))

    return {
        "model": "Kokoro-82M packaged runtime",
        "runtime": "PyTorch KokoroVoice",
        "precision": voice.precision,
        "device": device,
        "model_load_seconds": round(load_seconds, 6),
        "rss_after_load_mb": rss_after_load,
        "iterations": iterations,
        "audio_seconds": round(float(np.mean(audio_seconds)), 4),
        "mean_seconds": round(float(np.mean(timings)), 6),
        "median_seconds": round(float(np.median(timings)), 6),
        "min_seconds": round(float(np.min(timings)), 6),
        "max_seconds": round(float(np.max(timings)), 6),
        "audio_seconds_per_second": round(float(np.mean(audio_seconds) / np.mean(timings)), 3),
        "real_time_factor": round(float(np.mean(timings) / np.mean(audio_seconds)), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--kokoro-device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--skip-kokoro", action="store_true")
    parser.add_argument("--skip-pocket", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.iterations < 1:
        raise ValueError("--iterations must be >= 1")
    torch = __import__("torch")
    kokoro_device = args.kokoro_device or ("cuda" if torch.cuda.is_available() else "cpu")
    result = {
        "hardware": {
            "cpu": os.cpu_count(),
            "gpu": str(__import__("torch").cuda.get_device_name(0)) if __import__("torch").cuda.is_available() else None,
        },
        "benchmark_text_characters": len(BENCHMARK_TEXT),
        "pocket": {"skipped": True} if args.skip_pocket else benchmark_pocket(args.iterations, args.threads),
        "kokoro": {"skipped": True} if args.skip_kokoro else benchmark_kokoro(args.iterations, kokoro_device),
    }
    if not args.skip_kokoro and not args.skip_pocket:
        result["comparison"] = {
            "pocket_vs_kokoro_warm_speedup": round(
                result["pocket"]["audio_seconds_per_second"]
                / result["kokoro"]["audio_seconds_per_second"],
                3,
            ),
            "pocket_vs_kokoro_warm_latency_ratio": round(
                result["pocket"]["mean_seconds"] / result["kokoro"]["mean_seconds"],
                3,
            ),
            "note": "Compare providers/devices from the fields above; Pocket is CPU ONNX and Kokoro uses the selected packaged device.",
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
