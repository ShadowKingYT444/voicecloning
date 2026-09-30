#!/usr/bin/env python3
"""Local Kokoro-82M + Inno voice-cloning setup and benchmark.

The script intentionally loads model files from this workspace so inference does
not depend on an online Hugging Face lookup after setup.
"""

from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from inno_kokoro.enroll import Tuner, enroll, read
from kokoro import KModel, KPipeline


ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "models" / "kokoro"
INNO_WEIGHTS = ROOT / "models" / "inno" / "model.safetensors"
DEFAULT_REFERENCE = ROOT / "artifacts" / "references" / "mommy_trimmed_mono_24k.wav"
DEFAULT_VOICE = ROOT / "voices" / "af_mommy.pt"
DEFAULT_TEST_RENDER = ROOT / "outputs" / "af_mommy_test.wav"
DEFAULT_BENCHMARK = ROOT / "outputs" / "kokoro_benchmark.json"
SAMPLE_RATE = 24_000

BENCHMARK_TEXT = """Kokoro is a small, efficient speech model designed for fast local synthesis.
This benchmark measures the time spent rendering a fixed passage after the model,
voice pack, phonemizer, and CUDA kernels have been warmed up. The result is
reported in phoneme tokens per second and generated audio seconds per second.
The same locally enrolled voice is used for every pass so that the measurement
represents the configured voice-cloning path rather than a stock voice."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--voice", type=Path, default=DEFAULT_VOICE)
    parser.add_argument("--test-render", type=Path, default=DEFAULT_TEST_RENDER)
    parser.add_argument("--benchmark-json", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--precision",
        choices=("fp32", "autocast_fp16"),
        default="autocast_fp16",
        help="Inference precision; autocast_fp16 is the default safe GPU optimization.",
    )
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--skip-enroll", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    return parser.parse_args()


def check_paths(reference: Path, voice: Path | None = None) -> None:
    required = [BASE_DIR / "config.json", BASE_DIR / "kokoro-v1_0.pth", INNO_WEIGHTS, reference]
    if voice is not None:
        required.append(voice)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing local setup file(s):\n" + "\n".join(missing))


def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def load_pipeline(device: str) -> KPipeline:
    if device.startswith("cuda"):
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        # Text lengths vary between requests; cuDNN autotuning can perform a
        # multi-second search for each new shape, which harms tail latency.
        torch.backends.cudnn.benchmark = False
    model = KModel(
        repo_id=str(BASE_DIR),
        config=str(BASE_DIR / "config.json"),
        model=str(BASE_DIR / "kokoro-v1_0.pth"),
    ).to(device).eval()
    return KPipeline(lang_code="a", repo_id=str(BASE_DIR), model=model, device=device)


def render(
    pipeline: KPipeline,
    text: str,
    voice: torch.Tensor,
    device: str,
    precision: str,
) -> tuple[np.ndarray, int]:
    chunks: list[np.ndarray] = []
    token_count = 0
    autocast_enabled = precision == "autocast_fp16" and device.startswith("cuda")
    autocast_dtype = torch.float16
    context = torch.autocast(device_type="cuda", dtype=autocast_dtype) if autocast_enabled else nullcontext()
    with torch.inference_mode(), context:
        for result in pipeline(text, voice=voice):
            if result.audio is not None:
                chunks.append(result.audio.detach().float().cpu().numpy())
            if result.tokens is not None:
                token_count += len(result.tokens)
    if not chunks:
        raise RuntimeError("Kokoro produced no audio")
    return np.concatenate(chunks), token_count


def enroll_voice(reference: Path, voice_path: Path, device: str) -> dict[str, object]:
    voice_path.parent.mkdir(parents=True, exist_ok=True)
    wav, sample_rate = read(str(reference))
    tuner = Tuner(str(INNO_WEIGHTS), device=device)
    synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        pack, stats = enroll(wav.to(device), sample_rate, tuner)
    synchronize(device)
    elapsed = time.perf_counter() - started
    torch.save(pack.detach().cpu(), voice_path)
    return {
        "reference": str(reference),
        "reference_sample_rate": sample_rate,
        "reference_seconds": round(float(wav.shape[0] / sample_rate), 3),
        "voice_pack": str(voice_path),
        "voice_pack_shape": list(pack.shape),
        "enrollment_seconds": round(elapsed, 4),
        "enrollment_stats": stats,
        "device": device,
    }


def benchmark(
    pipeline: KPipeline,
    voice: torch.Tensor,
    device: str,
    precision: str,
    iterations: int,
) -> dict[str, object]:
    # Trigger G2P/model setup and CUDA kernel selection outside timed passes.
    with torch.inference_mode():
        render(pipeline, "Warm up the local Kokoro voice.", voice, device, precision)
    synchronize(device)

    timings: list[float] = []
    tokens: list[int] = []
    audio_seconds: list[float] = []
    for _ in range(iterations):
        synchronize(device)
        started = time.perf_counter()
        audio, token_count = render(pipeline, BENCHMARK_TEXT, voice, device, precision)
        synchronize(device)
        elapsed = time.perf_counter() - started
        timings.append(elapsed)
        tokens.append(token_count)
        audio_seconds.append(float(audio.shape[0] / SAMPLE_RATE))

    elapsed = float(np.mean(timings))
    token_count = int(round(np.mean(tokens)))
    generated_audio_seconds = float(np.mean(audio_seconds))
    return {
        "device": device,
        "precision": precision,
        "iterations": iterations,
        "benchmark_text_characters": len(BENCHMARK_TEXT),
        "phoneme_tokens": token_count,
        "mean_seconds": round(elapsed, 6),
        "min_seconds": round(float(np.min(timings)), 6),
        "max_seconds": round(float(np.max(timings)), 6),
        "phoneme_tokens_per_second": round(token_count / elapsed, 2),
        "audio_seconds_per_second": round(generated_audio_seconds / elapsed, 2),
        "real_time_factor": round(elapsed / generated_audio_seconds, 4),
        "target_tokens_per_second": 100,
        "target_reached": token_count / elapsed >= 100,
    }


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    if args.iterations < 1:
        raise ValueError("--iterations must be at least 1")
    check_paths(args.reference, args.voice if args.skip_enroll else None)
    args.test_render.parent.mkdir(parents=True, exist_ok=True)
    args.benchmark_json.parent.mkdir(parents=True, exist_ok=True)

    if args.skip_enroll:
        enrollment = {"skipped": True, "voice_pack": str(args.voice)}
    else:
        enrollment = enroll_voice(args.reference, args.voice, args.device)

    pipeline = load_pipeline(args.device)
    # Keep the pack on CPU: Kokoro 0.9.x recognizes torch.FloatTensor voice
    # packs and moves them to the model device internally.
    voice = torch.load(args.voice, map_location="cpu", weights_only=True)

    with torch.inference_mode():
        test_audio, test_tokens = render(
            pipeline,
            "Hello from a locally enrolled Kokoro voice.",
            voice,
            args.device,
            args.precision,
        )
    sf.write(args.test_render, test_audio, SAMPLE_RATE)

    result: dict[str, object] = {
        "model": "hexgrad/Kokoro-82M",
        "tuner": "remsky/kokoro-inno-clone-tuner / inno-kokoro",
        "base_model_dir": str(BASE_DIR),
        "tuner_weights": str(INNO_WEIGHTS),
        "test_render": str(args.test_render),
        "test_render_seconds": round(float(test_audio.shape[0] / SAMPLE_RATE), 3),
        "test_render_tokens": test_tokens,
        "enrollment": enrollment,
    }
    if args.skip_benchmark:
        result["benchmark"] = {"skipped": True}
    else:
        result["benchmark"] = benchmark(
            pipeline,
            voice,
            args.device,
            args.precision,
            args.iterations,
        )
    args.benchmark_json.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
