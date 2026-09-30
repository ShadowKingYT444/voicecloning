#!/usr/bin/env python3
"""Benchmark Whisper_fast_package against its no-VAD path."""

from __future__ import annotations

import difflib
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "Whisper_fast_package"))

from whisper_fast_vad import WhisperFastVAD, load_audio  # noqa: E402


REFERENCE_AUDIO = ROOT / "artifacts" / "references" / "mommy_trimmed_mono_24k.wav"
DEFAULT_OUTPUT = ROOT / "Whisper_fast_package" / "benchmarks" / "whisper_fast_vad.json"


def rss_mb() -> float:
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text().splitlines():
            if line.startswith("VmRSS:"):
                return round(float(line.split()[1]) / 1024.0, 1)
    return float("nan")


def similarity(left: str, right: str) -> float:
    return round(difflib.SequenceMatcher(None, left.lower().split(), right.lower().split()).ratio(), 4)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, default=REFERENCE_AUDIO)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--silence-seconds", type=float, default=30.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    started = time.perf_counter()
    engine = WhisperFastVAD(cpu_threads=args.threads)
    load_seconds = time.perf_counter() - started
    audio = load_audio(args.audio)
    silence = np.zeros(max(0, int(args.silence_seconds * 16_000)), dtype=np.float32)
    padded = np.concatenate((silence, audio, silence))

    started = time.perf_counter()
    selected = engine.vad.segment(padded)
    vad_seconds = time.perf_counter() - started

    started = time.perf_counter()
    vad_result = engine.transcribe_array(padded, 16_000, return_result=True)
    vad_elapsed = time.perf_counter() - started

    started = time.perf_counter()
    full_text, full_rows = engine._decode(padded, "en", False)
    full_elapsed = time.perf_counter() - started

    expected_path = ROOT / "artifacts" / "transcript.txt"
    expected = expected_path.read_text().strip() if expected_path.exists() else ""
    result = {
        "model": "Systran/faster-whisper-tiny.en",
        "runtime": "Faster-Whisper / CTranslate2",
        "compute_type": "int8",
        "device": "cpu",
        "threads": args.threads,
        "input": str(args.audio),
        "input_audio_seconds": round(len(audio) / 16_000, 4),
        "benchmark_audio_seconds_with_silence": round(len(padded) / 16_000, 4),
        "silence_seconds_each_side": args.silence_seconds,
        "model_load_seconds": round(load_seconds, 6),
        "rss_after_load_mb": rss_mb(),
        "vad_segmentation_seconds": round(vad_seconds, 6),
        "vad_segments": len(selected),
        "vad_speech_seconds": round(sum(x.duration_seconds for x in selected), 4),
        "vad_elapsed_seconds": round(vad_elapsed, 6),
        "vad_real_time_factor": round(vad_result.real_time_factor, 4),
        "vad_text": vad_result.text,
        "full_decode_seconds": round(full_elapsed, 6),
        "full_real_time_factor": round(full_elapsed / (len(padded) / 16_000), 4),
        "full_text": full_text,
        "vad_speedup_vs_full": round(full_elapsed / vad_elapsed, 4),
        "expected_text_similarity_vad": similarity(vad_result.text, expected) if expected else None,
        "expected_text_similarity_full": similarity(full_text, expected) if expected else None,
        "full_segment_count": len(full_rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
