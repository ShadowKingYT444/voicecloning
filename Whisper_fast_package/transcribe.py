#!/usr/bin/env python3
"""CLI for the packaged Faster-Whisper tiny + WebRTC-VAD runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from whisper_fast_vad import WhisperFastVAD


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--language", default="en")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--vad-aggressiveness", type=int, default=2)
    parser.add_argument("--timestamps", action="store_true")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    engine = WhisperFastVAD(
        model_path=args.model,
        compute_type=args.compute_type,
        cpu_threads=args.threads,
    )
    result = engine.transcribe_file(
        args.audio,
        language=args.language,
        return_result=True,
        with_timestamps=args.timestamps,
    )
    if args.as_json:
        print(json.dumps(result.__dict__, indent=2))
    else:
        print(result.text)
        print(
            f"\n[{result.elapsed_seconds:.3f}s; {result.audio_seconds:.2f}s audio; "
            f"{result.speech_seconds:.2f}s speech; RTF {result.real_time_factor:.3f}]"
        )


if __name__ == "__main__":
    main()
