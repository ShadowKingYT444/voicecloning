"""Make loudness-matched listening copies using only constant gain.

The shared target is lowered if needed to preserve each clip's dynamics and a
four-times-oversampled -1 dBTP peak ceiling. No limiter, filter, or gate is used.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pyloudnorm as ln
import soundfile as sf
from scipy.signal import resample_poly


def measure(path):
    audio, sr = sf.read(path, dtype="float64", always_2d=False)
    if audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all():
        raise ValueError(f"Expected finite mono audio: {path}")
    loudness = float(ln.Meter(sr).integrated_loudness(audio))
    peak = float(20 * np.log10(max(np.max(np.abs(resample_poly(audio, 4, 1))), 1e-12)))
    if not math.isfinite(loudness):
        raise ValueError(f"Cannot measure loudness: {path}")
    return audio, sr, loudness, peak


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-lufs", type=float, default=-27.0)
    args = parser.parse_args()
    rows = json.loads(args.manifest.read_text())
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)) or any(Path(i).name != i for i in ids):
        raise ValueError("IDs must be unique safe filenames")
    measured = []
    target = args.target_lufs
    for row in rows:
        _, _, loudness, peak = measure(row["path"])
        target = min(target, loudness - 1.0 - peak)
        measured.append((loudness, peak))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = []
    for row, (original_lufs, original_peak) in zip(rows, measured):
        audio, sr, _, _ = measure(row["path"])
        gain = target - original_lufs
        path = args.output_dir / (row["id"] + ".wav")
        # R128's absolute gate can change membership after attenuation. A
        # single gain calculation is therefore not always loudness-exact.
        # Recompute the scalar gain from the ORIGINAL audio each time. This
        # preserves dynamics and never adds a compressor, gate, or limiter.
        for gain_iteration in range(8):
            sf.write(path, audio * (10 ** (gain / 20)), sr, subtype="PCM_24")
            _, _, achieved, peak = measure(path)
            if peak > -0.99:
                raise RuntimeError(f"Peak verification failed: {path}")
            if abs(achieved - target) <= 0.01:
                break
            gain += target - achieved
        else:
            raise RuntimeError(f"Loudness verification failed: {path}")
        output.append({**row, "path": str(path.resolve()),
                       "source_path": str(Path(row["path"]).resolve()),
                       "source_sha256": hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest(),
                       "processing": "constant gain only, PCM24 quantization",
                       "gain_db": gain, "original_lufs": original_lufs,
                       "gain_iterations": gain_iteration + 1,
                       "original_true_peak_dbtp": original_peak,
                       "target_lufs": target, "achieved_lufs": achieved,
                       "true_peak_dbtp_4x": peak})
    (args.output_dir / "manifest.json").write_text(json.dumps(output, indent=2))
    print(json.dumps({"count": len(output), "shared_target_lufs": target,
                      "maximum_loudness_error": max(abs(r["achieved_lufs"] - target) for r in output)}))


if __name__ == "__main__":
    main()
