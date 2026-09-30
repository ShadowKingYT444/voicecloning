"""Measure explicit HiFT source-noise changes with fixed mel and phase inputs.

Research only. Run under bounded_job.py. Outputs are intermediate waveforms;
use the normal Perth/master stage before publishing any listening samples.
"""
import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np

from onnx_staged_runtime import VocoderOrtRuntime, DEFAULT_MODEL_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.run_dir / "estimator/mel.npz") as data:
        mel = data["mel"]
    with np.load(args.run_dir / "vocoder/audio.npz") as data:
        phase, noise, baseline = data["phase_noise"], data["sine_noise"], data["audio"]
    runtime = VocoderOrtRuntime(DEFAULT_MODEL_DIR, intra_op_num_threads=1)
    rows = []
    for scale in (1.0, 0.5, 0.0):
        started = time.perf_counter()
        audio = runtime.synthesize(mel, phase_noise=phase, sine_noise=noise * scale).reshape(-1)
        if not np.isfinite(audio).all() or audio.shape != baseline.shape:
            raise RuntimeError("Invalid vocoder output")
        if scale == 1.0 and not np.array_equal(audio, baseline):
            raise RuntimeError("Scale 1 does not reproduce the saved baseline exactly")
        output = args.output_dir / f"scale_{scale:.1f}" / "vocoder"
        output.mkdir(parents=True, exist_ok=True)
        np.savez(output / "audio.npz", audio=audio)
        difference = audio.astype(np.float64) - baseline
        rows.append({
            "source_noise_scale": scale,
            "seconds": time.perf_counter() - started,
            "identical_to_baseline": bool(np.array_equal(audio, baseline)),
            "max_abs_difference": float(np.max(np.abs(difference))),
            "difference_rms": float(np.sqrt(np.mean(difference ** 2))),
            "relative_l2_difference": float(np.linalg.norm(difference) / max(np.linalg.norm(baseline), 1e-12)),
            "intermediate_waveform": str(output / "audio.npz"),
        })
        print(json.dumps(rows[-1]), flush=True)
    report = {"research_only": True, "input_run": str(args.run_dir),
              "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
              "watermark_stage_required_before_delivery": True, "runs": rows}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
