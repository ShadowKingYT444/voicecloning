"""Measure diagnostic prosody and harmonicity features for an audio manifest.

This utility is descriptive only.  It does not produce a quality score or an
accept/reject decision.  Pitch uses Praat autocorrelation with a 60--500 Hz
search range.  This range and the reported frame coverage are explicit because
whispered, creaky, breathy, and noisy ASMR can make autocorrelation pitch
unreliable.  Run it in the audio venv when a real report is authorized; this
module performs no audio work on import.

Manifest rows require ``id``, ``path``, ``text``, and ``source_group``.  A JSON
list is accepted, as are objects containing ``inputs``, ``rows``, or ``runs``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import resource
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TIME_STEP_S = 0.01
DEFAULT_PITCH_FLOOR_HZ = 60.0
DEFAULT_PITCH_CEILING_HZ = 500.0
PRAAT_HARMONICITY_UNDEFINED_DB = -200.0
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?")


def _resolve(value: str | os.PathLike[str], *, base: Path = ROOT) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1 << 20)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _rss_mib() -> float | None:
    """Return process peak RSS in MiB where the platform reports it."""

    try:
        # Linux reports KiB; macOS reports bytes.
        value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return round(value / (1024.0 if os.name == "posix" and Path("/proc/self/status").exists() else 1024.0**2), 3)
    except (AttributeError, TypeError, ValueError):
        return None


def _load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        values = payload
    elif isinstance(payload, Mapping):
        values = payload.get("inputs") or payload.get("rows") or payload.get("runs") or []
    else:
        raise ValueError("manifest must be a JSON list or object containing inputs/rows/runs")
    rows: list[dict[str, Any]] = []
    for index, value in enumerate(values):
        if not isinstance(value, Mapping):
            raise ValueError(f"manifest row {index} is not an object")
        missing = [key for key in ("id", "path", "text", "source_group") if key not in value]
        if missing:
            raise ValueError(f"manifest row {index} missing required fields: {', '.join(missing)}")
        rows.append(dict(value))
    return rows


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _distribution(values: Iterable[Any]) -> dict[str, Any]:
    finite = np.asarray([number for value in values if (number := _finite(value)) is not None], dtype=np.float64)
    if finite.size == 0:
        return {"count": 0, "missing_count": 0, "min": None, "p10": None, "p25": None, "median": None, "p75": None, "p90": None, "max": None, "mean": None, "std": None}
    return {
        "count": int(finite.size),
        "missing_count": 0,
        "min": float(np.min(finite)),
        "p10": float(np.percentile(finite, 10)),
        "p25": float(np.percentile(finite, 25)),
        "median": float(np.median(finite)),
        "p75": float(np.percentile(finite, 75)),
        "p90": float(np.percentile(finite, 90)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
    }


def _run_durations(mask: np.ndarray, step_s: float) -> list[float]:
    values = np.asarray(mask, dtype=bool).reshape(-1)
    if values.size == 0:
        return []
    starts = np.flatnonzero(values & np.concatenate(([True], ~values[:-1])))
    ends = np.flatnonzero(values & np.concatenate((~values[1:], [True])))
    return [float((end - start + 1) * step_s) for start, end in zip(starts, ends)]


def _pitch_stats(frequency: np.ndarray, floor_hz: float, ceiling_hz: float) -> dict[str, Any]:
    values = np.asarray(frequency, dtype=np.float64).reshape(-1)
    finite = np.isfinite(values)
    voiced = finite & (values >= floor_hz) & (values <= ceiling_hz) & (values > 0.0)
    voiced_values = values[voiced]
    result: dict[str, Any] = {
        "frame_count": int(values.size),
        "finite_frame_count": int(np.count_nonzero(finite)),
        "voiced_frame_count": int(np.count_nonzero(voiced)),
        "finite_frame_fraction": float(np.mean(finite)) if values.size else None,
        "voiced_frame_fraction": float(np.mean(voiced)) if values.size else None,
        "f0_hz": _distribution(voiced_values),
    }
    if voiced_values.size >= 2:
        adjacent = voiced[:-1] & voiced[1:]
        first = values[:-1][adjacent]
        second = values[1:][adjacent]
        semitone_delta = 12.0 * np.log2(np.maximum(second, 1e-9) / np.maximum(first, 1e-9))
        result["adjacent_voiced_frame_count"] = int(semitone_delta.size)
        result["semitone_delta"] = _distribution(semitone_delta)
        result["large_jump_fraction_abs_gt_4st"] = float(np.mean(np.abs(semitone_delta) > 4.0)) if semitone_delta.size else None
    else:
        result["adjacent_voiced_frame_count"] = 0
        result["semitone_delta"] = _distribution([])
        result["large_jump_fraction_abs_gt_4st"] = None
    return result


def _harmonicity_stats(values: np.ndarray) -> dict[str, Any]:
    """Separate Praat's finite -200 dB undefined sentinel from HNR frames."""

    array = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = np.isfinite(array)
    defined = finite & (array > PRAAT_HARMONICITY_UNDEFINED_DB)
    return {
        "frame_count": int(array.size),
        "raw_finite_frame_count": int(np.count_nonzero(finite)),
        "raw_finite_frame_fraction": float(np.mean(finite)) if array.size else None,
        "undefined_sentinel_db": PRAAT_HARMONICITY_UNDEFINED_DB,
        "undefined_sentinel_frame_count": int(np.count_nonzero(finite & ~defined)),
        "defined_frame_count": int(np.count_nonzero(defined)),
        "defined_frame_fraction": float(np.mean(defined)) if array.size else None,
        "harmonicity_db": _distribution(array[defined]),
    }


def _energy_stats(samples: np.ndarray, sample_rate: int, *, step_s: float = DEFAULT_TIME_STEP_S) -> dict[str, Any]:
    frame_length = max(1, int(round(sample_rate * 0.025)))
    hop = max(1, int(round(sample_rate * step_s)))
    if samples.size == 0:
        return {"frame_count": 0, "active_frame_fraction": None, "pause_frame_fraction": None, "pause_count": 0, "pause_duration_s": _distribution([]), "active_duration_s": _distribution([]), "threshold_dbfs": None, "threshold_method": "adaptive_rms_dbfs"}
    frame_count = max(1, int(math.ceil(max(0, samples.size - frame_length) / hop)) + 1)
    padded = np.pad(samples, (0, max(0, (frame_count - 1) * hop + frame_length - samples.size)))
    frames = np.lib.stride_tricks.sliding_window_view(padded, frame_length)[::hop][:frame_count]
    rms = np.sqrt(np.mean(np.square(frames), axis=1))
    dbfs = 20.0 * np.log10(np.maximum(rms, 1e-12))
    q20, q95 = np.percentile(dbfs, [20, 95])
    threshold = float(np.clip(max(q20 + 6.0, q95 - 35.0), -80.0, -12.0))
    active = dbfs > threshold
    try:
        from scipy import ndimage

        active = ndimage.binary_closing(active, structure=np.ones(3, dtype=bool))
        active = ndimage.binary_opening(active, structure=np.ones(2, dtype=bool))
    except ImportError:
        pass
    pause_runs = _run_durations(~active, step_s)
    active_runs = _run_durations(active, step_s)
    return {
        "frame_count": int(dbfs.size),
        "active_frame_fraction": float(np.mean(active)),
        "pause_frame_fraction": float(np.mean(~active)),
        "pause_count": int(len(pause_runs)),
        "pause_duration_s": _distribution(pause_runs),
        "active_duration_s": _distribution(active_runs),
        "threshold_dbfs": threshold,
        "threshold_method": "adaptive_rms_dbfs",
        "frame_step_s": step_s,
    }


def analyze_audio(path: Path, text: str, *, pitch_floor_hz: float = DEFAULT_PITCH_FLOOR_HZ, pitch_ceiling_hz: float = DEFAULT_PITCH_CEILING_HZ, time_step_s: float = DEFAULT_TIME_STEP_S) -> dict[str, Any]:
    """Analyze one file.  Imports audio dependencies only when called."""

    try:
        import parselmouth
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("prosody_audit requires parselmouth and soundfile in the audio venv") from exc
    path = path.resolve()
    samples, sample_rate = sf.read(path, always_2d=True, dtype="float64")
    samples = np.asarray(samples, dtype=np.float64)
    mono = np.nan_to_num(samples.mean(axis=1), copy=False)
    sound = parselmouth.Sound(mono, sampling_frequency=float(sample_rate))
    pitch = sound.to_pitch(time_step=time_step_s, pitch_floor=pitch_floor_hz, pitch_ceiling=pitch_ceiling_hz)
    frequency = np.asarray(pitch.selected_array["frequency"], dtype=np.float64)
    harmonicity = sound.to_harmonicity_cc(time_step=time_step_s, minimum_pitch=pitch_floor_hz, silence_threshold=0.1, periods_per_window=4.5)
    harmonicity_values = np.asarray(harmonicity.values, dtype=np.float64).reshape(-1)
    harmonicity_result = _harmonicity_stats(harmonicity_values)
    pitch_result = _pitch_stats(frequency, pitch_floor_hz, pitch_ceiling_hz)
    words = _WORD_RE.findall(str(text))
    energy = _energy_stats(mono, int(sample_rate), step_s=time_step_s)
    duration_s = float(len(mono) / sample_rate) if sample_rate else None
    active_duration_s = (float(energy["active_frame_fraction"]) * duration_s) if duration_s is not None and energy["active_frame_fraction"] is not None else 0.0
    return {
        "path": str(path),
        "duration_s": duration_s,
        "sample_rate_hz": int(sample_rate),
        "channels": int(samples.shape[1]),
        "source_sha256": _sha256(path),
        "text_word_count": len(words),
        "pitch": {
            "method": "Praat autocorrelation via parselmouth",
            "floor_hz": pitch_floor_hz,
            "ceiling_hz": pitch_ceiling_hz,
            "time_step_s": time_step_s,
            **pitch_result,
            "reliability_warning": "Whispered, creaky, breathy, and noisy ASMR can produce unreliable autocorrelation F0; frame coverage is diagnostic only.",
        },
        "harmonicity": {
            "method": "Praat cross-correlation harmonicity",
            "minimum_pitch_hz": pitch_floor_hz,
            "time_step_s": time_step_s,
            **harmonicity_result,
            "reliability_warning": "Harmonicity coverage is a diagnostic of periodic structure, not a realism or quality score.",
            "sentinel_warning": "Praat reports -200 dB for undefined or silent harmonicity frames; those finite sentinel values are excluded from the HNR distribution and counted separately.",
        },
        "pauses_and_duration": energy,
        "speech_rate_proxy": {
            "method": "word count divided by adaptive RMS active duration",
            "words_per_audio_second": len(words) / (len(mono) / sample_rate) if sample_rate and len(mono) else None,
            "active_duration_mean_s": active_duration_s,
            "words_per_active_second": len(words) / active_duration_s if active_duration_s > 0 else None,
        },
    }


def _group_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = {
        "duration_s": [row.get("duration_s") for row in rows],
        "voiced_frame_fraction": [((row.get("pitch") or {}).get("voiced_frame_fraction")) for row in rows],
        "harmonicity_defined_frame_fraction": [((row.get("harmonicity") or {}).get("defined_frame_fraction")) for row in rows],
        "harmonicity_raw_finite_frame_fraction": [((row.get("harmonicity") or {}).get("raw_finite_frame_fraction")) for row in rows],
        "pause_frame_fraction": [((row.get("pauses_and_duration") or {}).get("pause_frame_fraction")) for row in rows],
        "words_per_audio_second": [((row.get("speech_rate_proxy") or {}).get("words_per_audio_second")) for row in rows],
    }
    return {"count": len(rows), "distributions": {name: _distribution(values) for name, values in metrics.items()}, "ids": [row.get("id") for row in rows]}


def audit_manifest(manifest_path: Path, *, pitch_floor_hz: float = DEFAULT_PITCH_FLOOR_HZ, pitch_ceiling_hz: float = DEFAULT_PITCH_CEILING_HZ, time_step_s: float = DEFAULT_TIME_STEP_S) -> dict[str, Any]:
    """Analyze manifest files and return a descriptive, score-free report."""

    started = time.perf_counter()
    rows = _load_rows(manifest_path)
    analyzed: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for row in rows:
        path_value = row["path"]
        path = _resolve(str(path_value), base=manifest_path.parent)
        try:
            result = analyze_audio(path, str(row.get("text", "")), pitch_floor_hz=pitch_floor_hz, pitch_ceiling_hz=pitch_ceiling_hz, time_step_s=time_step_s)
            result.update({"id": row["id"], "source_group": row["source_group"], "text": row.get("text", "")})
            analyzed.append(result)
        except Exception as exc:  # report one bad row without hiding its identity
            errors.append({"id": row.get("id"), "path": str(path), "error": f"{type(exc).__name__}: {exc}"})
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in analyzed:
        groups.setdefault(str(row["source_group"]), []).append(row)
    return {
        "schema_version": 1,
        "kind": "prosody_diagnostic",
        "score_free": True,
        "acceptance_score": None,
        "interpretation": "Descriptive acoustic diagnostics only; they do not establish realism, identity, or quality.",
        "pitch": {"backend": "Praat autocorrelation via parselmouth", "floor_hz": pitch_floor_hz, "ceiling_hz": pitch_ceiling_hz, "time_step_s": time_step_s, "reliability_warning": "Autocorrelation F0 can be unreliable for whispered, creaky, breathy, or noisy speech."},
        "inputs": str(manifest_path.resolve()),
        "input_count": len(rows),
        "analyzed_count": len(analyzed),
        "error_count": len(errors),
        "errors": errors,
        "rows": analyzed,
        "groups": {name: _group_summary(group_rows) for name, group_rows in sorted(groups.items())},
        "elapsed_s": round(time.perf_counter() - started, 6),
        "peak_rss_mib": _rss_mib(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pitch-floor-hz", type=float, default=DEFAULT_PITCH_FLOOR_HZ)
    parser.add_argument("--pitch-ceiling-hz", type=float, default=DEFAULT_PITCH_CEILING_HZ)
    parser.add_argument("--time-step-s", type=float, default=DEFAULT_TIME_STEP_S)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    report = audit_manifest(args.manifest, pitch_floor_hz=args.pitch_floor_hz, pitch_ceiling_hz=args.pitch_ceiling_hz, time_step_s=args.time_step_s)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(f".{args.out.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, args.out)
    print(json.dumps({"out": str(args.out.resolve()), "analyzed_count": report["analyzed_count"], "error_count": report["error_count"], "peak_rss_mib": report["peak_rss_mib"]}, sort_keys=True), flush=True)
    return 0 if not report["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
