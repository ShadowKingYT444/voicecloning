"""Research-only smooth mel-envelope calibration for Chatterbox Nano.

The experiment estimates a voice-specific, time-independent correction in the
80 log-mel bands.  It is deliberately separate from normal synthesis.  The
``prepare`` command reconstructs acoustic features from genuine cached speech
token IDs with the decoder flow only, then stores those features beside the
source recording's normalized mel features.  It does not call the HiFT
vocoder and does not produce an audio sample.

The ``fit`` command is CPU-only NumPy code.  It uses TRAIN rows only, computes
time-averaged relative spectral envelopes (there is no DTW or frame alignment),
takes a per-band median source-minus-reconstruction error, smooths over mel
bands, removes the global mean, and bounds every correction to at most 0.35
natural-log units.  VALID rows are used only for a heldout diagnostic at
strengths 0, 0.5, and 1.0.  The result is a diagnostic hypothesis, not a
quality claim or a default production setting.

The optional ``configure_mel_calibration`` wrapper changes only the
``speech_feat`` argument passed to ``model.s3gen.mel2wav.inference``.  A
missing path or zero strength restores the original bound method and calls it
unchanged.  ``reset_mel_calibration`` is provided for sweeps that reuse one
model object across cases.

Model and audio imports stay inside ``prepare`` or the nonzero wrapper path so
that ``fit`` and the pure tests do not construct a Torch model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
VENDOR_SRC = ROOT / "vendor" / "chatterbox" / "src"
DEFAULT_MODEL_DIR = ROOT / "models" / "chatterbox-nano"
DEFAULT_CACHE = ROOT / "artifacts" / "nano_lab" / "adaptation_asmr_aligned_cache.json"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "nano_lab" / "mel_calibration"
MEL_BANDS = 80
SAMPLE_RATE = 24_000
S3GEN_SIL = 4299
SPEECH_VOCAB = 6561
_CALIBRATION_STATE = "_nano_mel_calibration_state_v1"


def _sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _sha256_tree(root: Path) -> str:
    """Hash checkpoint file names and bytes in stable relative-path order."""

    if root.is_file():
        return _sha256(root)
    digest = hashlib.sha256()
    paths = sorted(path for path in root.rglob("*") if path.is_file())
    for path in paths:
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            while True:
                block = handle.read(1 << 20)
                if not block:
                    break
                digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _save_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    np.savez_compressed(temporary, **{key: np.asarray(value) for key, value in arrays.items()})
    generated = temporary if temporary.exists() else Path(str(temporary) + ".npz")
    os.replace(generated, path)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as values:
        return {name: np.array(values[name], copy=True) for name in values.files}


def _resolve(value: str | os.PathLike[str], *, base: Path = ROOT) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _read_cache(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or not isinstance(payload.get("rows"), list):
        raise ValueError(f"adaptation cache must contain a rows list: {path}")
    if payload.get("format") != "nano_t3_adaptation_cache_v1":
        raise ValueError(f"unsupported adaptation cache format: {payload.get('format')!r}")
    return payload


def _validate_aligned_cache(cache: Mapping[str, Any]) -> None:
    """Require the repaired, inference-compatible cache contract."""

    target_rows = [row for row in cache["rows"] if row.get("split") in {"train", "valid", "test"}]
    from dataset_contract import require_audited_rows, require_inference_conditioning
    require_audited_rows(target_rows)
    require_inference_conditioning(target_rows)
    if not target_rows:
        raise ValueError("aligned cache has no train/valid/test target rows")
    if not any(row.get("split") == "train" for row in target_rows):
        raise ValueError("aligned cache has no train rows")
    if not any(row.get("split") == "valid" for row in target_rows):
        raise ValueError("aligned cache has no valid rows")
    references = cache.get("conditioning_references")
    if not isinstance(references, dict) or not references:
        raise ValueError("aligned cache is missing conditioning_references provenance")
    for row in target_rows:
        for key in ("id", "audio_path", "speech_tokens", "split", "reference_audio_path", "conditioning_provenance"):
            if key not in row:
                raise ValueError(f"aligned cache row {row.get('id')} is missing {key}")
        audit = row.get("transcript_audit") or {}
        if audit.get("status") != "accepted" or not audit.get("method"):
            raise ValueError(f"aligned cache row {row.get('id')} lacks an accepted transcript audit")
        provenance = row.get("conditioning_provenance") or {}
        preprocessing = provenance.get("preprocessing") or {}
        if preprocessing.get("entrypoint") != "ChatterboxTurboTTS.prepare_conditionals":
            raise ValueError(f"row {row.get('id')} does not use upstream reference conditioning")
        if preprocessing.get("norm_loudness") is not True:
            raise ValueError(f"row {row.get('id')} conditioning was not loudness-normalized")
        if preprocessing.get("prompt_tokens_unmodified") is not True or preprocessing.get("speaker_embedding_unmodified") is not True:
            raise ValueError(f"row {row.get('id')} conditioning provenance is not exact")
        if not isinstance(row.get("reference_sha256"), str) or len(row["reference_sha256"]) != 64:
            raise ValueError(f"row {row.get('id')} has no reference SHA-256")


def _select_rows(cache: Mapping[str, Any], speaker_id: str | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    rows = [dict(row) for row in cache["rows"] if row.get("split") in {"train", "valid", "test"}]
    speakers = sorted({str(row.get("speaker_id", "default")) for row in rows})
    if speaker_id is None:
        if len(speakers) != 1:
            raise ValueError(f"cache contains multiple speakers {speakers}; pass --speaker-id")
        selected = speakers[0]
    else:
        selected = str(speaker_id)
        if selected not in speakers:
            raise ValueError(f"unknown speaker {selected!r}; available={speakers}")
    train = [row for row in rows if row.get("split") == "train" and str(row.get("speaker_id", "default")) == selected]
    valid = [row for row in rows if row.get("split") == "valid" and str(row.get("speaker_id", "default")) == selected]
    if not train or not valid:
        raise ValueError(f"speaker {selected!r} needs train and valid rows (train={len(train)}, valid={len(valid)})")
    return train, valid, selected


def _validate_tokens(row: Mapping[str, Any]) -> np.ndarray:
    values = np.asarray(row["speech_tokens"])
    if values.ndim != 1 or values.size < 1:
        raise ValueError(f"row {row.get('id')} speech_tokens must be a non-empty vector")
    if not np.issubdtype(values.dtype, np.integer):
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise ValueError(f"row {row.get('id')} speech_tokens are not integer IDs")
    values = values.astype(np.int64, copy=False)
    if np.any(values < 0) or np.any(values >= SPEECH_VOCAB):
        raise ValueError(f"row {row.get('id')} has out-of-range/special speech token IDs")
    return values


def _load_audio_24k(path: Path) -> np.ndarray:
    # Imported only in prepare, so fit remains a pure NumPy operation.
    import librosa

    audio, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1 or audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError(f"source audio is empty or non-finite: {path}")
    return audio


def _vendor_mel(audio: np.ndarray) -> np.ndarray:
    """Call the exact vendor S3Gen mel extractor and return (80,T)."""

    if str(VENDOR_SRC) not in sys.path:
        sys.path.insert(0, str(VENDOR_SRC))
    import torch
    from chatterbox.models.s3gen.utils.mel import mel_spectrogram

    with torch.inference_mode():
        mel = mel_spectrogram(torch.from_numpy(audio)[None, :])
    array = mel.detach().cpu().numpy().astype(np.float32, copy=False)
    if array.ndim != 3 or array.shape[0] != 1 or array.shape[1] != MEL_BANDS:
        raise RuntimeError(f"vendor mel extractor returned {array.shape}, expected (1,80,T)")
    return np.array(array[0], dtype=np.float32, copy=True)


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _flow_mel(model: Any, tokens: np.ndarray, *, steps: int) -> np.ndarray:
    """Run only S3Gen's token-to-mel flow.  The HiFT vocoder is never called."""

    import torch

    device = getattr(model, "device", None)
    if device is None:
        device = next(model.t3.parameters()).device
    speech = torch.from_numpy(tokens[None, :]).to(device=device, dtype=torch.long)
    ref_dict = model.conds.gen
    with torch.inference_mode():
        mel = model.s3gen.flow_inference(
            speech_tokens=speech,
            ref_dict=ref_dict,
            n_cfm_timesteps=int(steps),
            finalize=True,
        )
    if not torch.is_tensor(mel):
        raise RuntimeError("S3Gen flow_inference did not return a tensor")
    array = mel.detach().cpu().numpy().astype(np.float32, copy=False)
    if array.ndim != 3 or array.shape[0] != 1 or array.shape[1] != MEL_BANDS:
        raise RuntimeError(f"S3Gen flow returned {array.shape}, expected (1,80,T)")
    return np.array(array[0], dtype=np.float32, copy=True)


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    """Prepare genuine source/reconstructed mel arrays under a bounded process."""

    if int(args.steps) != 2:
        raise ValueError("the calibration experiment is fixed to exactly two mean-flow steps")
    cache_path = _resolve(args.cache)
    cache = _read_cache(cache_path)
    _validate_aligned_cache(cache)
    train_rows, valid_rows, selected_speaker = _select_rows(cache, args.speaker_id)
    all_rows = train_rows + valid_rows
    if args.max_rows is not None:
        if args.max_rows < 2:
            raise ValueError("--max-rows must be at least 2")
        all_rows = all_rows[: int(args.max_rows)]
        if not any(row.get("split") == "train" for row in all_rows) or not any(row.get("split") == "valid" for row in all_rows):
            raise ValueError("--max-rows selection must retain both train and valid rows")
        train_rows = [row for row in all_rows if row.get("split") == "train"]
        valid_rows = [row for row in all_rows if row.get("split") == "valid"]
    conditionals_path = _resolve(args.conditionals_path)
    if not conditionals_path.exists():
        raise FileNotFoundError(conditionals_path)
    model_dir = _resolve(args.model_dir)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # All model/runtime imports are intentionally inside this model-backed
    # command.  conditionals_path makes the optimized loader skip reference
    # encoders and use the already audited decoder reference.
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    from runtime import NanoEngine

    engine = NanoEngine.from_pretrained(
        model_dir,
        device=args.device,
        dtype="fp32",
        optimized=True,
        decoder="meanflow",
        conditionals_path=conditionals_path,
        cpu_threads=2,
    )
    if not engine.load_report.get("conditionals_path"):
        raise RuntimeError("optimized loader did not record a saved conditionals path")
    model = engine.model
    if model.conds is None or not getattr(model.conds, "gen", None):
        raise RuntimeError("saved conditionals did not provide a decoder reference")
    cached_speaker = model.conds.t3.speaker_emb.detach().cpu().float().numpy().reshape(-1)
    cached_prompt = model.conds.t3.cond_prompt_speech_tokens.detach().cpu().reshape(-1).tolist()
    for row in all_rows:
        if not np.array_equal(cached_speaker, np.asarray(row["speaker_emb"], dtype=np.float32)) or cached_prompt != row["cond_prompt_speech_tokens"]:
            raise ValueError(f"saved inference conditioning differs from training row {row['id']}")
    model_checkpoint_sha = _sha256_tree(model_dir)
    conditionals_sha = _sha256(conditionals_path)
    started = time.perf_counter()
    records: list[dict[str, Any]] = []
    for ordinal, row in enumerate(all_rows):
        row_id = str(row["id"])
        source_path = _resolve(str(row["audio_path"]))
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        raw_audio = _load_audio_24k(source_path)
        # This is the same per-clip loudness operation used by the upstream
        # reference path before its 24 kHz mel extraction.
        normalized_audio = np.asarray(model.norm_loudness(raw_audio, SAMPLE_RATE), dtype=np.float32)
        if normalized_audio.shape != raw_audio.shape or not np.isfinite(normalized_audio).all():
            raise RuntimeError(f"norm_loudness returned invalid audio for {row_id}")
        source_mel = _vendor_mel(normalized_audio)
        source_tokens = _validate_tokens(row)
        output_tokens = np.concatenate((source_tokens, np.full((3,), S3GEN_SIL, dtype=np.int64)))
        row_seed = int(args.seed) + ordinal * 1009
        _seed_all(row_seed)
        reconstructed_mel = _flow_mel(model, output_tokens, steps=int(args.steps))
        arrays_path = output_dir / f"{row_id}.mel.npz"
        _save_npz(arrays_path, {"source_mel": source_mel, "reconstructed_mel": reconstructed_mel})
        reference_provenance = row.get("conditioning_provenance") or {}
        records.append(
            {
                "id": row_id,
                "split": row.get("split"),
                "speaker_id": row.get("speaker_id"),
                "audio_path": str(source_path),
                "source_sha256": _sha256(source_path),
                "source_text": row.get("source_text"),
                "speech_token_count": int(source_tokens.size),
                "output_token_count": int(output_tokens.size),
                "trailing_silence": {"value": S3GEN_SIL, "count": 3},
                "mel_path": str(arrays_path),
                "source_mel_shape": list(source_mel.shape),
                "reconstructed_mel_shape": list(reconstructed_mel.shape),
                "seed": row_seed,
                "cfm_steps": int(args.steps),
                "conditioning_sha256": conditionals_sha,
                "conditioning_reference_path": row.get("reference_audio_path"),
                "conditioning_reference_sha256": row.get("reference_sha256"),
                "conditioning_reference_provenance": reference_provenance,
                "source_preprocessing": {
                    "loader": "librosa.load(sr=24000, mono=True)",
                    "normalizer": "ChatterboxTurboTTS.norm_loudness(target_lufs=-27)",
                    "mel_extractor": "vendor/chatterbox/src/chatterbox/models/s3gen/utils/mel.py::mel_spectrogram",
                },
            }
        )
        print(json.dumps({"event": "prepared", "id": row_id, "split": row.get("split"), "source_frames": int(source_mel.shape[1]), "reconstructed_frames": int(reconstructed_mel.shape[1])}), flush=True)

    manifest = {
        "format": "nano_mel_calibration_prepare_v1",
        "diagnostic_only": True,
        "synthetic_audio": False,
        "cache": str(cache_path),
        "cache_sha256": _sha256(cache_path),
        "model_dir": str(model_dir),
        "model_checkpoint_sha256": model_checkpoint_sha,
        "conditionals_path": str(conditionals_path),
        "conditioning_sha256": conditionals_sha,
        "conditioning_reference_provenance": cache.get("conditioning_references"),
        "speaker_id": selected_speaker,
        "source_sample_rate": SAMPLE_RATE,
        "mel_bands": MEL_BANDS,
        "mel_layout": "bands,time; raw log-mel values",
        "flow_path": "model.s3gen.flow_inference only; HiFT vocoder not called",
        "cfm_steps": int(args.steps),
        "seed": int(args.seed),
        "source_loudness_normalization": "upstream norm_loudness target_lufs=-27",
        "rows": records,
        "elapsed_seconds": time.perf_counter() - started,
        "created_unix": time.time(),
        "command": list(sys.argv),
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def _mel_2d(value: Any, *, name: str = "mel") -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2 or array.shape[0] != MEL_BANDS or array.shape[1] < 1:
        raise ValueError(f"{name} must have shape (80,T) or (1,80,T), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def relative_spectral_envelope(mel: Any) -> np.ndarray:
    """Return per-band time mean minus the global mean, without alignment."""

    array = _mel_2d(mel)
    band_mean = np.mean(array, axis=1)
    return band_mean - np.mean(band_mean)


def _smooth_bands(values: np.ndarray, sigma: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size != MEL_BANDS:
        raise ValueError(f"correction must have 80 bands, got {values.size}")
    if sigma <= 0:
        return values.copy()
    radius = max(1, int(math.ceil(3.0 * float(sigma))))
    positions = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (positions / float(sigma)) ** 2)
    kernel /= np.sum(kernel)
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _bound_zero_mean(values: np.ndarray, max_delta: float) -> np.ndarray:
    if not np.isfinite(max_delta) or max_delta < 0 or max_delta > 0.35:
        raise ValueError("max_delta must be finite and in [0, 0.35]")
    result = np.asarray(values, dtype=np.float64).reshape(-1)
    result = result - np.mean(result)
    if max_delta == 0:
        return np.zeros_like(result)
    peak = float(np.max(np.abs(result))) if result.size else 0.0
    if peak > max_delta:
        result = result * (max_delta / peak)
    # Uniform scaling keeps the smoothed shape and preserves the exact
    # zero-mean constraint.  Per-band clipping followed by recentering can
    # distort neighbouring bands and can require a second bound projection.
    return result - np.mean(result)


def fit_delta(
    records: Sequence[Mapping[str, Any]],
    *,
    sigma: float = 2.0,
    max_delta: float = 0.35,
) -> np.ndarray:
    """Fit a bounded, smooth, zero-mean delta from TRAIN records only."""

    train = [row for row in records if row.get("split") == "train"]
    if not train:
        raise ValueError("fit_delta requires at least one train record")
    differences = []
    for row in train:
        source = _mel_2d(row["source_mel"], name=f"{row.get('id')} source_mel")
        reconstructed = _mel_2d(row["reconstructed_mel"], name=f"{row.get('id')} reconstructed_mel")
        differences.append(relative_spectral_envelope(source) - relative_spectral_envelope(reconstructed))
    median_difference = np.median(np.stack(differences, axis=0), axis=0)
    smoothed = _smooth_bands(median_difference, sigma)
    return _bound_zero_mean(smoothed, max_delta)


def envelope_mse(source_mel: Any, reconstructed_mel: Any, delta: Any, *, strength: float) -> float:
    source = relative_spectral_envelope(source_mel)
    reconstructed = _mel_2d(reconstructed_mel, name="reconstructed_mel")
    correction = np.asarray(delta, dtype=np.float64).reshape(-1)
    if correction.size != MEL_BANDS:
        raise ValueError("delta must contain 80 values")
    adjusted = reconstructed + float(strength) * correction[:, None]
    estimate = relative_spectral_envelope(adjusted)
    return float(np.mean(np.square(source - estimate)))


def _load_prepare_records(manifest: Mapping[str, Any], *, base: Path) -> list[dict[str, Any]]:
    records = []
    for row in manifest.get("rows", []):
        arrays = _load_npz(_resolve(row["mel_path"], base=base))
        if "source_mel" not in arrays or "reconstructed_mel" not in arrays:
            raise ValueError(f"prepared row {row.get('id')} lacks raw source/reconstructed mel arrays")
        record = dict(row)
        record["source_mel"] = arrays["source_mel"]
        record["reconstructed_mel"] = arrays["reconstructed_mel"]
        records.append(record)
    if not records:
        raise ValueError("prepare manifest has no rows")
    return records


def fit(args: argparse.Namespace) -> dict[str, Any]:
    prepare_dir = _resolve(args.prepare_dir)
    manifest_path = prepare_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != "nano_mel_calibration_prepare_v1":
        raise ValueError("fit expects a mel_calibration prepare manifest")
    records = _load_prepare_records(manifest, base=ROOT)
    train = [row for row in records if row.get("split") == "train"]
    valid = [row for row in records if row.get("split") == "valid"]
    if not train or not valid:
        raise ValueError(f"fit requires train and valid rows (train={len(train)}, valid={len(valid)})")
    train_ids = {str(row["id"]) for row in train}
    valid_ids = {str(row["id"]) for row in valid}
    if train_ids & valid_ids:
        raise ValueError("train and valid IDs overlap")
    delta = fit_delta(records, sigma=float(args.sigma), max_delta=float(args.max_delta))
    if delta.shape != (MEL_BANDS,) or not np.isfinite(delta).all() or np.max(np.abs(delta)) > float(args.max_delta) + 1e-10:
        raise RuntimeError("fitted delta violated shape/finite/bound constraints")
    strengths = [0.0, 0.5, 1.0]
    heldout_rows = []
    for strength in strengths:
        errors = [envelope_mse(row["source_mel"], row["reconstructed_mel"], delta, strength=strength) for row in valid]
        heldout_rows.append(
            {
                "strength": strength,
                "mean_source_envelope_mse": float(np.mean(errors)),
                "per_row": {str(row["id"]): float(error) for row, error in zip(valid, errors)},
            }
        )
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    delta_path = output_dir / "delta.npy"
    np.save(delta_path, delta.astype(np.float32))
    report = {
        "format": "nano_mel_calibration_fit_v1",
        "diagnostic_only": True,
        "claim_status": "diagnostic_only_no_realism_claim",
        "prepare_manifest": str(manifest_path.resolve()),
        "prepare_manifest_sha256": _sha256(manifest_path),
        "model_checkpoint_sha256": manifest.get("model_checkpoint_sha256"),
        "conditioning_sha256": manifest.get("conditioning_sha256"),
        "conditioning_reference_provenance": manifest.get("conditioning_reference_provenance"),
        "speaker_id": manifest.get("speaker_id"),
        "fit_formula": "median_train(relative(source_mel) - relative(reconstructed_mel)), Gaussian band smoothing, zero mean, bounded ±max_delta",
        "time_alignment": "none; source and reconstructed frames are independently time-averaged",
        "fit_rows": [str(row["id"]) for row in train],
        "heldout_rows": [str(row["id"]) for row in valid],
        "sigma_bands": float(args.sigma),
        "max_delta": float(args.max_delta),
        "delta_path": str(delta_path.resolve()),
        "delta": [float(value) for value in delta.tolist()],
        "delta_mean": float(np.mean(delta)),
        "delta_max_abs": float(np.max(np.abs(delta))),
        "heldout_source_envelope_mse": heldout_rows,
        "no_explicit_pitch_or_time_transform": True,
        "vocoder_pitch_effects": "not measured; changing mel features can change the predicted waveform",
        "created_unix": time.time(),
    }
    _write_json(output_dir / "report.json", report)
    return report


def _load_delta(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        delta = np.load(path, allow_pickle=False)
    elif path.suffix.lower() in {".json", ".jsn"}:
        payload = json.loads(path.read_text())
        delta = payload.get("delta") if isinstance(payload, Mapping) else payload
    else:
        raise ValueError("calibration path must be .npy or .json")
    delta = np.asarray(delta, dtype=np.float64).reshape(-1)
    if delta.size != MEL_BANDS or not np.isfinite(delta).all():
        raise ValueError("calibration delta must contain 80 finite values")
    if np.max(np.abs(delta)) > 0.35 + 1e-10:
        raise ValueError("calibration delta exceeds the ±0.35 natural-log safety bound")
    return delta


def reset_mel_calibration(model: Any) -> Any:
    """Restore the original HiFT inference method after a sweep case."""

    mel2wav = getattr(getattr(model, "s3gen", None), "mel2wav", None)
    if mel2wav is None:
        return model
    state = getattr(mel2wav, _CALIBRATION_STATE, None)
    if state is not None:
        mel2wav.inference = state["original"]
        delattr(mel2wav, _CALIBRATION_STATE)
    return model


def _apply_delta_to_features(speech_feat: Any, delta: np.ndarray, strength: float) -> Any:
    if hasattr(speech_feat, "device") and hasattr(speech_feat, "dtype") and speech_feat.__class__.__module__.split(".")[0] == "torch":
        import torch

        tensor_delta = torch.as_tensor(delta, device=speech_feat.device, dtype=speech_feat.dtype).view(1, MEL_BANDS, 1)
        return speech_feat + float(strength) * tensor_delta
    array = np.asarray(speech_feat)
    if array.ndim not in {2, 3} or array.shape[-2] != MEL_BANDS:
        raise ValueError(f"speech_feat must have shape (80,T) or (B,80,T), got {array.shape}")
    reshape = (MEL_BANDS, 1) if array.ndim == 2 else (1, MEL_BANDS, 1)
    return array + np.asarray(delta, dtype=array.dtype).reshape(reshape) * np.asarray(strength, dtype=array.dtype)


def configure_mel_calibration(model: Any, path: str | os.PathLike[str] | None = None, strength: float = 1.0) -> Any:
    """Install or clear a mel correction around only HiFT's ``speech_feat``."""

    strength = float(strength)
    if not np.isfinite(strength) or strength < 0.0 or strength > 1.0:
        raise ValueError("mel calibration strength must be in [0, 1]")
    reset_mel_calibration(model)
    if path is None or strength == 0.0:
        return model
    delta = _load_delta(Path(path).expanduser().resolve())
    mel2wav = getattr(getattr(model, "s3gen", None), "mel2wav", None)
    if mel2wav is None or not hasattr(mel2wav, "inference"):
        raise AttributeError("model.s3gen.mel2wav.inference is required for mel calibration")
    original = mel2wav.inference

    def calibrated_inference(*args: Any, **kwargs: Any) -> Any:
        if "speech_feat" in kwargs:
            kwargs = dict(kwargs)
            kwargs["speech_feat"] = _apply_delta_to_features(kwargs["speech_feat"], delta, strength)
            return original(*args, **kwargs)
        if args:
            mutable = list(args)
            mutable[0] = _apply_delta_to_features(mutable[0], delta, strength)
            return original(*mutable, **kwargs)
        return original(*args, **kwargs)

    setattr(mel2wav, _CALIBRATION_STATE, {"original": original, "delta": delta.copy(), "strength": strength})
    mel2wav.inference = calibrated_inference
    return model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="save source and flow-reconstructed log-mel arrays")
    prep.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    prep.add_argument("--conditionals-path", type=Path, required=True)
    prep.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    prep.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    prep.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    prep.add_argument("--speaker-id")
    prep.add_argument("--seed", type=int, default=31)
    prep.add_argument("--steps", type=int, default=2)
    prep.add_argument("--max-rows", type=int)
    fit_parser = sub.add_parser("fit", help="fit a bounded envelope delta and heldout diagnostic")
    fit_parser.add_argument("--prepare-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    fit_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    fit_parser.add_argument("--sigma", type=float, default=2.0)
    fit_parser.add_argument("--max-delta", type=float, default=0.35)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        prepare(args)
    elif args.command == "fit":
        report = fit(args)
        print(json.dumps({"event": "fit", "delta_path": report["delta_path"], "heldout": report["heldout_source_envelope_mse"]}, indent=2))
    else:
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
