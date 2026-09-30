"""Serial diagnostics for Nano's acoustic codec and HiFT vocoder.

The utility isolates three measurable paths without changing the production
pipeline:

``prepare``
    Read the adaptation cache and export two genuine target speech-token
    sequences with the normal three ``S3GEN_SIL`` IDs appended.  These arrays
    describe source recordings.  They are never labelled as TTS output.

``source-mel``
    Load one source recording, resample it exactly as Chatterbox's local
    ``prepare_conditionals`` path, and call the local S3Gen mel extractor.
    No checkpoint or full S3Gen model is constructed.

``vocoder-reference``
    In a fresh process, load only the staged HiFT vocoder and run either a
    source mel or a saved ONNX-pipeline mel with explicit phase and sine noise.
    Save the native waveform and source/F0 diagnostics.

``vocoder-check``
    In another process, run the staged ORT vocoder with the exact saved mel and
    noise arrays, then compare its waveform with the native reference.

The native and ORT processes are deliberately separate.  This keeps Torch
weights and the ORT session from coexisting under the desktop's 1280 MiB
small-job cap.  The script performs no model work when imported.  Use
``bounded_job.py --small-job`` for every stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
VENDOR_SRC = ROOT / "vendor" / "chatterbox" / "src"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

DEFAULT_CACHE = ROOT / "artifacts" / "nano_lab" / "adaptation_asmr_cache.json"
DEFAULT_DIAGNOSTICS = ROOT / "artifacts" / "nano_lab" / "codec_diagnostics"
DEFAULT_MODEL_DIR = ROOT / "artifacts" / "nano_lab" / "onnx_staged"
DEFAULT_CHECKPOINT_DIR = ROOT / "models" / "chatterbox-nano"
SAMPLE_RATE = 24_000
SPEECH_VOCAB = 6561
S3GEN_SIL = 4299


def _ensure_vendor_path() -> None:
    if str(VENDOR_SRC) not in sys.path:
        sys.path.insert(0, str(VENDOR_SRC))


def _sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _save_npy(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npy")
    np.save(temporary, np.asarray(value))
    os.replace(temporary, path)


def _load_npy(path: Path, *, dtype: np.dtype | None = None) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    value = np.load(path, allow_pickle=False)
    return np.asarray(value, dtype=dtype) if dtype is not None else np.asarray(value)


def _save_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    np.savez(temporary, **{key: np.asarray(value) for key, value in arrays.items()})
    generated = temporary if temporary.exists() else Path(str(temporary) + ".npz")
    os.replace(generated, path)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as values:
        return {name: np.array(values[name], copy=True) for name in values.files}


def _rss_bytes() -> int:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, OSError, ValueError):
        pass
    return 0


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value * 1024 if sys.platform != "darwin" else value


def _resource_report() -> dict[str, int]:
    return {"rss_bytes": _rss_bytes(), "peak_rss_bytes": _peak_rss_bytes()}


def _resolve(value: str | os.PathLike[str], *, base: Path = ROOT) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _cache_rows(cache_path: Path) -> list[dict[str, Any]]:
    data = json.loads(cache_path.read_text())
    rows = data.get("rows") if isinstance(data, Mapping) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"cache must contain a non-empty rows list: {cache_path}")
    result = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"cache row {index} is not an object")
        required = [key for key in ("id", "audio_path", "speech_tokens") if key not in row]
        if required:
            raise ValueError(f"cache row {index} is missing {required}")
        result.append(dict(row))
    return result


def _validate_source_tokens(value: Any, *, row_id: str) -> np.ndarray:
    tokens = np.asarray(value)
    if tokens.ndim != 1 or tokens.size < 1:
        raise ValueError(f"cache row {row_id} has invalid speech_tokens shape {tokens.shape}")
    if not np.issubdtype(tokens.dtype, np.integer):
        if not np.isfinite(tokens).all() or not np.equal(tokens, np.floor(tokens)).all():
            raise ValueError(f"cache row {row_id} speech_tokens are not integer IDs")
    tokens = tokens.astype(np.int64, copy=False)
    if np.any(tokens < 0) or np.any(tokens >= SPEECH_VOCAB):
        raise ValueError(f"cache row {row_id} contains a special/out-of-range speech token")
    return tokens


def _row_by_id(rows: Sequence[Mapping[str, Any]], row_id: str) -> dict[str, Any]:
    for row in rows:
        if str(row.get("id")) == str(row_id):
            return dict(row)
    raise KeyError(f"cache row not found: {row_id}")


def _short_rows(rows: Sequence[Mapping[str, Any]], count: int) -> list[dict[str, Any]]:
    candidates = [row for row in rows if str(row.get("split", "train")) in {"train", "valid", "test"}]
    ordered = sorted(candidates, key=lambda row: (len(row.get("speech_tokens", [])), str(row.get("id"))))
    return [dict(row) for row in ordered[:count]]


def _source_metadata(row: Mapping[str, Any], *, cache_path: Path, source_tokens: np.ndarray, output_tokens: np.ndarray) -> dict[str, Any]:
    audio = _resolve(str(row["audio_path"]))
    reference = _resolve(str(row["reference_audio_path"])) if row.get("reference_audio_path") else None
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "kind": "genuine_source_speech_tokens",
        "synthetic_tts": False,
        "description": "Speech-token IDs copied from an adaptation-cache target recording; not generated TTS.",
        "row_id": str(row["id"]),
        "split": row.get("split"),
        "speaker_id": row.get("speaker_id"),
        "cache": str(cache_path.resolve()),
        "cache_sha256": _sha256(cache_path),
        "audio_path": str(audio),
        "audio_sha256": _sha256(audio) if audio.exists() else "missing",
        "reference_audio_path": str(reference) if reference else None,
        "reference_distinct_from_target": bool(reference and reference.resolve() != audio.resolve()),
        "source_text": row.get("source_text"),
        "text": row.get("text"),
        "source_token_count": int(source_tokens.size),
        "output_token_count": int(output_tokens.size),
        "trailing_silence": {"value": S3GEN_SIL, "count": 3, "appended": True},
    }
    return metadata


def stage_prepare(args: argparse.Namespace) -> dict[str, Any]:
    cache_path = _resolve(args.cache)
    rows = _cache_rows(cache_path)
    selected = [_row_by_id(rows, row_id) for row_id in args.ids] if args.ids else _short_rows(rows, int(args.count))
    if len(selected) < 2:
        raise ValueError("prepare requires at least two cache rows for a useful codec comparison")
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for row in selected:
        row_id = str(row["id"])
        source_tokens = _validate_source_tokens(row["speech_tokens"], row_id=row_id)
        output_tokens = np.concatenate((source_tokens, np.full((3,), S3GEN_SIL, dtype=np.int64)))
        token_path = output_dir / f"{row_id}.speech_tokens.npy"
        metadata_path = output_dir / f"{row_id}.source.json"
        _save_npy(token_path, output_tokens)
        metadata = _source_metadata(row, cache_path=cache_path, source_tokens=source_tokens, output_tokens=output_tokens)
        metadata["tokens_path"] = str(token_path)
        _write_json(metadata_path, metadata)
        records.append({"id": row_id, "tokens": str(token_path), "metadata": str(metadata_path), "source_token_count": int(source_tokens.size)})
    manifest = {
        "schema_version": 1,
        "kind": "codec_source_diagnostics",
        "synthetic_tts": False,
        "cache": str(cache_path),
        "cache_sha256": _sha256(cache_path),
        "selected": records,
        "selection": "explicit IDs" if args.ids else f"shortest {args.count} rows",
        "s3gen_sil": S3GEN_SIL,
        "created_unix": time.time(),
        "command": list(sys.argv),
    }
    _write_json(output_dir / "manifest.json", manifest)
    return {"output_dir": str(output_dir), "selected": records, "synthetic_tts": False}


def _load_source_audio(path: Path) -> tuple[np.ndarray, int]:
    # librosa.load(sr=24000, mono=True) is the exact upstream
    # prepare_conditionals loader.  It also avoids carrying torchaudio model
    # state into this stage.
    import librosa

    audio, sample_rate = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1 or audio.size < 1 or not np.isfinite(audio).all():
        raise ValueError(f"source audio is empty or non-finite: {path}")
    return audio, int(sample_rate)


def stage_source_mel(args: argparse.Namespace) -> dict[str, Any]:
    diagnostics_dir = _resolve(args.output_dir)
    manifest_path = diagnostics_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"prepare manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    rows = _cache_rows(_resolve(args.cache))
    row = _row_by_id(rows, args.id)
    audio_path = _resolve(str(row["audio_path"]))
    audio, sample_rate = _load_source_audio(audio_path)
    if args.max_seconds is not None:
        if args.max_seconds <= 0:
            raise ValueError("--max-seconds must be positive")
        audio = audio[: int(round(float(args.max_seconds) * sample_rate))]
    _ensure_vendor_path()
    import torch
    from chatterbox.models.s3gen.utils.mel import mel_spectrogram

    with torch.inference_mode():
        mel_bft = mel_spectrogram(torch.from_numpy(audio)[None, :]).detach().cpu().numpy().astype(np.float32)
    if mel_bft.ndim != 3 or mel_bft.shape[0] != 1 or mel_bft.shape[1] != 80:
        raise RuntimeError(f"local S3Gen mel extractor returned unexpected shape {mel_bft.shape}")
    if args.prefix_mel_frames is not None:
        frames = int(args.prefix_mel_frames)
        if frames < 1 or frames > mel_bft.shape[2]:
            raise ValueError(f"--prefix-mel-frames must be in 1..{mel_bft.shape[2]}")
        mel_bft = mel_bft[:, :, :frames]
    output_path = diagnostics_dir / f"{args.id}.source_mel.npz"
    _save_npz(output_path, {"mel": mel_bft, "audio": audio})
    report = {
        "schema_version": 1,
        "kind": "genuine_source_mel",
        "synthetic_tts": False,
        "row_id": args.id,
        "source_audio": str(audio_path),
        "source_audio_sha256": _sha256(audio_path),
        "sample_rate": sample_rate,
        "mel_extractor": "vendor/chatterbox/src/chatterbox/models/s3gen/utils/mel.py::mel_spectrogram",
        "mel_layout": "B,80,T",
        "mel_shape": list(mel_bft.shape),
        "audio_samples_after_crop": int(audio.size),
        "prefix_mel_frames": int(mel_bft.shape[2]),
        "mel_params": {"n_fft": 1920, "num_mels": 80, "hop_size": 480, "win_size": 1920, "fmin": 0, "fmax": 8000, "center": False},
        "pipeline_manifest": str(manifest_path),
        "command": list(sys.argv),
    }
    _write_json(diagnostics_dir / f"{args.id}.source_mel.json", report)
    return {"id": args.id, "mel": str(output_path), "mel_shape": list(mel_bft.shape), "synthetic_tts": False}


def _pipeline_inputs(run_dir: Path, *, prefix_mel_frames: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    estimator_path = run_dir / "estimator" / "mel.npz"
    vocoder_path = run_dir / "vocoder" / "audio.npz"
    if not estimator_path.exists() or not vocoder_path.exists():
        raise FileNotFoundError(f"pipeline estimator/vocoder artifacts not found under {run_dir}")
    estimator = _load_npz(estimator_path)
    vocoder = _load_npz(vocoder_path)
    mel = np.asarray(estimator["mel"], dtype=np.float32)
    phase = np.asarray(vocoder["phase_noise"], dtype=np.float32)
    sine = np.asarray(vocoder["sine_noise"], dtype=np.float32)
    if mel.ndim != 3 or mel.shape[0] != 1 or mel.shape[1] != 80:
        raise ValueError(f"pipeline mel must have shape (1,80,T), got {mel.shape}")
    if phase.shape != (1, 9, 1):
        raise ValueError(f"pipeline phase noise must have shape (1,9,1), got {phase.shape}")
    expected_noise = mel.shape[2] * 480
    if sine.shape != (1, 9, expected_noise):
        raise ValueError(f"pipeline sine noise shape {sine.shape} does not match mel length {mel.shape[2]}")
    full_frames = mel.shape[2]
    if prefix_mel_frames is not None:
        frames = int(prefix_mel_frames)
        if frames < 1 or frames > full_frames:
            raise ValueError(f"--prefix-mel-frames must be in 1..{full_frames}")
        mel = mel[:, :, :frames]
        sine = sine[:, :, : frames * 480]
    config = {"run_dir": str(run_dir), "full_mel_frames": full_frames, "used_mel_frames": int(mel.shape[2]), "source": "onnx_pipeline estimator/mel.npz + vocoder/audio.npz"}
    return mel, phase, sine, config


def _source_inputs(source_mel_path: Path, *, seed: int, prefix_mel_frames: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    values = _load_npz(source_mel_path)
    mel = np.asarray(values["mel"], dtype=np.float32)
    if mel.ndim != 3 or mel.shape[0] != 1 or mel.shape[1] != 80:
        raise ValueError(f"source mel must have shape (1,80,T), got {mel.shape}")
    if prefix_mel_frames is not None:
        frames = int(prefix_mel_frames)
        if frames < 1 or frames > mel.shape[2]:
            raise ValueError(f"--prefix-mel-frames must be in 1..{mel.shape[2]}")
        mel = mel[:, :, :frames]
    rng = np.random.default_rng(int(seed))
    phase = rng.uniform(-np.pi, np.pi, size=(1, 9, 1)).astype(np.float32)
    sine = rng.normal(size=(1, 9, mel.shape[2] * 480)).astype(np.float32)
    return mel, phase, sine, {"source_mel": str(source_mel_path), "seed": int(seed), "used_mel_frames": int(mel.shape[2]), "noise": "explicit NumPy phase uniform and sine normal"}


def _source_branch_diagnostics(model: Any, mel: Any, phase: Any, sine: Any) -> tuple[Any, dict[str, float]]:
    """Run the staged native source branch and return f0/source norms."""

    import torch
    import numpy as np_local

    module = model.mel2wav
    f0 = module.f0_predictor(mel)
    f0_upsampled = module.f0_upsamp(f0[:, None]).transpose(1, 2)
    sine_gen = module.m_source.l_sin_gen
    harmonic_count = int(sine_gen.harmonic_num) + 1
    harmonics = torch.arange(1, harmonic_count + 1, device=f0_upsampled.device, dtype=f0_upsampled.dtype).view(1, harmonic_count, 1)
    f_mat = f0_upsampled.transpose(1, 2) * harmonics / sine_gen.sampling_rate
    theta = 2.0 * float(np_local.pi) * (torch.cumsum(f_mat, dim=-1) % 1)
    phase_input = torch.cat((torch.zeros_like(phase[:, :1, :]), phase[:, 1:harmonic_count, :]), dim=1)
    sine_waves = sine_gen.sine_amp * torch.sin(theta + phase_input)
    uv = (f0_upsampled.transpose(1, 2) > sine_gen.voiced_threshold).to(sine_waves.dtype)
    noise_amp = uv * sine_gen.noise_std + (1 - uv) * sine_gen.sine_amp / 3
    sine_waves = sine_waves * uv + noise_amp * sine
    sine_waves_t = sine_waves.transpose(1, 2)
    sine_merge = module.m_source.l_tanh(module.m_source.l_linear(sine_waves_t))
    source = sine_merge.transpose(1, 2)
    diagnostics = {
        "f0_min_hz": float(f0.min().detach().cpu()),
        "f0_max_hz": float(f0.max().detach().cpu()),
        "f0_mean_hz": float(f0.mean().detach().cpu()),
        "voiced_fraction": float(uv.mean().detach().cpu()),
        "f0_frames": int(f0.shape[-1]),
        "sine_waves_l2": float(torch.linalg.vector_norm(sine_waves).detach().cpu()),
        "sine_waves_rms": float(torch.sqrt(torch.mean(sine_waves.square())).detach().cpu()),
        "source_branch_l2": float(torch.linalg.vector_norm(source).detach().cpu()),
        "source_branch_rms": float(torch.sqrt(torch.mean(source.square())).detach().cpu()),
        "phase_noise_l2": float(torch.linalg.vector_norm(phase).detach().cpu()),
        "sine_noise_l2": float(torch.linalg.vector_norm(sine).detach().cpu()),
    }
    return source, diagnostics


def stage_vocoder_reference(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = int(args.prefix_mel_frames) if args.prefix_mel_frames is not None else None
    if args.pipeline_run_dir:
        mel, phase, sine, provenance = _pipeline_inputs(_resolve(args.pipeline_run_dir), prefix_mel_frames=prefix)
        branch = "pipeline_mel"
    elif args.source_mel:
        mel, phase, sine, provenance = _source_inputs(_resolve(args.source_mel), seed=args.seed, prefix_mel_frames=prefix)
        branch = "source_mel"
    else:
        raise ValueError("vocoder-reference requires --pipeline-run-dir or --source-mel")
    _ensure_vendor_path()
    import torch
    from onnx_staged import _load_vocoder

    checkpoint_dir = _resolve(args.checkpoint_dir)
    model, load_report = _load_vocoder(checkpoint_dir, torch.device("cpu"))
    model.eval()
    mel_t = torch.from_numpy(mel)
    phase_t = torch.from_numpy(phase)
    sine_t = torch.from_numpy(sine)
    with torch.inference_mode():
        source_branch, diagnostics = _source_branch_diagnostics(model, mel_t, phase_t, sine_t)
        waveform = model(mel_t, phase_t, sine_t)
    waveform_np = np.asarray(waveform.detach().cpu().numpy(), dtype=np.float32)
    if waveform_np.ndim == 2 and waveform_np.shape[0] == 1:
        waveform_np = waveform_np[0]
    if waveform_np.ndim != 1 or waveform_np.size < 1 or not np.isfinite(waveform_np).all():
        raise RuntimeError(f"native vocoder returned invalid waveform shape {waveform_np.shape}")
    branch_np = np.asarray(source_branch.detach().cpu().numpy(), dtype=np.float32)
    _save_npy(output_dir / f"{args.label}.native_waveform.npy", waveform_np)
    _save_npy(output_dir / f"{args.label}.source_branch.npy", branch_np)
    _save_npy(output_dir / f"{args.label}.mel.npy", mel)
    _save_npy(output_dir / f"{args.label}.phase_noise.npy", phase)
    _save_npy(output_dir / f"{args.label}.sine_noise.npy", sine)
    report = {
        "schema_version": 1,
        "kind": "native_vocoder_reference",
        "branch": branch,
        "synthetic_tts": branch == "pipeline_mel",
        "mel_layout": "B,80,T",
        "mel_shape": list(mel.shape),
        "waveform_shape": list(waveform_np.shape),
        "sample_rate": SAMPLE_RATE,
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_load": load_report,
        "provenance": provenance,
        "native_waveform": str(output_dir / f"{args.label}.native_waveform.npy"),
        "source_branch": str(output_dir / f"{args.label}.source_branch.npy"),
        "mel": str(output_dir / f"{args.label}.mel.npy"),
        "phase_noise": str(output_dir / f"{args.label}.phase_noise.npy"),
        "sine_noise": str(output_dir / f"{args.label}.sine_noise.npy"),
        "source_branch_diagnostics": diagnostics,
        "command": list(sys.argv),
    }
    _write_json(output_dir / f"{args.label}.native.json", report)
    del model
    return {"label": args.label, "branch": branch, "native_report": str(output_dir / f"{args.label}.native.json"), "waveform_samples": int(waveform_np.size), "diagnostics": diagnostics}


def _relative_l2(actual: np.ndarray, expected: np.ndarray) -> float:
    diff = np.asarray(actual, dtype=np.float64) - np.asarray(expected, dtype=np.float64)
    denominator = max(float(np.linalg.norm(np.asarray(expected, dtype=np.float64))), 1e-12)
    return float(np.linalg.norm(diff) / denominator)


def stage_vocoder_check(args: argparse.Namespace) -> dict[str, Any]:
    native_report_path = _resolve(args.native_report)
    native = json.loads(native_report_path.read_text())
    mel = _load_npy(_resolve(native["mel"]), dtype=np.dtype("float32"))
    phase = _load_npy(_resolve(native["phase_noise"]), dtype=np.dtype("float32"))
    sine = _load_npy(_resolve(native["sine_noise"]), dtype=np.dtype("float32"))
    expected = _load_npy(_resolve(native["native_waveform"]), dtype=np.dtype("float32"))
    from onnx_staged_runtime import VocoderOrtRuntime

    runtime = VocoderOrtRuntime(_resolve(args.model_dir), intra_op_num_threads=args.ort_threads, inter_op_num_threads=1)
    actual = np.asarray(runtime.synthesize(mel, phase_noise=phase, sine_noise=sine), dtype=np.float32)
    if actual.ndim == 2 and actual.shape[0] == 1:
        actual = actual[0]
    if actual.shape != expected.shape:
        raise RuntimeError(f"ORT waveform shape {actual.shape} differs from native {expected.shape}")
    if not np.isfinite(actual).all():
        raise RuntimeError("ORT waveform contains NaN or infinity")
    difference = actual - expected
    report = {
        "schema_version": 1,
        "kind": "ort_vocoder_check",
        "status": "verified" if np.allclose(actual, expected, atol=3e-4, rtol=3e-4) else "failed",
        "atol": 3e-4,
        "rtol": 3e-4,
        "native_report": str(native_report_path),
        "branch": native.get("branch"),
        "graph_model_dir": str(_resolve(args.model_dir)),
        "waveform_shape": list(actual.shape),
        "max_abs_error": float(np.max(np.abs(difference))),
        "relative_l2_error": _relative_l2(actual, expected),
        "max_relative_error": float(np.max(np.abs(difference) / np.maximum(np.abs(expected), 1e-6))),
        "source_branch_diagnostics": native.get("source_branch_diagnostics"),
        "native_waveform": str(_resolve(native["native_waveform"])),
        "ort_waveform": str(_resolve(args.output_dir) / f"{native_report_path.stem}.ort_waveform.npy"),
        "command": list(sys.argv),
    }
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_npy(output_dir / f"{native_report_path.stem}.ort_waveform.npy", actual)
    _write_json(output_dir / f"{native_report_path.stem}.ort.json", report)
    return report


def _dispatch(args: argparse.Namespace) -> int:
    started = time.time()
    try:
        if args.command == "prepare":
            result = stage_prepare(args)
        elif args.command == "source-mel":
            result = stage_source_mel(args)
        elif args.command == "vocoder-reference":
            result = stage_vocoder_reference(args)
        elif args.command == "vocoder-check":
            result = stage_vocoder_check(args)
        else:  # pragma: no cover - argparse enforces command choices
            raise ValueError(args.command)
        failed = args.command == "vocoder-check" and result["status"] != "verified"
        report = {"status": "error" if failed else "ok", "command": list(sys.argv), "elapsed_seconds": time.time() - started, "resources": _resource_report(), "result": result}
        output_dir = _resolve(getattr(args, "output_dir", DEFAULT_DIAGNOSTICS))
        _write_json(output_dir / f"{args.command}.stage.json", report)
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        return 1 if failed else 0
    except Exception as exc:
        report = {"status": "error", "command": list(sys.argv), "elapsed_seconds": time.time() - started, "resources": _resource_report(), "error": f"{type(exc).__name__}: {exc}"}
        output_dir = _resolve(getattr(args, "output_dir", DEFAULT_DIAGNOSTICS))
        _write_json(output_dir / f"{args.command}.stage.json", report)
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        return 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="export genuine cache target tokens and provenance")
    prep.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    prep.add_argument("--output-dir", type=Path, default=DEFAULT_DIAGNOSTICS)
    prep.add_argument(
        "--ids",
        nargs="+",
        default=["asmr7_train_01", "asmr7_train_02"],
        help="explicit cache row IDs (default: asmr7_train_01 asmr7_train_02)",
    )
    prep.add_argument("--count", type=int, default=2)

    mel = sub.add_parser("source-mel", help="extract exact local S3Gen mel features from one source recording")
    mel.add_argument("--id", required=True)
    mel.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    mel.add_argument("--output-dir", type=Path, default=DEFAULT_DIAGNOSTICS)
    mel.add_argument("--max-seconds", type=float)
    mel.add_argument("--prefix-mel-frames", type=int)

    ref = sub.add_parser("vocoder-reference", help="fresh native HiFT reference; no ORT session")
    ref.add_argument("--label", required=True)
    ref.add_argument("--output-dir", type=Path, default=DEFAULT_DIAGNOSTICS)
    ref.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    ref.add_argument("--pipeline-run-dir", type=Path)
    ref.add_argument("--source-mel", type=Path)
    ref.add_argument("--prefix-mel-frames", type=int)
    ref.add_argument("--seed", type=int, default=20231)

    check = sub.add_parser("vocoder-check", help="fresh ORT HiFT parity check against a native report")
    check.add_argument("--native-report", type=Path, required=True)
    check.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    check.add_argument("--output-dir", type=Path, default=DEFAULT_DIAGNOSTICS)
    check.add_argument("--ort-threads", type=int, default=1)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    return _dispatch(_parser().parse_args(list(argv) if argv is not None else None))


if __name__ == "__main__":
    raise SystemExit(main())
