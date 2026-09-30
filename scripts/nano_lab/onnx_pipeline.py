"""Bounded, staged Nano speech pipeline.

This command is an experimental integration layer around the independently
verified ONNX stages in :mod:`onnx_staged_runtime`.  Each stage runs in a new
child process.  A child owns one ORT session (or one small preparation,
tokenisation, or watermark operation), writes its artifacts, and exits before
the next child starts.  This keeps the resident set of large sessions from
accumulating on a desktop with limited memory.

The T3 graph is intentionally gated behind ``--experimental-t3``.  The short
T3 checks pass, but the saved long reference still has a small cache mismatch
at length 400.  The pipeline therefore records this limitation in every run
report and does not call the stage numerically fully verified.

No PyTorch, Transformers, ONNX Runtime, Perth, SciPy, or SoundFile module is
imported at module import time.  The stages that need those packages import
them inside the child process only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from file_fingerprint import fingerprint_file
from profile_conditioning import resolve_conditioning_cache

DEFAULT_MODEL_DIR = ROOT / "artifacts" / "nano_lab" / "onnx_staged"
DEFAULT_CHECKPOINT_DIR = ROOT / "models" / "chatterbox-nano"
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "nano_lab" / "onnx_pipeline"
SAMPLE_RATE = 24_000
SPEECH_BOS = 6561
SPEECH_EOS = 6562
SPEECH_VOCAB = 6563
S3GEN_SIL = 4299
MAX_TEXT_TOKENS = 350
MAX_GENERATED_TOKENS = 700
EXPERIMENTAL_T3_LIMITATION = (
    "The long T3 reference at sequence length 400 is not within the strict "
    "3e-4 cache gate: 176 of 7.4M cache values exceeded the gate (max abs "
    "0.000989; relative L2 1.44e-5). Logits, lengths 32/128, and decode "
    "cases passed. This integration run is experimental and does not claim "
    "full numerical T3 verification."
)


# The normal staged pipeline creates one runtime in each child. The explicit
# batch benchmark can install a process-local factory so the same four runtime
# objects are reused for several prepared cases. None keeps all existing
# single-request paths on their original constructors.
_RUNTIME_FACTORY: Callable[..., Any] | None = None


def _ort_runtime_kwargs(args: argparse.Namespace, stage: str | None = None) -> dict[str, Any]:
    """Return the explicit provider controls shared by every ORT stage."""

    profile_prefix = None
    if bool(getattr(args, "ort_profile", False)):
        stage_name = stage or "stage"
        profile_prefix = Path(args.run_dir).resolve() / "ort_profiles" / f"{stage_name}_{os.getpid()}"
    return {
        "intra_op_num_threads": getattr(args, "ort_threads", 1),
        "inter_op_num_threads": 1,
        "ort_provider": getattr(args, "ort_provider", "cpu"),
        "cuda_device_id": getattr(args, "cuda_device_id", 0),
        "gpu_mem_limit_mib": getattr(args, "gpu_mem_limit_mib", 2048),
        "arena_extend_strategy": getattr(args, "arena_extend_strategy", "kSameAsRequested"),
        "cudnn_conv_algo_search": getattr(args, "cudnn_conv_algo_search", "HEURISTIC"),
        "do_copy_in_default_stream": getattr(args, "do_copy_in_default_stream", True),
        "ort_profile": bool(getattr(args, "ort_profile", False)),
        "profile_prefix": profile_prefix,
        "cuda_kv_resident": bool(getattr(args, "cuda_kv_resident", False)),
    }


def _runtime_info(runtime: Any, args: argparse.Namespace) -> dict[str, Any]:
    """Collect JSON-safe provider diagnostics from a stage adapter."""

    info = runtime.info() if callable(getattr(runtime, "info", None)) else {}
    if not isinstance(info, Mapping):
        info = {}
    value = dict(info)
    ort_provider = getattr(args, "ort_provider", "cpu")
    value.setdefault("ort_provider", ort_provider)
    value.setdefault("requested_provider_options", {
        "cuda_device_id": getattr(args, "cuda_device_id", 0),
        "gpu_mem_limit_mib": getattr(args, "gpu_mem_limit_mib", 2048),
        "arena_extend_strategy": getattr(args, "arena_extend_strategy", "kSameAsRequested"),
        "cudnn_conv_algo_search": getattr(args, "cudnn_conv_algo_search", "HEURISTIC"),
        "do_copy_in_default_stream": getattr(args, "do_copy_in_default_stream", True),
    } if ort_provider == "cuda" else None)
    value.setdefault("profile_requested", bool(getattr(args, "ort_profile", False)))
    return value


def set_runtime_factory(factory: Callable[..., Any] | None) -> None:
    """Install an optional runtime factory for the current process.

    The factory receives (stage, model_dir, **runtime_kwargs) and must return
    an object implementing the stage runtime API. This hook is used by
    onnx_batch only. Passing None restores the default constructors and is
    safe for tests that run several pipeline calls in one process.
    """

    global _RUNTIME_FACTORY
    _RUNTIME_FACTORY = factory


def _make_runtime(stage: str, args: argparse.Namespace) -> Any:
    """Construct one stage runtime through the optional injectable factory."""

    model_dir = Path(args.model_dir).resolve()
    kwargs = _ort_runtime_kwargs(args, stage)
    if _RUNTIME_FACTORY is not None:
        runtime = _RUNTIME_FACTORY(stage, model_dir, **kwargs)
        if runtime is None:
            raise RuntimeError(f"runtime factory returned None for stage {stage!r}")
        return runtime

    # Keep imports lazy. The parent and the preparation/finish children do not
    # import ONNX Runtime.
    from onnx_staged_runtime import (
        FlowEncoderOrtRuntime,
        MeanflowEstimatorOrtRuntime,
        T3UnifiedOrtRuntime,
        VocoderOrtRuntime,
    )

    classes = {
        "t3": T3UnifiedOrtRuntime,
        "flow_encoder": FlowEncoderOrtRuntime,
        "meanflow_estimator": MeanflowEstimatorOrtRuntime,
        "vocoder": VocoderOrtRuntime,
    }
    try:
        runtime_class = classes[stage]
    except KeyError as exc:
        raise ValueError(f"unknown ONNX runtime stage: {stage!r}") from exc
    return runtime_class(model_dir, **kwargs)


def _finish_runtime(runtime: Any, args: argparse.Namespace) -> dict[str, Any]:
    """Flush optional ORT profiling, then return final provider metadata."""

    close = getattr(runtime, "close", None)
    # Persistent batch workers keep each session alive between cases. The
    # worker closes all four objects in its finally block after the last case.
    if callable(close) and not bool(getattr(runtime, "_nano_persistent", False)):
        close()
    return _runtime_info(runtime, args)


def _cache_length(runtime: Any, cache: Sequence[Any]) -> int:
    """Read T3 cache length without copying CUDA OrtValues to NumPy."""

    helper = getattr(runtime, "cache_length", None)
    if callable(helper):
        return int(helper(cache))
    first = cache[0]
    shape_method = getattr(first, "shape", None)
    shape = shape_method() if callable(shape_method) else shape_method
    return int(shape[2])


def _now() -> float:
    return time.time()


def _rss_bytes() -> int:
    """Return the current process RSS where procfs is available."""

    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, OSError, ValueError):
        pass
    return 0


def _peak_rss_bytes() -> int:
    # Linux reports ru_maxrss in KiB.  Keep the fallback conservative for
    # platforms where the unit differs.
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value * 1024 if sys.platform != "darwin" else value


def _rss_report() -> dict[str, int]:
    return {"rss_bytes": _rss_bytes(), "peak_rss_bytes": _peak_rss_bytes()}


def _tree_rss_bytes(pid: int) -> int:
    """Sample this pipeline's process tree only; Linux procfs, no model imports."""
    total = 0
    pending = [pid]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        try:
            for line in Path(f"/proc/{current}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1]) * 1024
                    break
            children = Path(f"/proc/{current}/task/{current}/children").read_text()
            pending.extend(int(value) for value in children.split())
        except (OSError, ValueError):
            # A child may exit between the two procfs reads.
            continue
    return total


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _sha256(path: Path, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(path: Path) -> str:
    """Hash tokenizer assets without reading model checkpoint weights."""

    if path.is_file():
        return _sha256(path)
    digest = hashlib.sha256()
    if not path.exists():
        return "missing"
    tokenizer_names = {
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "spiece.model",
        "sentencepiece.bpe.model",
        "tokenizer.model",
    }
    count = 0
    for child in sorted(p for p in path.rglob("*") if p.is_file() and p.name in tokenizer_names):
        count += 1
        digest.update(str(child.relative_to(path)).encode("utf-8"))
        digest.update(_sha256(child).encode("ascii"))
    return digest.hexdigest() if count else "no-tokenizer-assets"


def _resolve_path(value: str | os.PathLike[str], *, base: Path = ROOT) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path)


def _resolve_profile(voice: str | os.PathLike[str]) -> Path:
    candidate = Path(voice).expanduser()
    if candidate.exists():
        return candidate.resolve()
    if candidate.suffix != ".json":
        candidate = ROOT / "voices" / "nano" / f"{candidate.name}.json"
    else:
        candidate = ROOT / "voices" / "nano" / candidate.name
    if not candidate.exists():
        raise FileNotFoundError(f"Nano voice profile not found: {voice}")
    return candidate.resolve()


def _load_profile(
    voice: str | os.PathLike[str] | None = None,
    *,
    path: Path | None = None,
    model_dir: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    profile_path = path.resolve() if path is not None else _resolve_profile(voice or "asmr_conversational")
    profile = json.loads(profile_path.read_text())
    if not isinstance(profile, dict):
        raise ValueError(f"Voice profile must be a JSON object: {profile_path}")
    if profile.get("status") == "rejected_training_label_mismatch":
        raise ValueError(profile["rejection_reason"])
    if model_dir is None:
        # Preserve the old fail-closed behavior for callers that do not provide
        # a model root. Stage preparation and tokenisation pass model_dir and
        # perform the stronger graph/provenance validation below.
        if profile.get("adapter") and float(profile.get("adapter_scale", 1.0)) != 0.0:
            raise ValueError("This profile requires fitted adapter provenance. Use nano-clone with a matching patched ONNX model_dir.")
        if profile.get("mel_calibration") and float(profile.get("mel_calibration_strength", 1.0)) != 0.0:
            raise ValueError("This profile requires mel calibration. Use nano-clone with a staged model_dir.")
    else:
        # The stage entrypoints call _validate_profile_for_model immediately
        # after loading. Keep this branch permissive so the expensive graph
        # hash is computed exactly once and the validation result can be
        # recorded in the stage metadata.
        pass
    return profile_path, profile


def _float_field(value: Any, *, name: str, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _profile_adapter_enabled(profile: Mapping[str, Any]) -> tuple[bool, float, Path | None]:
    adapter_value = profile.get("adapter")
    adapter_path = None
    if adapter_value not in (None, ""):
        adapter_path = _resolve_path(str(adapter_value)).resolve()
    default_scale = 1.0 if adapter_path is not None else 0.0
    scale = _float_field(profile.get("adapter_scale"), name="adapter_scale", default=default_scale)
    enabled = abs(scale) > 1e-12
    return enabled, scale, adapter_path


def _load_t3_manifest(model_dir: Path) -> tuple[Path, dict[str, Any]]:
    manifest_path = model_dir / "t3" / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"staged T3 manifest not found: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid staged T3 manifest: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"staged T3 manifest must be an object: {manifest_path}")
    if manifest.get("stage") != "t3" or manifest.get("status") != "exported":
        raise ValueError(f"staged T3 manifest is not an exported t3 graph: {manifest_path}")
    return manifest_path, manifest


def _manifest_value(mapping: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        value = mapping.get(name)
        if value not in (None, ""):
            return value
    return None


def _validate_adapter_provenance(profile: Mapping[str, Any], model_dir: Path) -> dict[str, Any]:
    enabled, scale, adapter_path = _profile_adapter_enabled(profile)
    manifest_path, manifest = _load_t3_manifest(model_dir)
    patch = manifest.get("adapter_patch")
    if patch is not None and not isinstance(patch, Mapping):
        raise ValueError("staged T3 adapter_patch metadata must be an object")
    patch_present = isinstance(patch, Mapping)
    if not enabled:
        if patch_present:
            raise ValueError(
                "base voice profile cannot run against a patched T3 graph; use the matching fitted profile explicitly"
            )
        return {
            "enabled": False,
            "profile_adapter": None,
            "profile_scale": 0.0,
            "manifest": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "graph_sha256": None,
            "adapter_sha256": None,
        }
    if adapter_path is None:
        raise ValueError("fitted profile has nonzero adapter_scale but no adapter path")
    if not patch_present:
        raise ValueError("fitted profile requires a staged T3 adapter_patch manifest; base graph was selected")
    graph = manifest.get("graph")
    if not isinstance(graph, Mapping) or not graph.get("path"):
        raise ValueError("staged T3 manifest has no graph path for adapter validation")
    graph_path = Path(str(graph["path"])).expanduser()
    if not graph_path.is_absolute():
        graph_path = (model_dir / "t3" / graph_path).resolve()
    if not graph_path.exists():
        raise FileNotFoundError(f"staged T3 graph not found: {graph_path}")
    # The T3 graph can be hundreds of MiB.  Keep the provenance check strict,
    # but reuse a metadata-keyed process-local digest when this process checks
    # the same graph more than once.  Adapter and manifest files remain on the
    # ordinary small-file hashing path below.
    actual_graph_sha = fingerprint_file(graph_path)
    declared_graph_sha = _manifest_value(graph, "sha256")
    patched_graph_sha = _manifest_value(patch, "patched_graph_sha256", "output_graph_sha256")
    if not declared_graph_sha or actual_graph_sha != str(declared_graph_sha):
        raise ValueError("staged T3 graph SHA-256 does not match its manifest")
    if not patched_graph_sha or actual_graph_sha != str(patched_graph_sha):
        raise ValueError("staged T3 graph SHA-256 does not match adapter_patch provenance")
    declared_adapter_sha = _manifest_value(patch, "adapter_sha256", "sha256")
    if not declared_adapter_sha or not adapter_path.exists():
        raise FileNotFoundError("fitted adapter file or adapter_patch SHA-256 is missing")
    actual_adapter_sha = _sha256(adapter_path)
    if actual_adapter_sha != str(declared_adapter_sha):
        raise ValueError("fitted adapter SHA-256 does not match staged adapter_patch provenance")
    declared_adapter_path = _manifest_value(patch, "adapter_path", "path")
    if declared_adapter_path is not None:
        expected_path = _resolve_path(str(declared_adapter_path)).resolve()
        if expected_path != adapter_path:
            raise ValueError("fitted adapter path does not match staged adapter_patch provenance")
    declared_scale = _manifest_value(patch, "checkpoint_scale", "scale")
    if declared_scale is None:
        raise ValueError("staged adapter_patch is missing checkpoint scale provenance")
    patch_scale = _float_field(declared_scale, name="adapter_patch scale")
    if not np.isclose(scale, patch_scale, rtol=0.0, atol=1e-9):
        raise ValueError(f"profile adapter_scale {scale} does not match staged adapter scale {patch_scale}")
    return {
        "enabled": True,
        "profile_adapter": str(adapter_path),
        "profile_scale": scale,
        "adapter_sha256": actual_adapter_sha,
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "graph": str(graph_path),
        "graph_sha256": actual_graph_sha,
        "patched_graph_sha256": str(patched_graph_sha),
        "adapter_patch_format": patch.get("format"),
    }


def _resolve_calibration_delta(report_path: Path, report: Mapping[str, Any]) -> tuple[Path, np.ndarray, str]:
    delta_value = report.get("delta_path")
    if delta_value in (None, ""):
        raise ValueError(f"mel calibration report has no delta_path: {report_path}")
    delta_path = Path(str(delta_value)).expanduser()
    if not delta_path.is_absolute():
        delta_path = (report_path.parent / delta_path).resolve()
    if not delta_path.exists():
        raise FileNotFoundError(f"mel calibration delta not found: {delta_path}")
    # Reuse the calibration module's shape, finite-value, and safety-bound
    # checks. Its import is NumPy-only at module import time.
    import mel_calibration

    delta = np.asarray(mel_calibration._load_delta(delta_path), dtype=np.float64).reshape(-1)
    if delta.shape != (80,):
        raise ValueError(f"mel calibration delta must have 80 values, got {delta.shape}")
    report_delta = report.get("delta")
    if report_delta is not None:
        listed = np.asarray(report_delta, dtype=np.float64).reshape(-1)
        if listed.shape != (80,) or not np.allclose(delta, listed, rtol=0.0, atol=1e-6):
            raise ValueError("mel calibration report delta does not match delta_path")
    return delta_path, delta, _sha256(delta_path)


def _validate_mel_calibration(profile: Mapping[str, Any]) -> dict[str, Any]:
    path_value = profile.get("mel_calibration")
    strength = _float_field(profile.get("mel_calibration_strength"), name="mel_calibration_strength", default=0.0)
    if strength < 0.0 or strength > 1.0:
        raise ValueError("mel_calibration_strength must be in [0, 1]")
    enabled = abs(strength) > 1e-12
    if not enabled:
        return {
            "enabled": False,
            "applied": False,
            "report": None,
            "report_sha256": None,
            "delta": None,
            "delta_sha256": None,
            "strength": 0.0,
        }
    if path_value in (None, ""):
        raise ValueError("nonzero mel_calibration_strength requires mel_calibration report")
    report_path = _resolve_path(str(path_value)).resolve()
    if not report_path.exists():
        raise FileNotFoundError(f"mel calibration report not found: {report_path}")
    try:
        report = json.loads(report_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid mel calibration report: {report_path}") from exc
    if not isinstance(report, Mapping) or report.get("format") != "nano_mel_calibration_fit_v1":
        raise ValueError("mel calibration report has unsupported format")
    delta_path, delta, delta_sha = _resolve_calibration_delta(report_path, report)
    return {
        "enabled": True,
        "applied": False,
        "report": str(report_path),
        "report_sha256": _sha256(report_path),
        "delta_path": str(delta_path),
        "delta_sha256": delta_sha,
        "strength": strength,
        "delta": delta,
    }


def _validate_profile_for_model(profile_path: Path, profile: Mapping[str, Any], model_dir: Path) -> dict[str, Any]:
    """Validate fitted graph and mel calibration provenance for one model root."""

    adapter = _validate_adapter_provenance(profile, model_dir)
    mel = _validate_mel_calibration(profile)
    from onnx_decoder_attention_profile import validate_decoder_attention_profile
    from profile_t3_donor import resolve_t3_donor
    acoustic = validate_decoder_attention_profile(profile, model_dir, ROOT)
    return {"profile": str(profile_path), "adapter": adapter, "mel_calibration": mel,
            "decoder_attention": acoustic, "t3_donor": resolve_t3_donor(profile, ROOT)}


def _serialise_profile_validation(validation: Mapping[str, Any]) -> dict[str, Any]:
    """Drop the in-memory mel vector before writing stage JSON."""

    value = dict(validation)
    mel = dict(value.get("mel_calibration") or {})
    mel.pop("delta", None)
    value["mel_calibration"] = mel
    return value


def _stage_dir(run_dir: Path, stage: str) -> Path:
    path = run_dir / stage
    path.mkdir(parents=True, exist_ok=True)
    return path


def _np_array(value: Any, *, name: str) -> np.ndarray:
    try:
        array = np.asarray(value)
    except Exception as exc:
        raise TypeError(f"conditioning field {name!r} is not array-like") from exc
    if array.dtype == object:
        raise TypeError(f"conditioning field {name!r} has object dtype")
    return array


def _flatten_conditionals(value: Any, prefix: str, out: dict[str, np.ndarray], skipped: list[str]) -> None:
    """Flatten a weights-only cache into safe NumPy arrays.

    ``Conditionals.save`` stores ``t3`` and ``gen`` dictionaries.  The cache
    contains tensors and a few ``None`` optional fields.  Optional ``None``
    values are recorded in metadata and omitted from the NPZ because they are
    not runtime inputs.
    """

    if value is None:
        skipped.append(prefix)
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            _flatten_conditionals(child, name, out, skipped)
        return
    if isinstance(value, (list, tuple)):
        try:
            array = np.asarray(value)
        except Exception as exc:
            raise TypeError(f"conditioning field {prefix!r} is not numeric") from exc
    elif hasattr(value, "detach") and callable(getattr(value, "detach")):
        # Torch is intentionally discovered only after the stage starts.  A
        # tensor's CPU copy is bounded by the small conditioning cache.
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if array.dtype == object:
        raise TypeError(f"conditioning field {prefix!r} has object dtype")
    out[prefix] = np.array(array, copy=True)


def _save_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    np.savez(temporary, **arrays)
    # numpy appends .npz when the temporary name does not end in that suffix.
    generated = Path(str(temporary) + ".npz") if not temporary.exists() else temporary
    os.replace(generated, path)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as loaded:
        return {name: np.array(loaded[name], copy=True) for name in loaded.files}


def _find_array(arrays: Mapping[str, np.ndarray], *names: str) -> np.ndarray:
    for name in names:
        if name in arrays:
            return arrays[name]
    available = ", ".join(sorted(arrays))
    raise KeyError(f"none of conditioning fields {names!r} were found; available: {available}")


def _as_batch_vector(array: np.ndarray, *, width: int, name: str, dtype: np.dtype = np.dtype("float32")) -> np.ndarray:
    value = np.asarray(array, dtype=dtype)
    if value.ndim == 1:
        value = value[None, :]
    if value.shape != (1, width):
        raise ValueError(f"{name} must have shape (1,{width}), got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return value


def _as_tokens(array: np.ndarray, *, name: str, allow_empty: bool = False) -> np.ndarray:
    value = np.asarray(array)
    if value.ndim == 2 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 1 or (not allow_empty and value.size < 1):
        raise ValueError(f"{name} must be a non-empty one-dimensional token array, got {value.shape}")
    if not np.issubdtype(value.dtype, np.integer):
        if not np.isfinite(value).all() or not np.equal(value, np.floor(value)).all():
            raise ValueError(f"{name} must contain integer token IDs")
    return value.astype(np.int64, copy=False)


def _normalise_prompt_feat(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array, dtype=np.float32)
    if value.ndim == 2 and value.shape[-1] == 80:
        value = value[None, :, :]
    elif value.ndim == 3 and value.shape[0] == 1 and value.shape[-1] == 80:
        pass
    elif value.ndim == 3 and value.shape[0] == 1 and value.shape[1] == 80:
        value = value.transpose(0, 2, 1)
    else:
        raise ValueError(f"gen.prompt_feat must have shape (T,80), (1,T,80), or (1,80,T), got {value.shape}")
    if value.shape[1] < 1 or not np.isfinite(value).all():
        raise ValueError("gen.prompt_feat is empty or non-finite")
    return value


def _load_prepared(run_dir: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    stage = run_dir / "prepare"
    config_path = stage / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"prepare stage is missing: {config_path}")
    config = json.loads(config_path.read_text())
    return config, _load_npz(stage / "conditioning.npz")


def _sampling_config(profile: Mapping[str, Any]) -> dict[str, float | int]:
    sampling = profile.get("sampling") or {}
    return {
        "temperature": float(sampling.get("temperature", 0.8)),
        "top_p": float(sampling.get("top_p", 0.95)),
        "top_k": int(sampling.get("top_k", 1000)),
        "repetition_penalty": float(sampling.get("repetition_penalty", 1.2)),
    }


def stage_prepare(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir).resolve()
    stage = _stage_dir(run_dir, "prepare")
    model_dir = Path(args.model_dir).resolve()
    profile_path, profile = _load_profile(args.voice, model_dir=model_dir)
    profile_validation = _validate_profile_for_model(profile_path, profile, model_dir)
    try:
        cache_info = resolve_conditioning_cache(profile, root=ROOT, require_exists=True)
    except (FileNotFoundError, ValueError) as exc:
        raise type(exc)(f"{exc} (profile: {profile_path})") from exc
    cache_path = cache_info["path"]

    # This is the only model-package import in this stage.  The cache is a
    # weights-only dictionary, so no Conditionals or model class is imported.
    import torch

    try:
        torch.set_num_threads(1)
    except RuntimeError:
        pass
    payload = torch.load(cache_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError(f"weights-only cache must be a dictionary, got {type(payload).__name__}")
    if profile_validation.get('t3_donor'):
        from profile_t3_donor import copy_t3_payload
        donor_info = profile_validation['t3_donor']
        donor_payload = torch.load(donor_info['path'], map_location='cpu', weights_only=True)
        copy_t3_payload(donor_payload, payload, donor_info['mode'])
        del donor_payload
    arrays: dict[str, np.ndarray] = {}
    skipped: list[str] = []
    _flatten_conditionals(payload, "", arrays, skipped)
    required = (
        ("t3.speaker_emb", "t3.speaker_emb"),
        ("t3.cond_prompt_speech_tokens", "t3.cond_prompt_speech_tokens"),
        ("gen.prompt_token", "gen.prompt_token"),
        ("gen.prompt_feat", "gen.prompt_feat"),
        ("gen.embedding", "gen.embedding"),
    )
    missing = [name for name, _ in required if name not in arrays]
    if missing:
        raise KeyError(f"conditioning cache is missing required fields: {missing}")
    _save_npz(stage / "conditioning.npz", arrays)
    profile_copy = dict(profile)
    profile_copy["profile_path"] = str(profile_path)
    profile_copy["conditioning_cache"] = str(cache_path)
    profile_copy["conditioning_cache_sha256"] = _sha256(cache_path)
    _write_json(stage / "profile.json", profile_copy)
    metadata = {
        "schema_version": 1,
        "profile": str(profile_path),
        "profile_sha256": _sha256(profile_path),
        "mode": profile.get("mode"),
        "conditioning_cache": str(cache_path),
        "conditioning_cache_sha256": profile_copy["conditioning_cache_sha256"],
        "keys": {name: {"shape": list(value.shape), "dtype": str(value.dtype)} for name, value in sorted(arrays.items())},
        "skipped_optional_fields": skipped,
        "sampling": _sampling_config(profile),
        "decoder": profile.get("decoder", "meanflow"),
        "steps": int(profile.get("steps", 2)),
        "profile_validation": _serialise_profile_validation(profile_validation),
    }
    _write_json(stage / "config.json", metadata)
    del payload, arrays
    return {
        "profile": str(profile_path),
        "conditioning_cache": str(cache_path),
        "field_count": len(metadata["keys"]),
        "skipped_optional_fields": skipped,
        "mode": profile.get("mode"),
        "profile_validation": _serialise_profile_validation(profile_validation),
    }


def _punc_norm(text: str) -> str:
    """Pure copy of chatterbox.tts_turbo.punc_norm.

    Keeping this small function here avoids importing the full model package in
    the token stage while preserving the exact upstream punctuation order.
    """

    if len(text) == 0:
        return "You need to add some text for me to talk."
    if text[0].islower():
        text = text[0].upper() + text[1:]
    text = " ".join(text.split())
    for old_char_sequence, new_char in (
        ("…", ", "),
        (":", ","),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        ("“", '"'),
        ("”", '"'),
        ("‘", "'"),
        ("’", "'"),
    ):
        text = text.replace(old_char_sequence, new_char)
    text = text.rstrip(" ")
    if not any(text.endswith(p) for p in {".", "!", "?", "-", ","}):
        text += "."
    return text


def _tokenise(text: str, tokenizer_dir: Path) -> tuple[str, np.ndarray]:
    # Tokenization uses no model framework. Keep Torch out of the ORT child.
    os.environ["USE_TORCH"] = "0"
    os.environ["USE_TF"] = "0"
    from transformers import AutoTokenizer

    normalized = _punc_norm(text)
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)
    encoded = tokenizer(normalized, add_special_tokens=True, truncation=False, return_attention_mask=False)
    ids = encoded.get("input_ids") if isinstance(encoded, Mapping) else getattr(encoded, "input_ids", None)
    if ids is None:
        raise RuntimeError("local tokenizer did not return input_ids")
    value = _as_tokens(np.asarray(ids, dtype=np.int64), name="text_tokens")
    if value.size > MAX_TEXT_TOKENS:
        raise ValueError(f"text token count {value.size} exceeds Nano limit {MAX_TEXT_TOKENS}; split the text")
    return normalized, value


def _save_token_artifact(stage: Path, tokens: np.ndarray, config: Mapping[str, Any]) -> None:
    temporary = stage / f".speech_tokens.{os.getpid()}.tmp.npy"
    np.save(temporary, np.asarray(tokens, dtype=np.int64))
    os.replace(temporary, stage / "speech_tokens.npy")
    _write_json(stage / "config.json", dict(config))


def stage_tokens(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir).resolve()
    stage = _stage_dir(run_dir, "tokens")
    prepared, arrays = _load_prepared(run_dir)
    profile_path = run_dir / "prepare" / "profile.json"
    profile_path, profile = _load_profile(path=profile_path, model_dir=Path(args.model_dir).resolve())
    profile_validation = _validate_profile_for_model(profile_path, profile, Path(args.model_dir).resolve())
    if args.tokens_file:
        path = Path(args.tokens_file).expanduser().resolve()
        if path.suffix.lower() != ".npy":
            raise ValueError("--tokens-file accepts a NumPy .npy array only; convert .pt tokens in a separate bounded step")
        tokens = _as_tokens(np.load(path, allow_pickle=False), name="supplied speech tokens")
        if np.any(tokens < 0) or np.any(tokens >= SPEECH_BOS):
            raise ValueError("supplied acoustic speech tokens must be in 0..6560; special T3 IDs are rejected")
        if tokens.size < 3 or not np.all(tokens[-3:] == S3GEN_SIL):
            raise ValueError(
                "supplied speech tokens must already contain exactly the trailing "
                f"S3GEN_SIL convention ({S3GEN_SIL}) in the final three positions"
            )
        _save_token_artifact(
            stage,
            tokens,
            {
                "schema_version": 1,
                "mode": "supplied_tokens_acoustic_verification",
                "source": str(path),
                "source_sha256": _sha256(path),
                "trailing_silence": {"value": S3GEN_SIL, "count": 3, "already_present": True},
                "t3_used": False,
                "experimental_t3": bool(args.experimental_t3),
                "profile_validation": _serialise_profile_validation(profile_validation),
            },
        )
        return {"mode": "supplied_tokens_acoustic_verification", "token_count": int(tokens.size), "t3_used": False}

    if not args.text and not args.text_file:
        raise ValueError("tokens requires --text or --text-file")
    if not args.experimental_t3:
        raise ValueError("T3 generation requires --experimental-t3")
    text = Path(args.text_file).read_text() if args.text_file else str(args.text)
    tokenizer_dir = _resolve_path(args.tokenizer_dir or DEFAULT_CHECKPOINT_DIR)
    normalized, text_tokens = _tokenise(text, tokenizer_dir)
    cond_tokens = _as_tokens(
        _find_array(arrays, "t3.cond_prompt_speech_tokens", "t3.cond_prompt_speech_tokens"),
        name="cond_prompt_speech_tokens",
    )
    _save_token_artifact(
        stage,
        cond_tokens,
        {
            "schema_version": 1,
            "mode": "t3_pending",
            "text": text,
            "normalized_text": normalized,
            "text_tokens": text_tokens.tolist(),
            "text_token_count": int(text_tokens.size),
            "prompt_speech_token_count": int(cond_tokens.size),
            "tokenizer_dir": str(tokenizer_dir),
            "tokenizer_sha256": _tree_sha256(tokenizer_dir),
            "sampling": _sampling_config(profile),
            "seed": int(args.seed),
            "max_generated_tokens": MAX_GENERATED_TOKENS,
            "bos": SPEECH_BOS,
            "eos": SPEECH_EOS,
            "silence": {"value": S3GEN_SIL, "count": 3, "appended_by_t3": True},
            "t3_used": True,
            "profile_validation": _serialise_profile_validation(profile_validation),
        },
    )
    # Store text tokens separately.  This avoids embedding a large JSON list in
    # later reports and makes the exact input visible to the T3 child.
    temporary = stage / f".text_tokens.{os.getpid()}.tmp.npy"
    np.save(temporary, text_tokens)
    os.replace(temporary, stage / "text_tokens.npy")
    # T3 is part of the token stage.  This keeps the public command surface
    # small (prepare/tokens/flow/estimator/vocoder/finish/run) while the
    # entire autoregressive ORT session still lives in its own child process.
    return _run_t3_generation(args, token_config_path=stage / "config.json")


def _assemble_t3_embeddings(model_dir: Path, cond: Mapping[str, np.ndarray], text_tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    stage = model_dir / "t3"
    speech_table = np.load(stage / "speech_embedding.npy", mmap_mode="r")
    text_table = np.load(stage / "text_embedding.npy", mmap_mode="r")
    projection = np.load(stage / "speaker_projection_weight.npy", mmap_mode="r")
    bias = np.load(stage / "speaker_projection_bias.npy", mmap_mode="r")
    if speech_table.shape != (SPEECH_VOCAB, 768) or text_table.shape[1] != 768:
        raise ValueError("staged T3 embedding tables have unexpected shapes")
    if projection.shape != (768, 256) or bias.shape != (768,):
        raise ValueError("staged T3 speaker projection has unexpected shapes")
    speaker = _as_batch_vector(_find_array(cond, "t3.speaker_emb"), width=256, name="t3.speaker_emb")
    prompt = _as_tokens(_find_array(cond, "t3.cond_prompt_speech_tokens"), name="cond_prompt_speech_tokens")
    if np.any(prompt < 0) or np.any(prompt >= SPEECH_VOCAB):
        raise ValueError("conditioning speech tokens contain an out-of-range ID")
    text = _as_tokens(text_tokens, name="text_tokens")
    if np.any(text < 0) or np.any(text >= text_table.shape[0]):
        raise ValueError("text tokenizer produced an ID outside the external embedding table")
    speaker_hidden = np.matmul(speaker, np.asarray(projection, dtype=np.float32).T) + np.asarray(bias, dtype=np.float32)
    prompt_hidden = np.asarray(speech_table[prompt], dtype=np.float32)[None, :, :]
    text_hidden = np.asarray(text_table[text], dtype=np.float32)[None, :, :]
    bos_hidden = np.asarray(speech_table[[SPEECH_BOS]], dtype=np.float32)[None, :, :]
    embeds = np.concatenate((speaker_hidden[:, None, :], prompt_hidden, text_hidden, bos_hidden), axis=1)
    if not np.isfinite(embeds).all():
        raise ValueError("assembled T3 embeddings contain NaN or infinity")
    return embeds.astype(np.float32, copy=False), prompt


def _run_t3_generation(args: argparse.Namespace, *, token_config_path: Path | None = None) -> dict[str, Any]:
    run_dir = Path(args.run_dir).resolve()
    stage = _stage_dir(run_dir, "tokens")
    prepared, cond = _load_prepared(run_dir)
    token_config_path = token_config_path or (stage / "config.json")
    if not token_config_path.exists():
        raise FileNotFoundError(f"tokens stage is missing: {token_config_path}")
    token_config = json.loads(token_config_path.read_text())
    if token_config.get("mode") != "t3_pending":
        return {"mode": token_config.get("mode"), "t3_used": False, "skipped": True}
    text_tokens = np.load(stage / "text_tokens.npy", allow_pickle=False)
    embeds, _ = _assemble_t3_embeddings(Path(args.model_dir).resolve(), cond, text_tokens)
    from onnx_runtime import _sample

    speech_table = np.load(Path(args.model_dir).resolve() / "t3" / "speech_embedding.npy", mmap_mode="r")
    runtime = _make_runtime("t3", args)
    runtime_meta = _runtime_info(runtime, args)
    logits, cache = runtime.prefill(embeds, np.arange(embeds.shape[1], dtype=np.int64))
    rng = np.random.default_rng(int(args.seed))
    sampling = _sampling_config(json.loads((run_dir / "prepare" / "profile.json").read_text()))
    # Keep raw sampled IDs for repetition penalties.  Final acoustic output
    # filters special IDs only after the autoregressive loop, matching the
    # upstream ``speech_tokens[speech_tokens < 6561]`` operation.
    raw_generated: list[int] = []
    generated: list[int] = []
    decode_seconds = 0.0
    prefill_logits = np.asarray(logits)[0]
    hit_limit = False
    eos_seen = False
    for _ in range(MAX_GENERATED_TOKENS):
        history = [SPEECH_BOS] if not raw_generated else raw_generated
        token = int(
            _sample(
                prefill_logits if not raw_generated else logits[0],
                history,
                temperature=float(sampling["temperature"]),
                top_k=int(sampling["top_k"]),
                top_p=float(sampling["top_p"]),
                repetition_penalty=float(sampling["repetition_penalty"]),
                rng=rng,
            )
        )
        if token < 0 or token >= SPEECH_VOCAB:
            raise RuntimeError(f"T3 sampler returned an invalid token ID {token}")
        raw_generated.append(token)
        if token == SPEECH_EOS:
            eos_seen = True
            break
        if token < SPEECH_BOS:
            generated.append(token)
        # Special IDs other than EOS remain in the sampler history and are fed
        # through the embedding table.  They are filtered from the final
        # acoustic token stream only after this loop.
        next_embed = np.asarray(speech_table[[token]], dtype=np.float32)[None, :, :]
        started = time.perf_counter()
        cache_length = _cache_length(runtime, cache)
        logits, cache = runtime.decode(next_embed, np.asarray([cache_length], dtype=np.int64), cache)
        decode_seconds += time.perf_counter() - started
    else:
        hit_limit = True
    if hit_limit and not eos_seen:
        raise RuntimeError(f"T3 reached the hard {MAX_GENERATED_TOKENS}-token limit without EOS")
    runtime_meta = _finish_runtime(runtime, args)
    speech = np.asarray(generated, dtype=np.int64)
    speech = np.concatenate((speech, np.full((3,), S3GEN_SIL, dtype=np.int64)))
    if speech.size < 4:
        raise RuntimeError("T3 produced no valid speech tokens before EOS")
    _save_token_artifact(
        stage,
        speech,
        {
            **token_config,
            "mode": "t3_generated_numpy_sampler",
            "t3_used": True,
            "generated_token_count": int(len(generated)),
            "raw_sampled_token_count": int(len(raw_generated)),
            "eos_seen": bool(eos_seen),
            "length_limit_reached": bool(hit_limit),
            "speech_token_count_with_silence": int(speech.size),
            "rng": "numpy.default_rng; not Torch-identical",
            "decode_seconds": decode_seconds,
            "prefill_sequence_length": int(embeds.shape[1]),
            "experimental_t3": True,
            "experimental_t3_limitation": EXPERIMENTAL_T3_LIMITATION,
            "runtime": runtime_meta,
        },
    )
    return {
        "mode": "t3_generated_numpy_sampler",
        "t3_used": True,
        "generated_token_count": int(len(generated)),
        "raw_sampled_token_count": int(len(raw_generated)),
        "eos_seen": bool(eos_seen),
        "length_limit_reached": bool(hit_limit),
        "speech_token_count_with_silence": int(speech.size),
        "prefill_sequence_length": int(embeds.shape[1]),
        "experimental_t3": True,
        "runtime": runtime_meta,
    }


def stage_t3(args: argparse.Namespace) -> dict[str, Any]:
    """Compatibility alias for older scripts that invoke the private T3 stage."""

    return _run_t3_generation(args)


def _load_speech_tokens(run_dir: Path) -> tuple[np.ndarray, dict[str, Any]]:
    stage = run_dir / "tokens"
    config = json.loads((stage / "config.json").read_text())
    tokens = _as_tokens(np.load(stage / "speech_tokens.npy", allow_pickle=False), name="speech_tokens")
    if np.any(tokens < 0) or np.any(tokens >= SPEECH_VOCAB):
        raise ValueError("speech tokens contain an out-of-range ID")
    return tokens, config


def stage_flow(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir).resolve()
    cond_config, cond = _load_prepared(run_dir)
    speech_tokens, token_config = _load_speech_tokens(run_dir)
    prompt_token = _as_tokens(_find_array(cond, "gen.prompt_token"), name="gen.prompt_token")
    full_tokens = np.concatenate((prompt_token, speech_tokens)).astype(np.int64, copy=False)
    if full_tokens.size < 1:
        raise ValueError("flow token sequence is empty")
    runtime = _make_runtime("flow_encoder", args)
    runtime_meta = _runtime_info(runtime, args)
    mu_btf, mask = runtime.encode(full_tokens[None, :], np.asarray([full_tokens.size], dtype=np.int64))
    runtime_meta = _finish_runtime(runtime, args)
    mu = np.asarray(mu_btf, dtype=np.float32).transpose(0, 2, 1)
    mask = np.asarray(mask, dtype=np.float32)
    prompt_feat = _normalise_prompt_feat(_find_array(cond, "gen.prompt_feat"))
    if prompt_feat.shape[1] > mu.shape[2]:
        raise ValueError(f"prompt feature length {prompt_feat.shape[1]} exceeds flow mel length {mu.shape[2]}")
    cond_mel = np.zeros_like(mu, dtype=np.float32)
    cond_mel[:, :, : prompt_feat.shape[1]] = prompt_feat.transpose(0, 2, 1)
    speaker = _as_batch_vector(_find_array(cond, "gen.embedding"), width=192, name="gen.embedding")
    if mask.shape != (1, 1, mu.shape[2]):
        raise ValueError(f"flow mask has unexpected shape {mask.shape}; expected (1,1,{mu.shape[2]})")
    flow_dir = _stage_dir(run_dir, "flow")
    _save_npz(
        flow_dir / "flow_inputs.npz",
        {
            "mu": mu,
            "mask": mask,
            "cond": cond_mel,
            "speaker_embedding": speaker,
            "full_tokens": full_tokens,
        },
    )
    _write_json(
        flow_dir / "config.json",
        {
            "schema_version": 1,
            "prompt_token_count": int(prompt_token.size),
            "speech_token_count": int(speech_tokens.size),
            "full_token_count": int(full_tokens.size),
            "prompt_feat_length": int(prompt_feat.shape[1]),
            "mel_length": int(mu.shape[2]),
            "token_mode": token_config.get("mode"),
            "mu_layout": "B,80,T",
            "mask_layout": "B,1,T",
            "runtime": runtime_meta,
        },
    )
    return {
        "prompt_token_count": int(prompt_token.size),
        "speech_token_count": int(speech_tokens.size),
        "mel_length": int(mu.shape[2]),
        "prompt_feat_length": int(prompt_feat.shape[1]),
        "runtime": runtime_meta,
    }


def stage_estimator(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir).resolve()
    flow_dir = run_dir / "flow"
    config = json.loads((flow_dir / "config.json").read_text())
    values = _load_npz(flow_dir / "flow_inputs.npz")
    mu = np.asarray(values["mu"], dtype=np.float32)
    mask = np.asarray(values["mask"], dtype=np.float32)
    cond = np.asarray(values["cond"], dtype=np.float32)
    speaker = _as_batch_vector(values["speaker_embedding"], width=192, name="gen.embedding")
    # S3Gen's upstream ``flow_inference`` samples noise for the complete
    # speech-token sequence passed to it.  That sequence includes the three
    # trailing S3GEN_SIL IDs, so keep those IDs in this length calculation.
    speech_token_count = int(config["speech_token_count"])
    if speech_token_count < 1:
        raise ValueError("flow stage has no speech tokens")
    if mu.ndim != 3 or mu.shape[1] != 80:
        raise ValueError(f"flow mu must have shape (1,80,T), got {mu.shape}")
    nsteps = int(args.steps)
    if nsteps != 2:
        raise ValueError("the staged meanflow pipeline currently requires exactly 2 Euler steps")
    rng = np.random.default_rng(int(args.seed))
    speech_noise = rng.normal(size=(1, 80, speech_token_count * 2)).astype(np.float32)
    if speech_noise.shape[2] > mu.shape[2]:
        raise ValueError("generated speech noise is longer than the full flow state")
    full_noise = rng.normal(size=mu.shape).astype(np.float32)
    noise_offset = mu.shape[2] - speech_noise.shape[2]
    # Upstream places supplied noise by tail length, but crops the final mel
    # by the actual reference feature length. Odd reference mel lengths can
    # differ by one frame from twice the reference token count (e.g. Harvey).
    prompt_len = int(config["prompt_feat_length"])
    full_noise[:, :, noise_offset:] = speech_noise
    runtime = _make_runtime("meanflow_estimator", args)
    runtime_meta = _runtime_info(runtime, args)
    x = full_noise
    times = np.linspace(0.0, 1.0, nsteps + 1, dtype=np.float32)
    for index in range(nsteps):
        t = np.asarray([times[index]], dtype=np.float32)
        r = np.asarray([times[index + 1]], dtype=np.float32)
        dxdt = runtime.estimate(x, mask, mu, t, speaker, cond, r)
        if dxdt.shape != x.shape or not np.isfinite(dxdt).all():
            raise RuntimeError(f"meanflow estimator returned invalid shape or values: {dxdt.shape}")
        x = (x + (r[0] - t[0]) * dxdt).astype(np.float32, copy=False)
    runtime_meta = _finish_runtime(runtime, args)
    mel = x[:, :, prompt_len:]
    if mel.shape[2] < 1 or not np.isfinite(mel).all():
        raise RuntimeError("meanflow output is empty or non-finite")
    estimator_dir = _stage_dir(run_dir, "estimator")
    _save_npz(estimator_dir / "mel.npz", {"mel": mel, "noise_speech": speech_noise, "noise_full": full_noise})
    _write_json(
        estimator_dir / "config.json",
        {
            "schema_version": 1,
            "steps": nsteps,
            "time_grid": times.tolist(),
            "seed": int(args.seed),
            "rng": "numpy.default_rng; explicit state and tail noise; not Torch-identical",
            "speech_token_count": speech_token_count,
            "generated_token_count": int(config.get("generated_token_count", max(0, speech_token_count - 3))),
            "prompt_feat_length": prompt_len,
            "noise_tail_offset": noise_offset,
            "mel_length": int(mel.shape[2]),
            "layout": "B,80,T",
            "cfg": "disabled; meanflow graph receives one conditional estimate per Euler step",
            "runtime": runtime_meta,
        },
    )
    return {
        "steps": nsteps,
        "mel_length": int(mel.shape[2]),
        "prompt_feat_length": prompt_len,
        "speech_token_count": speech_token_count,
        "seed": int(args.seed),
        "runtime": runtime_meta,
    }


def _mel_calibration_for_run(run_dir: Path, model_dir: Path) -> dict[str, Any]:
    """Reload and validate the profile before the final vocoder call."""

    profile_path = run_dir / "prepare" / "profile.json"
    if not profile_path.exists():
        return {
            "enabled": False,
            "applied": False,
            "report": None,
            "report_sha256": None,
            "delta_path": None,
            "delta_sha256": None,
            "strength": 0.0,
        }
    loaded_path, profile = _load_profile(path=profile_path, model_dir=model_dir)
    validation = _validate_profile_for_model(loaded_path, profile, model_dir)
    return dict(validation["mel_calibration"])


def stage_vocoder(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir).resolve()
    estimator_dir = run_dir / "estimator"
    values = _load_npz(estimator_dir / "mel.npz")
    mel = np.asarray(values["mel"], dtype=np.float32)
    if mel.ndim != 3 or mel.shape[0] != 1 or mel.shape[1] != 80:
        raise ValueError(f"mel must have shape (1,80,T), got {mel.shape}")
    rng = np.random.default_rng(int(args.seed))
    phase_noise = rng.uniform(-np.pi, np.pi, size=(1, 9, 1)).astype(np.float32)
    sine_noise = rng.normal(size=(1, 9, mel.shape[2] * 480)).astype(np.float32)
    mel_calibration = _mel_calibration_for_run(run_dir, Path(args.model_dir).resolve())
    if mel_calibration.get("enabled") and float(mel_calibration.get("strength", 0.0)) != 0.0:
        delta = np.asarray(mel_calibration.get("delta"), dtype=np.float32)
        if delta.shape != (80,):
            raise ValueError(f"mel calibration delta must have shape (80,), got {delta.shape}")
        # Keep estimator/mel.npz unchanged. Apply the correction only at the
        # final handoff to the vocoder, after all deterministic noise is made.
        mel = np.array(mel, dtype=np.float32, copy=True)
        mel += delta.reshape(1, 80, 1) * np.float32(mel_calibration["strength"])
        if not np.isfinite(mel).all():
            raise ValueError("mel calibration produced non-finite vocoder features")
        mel_calibration["applied"] = True
    else:
        mel_calibration["applied"] = False
    runtime = _make_runtime("vocoder", args)
    runtime_meta = _runtime_info(runtime, args)
    waveform = np.asarray(runtime.synthesize(mel, phase_noise=phase_noise, sine_noise=sine_noise), dtype=np.float32)
    runtime_meta = _finish_runtime(runtime, args)
    if waveform.ndim == 2 and waveform.shape[0] == 1:
        waveform = waveform[0]
    if waveform.ndim != 1 or waveform.size < 1 or not np.isfinite(waveform).all():
        raise RuntimeError(f"vocoder returned invalid waveform shape {waveform.shape}")
    vocoder_dir = _stage_dir(run_dir, "vocoder")
    _save_npz(vocoder_dir / "audio.npz", {"audio": waveform, "phase_noise": phase_noise, "sine_noise": sine_noise})
    _write_json(
        vocoder_dir / "config.json",
        {
            "schema_version": 1,
            "sample_rate": SAMPLE_RATE,
            "mel_length": int(mel.shape[2]),
            "audio_samples": int(waveform.size),
            "seed": int(args.seed),
            "phase_noise": "uniform[-pi,pi), shape (1,9,1)",
            "sine_noise": "normal, shape (1,9,mel_length*480)",
            "watermark": "not yet applied",
            "mel_calibration": _serialise_profile_validation({"mel_calibration": mel_calibration})["mel_calibration"],
            "runtime": runtime_meta,
        },
    )
    return {
        "audio_samples": int(waveform.size),
        "audio_seconds": float(waveform.size / SAMPLE_RATE),
        "mel_length": int(mel.shape[2]),
        "mel_calibration": _serialise_profile_validation({"mel_calibration": mel_calibration})["mel_calibration"],
        "runtime": runtime_meta,
    }


def _trim_fade(audio: np.ndarray) -> np.ndarray:
    value = np.asarray(audio, dtype=np.float32).copy()
    n_trim = SAMPLE_RATE // 50
    if value.size < 1:
        return value
    trim = np.zeros(2 * n_trim, dtype=np.float32)
    trim[n_trim:] = (np.cos(np.linspace(np.pi, 0.0, n_trim, dtype=np.float32)) + 1.0) / 2.0
    value[: min(value.size, trim.size)] *= trim[: min(value.size, trim.size)]
    return value


def _master_light(audio: np.ndarray, sr: int = SAMPLE_RATE) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the quality sweep delivery policy without importing that module.

    The high-pass, LUFS gain, true-peak ceiling, and endpoint fades match
    ``quality_sweep.master``.  Missing mastering dependencies are an explicit
    stage error.  The pipeline never silently ships an unmastered fallback.
    """

    value = np.asarray(audio, dtype=np.float32).copy()
    gain_db = 0.0
    achieved_lufs: float | None = None
    from scipy import signal
    import pyloudnorm as ln

    value = signal.sosfilt(signal.butter(2, 45, btype="highpass", fs=sr, output="sos"), value).astype(np.float32)
    loudness = float(ln.Meter(sr).integrated_loudness(value))
    gain_db = min(12.0, -19.0 - loudness) if np.isfinite(loudness) else 0.0
    true_peak = float(np.max(np.abs(signal.resample_poly(value, 4, 1))))
    gain_db = min(gain_db, -1.0 - 20.0 * np.log10(max(true_peak, 1e-9)))
    value *= np.float32(10.0 ** (gain_db / 20.0))
    fade = min(int(0.005 * sr), value.size // 2)
    if fade:
        value[:fade] *= np.linspace(0.0, 1.0, fade)
        value[-fade:] *= np.linspace(1.0, 0.0, fade)
    achieved_lufs = float(ln.Meter(sr).integrated_loudness(value))
    return value, {
        "highpass_hz": 45,
        "gain_db": float(gain_db),
        "target_lufs": -19,
        "achieved_lufs": achieved_lufs,
        "true_peak_ceiling_dbtp": -1,
        "true_peak_oversampling": 4,
        "noise_gate": False,
    }


def stage_finish(args: argparse.Namespace, *, watermarker: Any | None = None) -> dict[str, Any]:
    """Apply Perth, mastering, and WAV output for one prepared run.

    ``watermarker`` is injectable for the persistent batch worker. The default
    path still constructs one marker for this finish child, preserving the
    original staged launcher lifecycle.
    """

    run_dir = Path(args.run_dir).resolve()
    values = _load_npz(run_dir / "vocoder" / "audio.npz")
    raw = np.asarray(values["audio"], dtype=np.float32)
    if raw.ndim == 2 and raw.shape[0] == 1:
        raw = raw[0]
    raw = _trim_fade(raw)
    if not np.isfinite(raw).all() or raw.size < 1:
        raise ValueError("vocoder waveform is empty or non-finite")

    # Preserve the original Perth call. This import is isolated to the finish
    # child in the default path. The persistent worker passes one marker for
    # all cases, so construction cost is measured once and no ORT session is
    # recreated.
    if watermarker is None:
        import perth

        watermarker = perth.PerthImplicitWatermarker()
    import torch

    with torch.inference_mode():
        watermarked = watermarker.apply_watermark(raw, sample_rate=SAMPLE_RATE)
    audio = np.asarray(watermarked, dtype=np.float32)
    if audio.ndim == 2 and audio.shape[0] == 1:
        audio = audio[0]
    if audio.ndim != 1 or audio.size < 1 or not np.isfinite(audio).all():
        raise RuntimeError(f"Perth returned invalid waveform shape {audio.shape}")
    # The normal Perth package accepts NumPy, matching
    # ChatterboxTurboTTS.generate.  Do not silently ship an unwatermarked file.
    mastered, processing = _master_light(audio)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    import soundfile as sf

    sf.write(temporary, mastered, SAMPLE_RATE, subtype="PCM_24", format="WAV")
    os.replace(temporary, output)
    finish_dir = _stage_dir(run_dir, "finish")
    report = {
        "schema_version": 1,
        "output": str(output),
        "sample_rate": SAMPLE_RATE,
        "audio_samples": int(mastered.size),
        "audio_seconds": float(mastered.size / SAMPLE_RATE),
        "watermark": "PerthImplicitWatermarker",
        "watermark_applied": True,
        "trim_fade": {"n_trim": SAMPLE_RATE // 50, "fade_ms": 20},
        "master_processing": processing,
        "source": str(run_dir / "vocoder" / "audio.npz"),
    }
    _write_json(finish_dir / "config.json", report)
    return report


def _stage_report(run_dir: Path, stage_name: str, started: float, result: Mapping[str, Any] | None, error: BaseException | None) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": 1,
        "stage": stage_name,
        "started_unix": started,
        "finished_unix": _now(),
        "elapsed_seconds": max(0.0, _now() - started),
        "status": "error" if error else "ok",
        "rss": _rss_report(),
    }
    if result:
        report["result"] = dict(result)
    if error:
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
    _write_json(_stage_dir(run_dir, stage_name) / "stage_report.json", report)
    return report


def _dispatch_stage(args: argparse.Namespace) -> int:
    started = _now()
    run_dir = Path(args.run_dir).resolve()
    stage_name = args.command
    functions = {
        "prepare": stage_prepare,
        "tokens": stage_tokens,
        "t3": stage_t3,
        "flow": stage_flow,
        "estimator": stage_estimator,
        "vocoder": stage_vocoder,
        "finish": stage_finish,
    }
    try:
        result = functions[stage_name](args)
    except Exception as exc:
        report = _stage_report(run_dir, stage_name, started, None, exc)
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        return 1
    report = _stage_report(run_dir, stage_name, started, result, None)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


def _child_command(args: argparse.Namespace, command: str, run_dir: Path, extra: Sequence[str] = ()) -> list[str]:
    command_args = [sys.executable, str(Path(__file__).resolve()), command, "--run-dir", str(run_dir)]
    command_args.extend([
        "--model-dir", str(Path(args.model_dir).resolve()),
        "--ort-threads", str(args.ort_threads),
        "--ort-provider", str(args.ort_provider),
        "--cuda-device-id", str(args.cuda_device_id),
        "--gpu-mem-limit-mib", str(args.gpu_mem_limit_mib),
        "--arena-extend-strategy", str(args.arena_extend_strategy),
        "--cudnn-conv-algo-search", str(args.cudnn_conv_algo_search),
    ])
    if args.do_copy_in_default_stream:
        command_args.append("--do-copy-in-default-stream")
    else:
        command_args.append("--no-copy-in-default-stream")
    if args.ort_profile:
        command_args.append("--ort-profile")
    if args.cuda_kv_resident:
        command_args.append("--cuda-kv-resident")
    if command in {"prepare", "tokens"}:
        command_args.extend(["--voice", str(args.voice)])
    if command == "tokens":
        if args.text is not None:
            command_args.extend(["--text", str(args.text)])
        if args.text_file:
            command_args.extend(["--text-file", str(Path(args.text_file).resolve())])
        if args.tokenizer_dir:
            command_args.extend(["--tokenizer-dir", str(Path(args.tokenizer_dir).resolve())])
        command_args.extend(["--seed", str(args.seed)])
        if args.tokens_file:
            command_args.extend(["--tokens-file", str(Path(args.tokens_file).resolve())])
        if args.experimental_t3:
            command_args.append("--experimental-t3")
    if command == "t3":
        command_args.extend(["--seed", str(args.seed)])
    if command == "estimator":
        command_args.extend(["--seed", str(args.seed), "--steps", str(args.steps)])
    if command == "vocoder":
        command_args.extend(["--seed", str(args.seed)])
    if command == "finish":
        command_args.extend(["--output", str(Path(args.output).resolve())])
    command_args.extend(extra)
    return command_args


def _read_stage_report(run_dir: Path, stage: str) -> dict[str, Any]:
    path = run_dir / stage / "stage_report.json"
    if not path.exists():
        raise RuntimeError(f"child stage {stage} did not write a report: {path}")
    return json.loads(path.read_text())


def _summarise_ort_profile(path: str | os.PathLike[str] | None) -> dict[str, Any]:
    """Summarise provider kernel evidence after an ORT child exits.

    ORT writes a JSON event list when ``SessionOptions.enable_profiling`` is
    enabled.  Parsing happens in the lightweight pipeline parent, never in the
    model child, so a large profile cannot extend the stage's model RSS peak.
    """

    if not path:
        return {"status": "not_requested", "profile_path": None}
    profile_path = Path(path).expanduser().resolve()
    result: dict[str, Any] = {
        "status": "pending",
        "profile_path": str(profile_path),
        "provider_event_counts": {},
        "provider_duration_us": {},
        "cuda_kernel_count": 0,
        "cuda_duration_us": 0.0,
    }
    if not profile_path.exists():
        result.update(status="missing", error="ORT profile file was not written")
        return result
    try:
        payload = json.loads(profile_path.read_text())
        events = payload if isinstance(payload, list) else payload.get("events", [])
        if not isinstance(events, list):
            raise ValueError("profile JSON does not contain an event list")
    except Exception as exc:
        result.update(status="invalid", error=f"{type(exc).__name__}: {exc}")
        return result
    counts: dict[str, int] = {}
    durations: dict[str, float] = {}
    cuda_count = 0
    cuda_duration = 0.0
    for event in events:
        if not isinstance(event, Mapping):
            continue
        args = event.get("args") if isinstance(event.get("args"), Mapping) else {}
        provider = (
            args.get("provider")
            or args.get("execution_provider")
            or args.get("ExecutionProvider")
            or event.get("provider")
        )
        name = str(event.get("name", ""))
        if provider is None:
            if "CUDAExecutionProvider" in name:
                provider = "CUDAExecutionProvider"
            elif "CPUExecutionProvider" in name:
                provider = "CPUExecutionProvider"
        if provider is None:
            continue
        provider = str(provider)
        duration = float(event.get("dur", 0.0) or 0.0)
        counts[provider] = counts.get(provider, 0) + 1
        durations[provider] = durations.get(provider, 0.0) + duration
        if provider == "CUDAExecutionProvider":
            # Node/kernel events have provider metadata; session bookkeeping
            # does not.  The conservative count treats only named Node events
            # or provider-suffixed kernel events as actual kernels.
            category = str(event.get("cat", ""))
            if category.lower() == "node" or "_kernel_time" in name:
                cuda_count += 1
                cuda_duration += duration
    result.update(
        status="ok",
        event_count=len(events),
        provider_event_counts=counts,
        provider_duration_us=durations,
        cuda_kernel_count=cuda_count,
        cuda_duration_us=cuda_duration,
        cuda_used=bool(cuda_count > 0),
        profile_bytes=profile_path.stat().st_size,
    )
    return result


def _run_pipeline(args: argparse.Namespace) -> int:
    if not args.experimental_t3:
        raise SystemExit("run requires --experimental-t3 because the long T3 cache reference is not fully verified")
    if args.cuda_kv_resident and args.ort_provider != "cuda":
        raise SystemExit("--cuda-kv-resident requires --ort-provider cuda")
    run_root = Path(args.output_root).expanduser().resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else run_root / f"{args.voice}_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    stages = ["prepare", "tokens", "flow", "estimator", "vocoder", "finish"]
    reports: list[dict[str, Any]] = []
    tree_rss_peak = _tree_rss_bytes(os.getpid())
    overall_started = _now()
    run_meta: dict[str, Any] = {
        "schema_version": 1,
        "voice": str(args.voice),
        "run_dir": str(run_dir),
        "model_dir": str(Path(args.model_dir).resolve()),
        "pipeline_source_sha256": _sha256(Path(__file__)),
        "python_executable": sys.executable,
        "seed": int(args.seed),
        "model_manifest_sha256": {
            name: _sha256(Path(args.model_dir) / name / "manifest.json")
            for name in ("t3", "flow_encoder", "meanflow_estimator", "vocoder")
        },
        "experimental_t3": True,
        "experimental_t3_limitation": EXPERIMENTAL_T3_LIMITATION,
        "ort_provider": args.ort_provider,
        "ort_provider_options": {
            "cuda_device_id": args.cuda_device_id,
            "gpu_mem_limit_mib": args.gpu_mem_limit_mib,
            "arena_extend_strategy": args.arena_extend_strategy,
            "cudnn_conv_algo_search": args.cudnn_conv_algo_search,
            "do_copy_in_default_stream": args.do_copy_in_default_stream,
        } if args.ort_provider == "cuda" else None,
        "ort_profile": bool(args.ort_profile),
        "cuda_kv_resident": bool(args.cuda_kv_resident),
        "ort_profiles": [],
        "supplied_tokens_mode": bool(args.tokens_file),
        "numpy_sampler": "not Torch-identical",
        "stages": [],
        "status": "running",
    }
    _write_json(run_dir / "run.json", run_meta)
    try:
        # Preparation and tokenisation are separate child processes so Torch
        # and Transformers allocations are released before ORT is created.
        for command in stages:
            child_args = _child_command(args, command, run_dir)
            started = _now()
            stage_tree_rss_peak = _tree_rss_bytes(os.getpid())
            with subprocess.Popen(child_args, cwd=str(ROOT)) as child:
                while child.poll() is None:
                    stage_tree_rss_peak = max(stage_tree_rss_peak, _tree_rss_bytes(os.getpid()))
                    time.sleep(0.1)
                returncode = child.returncode
            tree_rss_peak = max(tree_rss_peak, stage_tree_rss_peak)
            child_report = _read_stage_report(run_dir, command)
            child_report["parent_elapsed_seconds"] = _now() - started
            child_report["returncode"] = int(returncode)
            child_report["sampled_peak_tree_rss_bytes"] = stage_tree_rss_peak
            runtime_meta = (child_report.get("result") or {}).get("runtime") or {}
            profile_meta = (runtime_meta.get("profiling") or {}) if isinstance(runtime_meta, Mapping) else {}
            profile_summary = _summarise_ort_profile(profile_meta.get("profile_path"))
            child_report["ort_profile"] = profile_summary
            if args.ort_profile:
                run_meta["ort_profiles"].append({"stage": command, **profile_summary})
                if returncode == 0 and child_report.get("status") == "ok" and args.ort_provider == "cuda" and command in {"tokens", "flow", "estimator", "vocoder"}:
                    if profile_summary.get("status") != "ok" or int(profile_summary.get("cuda_kernel_count", 0)) <= 0:
                        raise RuntimeError(
                            f"CUDA profiling for stage {command} did not show a CUDA kernel: {profile_summary}"
                        )
            reports.append(child_report)
            run_meta["stages"] = reports
            run_meta["sampled_peak_tree_rss_bytes"] = tree_rss_peak
            run_meta["rss_sample_interval_seconds"] = 0.1
            _write_json(run_dir / "run.json", run_meta)
            if returncode != 0 or child_report.get("status") != "ok":
                raise RuntimeError(f"stage {command} failed with return code {returncode}")
        run_meta["status"] = "ok"
        run_meta["finished_unix"] = _now()
        run_meta["elapsed_seconds"] = _now() - overall_started
        finish_report = _read_stage_report(run_dir, "finish")
        output_seconds = float((finish_report.get("result") or {}).get("audio_seconds", 0.0))
        run_meta["rtf"] = run_meta["elapsed_seconds"] / output_seconds if output_seconds > 0 else None
        run_meta["output"] = str(Path(args.output).resolve())
    except Exception as exc:
        run_meta["status"] = "error"
        run_meta["error"] = f"{type(exc).__name__}: {exc}"
        run_meta["traceback"] = traceback.format_exc()
        run_meta["finished_unix"] = _now()
        run_meta["elapsed_seconds"] = _now() - overall_started
        _write_json(run_dir / "run.json", run_meta)
        print(json.dumps(run_meta, indent=2, sort_keys=True), flush=True)
        return 1
    _write_json(run_dir / "run.json", run_meta)
    print(json.dumps(run_meta, indent=2, sort_keys=True), flush=True)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--run-dir", type=Path, required=True)
    common.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    common.add_argument("--ort-threads", type=int, default=1)
    common.add_argument("--ort-provider", choices=("cpu", "cuda"), default="cpu")
    common.add_argument("--cuda-device-id", type=int, default=0)
    common.add_argument("--gpu-mem-limit-mib", type=int, default=2048)
    common.add_argument("--arena-extend-strategy", default="kSameAsRequested")
    common.add_argument("--cudnn-conv-algo-search", default="HEURISTIC")
    common.add_argument("--ort-profile", action="store_true", help="write an ORT JSON profile for this stage")
    common.add_argument("--cuda-kv-resident", action="store_true", help="keep T3 KV cache in CUDA OrtValues (CUDA only)")
    copy_group = common.add_mutually_exclusive_group()
    copy_group.add_argument("--do-copy-in-default-stream", dest="do_copy_in_default_stream", action="store_true", default=True)
    copy_group.add_argument("--no-copy-in-default-stream", dest="do_copy_in_default_stream", action="store_false")

    prep = sub.add_parser("prepare", parents=[common])
    prep.add_argument("--voice", default="asmr_conversational")

    tok = sub.add_parser("tokens", parents=[common])
    tok.add_argument("--voice", default="asmr_conversational")
    tok.add_argument("--text")
    tok.add_argument("--text-file", type=Path)
    tok.add_argument("--tokenizer-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    tok.add_argument("--tokens-file", type=Path)
    tok.add_argument("--seed", type=int, default=31)
    tok.add_argument("--experimental-t3", action="store_true")

    t3 = sub.add_parser("t3", parents=[common])
    t3.add_argument("--seed", type=int, default=31)

    flow = sub.add_parser("flow", parents=[common])

    est = sub.add_parser("estimator", parents=[common])
    est.add_argument("--seed", type=int, default=10031)
    est.add_argument("--steps", type=int, default=2)

    voc = sub.add_parser("vocoder", parents=[common])
    voc.add_argument("--seed", type=int, default=20031)

    fin = sub.add_parser("finish", parents=[common])
    fin.add_argument("--output", type=Path, required=True)

    run = sub.add_parser("run")
    run.add_argument("--voice", default="asmr_conversational")
    run.add_argument("--text")
    run.add_argument("--text-file", type=Path)
    run.add_argument("--tokens-file", type=Path, help=".npy speech IDs with three trailing S3GEN_SIL IDs")
    run.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    run.add_argument("--tokenizer-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    run.add_argument("--run-dir", type=Path)
    run.add_argument("--seed", type=int, default=31)
    run.add_argument("--steps", type=int, default=2)
    run.add_argument("--ort-threads", type=int, default=1)
    run.add_argument("--ort-provider", choices=("cpu", "cuda"), default="cpu")
    run.add_argument("--cuda-device-id", type=int, default=0)
    run.add_argument("--gpu-mem-limit-mib", type=int, default=2048)
    run.add_argument("--arena-extend-strategy", default="kSameAsRequested")
    run.add_argument("--cudnn-conv-algo-search", default="HEURISTIC")
    run.add_argument("--ort-profile", action="store_true", help="write and summarise ORT JSON profiles per stage")
    run.add_argument("--cuda-kv-resident", action="store_true", help="keep T3 KV cache in CUDA OrtValues (CUDA only)")
    copy_group = run.add_mutually_exclusive_group()
    copy_group.add_argument("--do-copy-in-default-stream", dest="do_copy_in_default_stream", action="store_true", default=True)
    copy_group.add_argument("--no-copy-in-default-stream", dest="do_copy_in_default_stream", action="store_false")
    run.add_argument("--experimental-t3", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    if args.command == "run":
        return _run_pipeline(args)
    return _dispatch_stage(args)


if __name__ == "__main__":
    raise SystemExit(main())
