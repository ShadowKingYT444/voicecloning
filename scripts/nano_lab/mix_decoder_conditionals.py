"""Mix saved S3Gen decoder conditionals without loading Chatterbox models.

This is a controlled inference experiment.  It combines two previously
prepared ``Conditionals.save`` files and writes a new file.  The T3 branch is
copied from the base file byte-for-byte at the tensor level.  The default
``embedding`` mode only changes ``gen.embedding``.  ``prompt`` changes the
decoder prompt token/features and their lengths.  ``all_gen`` copies the full
decoder branch from the donor.

The module keeps Torch imports lazy.  The NumPy interpolation helper is safe
to use from lightweight tests and does not construct a model or import
Chatterbox.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
GEN_EMBEDDING_DIM = 192
PROMPT_TOKEN_KEYS = ("prompt_token", "prompt_token_len")
PROMPT_FEATURE_KEYS = ("prompt_feat", "prompt_feat_len")
PROMPT_KEYS = PROMPT_TOKEN_KEYS + PROMPT_FEATURE_KEYS
MODES = ("embedding", "prompt", "all_gen")


def _as_finite_embedding(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (1, GEN_EMBEDDING_DIM):
        raise ValueError(f"{name} must have shape (1, {GEN_EMBEDDING_DIM}), got {array.shape}")
    if not np.issubdtype(array.dtype, np.floating):
        raise ValueError(f"{name} must be floating point, got {array.dtype}")
    array = np.asarray(array, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def interpolate_embedding_numpy(
    base: Any,
    donor: Any,
    strength: float,
    *,
    zero_epsilon: float = 1e-8,
) -> np.ndarray:
    """Interpolate two 192-D embeddings on the unit sphere.

    Interior strengths interpolate normalized vectors and rescale the result
    to the base embedding's original L2 norm.  Exact endpoint strengths return
    independent copies of the original arrays, preserving their values and
    dtype semantics for reproducible no-op and full-donor controls.
    """

    alpha = float(strength)
    if not math.isfinite(alpha) or alpha < 0.0 or alpha > 1.0:
        raise ValueError("strength must be finite and in [0, 1]")
    base_array = np.asarray(base)
    donor_array = np.asarray(donor)
    base_float = _as_finite_embedding(base_array, name="base embedding")
    donor_float = _as_finite_embedding(donor_array, name="donor embedding")
    if alpha == 0.0:
        return np.array(base_array, copy=True)
    if alpha == 1.0:
        return np.array(donor_array, copy=True)
    base_norm = float(np.linalg.norm(base_float))
    donor_norm = float(np.linalg.norm(donor_float))
    if base_norm <= float(zero_epsilon) or donor_norm <= float(zero_epsilon):
        raise ValueError("base and donor embeddings must have nonzero norm")
    mixed = (1.0 - alpha) * (base_float / base_norm) + alpha * (donor_float / donor_norm)
    mixed_norm = float(np.linalg.norm(mixed))
    if mixed_norm <= float(zero_epsilon):
        raise ValueError("interpolated embeddings are nearly opposed; choose a different strength")
    result = mixed / mixed_norm * base_norm
    # Keep the base dtype for a drop-in conditionals file.  Endpoint branches
    # already return exact independent copies above.
    return result.astype(base_array.dtype, copy=False)


def _require_strength(strength: float, mode: str) -> float:
    alpha = float(strength)
    if not math.isfinite(alpha) or alpha < 0.0 or alpha > 1.0:
        raise ValueError("strength must be finite and in [0, 1]")
    if mode in {"prompt", "all_gen"} and not math.isclose(alpha, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"mode={mode!r} requires strength=1 so the donor prompt is unambiguous")
    return alpha


def _import_torch():
    import torch

    return torch


def _is_tensor(value: Any, torch: Any) -> bool:
    return bool(torch.is_tensor(value))


def _clone(value: Any, torch: Any) -> Any:
    if _is_tensor(value, torch):
        return value.detach().clone().cpu()
    if isinstance(value, Mapping):
        return {key: _clone(item, torch) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone(item, torch) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone(item, torch) for item in value)
    return value


def _exact_equal(left: Any, right: Any, torch: Any) -> bool:
    if _is_tensor(left, torch) or _is_tensor(right, torch):
        return bool(_is_tensor(left, torch) and _is_tensor(right, torch) and torch.equal(left, right))
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping) or set(left) != set(right):
            return False
        return all(_exact_equal(left[key], right[key], torch) for key in left)
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)) or len(left) != len(right):
            return False
        return all(_exact_equal(a, b, torch) for a, b in zip(left, right))
    return left == right


def _validate_tensor_finite(value: Any, *, name: str, torch: Any) -> None:
    if not _is_tensor(value, torch):
        return
    if value.is_floating_point() and not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains non-finite values")


def _validate_embedding(value: Any, *, name: str, torch: Any) -> None:
    if not _is_tensor(value, torch):
        raise ValueError(f"{name} must be a Torch tensor")
    if tuple(value.shape) != (1, GEN_EMBEDDING_DIM):
        raise ValueError(f"{name} must have shape (1, {GEN_EMBEDDING_DIM}), got {tuple(value.shape)}")
    if not value.is_floating_point():
        raise ValueError(f"{name} must be floating point")
    _validate_tensor_finite(value, name=name, torch=torch)
    if float(value.detach().float().norm().item()) <= 1e-8:
        raise ValueError(f"{name} must have nonzero norm")


def _validate_prompt(gen: Mapping[str, Any], *, name: str, torch: Any) -> None:
    missing = [key for key in PROMPT_KEYS if key not in gen]
    if missing:
        raise ValueError(f"{name} is missing decoder prompt fields: {', '.join(missing)}")
    tokens = gen["prompt_token"]
    token_len = gen["prompt_token_len"]
    feat = gen["prompt_feat"]
    feat_len = gen["prompt_feat_len"]
    if not _is_tensor(tokens, torch) or tokens.ndim != 2 or int(tokens.shape[0]) != 1:
        raise ValueError(f"{name}.prompt_token must have shape (1, T)")
    if tokens.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError(f"{name}.prompt_token must be an integer tensor")
    if not _is_tensor(token_len, torch) or token_len.ndim != 1 or tuple(token_len.shape) != (1,):
        raise ValueError(f"{name}.prompt_token_len must have shape (1,)")
    if token_len.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError(f"{name}.prompt_token_len must be an integer tensor")
    token_count = int(token_len.detach().cpu().item())
    if token_count < 1 or token_count > int(tokens.shape[1]):
        raise ValueError(f"{name}.prompt_token_len is outside prompt_token: {token_count}")
    if not _is_tensor(feat, torch) or feat.ndim != 3 or tuple(feat.shape[:1]) != (1,) or int(feat.shape[2]) != 80:
        raise ValueError(f"{name}.prompt_feat must have shape (1, T, 80)")
    _validate_tensor_finite(feat, name=f"{name}.prompt_feat", torch=torch)
    if feat_len is not None:
        if not _is_tensor(feat_len, torch) or feat_len.ndim != 1 or tuple(feat_len.shape) != (1,):
            raise ValueError(f"{name}.prompt_feat_len must be None or shape (1,)")
        if feat_len.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise ValueError(f"{name}.prompt_feat_len must be an integer tensor")
        feat_count = int(feat_len.detach().cpu().item())
        if feat_count < 1 or feat_count > int(feat.shape[1]):
            raise ValueError(f"{name}.prompt_feat_len is outside prompt_feat: {feat_count}")
    _validate_tensor_finite(tokens, name=f"{name}.prompt_token", torch=torch)
    _validate_tensor_finite(token_len, name=f"{name}.prompt_token_len", torch=torch)


def _validate_conditionals(value: Any, *, name: str, torch: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a dictionary saved by Conditionals.save")
    if set(value) != {"t3", "gen"}:
        raise ValueError(f"{name} must contain exactly t3 and gen branches")
    t3 = value["t3"]
    gen = value["gen"]
    if not isinstance(t3, Mapping) or not isinstance(gen, Mapping):
        raise ValueError(f"{name}.t3 and {name}.gen must be dictionaries")
    if "speaker_emb" not in t3:
        raise ValueError(f"{name}.t3 is missing speaker_emb")
    if not _is_tensor(t3["speaker_emb"], torch):
        raise ValueError(f"{name}.t3.speaker_emb must be a tensor")
    _validate_tensor_finite(t3["speaker_emb"], name=f"{name}.t3.speaker_emb", torch=torch)
    if "embedding" not in gen:
        raise ValueError(f"{name}.gen is missing embedding")
    _validate_embedding(gen["embedding"], name=f"{name}.gen.embedding", torch=torch)
    _validate_prompt(gen, name=f"{name}.gen", torch=torch)
    for key, item in t3.items():
        _validate_tensor_finite(item, name=f"{name}.t3.{key}", torch=torch)
    for key, item in gen.items():
        _validate_tensor_finite(item, name=f"{name}.gen.{key}", torch=torch)
    return {"t3": dict(t3), "gen": dict(gen)}


def _load_conditionals(path: Path) -> dict[str, Any]:
    torch = _import_torch()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(path)
    # weights_only=True prevents arbitrary Python object reconstruction.  The
    # saved Conditionals contract is a plain dictionary of tensors and None.
    value = torch.load(str(path), map_location="cpu", weights_only=True)
    return _validate_conditionals(value, name=str(path), torch=torch)


def _embedding_from_strength(base: Any, donor: Any, strength: float, torch: Any) -> Any:
    if float(strength) == 0.0:
        return base.detach().clone().cpu()
    if float(strength) == 1.0:
        return donor.detach().clone().cpu()
    base_np = base.detach().cpu().numpy()
    donor_np = donor.detach().cpu().numpy()
    mixed_np = interpolate_embedding_numpy(base_np, donor_np, strength)
    return torch.from_numpy(np.asarray(mixed_np)).to(dtype=base.dtype)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_value(value: Any, torch: Any) -> dict[str, Any]:
    if _is_tensor(value, torch):
        tensor = value.detach().cpu().contiguous()
        try:
            raw = tensor.numpy().tobytes(order="C")
        except TypeError:
            # NumPy has no native bfloat16 scalar type.  Hash its exact bytes
            # through a uint8 view instead of silently converting the tensor.
            raw = tensor.view(torch.uint8).numpy().tobytes(order="C")
        digest = hashlib.sha256()
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(raw)
        return {"kind": "tensor", "dtype": str(tensor.dtype), "shape": list(tensor.shape), "sha256": digest.hexdigest()}
    if value is None:
        return {"kind": "none", "sha256": hashlib.sha256(b"null").hexdigest()}
    if isinstance(value, Mapping):
        children = {str(key): _hash_value(item, torch) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
        raw = json.dumps(children, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return {"kind": "mapping", "keys": sorted(children), "sha256": hashlib.sha256(raw).hexdigest(), "children": children}
    if isinstance(value, (list, tuple)):
        children = [_hash_value(item, torch) for item in value]
        raw = json.dumps(children, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return {"kind": "sequence", "length": len(children), "sha256": hashlib.sha256(raw).hexdigest(), "children": children}
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"kind": type(value).__name__, "sha256": hashlib.sha256(raw).hexdigest(), "value": value}


def _resolve_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (ROOT / path)


def mix_conditionals(
    base_cache: str | os.PathLike[str],
    donor_cache: str | os.PathLike[str],
    output: str | os.PathLike[str],
    *,
    strength: float = 1.0,
    mode: str = "embedding",
    report: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Create one provenance-recorded mixed conditionals file."""

    torch = _import_torch()
    torch.set_num_threads(2)
    mode = str(mode).strip().lower()
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    alpha = _require_strength(strength, mode)
    base_path = _resolve_path(base_cache).resolve()
    donor_path = _resolve_path(donor_cache).resolve()
    output_path = _resolve_path(output).resolve()
    report_path = (_resolve_path(report).resolve() if report is not None else output_path.with_suffix(".report.json"))
    if output_path in {base_path, donor_path}:
        raise ValueError("output must be a new path; refusing to overwrite a source cache")
    if report_path in {base_path, donor_path, output_path}:
        raise ValueError("report must be a new path distinct from source and output caches")
    if output_path.exists():
        raise FileExistsError(f"output already exists; refusing overwrite: {output_path}")
    if report_path.exists():
        raise FileExistsError(f"report already exists; refusing overwrite: {report_path}")

    base = _load_conditionals(base_path)
    donor = _load_conditionals(donor_path)
    output_state = _clone(base, torch)
    base_embedding = base["gen"]["embedding"]
    donor_embedding = donor["gen"]["embedding"]
    if mode == "embedding":
        output_state["gen"]["embedding"] = _embedding_from_strength(base_embedding, donor_embedding, alpha, torch)
    elif mode == "prompt":
        for key in PROMPT_KEYS:
            output_state["gen"][key] = _clone(donor["gen"][key], torch)
    else:
        output_state["gen"] = _clone(donor["gen"], torch)

    if not _exact_equal(output_state["t3"], base["t3"], torch):
        raise RuntimeError("internal error: output T3 branch changed")
    if mode == "embedding" and not all(
        _exact_equal(output_state["gen"][key], base["gen"][key], torch)
        for key in output_state["gen"]
        if key != "embedding"
    ):
        raise RuntimeError("internal error: embedding mode changed a non-embedding gen field")
    if mode == "prompt" and not _exact_equal(output_state["gen"]["embedding"], base_embedding, torch):
        raise RuntimeError("internal error: prompt mode changed the base embedding")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    torch.save(output_state, temporary)
    os.replace(temporary, output_path)
    report_payload: dict[str, Any] = {
        "format": "nano_decoder_conditionals_mix_report_v1",
        "status": "created",
        "mode": mode,
        "strength": alpha,
        "fit": False,
        "quality_claim": "No general quality claim; this is a controlled conditionals mix.",
        "base_cache": str(base_path),
        "base_cache_sha256": _sha256_file(base_path),
        "donor_cache": str(donor_path),
        "donor_cache_sha256": _sha256_file(donor_path),
        "output": str(output_path),
        "output_sha256": _sha256_file(output_path),
        "report": str(report_path),
        "same_t3_exact": True,
        "same_t3_proof": {
            "output_matches_base": _exact_equal(output_state["t3"], base["t3"], torch),
            "base_vs_donor": _exact_equal(base["t3"], donor["t3"], torch),
        },
        "tensor_hashes": {
            "base": _hash_value(base, torch),
            "donor": _hash_value(donor, torch),
            "output": _hash_value(output_state, torch),
        },
        "embedding_contract": {
            "shape": [1, GEN_EMBEDDING_DIM],
            "interior": "normalized linear interpolation, then rescale to base L2 norm",
            "endpoint_zero": "exact independent copy of base embedding",
            "endpoint_one": "exact independent copy of donor embedding",
        },
        "prompt_contract": {
            "fields": list(PROMPT_KEYS),
            "prompt_and_feature_lengths_copied_together": True,
        },
    }
    report_temporary = report_path.with_name(f".{report_path.name}.{os.getpid()}.tmp")
    report_temporary.write_text(json.dumps(report_payload, indent=2, sort_keys=True) + "\n")
    os.replace(report_temporary, report_path)
    return report_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-cache", type=Path, required=True)
    parser.add_argument("--donor-cache", type=Path, required=True)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--mode", choices=MODES, default="embedding")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = mix_conditionals(
        args.base_cache,
        args.donor_cache,
        args.output,
        strength=args.strength,
        mode=args.mode,
        report=args.report,
    )
    print(json.dumps({"status": payload["status"], "output": payload["output"], "report": payload["report"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
