"""Fold a fitted S3Gen decoder final projection into a Nano model.

The adapter is a small, provenance-recorded LoRA factor for
``flow.decoder.estimator.final_proj.weight``.  This module applies the merged
delta directly to that weight, so inference has no extra module or call.  The
original CPU weight is retained once per projection module and every switch
starts from that original tensor.  No adapter deltas can accumulate.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
_STATE_ATTR = "_nano_decoder_projection_state"
_FORMAT = "nano_decoder_projection_lora_v1"
_TARGET_KEY = "flow.decoder.estimator.final_proj.weight"
_TARGET_SHAPE = (80, 256, 1)
_RANK = 2
_ALPHA = 2.0
_FILE_HASH_CACHE: dict[tuple[str, int, int, int, int, int], str] = {}
_FILE_HASH_CACHE_LIMIT = 64


def _file_identity(path: Path) -> tuple[str, int, int, int, int, int]:
    stat = path.stat()
    return (str(path.resolve()), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _sha256_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    identity = _file_identity(path)
    cached = _FILE_HASH_CACHE.get(identity)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    after = _file_identity(path)
    if after != identity:
        raise RuntimeError(f"file changed while hashing: {path}")
    result = digest.hexdigest()
    if len(_FILE_HASH_CACHE) >= _FILE_HASH_CACHE_LIMIT:
        _FILE_HASH_CACHE.pop(next(iter(_FILE_HASH_CACHE)))
    _FILE_HASH_CACHE[identity] = result
    return result


def _sha256_tensor_fp32(torch: object, tensor: object) -> str:
    """Hash contiguous CPU fp32 tensor bytes, independent of model device."""

    value = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()  # type: ignore[attr-defined]
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def _hash_field(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"decoder projection {name} must be a 64-character hexadecimal SHA-256 string")
    normalised = value.lower()
    if any(character not in "0123456789abcdef" for character in normalised):
        raise ValueError(f"decoder projection {name} must be a 64-character hexadecimal SHA-256 string")
    return normalised


def _resolve_path(value: str | Path, *, base: Path = ROOT) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _projection_module(model: object) -> object:
    try:
        return model.s3gen.flow.decoder.estimator.final_proj  # type: ignore[attr-defined]
    except AttributeError as exc:
        raise ValueError(f"model has no {_TARGET_KEY} module") from exc


def _validate_scalar(value: object, *, name: str, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"decoder projection {name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"decoder projection {name} must be finite") from exc
    if not math.isfinite(result) or (positive and result <= 0.0):
        suffix = " and positive" if positive else ""
        raise ValueError(f"decoder projection {name} must be finite{suffix}")
    return result


def _as_factor(torch: object, value: object, *, name: str, shape: tuple[int, ...]) -> object:
    try:
        tensor = torch.as_tensor(value, dtype=torch.float32, device="cpu")  # type: ignore[attr-defined]
    except Exception as exc:
        raise ValueError(f"decoder projection {name} is not a tensor") from exc
    if tuple(tensor.shape) != shape:  # type: ignore[attr-defined]
        raise ValueError(f"decoder projection {name} must have shape {shape}, got {tuple(tensor.shape)}")  # type: ignore[attr-defined]
    if not bool(torch.isfinite(tensor).all()):  # type: ignore[attr-defined]
        raise ValueError(f"decoder projection {name} contains non-finite values")
    return tensor


def _load_adapter(torch: object, path: Path) -> tuple[dict[str, object], object, object, object]:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"decoder projection adapter not found: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)  # type: ignore[attr-defined]
    except Exception as exc:
        raise ValueError(f"could not load decoder projection adapter: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("decoder projection adapter must be a dictionary")
    if payload.get("format") != _FORMAT:
        raise ValueError(f"unsupported decoder projection format: {payload.get('format')!r}")
    required = (
        "down_weight",
        "up_weight",
        "merged_delta_weight",
        "base_target_weight",
        "target_weight_key",
        "target_shape",
        "rank",
        "alpha",
        "scale",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"decoder projection adapter is missing required fields: {missing}")
    if payload.get("target_weight_key") != _TARGET_KEY:
        raise ValueError(f"decoder projection target_weight_key must be {_TARGET_KEY!r}")
    target_shape = tuple(payload.get("target_shape") or ())
    if target_shape != _TARGET_SHAPE:
        raise ValueError(f"decoder projection target_shape must be {_TARGET_SHAPE}, got {target_shape}")
    rank_value = payload.get("rank")
    if isinstance(rank_value, bool):
        raise ValueError("decoder projection rank must be a positive integer")
    try:
        rank = int(rank_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("decoder projection rank must be a positive integer") from exc
    if rank != _RANK or float(rank) != float(rank_value):
        raise ValueError("decoder projection checkpoint requires rank 2")
    alpha = _validate_scalar(payload.get("alpha"), name="alpha", positive=True)
    if not math.isclose(alpha, _ALPHA, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError("decoder projection checkpoint requires alpha=2")
    scale = _validate_scalar(payload.get("scale"), name="scale", positive=True)
    # The fit format stores the training scale in the merged factor itself.
    # Keep it in the provenance record, but reject a missing/non-unit scale so
    # a caller cannot silently apply a differently scaled checkpoint.
    if not math.isclose(scale, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("decoder projection scale must be exactly 1.0 for this runtime")
    down = _as_factor(torch, payload["down_weight"], name="down_weight", shape=(rank, 256, 1))
    up = _as_factor(torch, payload["up_weight"], name="up_weight", shape=(80, rank, 1))
    delta = _as_factor(torch, payload["merged_delta_weight"], name="merged_delta_weight", shape=_TARGET_SHAPE)
    base_target = _as_factor(torch, payload["base_target_weight"], name="base_target_weight", shape=_TARGET_SHAPE)
    expected = (alpha / float(rank)) * torch.matmul(up[:, :, 0], down[:, :, 0]).unsqueeze(-1)  # type: ignore[attr-defined]
    if not bool(torch.allclose(delta, expected, rtol=0.0, atol=1e-7)):  # type: ignore[attr-defined]
        # Match the fit-time fp32 validator.  Small serialization roundoff is
        # acceptable, but a meaningful factor mismatch is rejected.
        raise ValueError("decoder projection merged_delta_weight does not match alpha/r * up_weight @ down_weight")
    payload_base_hash = _hash_field(payload.get("base_target_weight_sha256"), name="base_target_weight_sha256")
    if payload_base_hash is None:
        raise ValueError("decoder projection base_target_weight_sha256 is required")
    if _sha256_tensor_fp32(torch, base_target) != payload_base_hash:
        raise ValueError("decoder projection base_target_weight does not match its SHA-256")
    metadata_payload = payload.get("metadata")
    if not isinstance(metadata_payload, dict):
        raise ValueError("decoder projection metadata must be a dictionary")
    checkpoint_hash = _hash_field(metadata_payload.get("model_checkpoint_sha256"), name="model_checkpoint_sha256")
    conditionals_hash = _hash_field(metadata_payload.get("conditionals_sha256"), name="conditionals_sha256")
    if checkpoint_hash is None or conditionals_hash is None:
        raise ValueError("decoder projection metadata must contain checkpoint and conditionals SHA-256 values")
    initial_source = metadata_payload.get("initial_conditionals")
    if initial_source is not None and not isinstance(initial_source, dict):
        raise ValueError("decoder projection metadata.initial_conditionals must be a dictionary or null")
    initial_hash = _hash_field(
        initial_source.get("sha256") if isinstance(initial_source, dict) else None,
        name="initial_conditionals.sha256",
    )
    hashes = {
        "base_target_weight_sha256": payload_base_hash,
        "model_checkpoint_sha256": checkpoint_hash,
        "conditionals_sha256": conditionals_hash,
        "initial_conditionals_sha256": initial_hash,
    }
    metadata = dict(payload)
    metadata["rank"] = rank
    metadata["alpha"] = alpha
    metadata["scale"] = scale
    metadata.update(hashes)
    metadata["adapter_delta_sha256"] = _sha256_tensor_fp32(torch, delta)
    return metadata, down, up, delta


def _state_for(torch: object, projection: object) -> dict[str, object]:
    state = getattr(projection, _STATE_ATTR, None)
    weight = getattr(projection, "weight", None)
    if weight is None:
        raise ValueError(f"model has no {_TARGET_KEY} weight")
    if tuple(weight.shape) != _TARGET_SHAPE:
        raise ValueError(f"model {_TARGET_KEY} must have shape {_TARGET_SHAPE}, got {tuple(weight.shape)}")
    if state is None:
        original = weight.detach().clone().to(device="cpu")
        state = {
            "original_weight": original,
            "base_target_weight_sha256": _sha256_tensor_fp32(torch, original),
        }
        setattr(projection, _STATE_ATTR, state)
    return state


def _restore(torch: object, projection: object, state: dict[str, object]) -> None:
    weight = projection.weight  # type: ignore[attr-defined]
    original = state["original_weight"].to(device=weight.device, dtype=weight.dtype)  # type: ignore[attr-defined]
    with torch.no_grad():  # type: ignore[attr-defined]
        weight.copy_(original)


def configure_decoder_projection(
    model: object,
    path: str | Path | None = None,
    *,
    strength: float = 1.0,
    conditionals_path: str | Path | None = None,
    model_checkpoint_path: str | Path | None = None,
) -> dict[str, object]:
    """Apply, switch, or restore one final-projection adapter.

    ``strength=0`` always restores the exact original model weight and does
    not load an adapter.  A nonzero strength requires an adapter path and a
    conditioning cache whose SHA-256 matches the adapter provenance.  The
    optional checkpoint path defaults to the repository's meanflow checkpoint
    and is included in the returned report.
    """

    try:
        value = float(strength)
    except (TypeError, ValueError) as exc:
        raise ValueError("decoder projection strength must be finite in [0,1]") from exc
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError("decoder projection strength must be finite in [0,1]")

    # A missing path is the public reset operation.  It is used between sweep
    # cases and must not require an adapter file or Torch import to exist.
    if path is None:
        state = None
        projection = None
        if hasattr(model, "s3gen"):
            try:
                projection = _projection_module(model)
                state = getattr(projection, _STATE_ATTR, None)
            except ValueError:
                state = None
        if state is not None and projection is not None:
            import torch

            _restore(torch, projection, state)
            state["active_path"] = None
            state["active_strength"] = 0.0
            return {
                "enabled": False,
                "restored": True,
                "strength": 0.0,
                "path": None,
                "base_target_weight_sha256": state["base_target_weight_sha256"],
            }
        return {"enabled": False, "restored": False, "strength": 0.0, "path": None}

    # A zero-strength path also restores exactly, without parsing the file.
    if value == 0.0:
        try:
            projection = _projection_module(model)
            state = getattr(projection, _STATE_ATTR, None)
        except ValueError:
            state = None
            projection = None
        if state is not None and projection is not None:
            import torch

            _restore(torch, projection, state)
            state["active_path"] = None
            state["active_strength"] = 0.0
            return {
                "enabled": False,
                "restored": True,
                "strength": 0.0,
                "path": None,
                "base_target_weight_sha256": state["base_target_weight_sha256"],
            }
        return {"enabled": False, "restored": False, "strength": 0.0, "path": None}

    if conditionals_path is None:
        raise ValueError("decoder projection requires conditionals_path for provenance validation")
    import torch

    adapter_path = _resolve_path(path)
    metadata, _down, _up, delta = _load_adapter(torch, adapter_path)
    projection = _projection_module(model)
    state = _state_for(torch, projection)
    base_hash = str(state["base_target_weight_sha256"])
    if base_hash != metadata["base_target_weight_sha256"]:
        raise ValueError("decoder projection base_target_weight_sha256 does not match model")

    cache_path = _resolve_path(conditionals_path)
    if not cache_path.exists() or not cache_path.is_file():
        raise FileNotFoundError(f"decoder projection conditionals cache not found: {cache_path}")
    cache_hash = _sha256_file(cache_path)
    expected_initial = metadata.get("initial_conditionals_sha256")
    expected_cache = expected_initial or metadata["conditionals_sha256"]
    if cache_hash != expected_cache:
        name = "initial_conditionals_sha256" if expected_initial else "conditionals_sha256"
        raise ValueError(f"decoder projection {name} does not match conditionals cache")

    checkpoint_path = _resolve_path(
        model_checkpoint_path or (ROOT / "models" / "chatterbox-nano" / "s3gen_meanflow.safetensors")
    )
    if not checkpoint_path.exists() or not checkpoint_path.is_file():
        raise FileNotFoundError(f"decoder projection model checkpoint not found: {checkpoint_path}")
    checkpoint_hash = _sha256_file(checkpoint_path)
    if checkpoint_hash != metadata["model_checkpoint_sha256"]:
        raise ValueError("decoder projection model_checkpoint_sha256 does not match checkpoint")

    _restore(torch, projection, state)
    weight = projection.weight
    base = state["original_weight"].to(device=weight.device, dtype=weight.dtype)  # type: ignore[attr-defined]
    applied_delta = delta.to(device=weight.device, dtype=weight.dtype) * value  # type: ignore[attr-defined]
    with torch.no_grad():
        weight.copy_(base + applied_delta)
    state["active_path"] = str(adapter_path)
    state["active_strength"] = value
    return {
        "enabled": True,
        "restored": False,
        "strength": value,
        "path": str(adapter_path),
        "adapter_sha256": _sha256_file(adapter_path),
        "base_target_weight_sha256": base_hash,
        "model_checkpoint_path": str(checkpoint_path),
        "model_checkpoint_sha256": checkpoint_hash,
        "conditionals_path": str(cache_path),
        "conditionals_sha256": cache_hash,
        "provenance_conditionals_sha256": metadata["conditionals_sha256"],
        "initial_conditionals_sha256": expected_initial,
        "target_weight_key": _TARGET_KEY,
        "target_shape": list(_TARGET_SHAPE),
        "rank": metadata["rank"],
        "alpha": metadata["alpha"],
        "scale": metadata["scale"],
        "delta_l2": float(delta.detach().float().norm().item()),  # type: ignore[attr-defined]
    }


__all__ = ["configure_decoder_projection"]
