"""Fold the fitted rank-two S3Gen self-attention adapter into Nano weights.

The checkpoint contains factors for the canonical 224 meanflow decoder
attention projections.  Runtime validation is fail-closed and completes for
all targets before the first model mutation.  Base weights are streamed from
the safetensors checkpoint one tensor at a time.  The runtime stores module
references and hashes only, so switching or restoring an adapter does not
retain a second dense decoder copy.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from decoder_projection import _sha256_file as _sha256_file_cached
from decoder_projection import _sha256_tensor_fp32


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_CHECKPOINT = ROOT / "models" / "chatterbox-nano" / "s3gen_meanflow.safetensors"
DEFAULT_INVENTORY = ROOT / "artifacts" / "nano_lab" / "decoder_attention_inventory.json"
FORMAT = "nano_decoder_attention_lora_v1"
TARGET_PREFIX = "flow.decoder.estimator."
RANK = 2
ALPHA = 2.0
SCALE = 1.0
TARGET_WEIGHT_COUNT = 224
TARGET_PARAMETER_COUNT = 344064
STATE_ATTR = "_nano_decoder_attention_state"


def _hex(value: object, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"decoder attention {name} must be a 64-character hexadecimal SHA-256 string")
    result = value.lower()
    if any(char not in "0123456789abcdef" for char in result):
        raise ValueError(f"decoder attention {name} must be a 64-character hexadecimal SHA-256 string")
    return result


def _resolve(value: str | Path, *, base: Path = ROOT) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _load_inventory(path: Path) -> tuple[dict[str, tuple[int, int]], str]:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"decoder attention inventory not found: {path}")
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"decoder attention inventory is invalid JSON: {path}") from exc
    if not isinstance(payload, Mapping) or len(payload) != TARGET_WEIGHT_COUNT:
        raise ValueError(f"decoder attention inventory must contain {TARGET_WEIGHT_COUNT} targets")
    result: dict[str, tuple[int, int]] = {}
    parameter_count = 0
    for key, raw_shape in payload.items():
        name = str(key)
        if not name.startswith(TARGET_PREFIX) or ".attn1." not in name or not name.endswith(".weight"):
            raise ValueError(f"decoder attention inventory contains a non-canonical target: {name}")
        try:
            shape = tuple(int(value) for value in raw_shape)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"decoder attention inventory shape is invalid for {name}") from exc
        if shape not in {(512, 256), (256, 512)}:
            raise ValueError(f"decoder attention inventory shape is invalid for {name}: {shape}")
        if name in result:
            raise ValueError(f"decoder attention inventory repeats target: {name}")
        result[name] = shape
        parameter_count += RANK * (shape[0] + shape[1])
    if parameter_count != TARGET_PARAMETER_COUNT:
        raise ValueError(f"decoder attention inventory parameter count is not {TARGET_PARAMETER_COUNT}")
    roots = ['down_blocks.0', *(f'mid_blocks.{i}' for i in range(12)), 'up_blocks.0']
    canonical = {f'{TARGET_PREFIX}{root}.1.{block}.attn1.{projection}.weight':
                 ((256, 512) if projection == 'to_out.0' else (512, 256))
                 for root in roots for block in range(4)
                 for projection in ('to_q', 'to_k', 'to_v', 'to_out.0')}
    if result != canonical:
        raise ValueError('decoder attention inventory differs from native target names')
    return dict(sorted(result.items())), _sha256_file_cached(path)


def _finite_scalar(value: object, *, name: str, expected: float | None = None) -> float:
    if isinstance(value, bool):
        raise ValueError(f"decoder attention {name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"decoder attention {name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"decoder attention {name} must be finite")
    if expected is not None and not math.isclose(result, expected, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(f"decoder attention {name} must be {expected:g}")
    return result


def _as_cpu_factor(torch: Any, value: object, *, shape: tuple[int, ...], name: str) -> Any:
    try:
        tensor = torch.as_tensor(value, dtype=torch.float32, device="cpu")
    except Exception as exc:
        raise ValueError(f"decoder attention {name} is not a tensor") from exc
    if tuple(tensor.shape) != shape:
        raise ValueError(f"decoder attention {name} must have shape {shape}, got {tuple(tensor.shape)}")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"decoder attention {name} contains NaN or infinity")
    return tensor.contiguous()


def _load_checkpoint(torch: Any, path: Path, inventory: Mapping[str, tuple[int, int]]) -> tuple[dict[str, Any], dict[str, dict[str, Any]], str]:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"decoder attention adapter not found: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"could not load decoder attention adapter: {path}") from exc
    if not isinstance(payload, dict) or payload.get("format") != FORMAT:
        raise ValueError(f"unsupported decoder attention adapter format: {payload.get('format') if isinstance(payload, dict) else type(payload).__name__}")
    if int(payload.get("rank", -1)) != RANK:
        raise ValueError("decoder attention checkpoint requires rank=2")
    _finite_scalar(payload.get("alpha"), name="alpha", expected=ALPHA)
    _finite_scalar(payload.get("scale"), name="scale", expected=SCALE)
    if payload.get("target_weight_count") != TARGET_WEIGHT_COUNT:
        raise ValueError(f"decoder attention checkpoint target_weight_count must be {TARGET_WEIGHT_COUNT}")
    if payload.get("target_parameter_count") != TARGET_PARAMETER_COUNT:
        raise ValueError(f"decoder attention checkpoint target_parameter_count must be {TARGET_PARAMETER_COUNT}")
    top_hashes = {
        key: _hex(payload.get(key), name=key)
        for key in (
            "model_checkpoint_sha256",
            "initial_conditionals_sha256",
            "prepared_conditionals_sha256",
            "cache_sha256",
            "inventory_sha256",
        )
    }
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("decoder attention checkpoint metadata must be a dictionary")
    for key in ("model_checkpoint_sha256", "conditionals_sha256", "cache_sha256", "inventory_sha256"):
        metadata_value = _hex(metadata.get(key), name=f"metadata.{key}")
        expected = top_hashes["prepared_conditionals_sha256"] if key == "conditionals_sha256" else top_hashes[key]
        if metadata_value != expected:
            raise ValueError(f"decoder attention {key} disagrees with metadata")
    initial_source = metadata.get("initial_conditionals")
    if not isinstance(initial_source, Mapping):
        raise ValueError("decoder attention metadata.initial_conditionals must be a dictionary")
    if _hex(initial_source.get("sha256"), name="initial_conditionals.sha256") != top_hashes["initial_conditionals_sha256"]:
        raise ValueError("decoder attention initial conditionals hash disagrees with metadata")
    targets = payload.get("targets")
    if not isinstance(targets, Mapping) or set(str(key) for key in targets) != set(inventory):
        raise ValueError("decoder attention target set differs from canonical inventory")
    if len(targets) != TARGET_WEIGHT_COUNT:
        raise ValueError(f"decoder attention checkpoint must contain {TARGET_WEIGHT_COUNT} targets")
    factors: dict[str, dict[str, Any]] = {}
    for name in sorted(inventory):
        raw = targets.get(name)
        if not isinstance(raw, Mapping):
            raise ValueError(f"decoder attention target {name} is not a dictionary")
        shape = tuple(int(value) for value in raw.get("shape", ()))
        expected_shape = inventory[name]
        if shape != expected_shape:
            raise ValueError(f"decoder attention target {name} shape differs from inventory")
        if "base_weight" in raw or "merged_delta_weight" in raw:
            raise ValueError(f"decoder attention target {name} stores forbidden dense weights")
        down = _as_cpu_factor(torch, raw.get("down_weight"), shape=(RANK, expected_shape[1]), name=f"{name}.down_weight")
        up = _as_cpu_factor(torch, raw.get("up_weight"), shape=(expected_shape[0], RANK), name=f"{name}.up_weight")
        base_hash = _hex(raw.get("base_weight_sha256"), name=f"{name}.base_weight_sha256")
        factors[name] = {"shape": expected_shape, "down": down, "up": up, "base_hash": base_hash}
    payload = dict(payload)
    payload.update(top_hashes)
    payload["metadata"] = dict(metadata)
    return payload, factors, _sha256_file_cached(path)


def _locate_module(model: object, target_name: str) -> object:
    parts = target_name.split(".")
    if not parts or parts[0] != "flow":
        raise ValueError(f"decoder attention target is not rooted at flow: {target_name}")
    current = model.s3gen  # type: ignore[attr-defined]
    for part in parts[:-1]:
        current = current[int(part)] if part.isdigit() else getattr(current, part)
    return current


def _target_modules(torch: Any, model: object, inventory: Mapping[str, tuple[int, int]]) -> dict[str, object]:
    modules: dict[str, object] = {}
    for name, shape in sorted(inventory.items()):
        try:
            module = _locate_module(model, name)
            weight = module.weight  # type: ignore[attr-defined]
        except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"decoder attention target module is missing: {name}") from exc
        if tuple(weight.shape) != shape:
            raise ValueError(f"decoder attention target {name} has shape {tuple(weight.shape)}, expected {shape}")
        if weight.dtype != torch.float32:
            raise ValueError(f"decoder attention target {name} must use fixed fp32 weights")
        modules[name] = module
    return modules


def _open_base(torch: Any, checkpoint_path: Path) -> Any:
    try:
        from safetensors import safe_open
    except Exception as exc:
        raise RuntimeError("decoder attention folding requires safetensors") from exc
    return safe_open(str(checkpoint_path), framework="pt", device="cpu")


def _verify_checkpoint_tensors(torch: Any, checkpoint_path: Path, factors: Mapping[str, Mapping[str, Any]]) -> None:
    with _open_base(torch, checkpoint_path) as source:
        names = set(source.keys())
        for name, spec in factors.items():
            if name not in names:
                raise ValueError(f"decoder attention checkpoint is missing base tensor: {name}")
            tensor = source.get_tensor(name)
            if tensor.dtype != torch.float32 or tuple(tensor.shape) != tuple(spec["shape"]):
                raise ValueError(f"decoder attention checkpoint base tensor is invalid: {name}")
            if _sha256_tensor_fp32(torch, tensor) != spec["base_hash"]:
                raise ValueError(f"decoder attention base weight hash mismatch: {name}")


def _verify_live_base(torch: Any, modules: Mapping[str, object], factors: Mapping[str, Mapping[str, Any]]) -> None:
    for name, module in modules.items():
        weight = module.weight  # type: ignore[attr-defined]
        if _sha256_tensor_fp32(torch, weight) != factors[name]["base_hash"]:
            raise ValueError(f"decoder attention live base weight hash mismatch: {name}")


def _restore_from_checkpoint(torch: Any, checkpoint_path: Path, modules: Mapping[str, object], factors: Mapping[str, Mapping[str, Any]]) -> None:
    # Verify every streamed tensor first.  A changed checkpoint cannot leave a
    # partially restored model.
    _verify_checkpoint_tensors(torch, checkpoint_path, factors)
    with _open_base(torch, checkpoint_path) as source:
        with torch.no_grad():
            for name, module in modules.items():
                weight = module.weight  # type: ignore[attr-defined]
                base = source.get_tensor(name).to(device=weight.device, dtype=weight.dtype)
                weight.copy_(base)


def _reset(model: object) -> dict[str, object]:
    state = getattr(model.s3gen.flow.decoder.estimator, STATE_ATTR, None)  # type: ignore[attr-defined]
    if state is None:
        return {"enabled": False, "restored": False, "strength": 0.0, "path": None}
    import torch

    checkpoint_path = Path(state["checkpoint_path"])
    if _sha256_file_cached(checkpoint_path) != state["model_checkpoint_sha256"]:
        raise ValueError("decoder attention model checkpoint SHA-256 changed before restore")
    _restore_from_checkpoint(torch, checkpoint_path, state["modules"], state["factors"])
    state["active_path"] = None
    state["active_adapter_sha256"] = None
    state["active_strength"] = 0.0
    return {
        "enabled": False,
        "restored": True,
        "strength": 0.0,
        "path": None,
        "target_weight_count": TARGET_WEIGHT_COUNT,
        "target_parameter_count": TARGET_PARAMETER_COUNT,
        "model_checkpoint_sha256": state["model_checkpoint_sha256"],
        "inventory_sha256": state["inventory_sha256"],
    }


def configure_decoder_attention(
    model: object,
    path: str | Path | None = None,
    *,
    strength: float = 1.0,
    conditionals_path: str | Path | None = None,
    model_checkpoint_path: str | Path | None = None,
    inventory_path: str | Path | None = None,
) -> dict[str, object]:
    """Apply, switch, or restore a factor-only decoder attention adapter."""

    try:
        value = float(strength)
    except (TypeError, ValueError) as exc:
        raise ValueError("decoder attention strength must be finite in [0,1]") from exc
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError("decoder attention strength must be finite in [0,1]")
    if path is None or value == 0.0:
        return _reset(model)
    if conditionals_path is None:
        raise ValueError("decoder attention requires input conditionals for provenance validation")
    import torch

    adapter_path = _resolve(path)
    checkpoint_path = _resolve(model_checkpoint_path or DEFAULT_MODEL_CHECKPOINT)
    inventory_file = _resolve(inventory_path or DEFAULT_INVENTORY)
    inventory, inventory_sha = _load_inventory(inventory_file)
    payload, factors, adapter_sha = _load_checkpoint(torch, adapter_path, inventory)
    if payload["inventory_sha256"] != inventory_sha:
        raise ValueError("decoder attention inventory SHA-256 does not match checkpoint")
    if not checkpoint_path.exists() or not checkpoint_path.is_file():
        raise FileNotFoundError(f"decoder attention model checkpoint not found: {checkpoint_path}")
    checkpoint_sha = _sha256_file_cached(checkpoint_path)
    if checkpoint_sha != payload["model_checkpoint_sha256"]:
        raise ValueError("decoder attention model checkpoint SHA-256 does not match checkpoint metadata")
    cache_path = _resolve(conditionals_path)
    if not cache_path.exists() or not cache_path.is_file():
        raise FileNotFoundError(f"decoder attention conditionals cache not found: {cache_path}")
    cache_sha = _sha256_file_cached(cache_path)
    if cache_sha != payload["initial_conditionals_sha256"]:
        raise ValueError("decoder attention initial conditionals SHA-256 does not match selected cache")

    estimator = model.s3gen.flow.decoder.estimator  # type: ignore[attr-defined]
    state = getattr(estimator, STATE_ATTR, None)
    modules = state["modules"] if state is not None else _target_modules(torch, model, inventory)
    if any(module.weight.device.type == 'cuda' for module in modules.values()):
        if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
            raise ValueError('decoder attention CUDA folding requires strict FP32: disable both matmul and cuDNN TF32')
    if state is None:
        # The two complete passes finish all validation before the first
        # mutation. Factor tensors remain CPU-resident only during this call.
        _verify_checkpoint_tensors(torch, checkpoint_path, factors)
        _verify_live_base(torch, modules, factors)
        state = {
            "checkpoint_path": str(checkpoint_path),
            "inventory_path": str(inventory_file),
            "inventory": inventory,
            "inventory_sha256": inventory_sha,
            "model_checkpoint_sha256": checkpoint_sha,
            "modules": modules,
            # Restoration needs only hashes and shapes. Learned factors are
            # transient during folding and do not stay resident afterwards.
            "factors": {name: {'shape': spec['shape'], 'base_hash': spec['base_hash']}
                        for name, spec in factors.items()},
            "active_path": None,
            "active_adapter_sha256": None,
            "active_strength": 0.0,
        }
        setattr(estimator, STATE_ATTR, state)
    else:
        if state["checkpoint_path"] != str(checkpoint_path) or state["inventory_sha256"] != inventory_sha:
            raise ValueError("decoder attention model or inventory changed during one runtime")
        if state["model_checkpoint_sha256"] != checkpoint_sha:
            raise ValueError("decoder attention checkpoint hash changed during one runtime")

    if state.get("active_adapter_sha256") == adapter_sha and math.isclose(float(state.get("active_strength", 0.0)), value, rel_tol=0.0, abs_tol=1e-12):
        return {
            "enabled": True,
            "restored": False,
            "idempotent": True,
            "strength": value,
            "path": str(adapter_path),
            "adapter_sha256": adapter_sha,
            "model_checkpoint_sha256": checkpoint_sha,
            "inventory_sha256": inventory_sha,
            "conditionals_path": str(cache_path),
            "initial_conditionals_sha256": payload["initial_conditionals_sha256"],
            "target_weight_count": TARGET_WEIGHT_COUNT,
            "target_parameter_count": TARGET_PARAMETER_COUNT,
        }

    # A new adapter must pass every base hash before any mutation, including
    # after a reset. Start each fold from the streamed original, so a retry
    # after an interrupted copy cannot accumulate a partial previous delta.
    _verify_checkpoint_tensors(torch, checkpoint_path, factors)
    state['active_adapter_sha256'] = None
    state['active_strength'] = 0.0
    with _open_base(torch, checkpoint_path) as source, torch.no_grad():
        for name, module in modules.items():
            weight = module.weight  # type: ignore[attr-defined]
            spec = factors[name]
            delta = (ALPHA / float(RANK)) * torch.matmul(spec["up"], spec["down"])
            base = source.get_tensor(name)
            folded = base + delta * value
            weight.copy_(folded.to(device=weight.device, dtype=weight.dtype))
    state["active_path"] = str(adapter_path)
    state["active_adapter_sha256"] = adapter_sha
    state["active_strength"] = value
    return {
        "enabled": True,
        "restored": False,
        "idempotent": False,
        "strength": value,
        "path": str(adapter_path),
        "adapter_sha256": adapter_sha,
        "model_checkpoint_path": str(checkpoint_path),
        "model_checkpoint_sha256": checkpoint_sha,
        "inventory_path": str(inventory_file),
        "inventory_sha256": inventory_sha,
        "conditionals_path": str(cache_path),
        "initial_conditionals_sha256": payload["initial_conditionals_sha256"],
        "prepared_conditionals_sha256": payload["prepared_conditionals_sha256"],
        "target_weight_count": TARGET_WEIGHT_COUNT,
        "target_parameter_count": TARGET_PARAMETER_COUNT,
        "target_weight_key_prefix": TARGET_PREFIX,
    }


__all__ = ["configure_decoder_attention"]
