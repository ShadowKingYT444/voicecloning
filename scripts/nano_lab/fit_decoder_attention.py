"""Fit a bounded rank-two LoRA over Nano S3Gen self-attention projections.

This is an experimental, speaker-specific decoder adapter.  It uses the
recorded acoustic-only preparation and the exact native two-draw noise replay
from :mod:`fit_decoder_embedding`.  The flow encoder, decoder base weights,
speaker embedding, and all conditioning tensors remain frozen.  Only rank-two
factors on the 224 ``attn1`` Q/K/V/output Linear weights are optimized.

The checkpoint stores factors and base-weight hashes only.  It does not copy
the decoder weights or store merged dense matrices.  A runtime loader can
stream the base S3Gen checkpoint and fold one factor pair at a time.

No model is loaded when this module is imported.  Run model work through
``bounded_job.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

import fit_decoder_embedding as embedding_fit
import fit_decoder_projection as projection_fit


ROOT = embedding_fit.ROOT
SCRIPT_DIR = embedding_fit.SCRIPT_DIR
DEFAULT_PREPARE_DIR = ROOT / "artifacts" / "nano_lab" / "acoustic_only_asmr" / "flow_v2"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "nano_lab" / "decoder_attention_fit"
DEFAULT_MODEL_DIR = ROOT / "models" / "chatterbox-nano"
DEFAULT_INITIAL_CONDITIONALS = ROOT / "artifacts" / "nano_lab" / "decoder_embedding_fit" / "conditionals.pt"
DEFAULT_INVENTORY = ROOT / "artifacts" / "nano_lab" / "decoder_attention_inventory.json"

FORMAT = "nano_decoder_attention_lora_v1"
CHECKPOINT_FORMAT = FORMAT
TARGET_PREFIX = "flow.decoder.estimator."
RANK = 2
ALPHA = 2.0
SCALE = ALPHA / RANK
DEFAULT_STEPS = embedding_fit.DEFAULT_STEPS
DEFAULT_MAX_STEPS = 30
DEFAULT_PATIENCE = 3
DEFAULT_LR = 1e-4
DEFAULT_REGULARIZER = 1e-4
DEFAULT_ENVELOPE_WEIGHT = 0.25
DEFAULT_DISTILL_WEIGHT = 0.5
DEFAULT_PARITY_ATOL = embedding_fit.DEFAULT_PARITY_ATOL
ADAPTER_INIT_SEED = 2701


def _resolve(value: str | os.PathLike[str], *, base: Path = ROOT) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha256(torch: Any, tensor: Any) -> str:
    if not torch.is_tensor(tensor):
        raise ValueError("tensor hash requires a Torch tensor")
    values = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _atomic_torch_save(torch: Any, payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


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
    return value if sys.platform == "darwin" else value * 1024


def _factor_delta_numpy(up: Any, down: Any, shape: Sequence[int], *, alpha: float = ALPHA) -> np.ndarray:
    """Return a merged LoRA delta with ``[out,in]`` target geometry."""

    target_shape = tuple(int(value) for value in shape)
    if len(target_shape) != 2:
        raise ValueError("attention target shape must have two dimensions")
    out_features, in_features = target_shape
    up_values = np.asarray(up, dtype=np.float32)
    down_values = np.asarray(down, dtype=np.float32)
    if up_values.shape != (out_features, RANK) or down_values.shape != (RANK, in_features):
        raise ValueError(f"LoRA factors have wrong shapes for target {target_shape}")
    if not np.isfinite(up_values).all() or not np.isfinite(down_values).all():
        raise ValueError("LoRA factors contain NaN or infinity")
    if not math.isfinite(float(alpha)) or float(alpha) <= 0:
        raise ValueError("alpha must be finite and positive")
    return (float(alpha) / float(RANK) * (up_values @ down_values)).astype(np.float32, copy=False)


def _fold_weight_numpy(base: Any, up: Any, down: Any, shape: Sequence[int], *, alpha: float = ALPHA) -> np.ndarray:
    target_shape = tuple(int(value) for value in shape)
    base_values = np.asarray(base, dtype=np.float32)
    if base_values.shape != target_shape or not np.isfinite(base_values).all():
        raise ValueError(f"base weight has invalid shape/value for target {target_shape}")
    return base_values + _factor_delta_numpy(up, down, target_shape, alpha=alpha)


def _apply_lora_numpy(base: Any, hidden: Any, up: Any, down: Any, shape: Sequence[int], *, alpha: float = ALPHA) -> np.ndarray:
    """Apply folded and factorized matrices to a final feature dimension."""

    target_shape = tuple(int(value) for value in shape)
    base_values = np.asarray(base, dtype=np.float32)
    hidden_values = np.asarray(hidden, dtype=np.float32)
    if hidden_values.shape[-1] != target_shape[1]:
        raise ValueError("hidden feature dimension does not match target")
    folded = _fold_weight_numpy(base_values, up, down, target_shape, alpha=alpha)
    return np.matmul(hidden_values, folded.T)


def _canonical_inventory() -> dict[str, tuple[int, int]]:
    roots = ['down_blocks.0', *(f'mid_blocks.{i}' for i in range(12)), 'up_blocks.0']
    return {f'{TARGET_PREFIX}{root}.1.{block}.attn1.{projection}.weight':
            ((256, 512) if projection == 'to_out.0' else (512, 256))
            for root in roots for block in range(4)
            for projection in ('to_q', 'to_k', 'to_v', 'to_out.0')}


def _inventory(path: Path) -> tuple[dict[str, tuple[int, int]], str]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text())
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError("attention inventory is empty")
    result: dict[str, tuple[int, int]] = {}
    for key, shape in payload.items():
        name = str(key)
        if not name.startswith(TARGET_PREFIX) or not name.endswith(".weight"):
            raise ValueError(f"inventory target is outside decoder self-attention: {name}")
        values = tuple(int(value) for value in shape)
        if len(values) != 2 or values not in {(512, 256), (256, 512)}:
            raise ValueError(f"inventory target has unsupported shape {name}: {values}")
        if name in result:
            raise ValueError(f"duplicate attention inventory target {name}")
        result[name] = values
    if len(result) != 224:
        raise ValueError(f"attention inventory must contain 224 targets, got {len(result)}")
    if result != _canonical_inventory():
        raise ValueError("attention inventory differs from the native self-attention target set")
    return dict(sorted(result.items())), _sha256(path)


def _validate_factor_payload(target_name: str, spec: Mapping[str, Any], expected_shape: Sequence[int]) -> None:
    shape = tuple(int(value) for value in spec.get("shape", ()))
    target_shape = tuple(int(value) for value in expected_shape)
    if shape != target_shape:
        raise ValueError(f"target {target_name} shape differs from inventory")
    down = np.asarray(spec.get("down_weight"), dtype=np.float32)
    up = np.asarray(spec.get("up_weight"), dtype=np.float32)
    if down.shape != (RANK, target_shape[1]) or up.shape != (target_shape[0], RANK):
        raise ValueError(f"target {target_name} factors have wrong shape")
    if not np.isfinite(down).all() or not np.isfinite(up).all():
        raise ValueError(f"target {target_name} factors contain NaN or infinity")
    base_sha = spec.get("base_weight_sha256")
    if not isinstance(base_sha, str) or len(base_sha) != 64 or any(char not in "0123456789abcdef" for char in base_sha.lower()):
        raise ValueError(f"target {target_name} base weight hash is missing or malformed")


def validate_attention_checkpoint(payload: Mapping[str, Any], inventory: Mapping[str, Sequence[int]] | None = None) -> dict[str, Any]:
    """Validate a factor-only checkpoint without importing Torch."""

    if not isinstance(payload, Mapping) or payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("unsupported decoder attention checkpoint format")
    if payload.get("rank") != RANK or abs(float(payload.get("alpha", float("nan"))) - ALPHA) > 1e-8:
        raise ValueError("attention checkpoint requires rank=2 and alpha=2")
    if not math.isfinite(float(payload.get("alpha", float("nan")))):
        raise ValueError("attention checkpoint alpha is non-finite")
    if payload.get("target_parameter_count") != 344064:
        raise ValueError("attention checkpoint target parameter count is not 344064")
    if payload.get("scale") != SCALE or payload.get("target_weight_count") != 224:
        raise ValueError("attention checkpoint scale or weight count is inconsistent")
    targets = payload.get("targets")
    if not isinstance(targets, Mapping) or not targets:
        raise ValueError("attention checkpoint has no target factors")
    expected = {str(key): tuple(int(value) for value in shape) for key, shape in (inventory or _canonical_inventory()).items()}
    if expected and set(targets) != set(expected):
        raise ValueError("attention checkpoint target set differs from inventory")
    if len(targets) != 224:
        raise ValueError("attention checkpoint must contain 224 targets")
    for name, spec in targets.items():
        if not isinstance(spec, Mapping):
            raise ValueError(f"target {name} factor record is not an object")
        if expected:
            shape = expected.get(str(name))
            if shape is None:
                raise ValueError(f"target {name} is not in inventory")
        else:
            shape = tuple(int(value) for value in spec.get("shape", ()))
        _validate_factor_payload(str(name), spec, shape)
        if "base_weight" in spec or "merged_delta_weight" in spec:
            raise ValueError(f"target {name} stores forbidden dense/base tensors")
    for key in ("model_checkpoint_sha256", "initial_conditionals_sha256", "prepared_conditionals_sha256", "cache_sha256", "inventory_sha256"):
        value = payload.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value.lower()):
            raise ValueError(f"checkpoint is missing {key}")
    return dict(payload)


def _locate_parent(module: Any, path: str) -> tuple[Any, str]:
    parts = path.split(".")
    parent = module
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    return parent, parts[-1]


def _get_child(parent: Any, name: str) -> Any:
    return parent[int(name)] if name.isdigit() else getattr(parent, name)


def _set_child(parent: Any, name: str, value: Any) -> None:
    if name.isdigit():
        parent[int(name)] = value
    else:
        setattr(parent, name, value)


def _attention_targets(estimator: Any, inventory: Mapping[str, Sequence[int]]) -> list[tuple[str, Any, tuple[int, int]]]:
    targets: list[tuple[str, Any, tuple[int, int]]] = []
    for full_name, shape_value in sorted(inventory.items()):
        # The streamed stage exposes a MeanflowEstimatorStage whose local
        # decoder root is ``estimator``.  Inventory names include the vendor
        # checkpoint prefix ``flow.decoder.estimator``.
        local_name = "estimator." + full_name[len(TARGET_PREFIX) :]
        module_path = local_name[: -len(".weight")]
        parent, leaf = _locate_parent(estimator, module_path)
        target = _get_child(parent, leaf)
        shape = tuple(int(value) for value in shape_value)
        weight = getattr(target, "weight", None)
        if weight is None or tuple(weight.shape) != shape:
            raise ValueError(f"inventory target {full_name} is missing or has shape {getattr(weight, 'shape', None)}")
        if not hasattr(target, "forward"):
            raise ValueError(f"inventory target {full_name} is not a module")
        targets.append((full_name, target, shape))
    return targets


def _build_attention_adapter(torch: Any, base: Any, *, init_seed: int) -> Any:
    import torch.nn as nn

    class AttentionLoRA(nn.Module):
        def __init__(self, original: Any) -> None:
            super().__init__()
            self.base = original
            for parameter in self.base.parameters():
                parameter.requires_grad_(False)
            out_features, in_features = tuple(int(value) for value in original.weight.shape)
            device = original.weight.device
            dtype = original.weight.dtype
            self.down = nn.Linear(in_features, RANK, bias=False, device=device, dtype=dtype)
            self.up = nn.Linear(RANK, out_features, bias=False, device=device, dtype=dtype)
            nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5.0))
            nn.init.zeros_(self.up.weight)
            self.alpha = float(ALPHA)
            self.scale = float(SCALE)
            self.init_seed = int(init_seed)

        def forward(self, hidden_states: Any, *args: Any, **kwargs: Any) -> Any:
            return self.base(hidden_states, *args, **kwargs) + self.scale * self.up(self.down(hidden_states))

        def merged_delta_weight(self) -> Any:
            return self.scale * torch.matmul(self.up.weight, self.down.weight)

    return AttentionLoRA(base)


def _install_attention_adapters(torch: Any, estimator: Any, inventory: Mapping[str, Sequence[int]], *, init_seed: int = ADAPTER_INIT_SEED) -> tuple[dict[str, Any], list[Any]]:
    targets = _attention_targets(estimator, inventory)
    devices = sorted({target.weight.device.index if target.weight.device.type == "cuda" and target.weight.device.index is not None else 0 for _, target, _ in targets if target.weight.device.type == "cuda"})
    context = torch.random.fork_rng(devices=devices)
    context.__enter__()
    try:
        torch.manual_seed(int(init_seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(init_seed))
        adapters: dict[str, Any] = {}
        factors: list[Any] = []
        for full_name, base, _shape in targets:
            local_name = "estimator." + full_name[len(TARGET_PREFIX) :]
            module_path = local_name[: -len(".weight")]
            parent, leaf = _locate_parent(estimator, module_path)
            adapter = _build_attention_adapter(torch, base, init_seed=init_seed)
            _set_child(parent, leaf, adapter)
            adapters[full_name] = adapter
            factors.extend([adapter.down.weight, adapter.up.weight])
    finally:
        context.__exit__(None, None, None)
    return adapters, factors


def _predict(torch: Any, estimator: Any, state: Mapping[str, Any], embedding: Any, noise: Any, *, steps: int) -> Any:
    return embedding_fit._aligned_prediction(
        embedding_fit._basic_euler(torch, estimator, state, embedding, noise, steps=steps)[:, :, state["prompt_len"] :],
        state,
    )


def _collect_predictions(torch: Any, estimator: Any, states: Sequence[Mapping[str, Any]], embedding: Any, noises: Mapping[str, Any], *, steps: int) -> dict[str, Any]:
    outputs: dict[str, Any] = {}
    with torch.no_grad():
        for state in states:
            row_id = str(state["id"])
            outputs[row_id] = _predict(torch, estimator, state, embedding, noises[row_id], steps=steps).detach()
    return outputs


def _delta_stats(torch: Any, predictions: Mapping[str, Any], anchors: Mapping[str, Any]) -> dict[str, float]:
    deltas = [prediction - anchors[row_id] for row_id, prediction in predictions.items()]
    if not deltas:
        return {"rms": 0.0, "max_abs": 0.0}
    joined = torch.cat([delta.detach().float().reshape(-1) for delta in deltas])
    return {"rms": float(torch.sqrt(torch.mean(joined.square())).cpu()), "max_abs": float(torch.max(torch.abs(joined)).cpu())}


def _attention_loss(torch: Any, predicted: Any, target: Any, anchor: Any, adapters: Mapping[str, Any], *, envelope_weight: float, distill_weight: float, regularizer: float) -> tuple[Any, dict[str, float]]:
    import torch.nn.functional as F

    raw = F.mse_loss(predicted, target)
    envelope = F.mse_loss(predicted.mean(dim=2), target.mean(dim=2))
    distill = F.mse_loss(predicted, anchor)
    factors = [factor for adapter in adapters.values() for factor in (adapter.down.weight, adapter.up.weight)]
    l2 = torch.stack([factor.square().mean() for factor in factors]).mean()
    total = raw + float(envelope_weight) * envelope + float(distill_weight) * distill + float(regularizer) * l2
    return total, {
        "total": float(total.detach().cpu()),
        "raw_mse": float(raw.detach().cpu()),
        "envelope_mse": float(envelope.detach().cpu()),
        "baseline_distill_mse": float(distill.detach().cpu()),
        "adapter_l2": float(l2.detach().cpu()),
        "regularization": float((float(regularizer) * l2).detach().cpu()),
    }


def _evaluate(torch: Any, estimator: Any, states: Sequence[Mapping[str, Any]], embedding: Any, noises: Mapping[str, Any], anchors: Mapping[str, Any], adapters: Mapping[str, Any], *, steps: int, envelope_weight: float, distill_weight: float, regularizer: float) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    predictions: dict[str, Any] = {}
    with torch.no_grad():
        for state in states:
            row_id = str(state["id"])
            predicted = _predict(torch, estimator, state, embedding, noises[row_id], steps=steps)
            predictions[row_id] = predicted.detach()
            _, metrics = _attention_loss(torch, predicted, state["target"], anchors[row_id], adapters, envelope_weight=envelope_weight, distill_weight=distill_weight, regularizer=regularizer)
            metrics["id"] = row_id
            rows.append(metrics)
    keys = ("total", "raw_mse", "envelope_mse", "baseline_distill_mse", "adapter_l2", "regularization")
    mean = {key: float(np.mean([row[key] for row in rows])) for key in keys}
    return {"mean": mean, "rows": rows, "delta": _delta_stats(torch, predictions, anchors)}


def _checkpoint_payload(torch: Any, adapters: Mapping[str, Any], inventory: Mapping[str, Sequence[int]], base_hashes: Mapping[str, str], *, metadata: Mapping[str, Any], step: int, valid: Mapping[str, Any]) -> dict[str, Any]:
    targets: dict[str, Any] = {}
    for name in sorted(inventory):
        adapter = adapters[name]
        down = adapter.down.weight.detach().cpu().float().contiguous()
        up = adapter.up.weight.detach().cpu().float().contiguous()
        targets[name] = {
            "shape": [int(value) for value in inventory[name]],
            "down_weight": down,
            "up_weight": up,
            "base_weight_sha256": base_hashes[name],
        }
    payload = {
        "format": CHECKPOINT_FORMAT,
        "rank": RANK,
        "alpha": ALPHA,
        "scale": SCALE,
        "target_parameter_count": 344064,
        "target_weight_count": len(targets),
        "targets": targets,
        "step": int(step),
        "valid": dict(valid),
        "metadata": dict(metadata),
        "model_checkpoint_sha256": metadata["model_checkpoint_sha256"],
        "initial_conditionals_sha256": metadata["initial_conditionals"]["sha256"],
        "prepared_conditionals_sha256": metadata["conditionals_sha256"],
        "cache_sha256": metadata["cache_sha256"],
        "inventory_sha256": metadata["inventory_sha256"],
    }
    validate_attention_checkpoint(payload, inventory)
    return payload


def _base_metadata(manifest: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], prepare_dir: Path, manifest_path: Path, conditionals_path: Path, initial_source: Mapping[str, Any], loader_report: Mapping[str, Any], model_dir: Path, inventory_path: Path, inventory_sha: str, base_hashes: Mapping[str, str], *, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "format": FORMAT,
        "prepare_dir": str(prepare_dir),
        "prepare_manifest": str(manifest_path),
        "prepare_manifest_sha256": _sha256(manifest_path),
        "cache": str(_resolve(manifest["cache"])),
        "cache_sha256": _sha256(_resolve(manifest["cache"])),
        "conditionals_path": str(conditionals_path),
        "conditionals_sha256": _sha256(conditionals_path),
        "initial_conditionals": dict(initial_source),
        "model_dir": str(model_dir),
        "model_checkpoint_sha256": loader_report.get("checkpoint_sha256"),
        "inventory_path": str(inventory_path),
        "inventory_sha256": inventory_sha,
        "target_weight_count": len(base_hashes),
        "target_parameter_count": 344064,
        "base_weight_sha256": dict(base_hashes),
        "train_ids": [str(row["id"]) for row in rows if row["split"] == "train"],
        "valid_ids": [str(row["id"]) for row in rows if row["split"] == "valid"],
        "native_row_seeds": {str(row["id"]): int(row["seed"]) for row in rows},
        "hyperparameters": {
            "rank": RANK,
            "alpha": ALPHA,
            "scale": SCALE,
            "steps": int(args.steps),
            "max_steps": int(args.max_steps),
            "patience": int(args.patience),
            "lr": float(args.lr),
            "regularizer": float(args.regularizer),
            "envelope_weight": float(args.envelope_weight),
            "distill_weight": float(args.distill_weight),
            "parity_atol": float(args.parity_atol),
        },
        "adapter_init_seed": ADAPTER_INIT_SEED,
        "frozen_policy": "all S3Gen base weights and the 192-D speaker embedding are frozen; only 224 rank-2 factor pairs train",
        "command": list(sys.argv),
    }


def _fit(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    started = time.perf_counter()
    if int(args.steps) != DEFAULT_STEPS:
        raise ValueError("--steps must equal 2 for native meanflow alignment")
    if int(args.max_steps) < 1 or int(args.max_steps) > DEFAULT_MAX_STEPS:
        raise ValueError(f"--max-steps must be in 1..{DEFAULT_MAX_STEPS}")
    if int(args.patience) < 1:
        raise ValueError("--patience must be >= 1")
    numeric = (args.lr, args.regularizer, args.envelope_weight, args.distill_weight, args.parity_atol)
    if not all(math.isfinite(float(value)) for value in numeric):
        raise ValueError("numeric hyperparameters must be finite")
    if args.lr <= 0 or args.lr > 1.0 or args.regularizer < 0 or args.envelope_weight < 0 or args.distill_weight < 0 or args.parity_atol <= 0 or args.parity_atol > 1.0:
        raise ValueError("hyperparameters are outside bounded ranges")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    if hasattr(torch, "set_num_threads"):
        torch.set_num_threads(2)
        try:
            torch.set_num_interop_threads(2)
        except RuntimeError:
            pass
    device = torch.device(args.device)
    prepare_dir = _resolve(args.prepare_dir)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory_path = _resolve(args.inventory)
    inventory, inventory_sha = _inventory(inventory_path)
    manifest, rows, conditionals_path, _cache, manifest_path = embedding_fit._validate_prepare_inputs(prepare_dir)
    initial_path = _resolve(args.initial_conditionals) if args.initial_conditionals else None
    prepared_payload, initial_payload, initial_source = projection_fit._load_conditionals_pair(torch, conditionals_path, initial_path)
    base_embedding = prepared_payload["gen"]["embedding"].detach().to(device=device, dtype=torch.float32)
    initial_embedding = initial_payload["gen"]["embedding"].detach().to(device=device, dtype=torch.float32)
    base_embedding.requires_grad_(False)
    initial_embedding.requires_grad_(False)
    model_dir = _resolve(args.model_dir)
    rss_before_load = _rss_bytes()
    flow_encoder, estimator, loader_report = embedding_fit._stream_s3gen_modules(torch, model_dir, device)
    all_states: list[dict[str, Any]] = []
    prepared_recon: dict[str, np.ndarray] = {}
    for row in rows:
        state = embedding_fit._prepare_flow_row(torch, flow_encoder, prepared_payload, row, device)
        all_states.append(state)
        prepared_recon[str(row["id"])] = np.asarray(row["prepared_reconstructed_mel"], dtype=np.float32)[None, ...]
    train_ids = {str(row["id"]) for row in rows if row["split"] == "train"}
    valid_ids = {str(row["id"]) for row in rows if row["split"] == "valid"}
    train_states = [state for state in all_states if str(state["id"]) in train_ids]
    valid_states = [state for state in all_states if str(state["id"]) in valid_ids]
    noises = {str(state["id"]): embedding_fit._fixed_noise(torch, state, seed=int(state["seed"])) for state in all_states}
    parity = embedding_fit._parity(torch, estimator, all_states, base_embedding, [noises[str(state["id"])] for state in all_states], prepared_recon, steps=args.steps, atol=args.parity_atol)
    native_precision = dict(matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                            cudnn_tf32=torch.backends.cudnn.allow_tf32)
    raw_targets = _attention_targets(estimator, inventory)
    base_hashes = {name: _tensor_sha256(torch, target.weight) for name, target, _shape in raw_targets}
    metadata = _base_metadata(manifest, rows, prepare_dir, manifest_path, conditionals_path, initial_source, loader_report, model_dir, inventory_path, inventory_sha, base_hashes, args=args)
    report: dict[str, Any] = {
        **metadata,
        "status": "parity_failed" if parity["status"] != "passed" else "parity_passed",
        "diagnostic_only": True,
        "experimental": True,
        "parity": parity,
        "native_parity_precision": native_precision,
        "initial_embedding": {"shape": list(initial_embedding.shape), "conditionals_sha256": initial_source["sha256"]},
        "prepared_embedding": {"shape": list(base_embedding.shape), "conditionals_sha256": _sha256(conditionals_path)},
        "rss_before_load_bytes": rss_before_load,
        "peak_rss_bytes": _peak_rss_bytes(),
        "elapsed_seconds": time.perf_counter() - started,
        "history": [],
        "disclosure": "This is a speaker-specific attention adapter trained on source-token reconstruction. It is experimental and does not establish zero-shot realism, timing, or pitch quality.",
    }
    _write_json(output_dir / "fit_report.json", report)
    # Do not install adapters before exact native parity has passed.
    if parity["status"] != "passed":
        return report
    # cuDNN TF32 amplified factorized-versus-folded perturbations to ~0.0038
    # log-mel units in the first smoke. Strict FP32 passed the unchanged gate.
    # Keep legacy replay above intact, then train and deploy in this precision.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    report['fit_precision'] = dict(matmul_tf32=False, cudnn_tf32=False)
    metadata['fit_precision'] = dict(matmul_tf32=False, cudnn_tf32=False)
    # Capture the unchanged fitted-embedding model BEFORE wrapping modules.
    # Comparing two wrapped runs would only test determinism, not zero parity.
    anchors = _collect_predictions(torch, estimator, all_states, initial_embedding, noises, steps=args.steps)
    adapters, factors = _install_attention_adapters(torch, estimator, inventory, init_seed=ADAPTER_INIT_SEED)
    zero_outputs = _collect_predictions(torch, estimator, all_states, initial_embedding, noises, steps=args.steps)
    zero_delta = _delta_stats(torch, zero_outputs, anchors)
    zero_exact = all(torch.equal(zero_outputs[row_id], anchors[row_id]) for row_id in anchors)
    report["zero_adapter"] = {"exact_equal_all_rows": bool(zero_exact), "delta": zero_delta, "rows": len(zero_outputs)}
    _write_json(output_dir / "fit_report.json", report)
    if not zero_exact:
        report["status"] = "zero_adapter_parity_failed"
        _write_json(output_dir / "fit_report.json", report)
        return report
    if args.parity_only:
        report["parity_only"] = True
        _write_json(output_dir / "fit_report.json", report)
        return report
    optimizer = torch.optim.AdamW(factors, lr=float(args.lr), weight_decay=0.0)
    epoch0_train = _evaluate(torch, estimator, train_states, initial_embedding, noises, anchors, adapters, steps=args.steps, envelope_weight=args.envelope_weight, distill_weight=args.distill_weight, regularizer=args.regularizer)
    epoch0_valid = _evaluate(torch, estimator, valid_states, initial_embedding, noises, anchors, adapters, steps=args.steps, envelope_weight=args.envelope_weight, distill_weight=args.distill_weight, regularizer=args.regularizer)
    report["epoch0"] = {"train": epoch0_train, "valid": epoch0_valid}
    best_valid = float(epoch0_valid["mean"]["total"])
    best_step = 0
    best_factors = {name: {"down": adapter.down.weight.detach().clone(), "up": adapter.up.weight.detach().clone()} for name, adapter in adapters.items()}
    checkpoint_path = output_dir / "best_attention.pt"
    _atomic_torch_save(torch, _checkpoint_payload(torch, adapters, inventory, base_hashes, metadata=metadata, step=0, valid=epoch0_valid["mean"]), checkpoint_path)
    report["best_checkpoint"] = str(checkpoint_path.resolve())
    _write_json(output_dir / "fit_report.json", report)
    stale = 0
    gradient_probe: dict[str, Any] | None = None
    for step in range(1, int(args.max_steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        train_metrics: list[dict[str, float]] = []
        for state in train_states:
            row_id = str(state["id"])
            prediction = _predict(torch, estimator, state, initial_embedding, noises[row_id], steps=args.steps)
            loss, metrics = _attention_loss(torch, prediction, state["target"], anchors[row_id], adapters, envelope_weight=args.envelope_weight, distill_weight=args.distill_weight, regularizer=args.regularizer)
            (loss / len(train_states)).backward()
            train_metrics.append(metrics)
        gradients = [factor.grad for factor in factors]
        finite = all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
        grad_norm = float(torch.sqrt(sum(torch.sum(gradient.square()) for gradient in gradients if gradient is not None)).detach().cpu())
        nonzero = grad_norm > 1e-10
        if not finite or not nonzero:
            raise RuntimeError("attention LoRA gradients are missing, non-finite, or zero")
        factor_ids = {id(factor) for factor in factors}
        frozen_grads = [parameter for parameter in estimator.parameters() if id(parameter) not in factor_ids and parameter.grad is not None]
        if frozen_grads or any(p.grad is not None for p in flow_encoder.parameters()) or initial_embedding.grad is not None:
            raise RuntimeError("a frozen decoder parameter received a gradient")
        if gradient_probe is None:
            # Up factors begin at zero, so their L2 gradient is exactly zero.
            # A nonzero initial up gradient therefore comes from reconstruction.
            up_grad_norm = float(torch.sqrt(sum(a.up.weight.grad.square().sum() for a in adapters.values())).detach().cpu())
            if not math.isfinite(up_grad_norm) or up_grad_norm <= 1e-10:
                raise RuntimeError("attention reconstruction gradient is absent")
            gradient_probe = {"finite": True, "nonzero": bool(nonzero), "gradient_norm": grad_norm, "initial_up_gradient_norm": up_grad_norm, "trainable_parameter_count": int(sum(parameter.numel() for parameter in factors))}
            report["gradient_probe"] = gradient_probe
            report["frozen_params_no_grads"] = True
        torch.nn.utils.clip_grad_norm_(factors, 1.0)
        optimizer.step()
        valid = _evaluate(torch, estimator, valid_states, initial_embedding, noises, anchors, adapters, steps=args.steps, envelope_weight=args.envelope_weight, distill_weight=args.distill_weight, regularizer=args.regularizer)
        train_mean = {key: float(np.mean([row[key] for row in train_metrics])) for key in ("total", "raw_mse", "envelope_mse", "baseline_distill_mse", "adapter_l2", "regularization")}
        report["history"].append({"step": step, "train": {"mean": train_mean, "rows": train_metrics}, "valid": valid, "gradient_norm": grad_norm})
        report["status"] = "running"
        report["elapsed_seconds"] = time.perf_counter() - started
        _write_json(output_dir / "fit_report.json", report)
        score = float(valid["mean"]["total"])
        if score < best_valid - 1e-8:
            best_valid = score
            best_step = step
            stale = 0
            best_factors = {name: {"down": adapter.down.weight.detach().clone(), "up": adapter.up.weight.detach().clone()} for name, adapter in adapters.items()}
            _atomic_torch_save(torch, _checkpoint_payload(torch, adapters, inventory, base_hashes, metadata=metadata, step=step, valid=valid["mean"]), checkpoint_path)
        else:
            stale += 1
        if stale >= int(args.patience):
            break
    with torch.no_grad():
        for name, adapter in adapters.items():
            adapter.down.weight.copy_(best_factors[name]["down"])
            adapter.up.weight.copy_(best_factors[name]["up"])
    final_train = _evaluate(torch, estimator, train_states, initial_embedding, noises, anchors, adapters, steps=args.steps, envelope_weight=args.envelope_weight, distill_weight=args.distill_weight, regularizer=args.regularizer)
    final_valid = _evaluate(torch, estimator, valid_states, initial_embedding, noises, anchors, adapters, steps=args.steps, envelope_weight=args.envelope_weight, distill_weight=args.distill_weight, regularizer=args.regularizer)
    report.update({
        "status": "fit_complete",
        "best_step": best_step,
        "best_valid_total": best_valid,
        "stopped_after_steps": len(report["history"]),
        "final_train": final_train,
        "final_valid": final_valid,
        "adoption_gate": {"improved_over_epoch0": bool(best_valid < float(epoch0_valid["mean"]["total"])), "status": "experimental_not_promoted"},
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "rss_after_fit_bytes": _rss_bytes(),
        "peak_rss_bytes": _peak_rss_bytes(),
        "elapsed_seconds": time.perf_counter() - started,
    })
    _write_json(output_dir / "fit_report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-dir", type=Path, default=DEFAULT_PREPARE_DIR)
    parser.add_argument("--initial-conditionals", type=Path, default=DEFAULT_INITIAL_CONDITIONALS)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--regularizer", type=float, default=DEFAULT_REGULARIZER)
    parser.add_argument("--envelope-weight", type=float, default=DEFAULT_ENVELOPE_WEIGHT)
    parser.add_argument("--distill-weight", type=float, default=DEFAULT_DISTILL_WEIGHT)
    parser.add_argument("--parity-atol", type=float, default=DEFAULT_PARITY_ATOL)
    parser.add_argument("--parity-only", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        report = _fit(args)
    except Exception as exc:
        report_path = output_dir / "fit_report.json"
        try:
            report = json.loads(report_path.read_text()) if report_path.exists() else {}
        except Exception:
            report = {}
        report.update({"format": FORMAT, "status": "failed", "error_type": type(exc).__name__, "error": str(exc), "peak_rss_bytes": _peak_rss_bytes()})
        _write_json(report_path, report)
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}, indent=2), file=sys.stderr)
        return 1
    print(json.dumps({"status": report.get("status"), "output_dir": str(output_dir), "zero_adapter": report.get("zero_adapter")}, indent=2))
    return 0 if report.get("status") in {"parity_passed", "fit_complete"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
