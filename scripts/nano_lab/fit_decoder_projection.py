"""Fit a tiny, speaker-specific LoRA on the S3Gen decoder output projection.

This experiment keeps the token, flow encoder, meanflow estimator, and speaker
embedding fixed.  It adds a rank-two residual to
``flow.decoder.estimator.final_proj``.  The residual is applied to the decoder
hidden sequence, so it can correct content-dependent spectral colour while a
single 192-D speaker vector cannot.  The residual is zero at step zero and can
be folded into the native/ONNX ``final_proj.weight`` at inference.

The input and alignment contract is the same as ``fit_decoder_embedding``:
audited source tokens, three native S3Gen silence IDs, native target-noise then
full-noise draw order, and no time warp.  This module does not load a model on
import.  Run model work only through ``bounded_job.py``.
"""

from __future__ import annotations

import argparse
import copy
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


ROOT = embedding_fit.ROOT
SCRIPT_DIR = embedding_fit.SCRIPT_DIR
DEFAULT_PREPARE_DIR = ROOT / "artifacts" / "nano_lab" / "mel_calibration_asmr"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "nano_lab" / "decoder_projection_fit"
DEFAULT_MODEL_DIR = ROOT / "models" / "chatterbox-nano"

FORMAT = "nano_decoder_projection_lora_v1"
CHECKPOINT_FORMAT = FORMAT
TARGET_WEIGHT_KEY = "flow.decoder.estimator.final_proj.weight"
INPUT_CHANNELS = 256
OUTPUT_CHANNELS = 80
RANK = 2
ALPHA = 2.0
SCALE = ALPHA / RANK
DEFAULT_STEPS = embedding_fit.DEFAULT_STEPS
DEFAULT_MAX_STEPS = 40
DEFAULT_PATIENCE = 3
DEFAULT_LR = 1e-3
DEFAULT_REGULARIZER = 1e-4
DEFAULT_ENVELOPE_WEIGHT = 0.25
DEFAULT_DISTILL_WEIGHT = 0.5
DEFAULT_PARITY_ATOL = embedding_fit.DEFAULT_PARITY_ATOL
ADAPTER_INIT_SEED = 1701


def _resolve(value: str | os.PathLike[str], *, base: Path = ROOT) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
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


def _tensor_sha256(torch: Any, tensor: Any) -> str:
    """Hash contiguous CPU fp32 tensor bytes, independent of Torch storage IDs."""

    if not torch.is_tensor(tensor):
        raise ValueError("tensor hash requires a Torch tensor")
    values = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def _factor_delta_numpy(up: Any, down: Any, *, alpha: float = ALPHA) -> np.ndarray:
    """Return the merged 1x1 Conv1d delta for pure tests and audits."""

    up_values = np.asarray(up, dtype=np.float32)
    down_values = np.asarray(down, dtype=np.float32)
    expected_up = (OUTPUT_CHANNELS, RANK, 1)
    expected_down = (RANK, INPUT_CHANNELS, 1)
    if up_values.shape != expected_up or down_values.shape != expected_down:
        raise ValueError(f"LoRA factors must be {expected_up} and {expected_down}")
    if not np.isfinite(up_values).all() or not np.isfinite(down_values).all():
        raise ValueError("LoRA factors contain NaN or infinity")
    if not math.isfinite(float(alpha)) or float(alpha) <= 0:
        raise ValueError("alpha must be finite and positive")
    scale = float(alpha) / float(RANK)
    return (scale * (up_values[:, :, 0] @ down_values[:, :, 0]))[:, :, None]


def _fold_projection_weight_numpy(base: Any, up: Any, down: Any, *, alpha: float = ALPHA) -> np.ndarray:
    """Pure NumPy weight fold used by tests and runtime integration review."""

    base_values = np.asarray(base, dtype=np.float32)
    if base_values.shape != (OUTPUT_CHANNELS, INPUT_CHANNELS, 1):
        raise ValueError("base final_proj weight must have shape (80,256,1)")
    if not np.isfinite(base_values).all():
        raise ValueError("base final_proj weight contains NaN or infinity")
    return base_values + _factor_delta_numpy(up, down, alpha=alpha)


def _projection_output_numpy(base: Any, hidden: Any, up: Any, down: Any, *, alpha: float = ALPHA) -> np.ndarray:
    """Apply base and folded 1x1 projections to ``[B,256,T]`` hidden data."""

    folded = _fold_projection_weight_numpy(base, up, down, alpha=alpha)
    hidden_values = np.asarray(hidden, dtype=np.float32)
    if hidden_values.ndim != 3 or hidden_values.shape[1] != INPUT_CHANNELS:
        raise ValueError("hidden must have shape [B,256,T]")
    base_matrix = np.asarray(base, dtype=np.float32)[:, :, 0]
    folded_matrix = folded[:, :, 0]
    return np.einsum("oc,bct->bot", folded_matrix, hidden_values) - np.einsum(
        "oc,bct->bot", base_matrix, hidden_values
    )


def validate_projection_checkpoint(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a saved checkpoint without importing Torch or loading a model."""

    if not isinstance(payload, Mapping) or payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("unsupported decoder projection checkpoint format")
    if payload.get("target_weight_key") != TARGET_WEIGHT_KEY:
        raise ValueError("checkpoint target weight key does not match final_proj")
    if tuple(payload.get("target_shape", ())) != (OUTPUT_CHANNELS, INPUT_CHANNELS, 1):
        raise ValueError("checkpoint target shape must be [80,256,1]")
    if int(payload.get("rank", -1)) != RANK:
        raise ValueError("decoder projection checkpoint requires rank 2")
    alpha = float(payload.get("alpha", float("nan")))
    if not math.isfinite(alpha) or abs(alpha - ALPHA) > 1e-8:
        raise ValueError("decoder projection checkpoint requires alpha=2")
    for key, shape in (
        ("down_weight", (RANK, INPUT_CHANNELS, 1)),
        ("up_weight", (OUTPUT_CHANNELS, RANK, 1)),
        ("merged_delta_weight", (OUTPUT_CHANNELS, INPUT_CHANNELS, 1)),
    ):
        if key not in payload:
            raise ValueError(f"checkpoint is missing {key}")
        values = np.asarray(payload[key], dtype=np.float32)
        if values.shape != shape:
            raise ValueError(f"checkpoint {key} has shape {values.shape}, expected {shape}")
        if not np.isfinite(values).all():
            raise ValueError(f"checkpoint {key} contains NaN or infinity")
    expected_delta = _factor_delta_numpy(payload["up_weight"], payload["down_weight"], alpha=alpha)
    if not np.allclose(np.asarray(payload["merged_delta_weight"], dtype=np.float32), expected_delta, rtol=0.0, atol=1e-7):
        raise ValueError("checkpoint merged delta does not match LoRA factors")
    if "base_target_weight" in payload:
        base = np.asarray(payload["base_target_weight"], dtype=np.float32)
        if base.shape != (OUTPUT_CHANNELS, INPUT_CHANNELS, 1) or not np.isfinite(base).all():
            raise ValueError("checkpoint base_target_weight has invalid shape or values")
    result = dict(payload)
    result["alpha"] = alpha
    result["scale"] = float(alpha) / float(RANK)
    return result


def _is_tensor(torch: Any, value: Any) -> bool:
    return bool(torch.is_tensor(value))


def _compare_conditionals_except_embedding(torch: Any, expected: Any, candidate: Any, path: tuple[str, ...] = ()) -> None:
    """Reject any initial conditionals drift except ``gen.embedding``."""

    if path == ("gen", "embedding"):
        return
    if _is_tensor(torch, expected) or _is_tensor(torch, candidate):
        if not (_is_tensor(torch, expected) and _is_tensor(torch, candidate)):
            raise ValueError(f"conditioning type mismatch at {'.'.join(path)}")
        if expected.dtype != candidate.dtype or tuple(expected.shape) != tuple(candidate.shape) or not torch.equal(expected, candidate):
            raise ValueError(f"conditioning tensor mismatch at {'.'.join(path)}")
        return
    if isinstance(expected, Mapping) or isinstance(candidate, Mapping):
        if not (isinstance(expected, Mapping) and isinstance(candidate, Mapping)):
            raise ValueError(f"conditioning mapping mismatch at {'.'.join(path)}")
        if set(expected) != set(candidate):
            raise ValueError(f"conditioning keys mismatch at {'.'.join(path)}")
        for key in expected:
            _compare_conditionals_except_embedding(torch, expected[key], candidate[key], path + (str(key),))
        return
    if expected != candidate:
        raise ValueError(f"conditioning value mismatch at {'.'.join(path)}")


def _load_conditionals_pair(torch: Any, prepared_path: Path, initial_path: Path | None) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load prepared and optional initial conditionals with strict content checks."""

    prepared = embedding_fit._load_conditionals_payload(torch, prepared_path)
    if initial_path is None:
        initial = copy.deepcopy(prepared)
        source = {"path": str(prepared_path.resolve()), "sha256": _sha256(prepared_path), "provided": False}
    else:
        initial_path = initial_path.resolve()
        if not initial_path.exists():
            raise FileNotFoundError(initial_path)
        initial = embedding_fit._load_conditionals_payload(torch, initial_path)
        _compare_conditionals_except_embedding(torch, prepared, initial)
        source = {"path": str(initial_path), "sha256": _sha256(initial_path), "provided": True}
    base_embedding = prepared["gen"]["embedding"]
    initial_embedding = initial["gen"]["embedding"]
    for label, value in (("prepared", base_embedding), ("initial", initial_embedding)):
        if tuple(value.shape) != (1, embedding_fit.EMBEDDING_DIM) or not torch.isfinite(value).all():
            raise ValueError(f"{label} conditionals embedding must be finite with shape (1,192)")
    return prepared, initial, source


def _build_projection_adapter(torch: Any, base_projection: Any, *, rank: int = RANK, alpha: float = ALPHA, init_seed: int = ADAPTER_INIT_SEED) -> Any:
    """Wrap final_proj with a zero-initialised rank-two residual."""

    import torch.nn as nn

    if int(rank) != RANK or not math.isfinite(float(alpha)) or abs(float(alpha) - ALPHA) > 1e-8:
        raise ValueError("projection experiment is fixed to rank=2, alpha=2")
    if int(init_seed) < 0:
        raise ValueError("adapter initialization seed must be non-negative")

    class FinalProjectionLoRA(nn.Module):
        def __init__(self, base: Any) -> None:
            super().__init__()
            self.base = base
            for parameter in self.base.parameters():
                parameter.requires_grad_(False)
            device = base.weight.device
            dtype = base.weight.dtype
            # Keep adapter initialization reproducible without consuming the
            # caller's global Torch RNG state.  The up branch is zero, so the
            # module is an exact no-op before the first optimizer step.
            devices = []
            if device.type == "cuda":
                devices = [device.index if device.index is not None else torch.cuda.current_device()]
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(int(init_seed))
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(int(init_seed))
                self.down = nn.Conv1d(INPUT_CHANNELS, RANK, kernel_size=1, bias=False, device=device, dtype=dtype)
                self.up = nn.Conv1d(RANK, OUTPUT_CHANNELS, kernel_size=1, bias=False, device=device, dtype=dtype)
                nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5.0))
                nn.init.zeros_(self.up.weight)
            self.alpha = float(alpha)
            self.scale = float(alpha) / float(RANK)
            self.init_seed = int(init_seed)

        def forward(self, hidden: Any) -> Any:
            return self.base(hidden) + self.scale * self.up(self.down(hidden))

        def merged_delta_weight(self) -> Any:
            return self.scale * torch.matmul(self.up.weight[:, :, 0], self.down.weight[:, :, 0]).unsqueeze(-1)

    return FinalProjectionLoRA(base_projection)


def _install_projection_adapter(estimator: Any, torch: Any, *, init_seed: int = ADAPTER_INIT_SEED) -> Any:
    base_projection = estimator.estimator.final_proj
    adapter = _build_projection_adapter(torch, base_projection, init_seed=init_seed)
    estimator.estimator.final_proj = adapter
    return adapter


def _predict(torch: Any, estimator: Any, state: Mapping[str, Any], embedding: Any, noise: Any, *, steps: int) -> Any:
    generated = embedding_fit._basic_euler(torch, estimator, state, embedding, noise, steps=steps)
    return embedding_fit._aligned_prediction(generated[:, :, state["prompt_len"]:], state)


def _collect_predictions(torch: Any, estimator: Any, states: Sequence[Mapping[str, Any]], embedding: Any, noises: Mapping[str, Any], *, steps: int) -> dict[str, Any]:
    outputs: dict[str, Any] = {}
    with torch.no_grad():
        for state in states:
            outputs[str(state["id"])] = _predict(torch, estimator, state, embedding, noises[str(state["id"])], steps=steps).detach()
    return outputs


def _delta_stats(torch: Any, predictions: Mapping[str, Any], anchors: Mapping[str, Any]) -> dict[str, float]:
    values: list[Any] = []
    for row_id, prediction in predictions.items():
        delta = prediction - anchors[row_id]
        values.append(delta.detach().float().reshape(-1))
    if not values:
        return {"rms": 0.0, "max_abs": 0.0}
    joined = torch.cat(values)
    return {"rms": float(torch.sqrt(torch.mean(joined.square())).cpu()), "max_abs": float(torch.max(torch.abs(joined)).cpu())}


def _projection_loss(torch: Any, predicted: Any, target: Any, anchor: Any, adapter: Any, *, envelope_weight: float, distill_weight: float, regularizer: float) -> tuple[Any, dict[str, float]]:
    import torch.nn.functional as F

    raw = F.mse_loss(predicted, target)
    envelope = F.mse_loss(predicted.mean(dim=2), target.mean(dim=2))
    distill = F.mse_loss(predicted, anchor)
    l2 = adapter.down.weight.square().mean() + adapter.up.weight.square().mean()
    total = raw + float(envelope_weight) * envelope + float(distill_weight) * distill + float(regularizer) * l2
    return total, {
        "total": float(total.detach().cpu()),
        "raw_mse": float(raw.detach().cpu()),
        "envelope_mse": float(envelope.detach().cpu()),
        "baseline_distill_mse": float(distill.detach().cpu()),
        "adapter_l2": float(l2.detach().cpu()),
        "regularization": float((float(regularizer) * l2).detach().cpu()),
    }


def _evaluate(torch: Any, estimator: Any, states: Sequence[Mapping[str, Any]], embedding: Any, noises: Mapping[str, Any], anchors: Mapping[str, Any], *, steps: int, envelope_weight: float, distill_weight: float, regularizer: float, adapter: Any) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    predictions: dict[str, Any] = {}
    with torch.no_grad():
        for state in states:
            row_id = str(state["id"])
            predicted = _predict(torch, estimator, state, embedding, noises[row_id], steps=steps)
            predictions[row_id] = predicted.detach()
            _, metrics = _projection_loss(torch, predicted, state["target"], anchors[row_id], adapter, envelope_weight=envelope_weight, distill_weight=distill_weight, regularizer=regularizer)
            metrics["id"] = row_id
            rows.append(metrics)
    keys = ("total", "raw_mse", "envelope_mse", "baseline_distill_mse", "adapter_l2", "regularization")
    mean = {key: float(np.mean([row[key] for row in rows])) for key in keys}
    return {"mean": mean, "rows": rows, "delta": _delta_stats(torch, predictions, anchors)}


def _atomic_torch_save(torch: Any, payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _checkpoint_payload(torch: Any, adapter: Any, base_weight: Any, *, metadata: Mapping[str, Any], step: int, valid: Mapping[str, Any]) -> dict[str, Any]:
    delta = adapter.merged_delta_weight().detach().cpu().float().contiguous()
    down = adapter.down.weight.detach().cpu().float().contiguous()
    up = adapter.up.weight.detach().cpu().float().contiguous()
    base = base_weight.detach().cpu().float().contiguous()
    payload = {
        "format": CHECKPOINT_FORMAT,
        "target_weight_key": TARGET_WEIGHT_KEY,
        "target_shape": [OUTPUT_CHANNELS, INPUT_CHANNELS, 1],
        "rank": RANK,
        "alpha": ALPHA,
        "scale": SCALE,
        "down_weight": down,
        "up_weight": up,
        "merged_delta_weight": delta,
        "base_target_weight": base,
        "base_target_weight_sha256": _tensor_sha256(torch, base),
        "step": int(step),
        "valid": dict(valid),
        "metadata": dict(metadata),
    }
    # Validate the serialized NumPy-equivalent fields before writing.  This
    # catches an accidental transpose or a stale adapter state at checkpoint
    # time, while keeping the runtime folding contract explicit.
    validate_projection_checkpoint(
        {
            "format": payload["format"],
            "target_weight_key": payload["target_weight_key"],
            "target_shape": payload["target_shape"],
            "rank": payload["rank"],
            "alpha": payload["alpha"],
            "down_weight": down.numpy(),
            "up_weight": up.numpy(),
            "merged_delta_weight": delta.numpy(),
        }
    )
    return payload


def _base_metadata(manifest: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], prepare_dir: Path, manifest_path: Path, conditionals_path: Path, initial_source: Mapping[str, Any], loader_report: Mapping[str, Any], model_dir: Path, base_weight_sha: str, *, args: argparse.Namespace) -> dict[str, Any]:
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
        "target_weight_key": TARGET_WEIGHT_KEY,
        "target_shape": [OUTPUT_CHANNELS, INPUT_CHANNELS, 1],
        "base_target_weight_sha256": base_weight_sha,
        "train_ids": [str(row["id"]) for row in rows if row["split"] == "train"],
        "valid_ids": [str(row["id"]) for row in rows if row["split"] == "valid"],
        "native_row_seeds": {str(row["id"]): int(row["seed"]) for row in rows},
        "target_alignment": {
            "speech_tokens_plus_s3gen_sil": 3,
            "source_frames": "2*N or 2*N-1",
            "generated_frames": "2*(N+3)",
            "trim_policy": "reuse fit_decoder_embedding._aligned_prediction; no time warp/resampling/DTW",
        },
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
        "loader": dict(loader_report),
        "adapter_init_seed": ADAPTER_INIT_SEED,
        "mask_policy": "ConditionalDecoder multiplies final_proj output by its mask after the adapter; adapter forward is unmasked.",
        "command": list(sys.argv),
    }


def _fit(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    started = time.perf_counter()
    if args.steps != DEFAULT_STEPS:
        raise ValueError("--steps must equal 2 for native meanflow alignment")
    if args.max_steps < 1 or args.max_steps > DEFAULT_MAX_STEPS:
        raise ValueError(f"--max-steps must be in 1..{DEFAULT_MAX_STEPS}")
    if args.patience < 1:
        raise ValueError("--patience must be >= 1")
    numeric_args = (args.lr, args.regularizer, args.envelope_weight, args.distill_weight, args.parity_atol)
    if not all(math.isfinite(float(value)) for value in numeric_args):
        raise ValueError("numeric hyperparameters must be finite")
    if args.lr <= 0 or args.lr > 1.0 or args.regularizer < 0 or args.regularizer > 10.0 or args.envelope_weight < 0 or args.envelope_weight > 10.0 or args.distill_weight < 0 or args.distill_weight > 10.0 or args.parity_atol <= 0 or args.parity_atol > 1.0:
        raise ValueError("learning rate must be positive; loss weights must be non-negative")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    prepare_dir = _resolve(args.prepare_dir)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest, rows, conditionals_path, _cache, manifest_path = embedding_fit._validate_prepare_inputs(prepare_dir)
    prepared_payload, initial_payload, initial_source = _load_conditionals_pair(torch, conditionals_path, _resolve(args.initial_conditionals) if args.initial_conditionals else None)
    base_embedding = prepared_payload["gen"]["embedding"].detach().to(device=device, dtype=torch.float32)
    initial_embedding = initial_payload["gen"]["embedding"].detach().to(device=device, dtype=torch.float32)
    initial_embedding.requires_grad_(False)
    model_dir = _resolve(args.model_dir)
    rss_before_load = _rss_bytes()
    flow_encoder, estimator, loader_report = embedding_fit._stream_s3gen_modules(torch, model_dir, device)
    torch.set_grad_enabled(True)
    all_states: list[dict[str, Any]] = []
    prepared_recon: dict[str, np.ndarray] = {}
    for row in rows:
        state = embedding_fit._prepare_flow_row(torch, flow_encoder, prepared_payload, row, device)
        all_states.append(state)
        prepared_recon[str(row["id"])] = np.asarray(row["prepared_reconstructed_mel"], dtype=np.float32)[None, ...]
    train_states = [state for state in all_states if str(state["id"]) in {str(row["id"]) for row in rows if row["split"] == "train"}]
    valid_states = [state for state in all_states if str(state["id"]) in {str(row["id"]) for row in rows if row["split"] == "valid"}]
    noises = {str(state["id"]): embedding_fit._fixed_noise(torch, state, seed=int(state["seed"])) for state in all_states}
    parity = embedding_fit._parity(torch, estimator, all_states, base_embedding, [noises[str(state["id"])] for state in all_states], prepared_recon, steps=args.steps, atol=args.parity_atol)
    base_projection = estimator.estimator.final_proj
    base_weight = base_projection.weight.detach().clone()
    if tuple(base_weight.shape) != (OUTPUT_CHANNELS, INPUT_CHANNELS, 1):
        raise RuntimeError(f"unexpected final_proj weight shape {tuple(base_weight.shape)}")
    base_weight_sha = _tensor_sha256(torch, base_weight)
    # The pre-adapter embedding output is the distillation anchor.  When an
    # optional fitted embedding is supplied, the original embedding still owns
    # the native parity gate above; the fitted embedding is not silently treated
    # as a cached native reconstruction.
    anchors = _collect_predictions(torch, estimator, all_states, initial_embedding, noises, steps=args.steps)
    adapter = _install_projection_adapter(estimator, torch, init_seed=ADAPTER_INIT_SEED)
    zero_outputs = _collect_predictions(torch, estimator, all_states, initial_embedding, noises, steps=args.steps)
    zero_delta = _delta_stats(torch, zero_outputs, anchors)
    zero_exact = all(torch.equal(zero_outputs[row_id], anchors[row_id]) for row_id in anchors)
    metadata = _base_metadata(manifest, rows, prepare_dir, manifest_path, conditionals_path, initial_source, loader_report, model_dir, base_weight_sha, args=args)
    report: dict[str, Any] = {
        **metadata,
        "status": "parity_failed" if parity["status"] != "passed" else "parity_passed",
        "diagnostic_only": True,
        "experimental": True,
        "initial_embedding": {"shape": list(initial_embedding.shape), "norm": float(torch.linalg.vector_norm(initial_embedding).cpu()), "conditionals_sha256": initial_source["sha256"]},
        "prepared_embedding": {"shape": list(base_embedding.shape), "norm": float(torch.linalg.vector_norm(base_embedding).cpu()), "conditionals_sha256": _sha256(conditionals_path)},
        "parity": parity,
        "zero_adapter": {"exact_equal_all_rows": bool(zero_exact), "delta": zero_delta, "rows": len(zero_outputs)},
        "rss_before_load_bytes": rss_before_load,
        "peak_rss_bytes": _peak_rss_bytes(),
        "elapsed_seconds": time.perf_counter() - started,
        "history": [],
        "disclosure": "This decoder projection fit is a timbre hypothesis. It uses source-token reconstruction targets and does not establish zero-shot realism or fix timing/pitch alignment.",
    }
    _write_json(output_dir / "fit_report.json", report)
    if parity["status"] != "passed":
        return report
    if not zero_exact:
        report["status"] = "zero_adapter_parity_failed"
        _write_json(output_dir / "fit_report.json", report)
        return report
    if args.parity_only:
        report["parity_only"] = True
        _write_json(output_dir / "fit_report.json", report)
        return report

    optimizer = torch.optim.AdamW([adapter.down.weight, adapter.up.weight], lr=float(args.lr), weight_decay=0.0)
    epoch0_train = _evaluate(torch, estimator, train_states, initial_embedding, noises, anchors, steps=args.steps, envelope_weight=args.envelope_weight, distill_weight=args.distill_weight, regularizer=args.regularizer, adapter=adapter)
    epoch0_valid = _evaluate(torch, estimator, valid_states, initial_embedding, noises, anchors, steps=args.steps, envelope_weight=args.envelope_weight, distill_weight=args.distill_weight, regularizer=args.regularizer, adapter=adapter)
    report["epoch0"] = {"train": epoch0_train, "valid": epoch0_valid}
    best_valid = float(epoch0_valid["mean"]["total"])
    best_step = 0
    best_adapter = {"down": adapter.down.weight.detach().clone(), "up": adapter.up.weight.detach().clone()}
    checkpoint_path = output_dir / "best_projection.pt"
    _atomic_torch_save(torch, _checkpoint_payload(torch, adapter, base_weight, metadata=metadata, step=0, valid=epoch0_valid["mean"]), checkpoint_path)
    report["best_checkpoint"] = str(checkpoint_path.resolve())
    report["elapsed_seconds"] = time.perf_counter() - started
    _write_json(output_dir / "fit_report.json", report)
    stale = 0
    gradient_probe: dict[str, Any] | None = None
    projection_params = [adapter.down.weight, adapter.up.weight]
    for step in range(1, int(args.max_steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        train_metrics: list[dict[str, float]] = []
        for state in train_states:
            row_id = str(state["id"])
            prediction = _predict(torch, estimator, state, initial_embedding, noises[row_id], steps=args.steps)
            loss, metrics = _projection_loss(torch, prediction, state["target"], anchors[row_id], adapter, envelope_weight=args.envelope_weight, distill_weight=args.distill_weight, regularizer=args.regularizer)
            (loss / len(train_states)).backward()
            train_metrics.append(metrics)
        grads = [parameter.grad for parameter in projection_params]
        finite_grad = all(gradient is not None and torch.isfinite(gradient).all() for gradient in grads)
        grad_norm = float(torch.sqrt(sum(torch.sum(gradient.square()) for gradient in grads if gradient is not None)).detach().cpu())
        nonzero_grad = grad_norm > 1e-10
        projection_ids = {id(parameter) for parameter in projection_params}
        frozen_grads = [parameter for parameter in estimator.parameters() if id(parameter) not in projection_ids and parameter.grad is not None]
        if not finite_grad or not nonzero_grad:
            raise RuntimeError("LoRA gradients are missing, non-finite, or zero")
        if frozen_grads:
            raise RuntimeError("a frozen decoder parameter received a gradient")
        if gradient_probe is None:
            gradient_probe = {"finite": True, "nonzero": bool(nonzero_grad), "gradient_norm": grad_norm, "trainable_parameter_count": int(sum(parameter.numel() for parameter in projection_params))}
            report["gradient_probe"] = gradient_probe
            report["frozen_params_no_grads"] = True
        torch.nn.utils.clip_grad_norm_(projection_params, 1.0)
        optimizer.step()
        valid = _evaluate(torch, estimator, valid_states, initial_embedding, noises, anchors, steps=args.steps, envelope_weight=args.envelope_weight, distill_weight=args.distill_weight, regularizer=args.regularizer, adapter=adapter)
        train_mean = {key: float(np.mean([row[key] for row in train_metrics])) for key in ("total", "raw_mse", "envelope_mse", "baseline_distill_mse", "adapter_l2", "regularization")}
        record = {"step": step, "train": {"mean": train_mean, "rows": train_metrics}, "valid": valid, "gradient_norm": grad_norm}
        report["history"].append(record)
        report["elapsed_seconds"] = time.perf_counter() - started
        report["status"] = "running"
        _write_json(output_dir / "fit_report.json", report)
        score = float(valid["mean"]["total"])
        if score < best_valid - 1e-8:
            best_valid = score
            best_step = step
            stale = 0
            best_adapter = {"down": adapter.down.weight.detach().clone(), "up": adapter.up.weight.detach().clone()}
            _atomic_torch_save(torch, _checkpoint_payload(torch, adapter, base_weight, metadata=metadata, step=step, valid=valid["mean"]), checkpoint_path)
        else:
            stale += 1
        if stale >= int(args.patience):
            break
    with torch.no_grad():
        adapter.down.weight.copy_(best_adapter["down"])
        adapter.up.weight.copy_(best_adapter["up"])
    final_valid = _evaluate(torch, estimator, valid_states, initial_embedding, noises, anchors, steps=args.steps, envelope_weight=args.envelope_weight, distill_weight=args.distill_weight, regularizer=args.regularizer, adapter=adapter)
    final_train = _evaluate(torch, estimator, train_states, initial_embedding, noises, anchors, steps=args.steps, envelope_weight=args.envelope_weight, distill_weight=args.distill_weight, regularizer=args.regularizer, adapter=adapter)
    delta = adapter.merged_delta_weight().detach().float()
    report.update({
        "status": "fit_complete",
        "best_step": best_step,
        "best_valid_total": best_valid,
        "stopped_after_steps": len(report["history"]),
        "final_train": final_train,
        "final_valid": final_valid,
        "output_delta_weight": {"rms": float(torch.sqrt(torch.mean(delta.square())).cpu()), "max_abs": float(torch.max(torch.abs(delta)).cpu()), "sha256": _tensor_sha256(torch, delta)},
        "adoption_gate": {"improved_over_epoch0": bool(best_valid < float(epoch0_valid["mean"]["total"])), "status": "experimental_not_promoted"},
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256(checkpoint_path) if checkpoint_path.exists() else None,
        "rss_after_fit_bytes": _rss_bytes(),
        "peak_rss_bytes": _peak_rss_bytes(),
        "elapsed_seconds": time.perf_counter() - started,
    })
    _write_json(output_dir / "fit_report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-dir", type=Path, default=DEFAULT_PREPARE_DIR)
    parser.add_argument("--initial-conditionals", type=Path, default=None, help="optional fitted conditionals; only gen.embedding may differ")
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
        _fit(args)
    except Exception as exc:
        report_path = output_dir / "fit_report.json"
        try:
            report = json.loads(report_path.read_text()) if report_path.exists() else {}
        except Exception:
            report = {}
        report.update({"format": FORMAT, "status": "failed", "error_type": type(exc).__name__, "error": str(exc), "elapsed_seconds": report.get("elapsed_seconds"), "peak_rss_bytes": _peak_rss_bytes()})
        _write_json(report_path, report)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
