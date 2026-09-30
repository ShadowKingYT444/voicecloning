"""Verify the fitted Nano T3 adapter against a staged ONNX graph.

The two commands run in separate processes by design:

``reference``
    Loads only Nano T3 with the optimized cached-T3 loader, attaches the
    requested LoRA checkpoint, and writes deterministic prefill/decode inputs
    and Torch outputs.  The baseline pass is generated before the adapter is
    attached, so the report also proves that the adapter has a non-zero effect.

``verify``
    Loads only :class:`T3UnifiedOrtRuntime` and compares the saved adapted
    outputs with pure ONNX Runtime.  Decode cases use the cache saved from the
    adapted Torch prefill.  They do not feed an ORT-generated cache forward.

The default gate is the existing 3e-4 absolute/relative ``allclose`` gate.
The stricter 1e-4 gate is available explicitly.  This helper only exercises
contexts 32 and 128.  The known base T3 context-400 cache mismatch remains a
separate disclosed limitation, so a short-context pass cannot promote the
streaming graph to production.

Neither command imports Torch and ONNX Runtime in the same process.
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


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = ROOT / "models" / "chatterbox-nano"
DEFAULT_ONNX_DIR = ROOT / "artifacts" / "nano_lab" / "onnx_staged"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "nano_lab" / "fitted_t3_onnx_reference"
DEFAULT_ADAPTER = ROOT / "artifacts" / "nano_lab" / "adapter_aligned_all_attn.pt"
T3_LAYERS = 12
T3_HEADS = 12
T3_HEAD_DIM = 64
T3_HIDDEN = 768
T3_OUTPUTS = 1 + T3_LAYERS * 2
DEFAULT_LENGTHS = (32, 128)
DEFAULT_SEED = 20260929
DEFAULT_GATE = 3e-4


def _sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_or_none(path: Path) -> str | None:
    return _sha256(path) if path.exists() else None


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


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _resolve(value: str | os.PathLike[str], *, base: Path = ROOT) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _as_float32(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return array


def _empty_cache(batch: int = 1) -> tuple[np.ndarray, ...]:
    return tuple(
        np.zeros((batch, T3_HEADS, 0, T3_HEAD_DIM), dtype=np.float32)
        for _ in range(T3_LAYERS * 2)
    )


def _output_arrays(value: Sequence[Any]) -> tuple[np.ndarray, ...]:
    converted = []
    for item in value:
        # Keep Torch lazy.  The reference command calls this helper with Torch
        # tensors, while the pure verifier calls it only with NumPy outputs.
        if hasattr(item, "detach"):
            item = item.detach().cpu().numpy()
        converted.append(np.asarray(item, dtype=np.float32).copy())
    arrays = tuple(converted)
    if len(arrays) != T3_OUTPUTS:
        raise ValueError(f"T3 output count {len(arrays)} != {T3_OUTPUTS}")
    for index, array in enumerate(arrays):
        if not np.isfinite(array).all():
            raise ValueError(f"T3 output {index} contains NaN or infinity")
    return arrays


def compare_array(actual: Any, expected: Any, *, atol: float, rtol: float) -> dict[str, Any]:
    """Return the same absolute/relative diagnostics used by staged checks."""

    got = np.asarray(actual, dtype=np.float64)
    want = np.asarray(expected, dtype=np.float64)
    if got.shape != want.shape:
        return {
            "status": "shape_mismatch",
            "actual_shape": list(got.shape),
            "expected_shape": list(want.shape),
            "elements": int(got.size),
        }
    difference = np.abs(got - want)
    max_abs = float(np.max(difference)) if difference.size else 0.0
    max_relative = float(np.max(difference / np.maximum(np.abs(want), 1e-5))) if difference.size else 0.0
    relative_l2 = float(np.linalg.norm(difference.ravel()) / max(np.linalg.norm(want.ravel()), 1e-12))
    outside = int(np.count_nonzero(~np.isclose(got, want, atol=float(atol), rtol=float(rtol))))
    return {
        "status": "passed" if outside == 0 else "mismatch",
        "elements": int(got.size),
        "max_abs": max_abs,
        "max_relative": max_relative,
        "relative_l2": relative_l2,
        "outside_tolerance_count": outside,
        "atol": float(atol),
        "rtol": float(rtol),
    }


def _output_metrics(actual: Sequence[Any], expected: Sequence[Any], *, atol: float, rtol: float) -> dict[str, Any]:
    if len(actual) != len(expected):
        return {
            "status": "output_count_mismatch",
            "actual_outputs": len(actual),
            "expected_outputs": len(expected),
        }
    logits = compare_array(actual[0], expected[0], atol=atol, rtol=rtol)
    cache = [compare_array(got, want, atol=atol, rtol=rtol) for got, want in zip(actual[1:], expected[1:])]
    all_metrics = [logits, *cache]
    return {
        "status": "passed" if all(metric["status"] == "passed" for metric in all_metrics) else "mismatch",
        "logits": logits,
        "cache": cache,
        "max_abs": max(float(metric.get("max_abs", 0.0)) for metric in all_metrics),
        "max_relative": max(float(metric.get("max_relative", 0.0)) for metric in all_metrics),
        "relative_l2_max": max(float(metric.get("relative_l2", 0.0)) for metric in all_metrics),
        "outside_tolerance_count": sum(int(metric.get("outside_tolerance_count", 0)) for metric in all_metrics),
    }


def _baseline_effect(actual: Sequence[Any], baseline: Sequence[Any]) -> dict[str, Any]:
    if len(actual) != len(baseline):
        return {"status": "output_count_mismatch", "actual_outputs": len(actual), "baseline_outputs": len(baseline)}
    metrics = []
    for index, (adapted, base) in enumerate(zip(actual, baseline)):
        got = np.asarray(adapted, dtype=np.float64)
        want = np.asarray(base, dtype=np.float64)
        if got.shape != want.shape:
            metrics.append({"index": index, "status": "shape_mismatch"})
            continue
        difference = np.abs(got - want)
        metrics.append(
            {
                "index": index,
                "max_abs": float(np.max(difference)) if difference.size else 0.0,
                "relative_l2": float(np.linalg.norm(difference.ravel()) / max(np.linalg.norm(want.ravel()), 1e-12)),
                "changed_elements_1e-8": int(np.count_nonzero(difference > 1e-8)),
                "elements": int(got.size),
            }
        )
    changed = sum(metric.get("changed_elements_1e-8", 0) for metric in metrics)
    max_abs = max((metric.get("max_abs", 0.0) for metric in metrics), default=0.0)
    return {"status": "nonzero" if changed else "zero", "max_abs": max_abs, "changed_elements_1e-8": changed, "outputs": metrics}


def _input_names() -> list[str]:
    return ["inputs_embeds", "cache_position", *[f"past_{index}" for index in range(T3_LAYERS * 2)]]


def _output_names() -> list[str]:
    return [f"output_{index}" for index in range(T3_OUTPUTS)]


def _torch_empty_cache(torch: Any, device: Any) -> tuple[Any, ...]:
    return tuple(
        torch.zeros((1, T3_HEADS, 0, T3_HEAD_DIM), device=device, dtype=torch.float32)
        for _ in range(T3_LAYERS * 2)
    )


def _case_name(length: int, kind: str) -> str:
    return f"L{int(length)}.{kind}"


def _adapter_sidecar(path: Path) -> dict[str, Any]:
    sidecar = path.with_suffix(".json")
    result: dict[str, Any] = {"path": str(path.resolve()), "sha256": _sha256(path)}
    if not sidecar.exists():
        result["sidecar_missing"] = True
        return result
    result["sidecar_path"] = str(sidecar.resolve())
    result["sidecar_sha256"] = _sha256(sidecar)
    payload = json.loads(sidecar.read_text())
    if isinstance(payload, dict):
        for key in ("format", "config", "metadata"):
            if key in payload:
                result[key] = payload[key]
    return result


def _save_examples(path: Path, cases: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {"path": path.name, "cases": {}}
    for case_name, case in cases.items():
        inputs = case["inputs"]
        outputs = case["outputs"]
        baseline = case.get("baseline_outputs")
        input_names = list(inputs)
        output_names = _output_names()
        for name, value in inputs.items():
            arrays[f"case.{case_name}.input.{name}"] = np.asarray(value)
        for index, value in enumerate(outputs):
            arrays[f"case.{case_name}.output.{output_names[index]}"] = np.asarray(value)
        if baseline is not None:
            for index, value in enumerate(baseline):
                arrays[f"case.{case_name}.baseline.{output_names[index]}"] = np.asarray(value)
        metadata["cases"][case_name] = {
            "input_names": input_names,
            "output_names": output_names,
            "baseline_output_names": output_names if baseline is not None else [],
        }
    np.savez(path, **arrays)
    metadata["bytes"] = path.stat().st_size
    return metadata


def reference(args: argparse.Namespace) -> dict[str, Any]:
    """Generate deterministic adapted Torch references and saved raw feeds."""

    if not args.lengths:
        raise ValueError("at least one context length is required")
    if any(int(value) < 1 for value in args.lengths):
        raise ValueError("context lengths must be positive")
    adapter_path = _resolve(args.adapter)
    model_dir = _resolve(args.model_dir)
    if not adapter_path.exists():
        raise FileNotFoundError(adapter_path)
    checkpoint = model_dir / "t3_nano_v1.safetensors"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    scale = float(args.scale)
    if not math.isfinite(scale) or scale < 0.0:
        raise ValueError("--scale must be finite and >= 0")
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))

    # Model-backed imports remain inside this command.  The verify command
    # imports only the NumPy ORT runtime below.
    import torch
    from adaptation import adapter_parameter_count, load_adapter
    from onnx_t3_core import NanoT3KV
    from t3_training import load_cached_t3

    torch.set_num_threads(max(1, min(int(args.threads), 2)))
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for reference generation but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    if args.seed is not None:
        torch.manual_seed(int(args.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.seed))
    started = time.perf_counter()
    rss_before = _rss_bytes()
    model = load_cached_t3(model_dir, device)
    model.t3.eval()
    wrapper = NanoT3KV(model.t3).eval()
    cases: dict[str, dict[str, Any]] = {}
    for length in [int(value) for value in args.lengths]:
        torch.manual_seed(int(args.seed) + length)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(args.seed) + length)
        embeds = torch.randn((1, length, T3_HIDDEN), device=device, dtype=torch.float32)
        positions = torch.arange(length, device=device, dtype=torch.long)
        decode_embeds = torch.randn((1, 1, T3_HIDDEN), device=device, dtype=torch.float32)
        decode_position = torch.tensor([length], device=device, dtype=torch.long)
        cases[str(length)] = {
            "inputs": {
                "inputs_embeds": embeds.detach().cpu().numpy(),
                "cache_position": positions.detach().cpu().numpy(),
                **{f"past_{index}": value for index, value in enumerate(_empty_cache())},
            },
            "decode_embeds": decode_embeds.detach().cpu().numpy(),
            "decode_position": decode_position.detach().cpu().numpy(),
        }

    # The frozen reference pass is generated before attaching any adapter.
    with torch.inference_mode():
        for length_key, case in cases.items():
            embeds = torch.from_numpy(case["inputs"]["inputs_embeds"]).to(device)
            positions = torch.from_numpy(case["inputs"]["cache_position"]).to(device)
            base_prefill_torch = wrapper(embeds, positions, *_torch_empty_cache(torch, device))
            base_prefill = _output_arrays(base_prefill_torch)
            # The wrapper returns (logits, key/value ...).  Save the base
            # cache only for the baseline diagnostic; adapted decode inputs are
            # populated after the adapter pass below.
            case["base_prefill"] = base_prefill
            case["base_cache"] = tuple(value.detach() for value in base_prefill_torch[1:])
            decode_embeds = torch.from_numpy(case["decode_embeds"]).to(device)
            decode_position = torch.from_numpy(case["decode_position"]).to(device)
            base_decode_torch = wrapper(decode_embeds, decode_position, *case["base_cache"])
            base_decode = _output_arrays(base_decode_torch)
            case["base_decode"] = base_decode

    adapters = load_adapter(model.t3, adapter_path)
    for module in adapters.values():
        module.scaling *= scale
    adapter_count = adapter_parameter_count(adapters)

    with torch.inference_mode():
        for length_key, case in cases.items():
            embeds = torch.from_numpy(case["inputs"]["inputs_embeds"]).to(device)
            positions = torch.from_numpy(case["inputs"]["cache_position"]).to(device)
            adapted_prefill_torch = wrapper(embeds, positions, *_torch_empty_cache(torch, device))
            adapted_prefill = _output_arrays(adapted_prefill_torch)
            case["adapted_prefill"] = adapted_prefill
            case["adapted_cache"] = tuple(value.detach() for value in adapted_prefill_torch[1:])
            decode_embeds = torch.from_numpy(case["decode_embeds"]).to(device)
            decode_position = torch.from_numpy(case["decode_position"]).to(device)
            adapted_decode_torch = wrapper(decode_embeds, decode_position, *case["adapted_cache"])
            adapted_decode = _output_arrays(adapted_decode_torch)
            case["adapted_decode"] = adapted_decode

    saved_cases: dict[str, dict[str, Any]] = {}
    baseline_effect: dict[str, Any] = {}
    for length_key, case in cases.items():
        prefill_inputs = dict(case["inputs"])
        decode_inputs = {
            "inputs_embeds": case["decode_embeds"],
            "cache_position": case["decode_position"],
            **{
                f"past_{index}": value.detach().cpu().numpy()
                if hasattr(value, "detach")
                else np.asarray(value)
                for index, value in enumerate(case["adapted_cache"])
            },
        }
        prefill_name = _case_name(int(length_key), "prefill")
        decode_name = _case_name(int(length_key), "decode")
        saved_cases[prefill_name] = {
            "inputs": prefill_inputs,
            "outputs": case["adapted_prefill"],
            "baseline_outputs": case["base_prefill"],
        }
        saved_cases[decode_name] = {
            "inputs": decode_inputs,
            "outputs": case["adapted_decode"],
            "baseline_outputs": case["base_decode"],
        }
        baseline_effect[prefill_name] = _baseline_effect(case["adapted_prefill"], case["base_prefill"])
        baseline_effect[decode_name] = _baseline_effect(case["adapted_decode"], case["base_decode"])

    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    examples = _save_examples(output_dir / "examples.npz", saved_cases)
    adapter_info = _adapter_sidecar(adapter_path)
    metadata = {
        "format": "nano_fitted_t3_onnx_reference_v1",
        "stage": "t3",
        "status": "reference_generated",
        "model_dir": str(model_dir),
        "model_checkpoint": str(checkpoint),
        "model_checkpoint_sha256": _sha256(checkpoint),
        "adapter": adapter_info,
        "adapter_scale": scale,
        "adapter_parameter_count": int(adapter_count),
        "reference_wrapper": "onnx_t3_core.NanoT3KV around t3_training.load_cached_t3",
        "device": str(device),
        "dtype": "torch.float32",
        "seed": int(args.seed),
        "threads": int(args.threads),
        "lengths": [int(value) for value in args.lengths],
        "decode_steps": 1,
        "cases": examples["cases"],
        "examples": examples,
        "baseline_adapter_effect": baseline_effect,
        "peak_rss_bytes": max(_peak_rss_bytes(), _rss_bytes()),
        "rss_before_load_bytes": rss_before,
        "rss_after_generation_bytes": _rss_bytes(),
        "elapsed_seconds": time.perf_counter() - started,
        "numeric_gate": {"supported": [1e-4, 3e-4], "note": "verify chooses a documented gate before comparison; no posthoc adjustment"},
        "production_status": "short_context_reference_only; context400_base_cache_gate_unresolved",
        "command": list(sys.argv),
    }
    _write_json(output_dir / "manifest.json", metadata)
    del wrapper, model
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return metadata


def _load_case(archive: Any, metadata: Mapping[str, Any], case_name: str) -> tuple[dict[str, np.ndarray], list[np.ndarray]]:
    case = metadata["cases"][case_name]
    inputs = {
        name: np.asarray(archive[f"case.{case_name}.input.{name}"]) for name in case["input_names"]
    }
    outputs = [
        np.asarray(archive[f"case.{case_name}.output.{name}"]) for name in case["output_names"]
    ]
    return inputs, outputs


def _reference_adapter_sha(metadata: Mapping[str, Any]) -> str | None:
    """Return the adapter digest recorded by the Torch reference manifest."""

    adapter = metadata.get("adapter")
    if isinstance(adapter, Mapping):
        value = adapter.get("sha256") or adapter.get("adapter_sha256")
        if value:
            return str(value)
    value = metadata.get("adapter_sha256")
    return str(value) if value else None


def _stage_provenance(reference_metadata: Mapping[str, Any], onnx_dir: Path) -> dict[str, Any]:
    """Check that a staged graph matches the saved Torch adapter reference.

    This check runs before importing ONNX Runtime.  It catches a graph from a
    different adapter, a changed graph file, a scale mismatch, or a different
    Nano checkpoint without opening a model session.  The returned manifest
    digest is the digest *before* a successful fitted-verification field may be
    appended.
    """

    stage_dir = onnx_dir / "t3"
    manifest_path = stage_dir / "manifest.json"
    result: dict[str, Any] = {
        "status": "mismatch",
        "stage": "t3",
        "stage_manifest_path": str(manifest_path),
        "stage_manifest_sha256": None,
        "graph_path": None,
        "graph_sha256": None,
        "graph_manifest_sha256": None,
        "adapter_sha256": None,
        "expected_adapter_sha256": _reference_adapter_sha(reference_metadata),
        "adapter_scale": None,
        "expected_adapter_scale": reference_metadata.get("adapter_scale"),
        "model_checkpoint_sha256": None,
        "expected_model_checkpoint_sha256": reference_metadata.get("model_checkpoint_sha256"),
        "errors": [],
    }

    def error(message: str) -> None:
        result["errors"].append(message)

    if not manifest_path.exists():
        error(f"staged T3 manifest not found: {manifest_path}")
        return result
    try:
        result["stage_manifest_sha256"] = _sha256(manifest_path)
        stage_manifest = json.loads(manifest_path.read_text())
    except Exception as exc:
        error(f"cannot read staged T3 manifest: {exc}")
        return result
    if stage_manifest.get("stage") != "t3":
        error(f"staged manifest identifies stage {stage_manifest.get('stage')!r}, expected 't3'")
    if stage_manifest.get("status") != "exported":
        error(f"staged T3 status is {stage_manifest.get('status')!r}, expected 'exported'")

    graph = stage_manifest.get("graph")
    if not isinstance(graph, Mapping) or not graph.get("path"):
        error("staged T3 manifest has no graph path")
        graph = {}
    raw_graph_path = graph.get("path")
    graph_path = Path(str(raw_graph_path)) if raw_graph_path else None
    if graph_path is not None and not graph_path.is_absolute():
        graph_path = stage_dir / graph_path
    result["graph_path"] = str(graph_path) if graph_path else None
    if graph_path is None or not graph_path.exists() or not graph_path.is_file():
        error(f"staged T3 graph not found: {graph_path}")
    else:
        result["graph_sha256"] = _sha256(graph_path)
    result["graph_manifest_sha256"] = graph.get("sha256")
    if not result["graph_sha256"]:
        error("staged graph SHA-256 could not be computed")
    if not result["graph_manifest_sha256"]:
        error("staged T3 manifest has no graph SHA-256")
    if result["graph_sha256"] and result["graph_manifest_sha256"] and result["graph_sha256"] != result["graph_manifest_sha256"]:
        error("staged graph SHA-256 does not match its T3 manifest")

    patch = stage_manifest.get("adapter_patch")
    if not isinstance(patch, Mapping):
        # Keep the helper useful for older patch roots whose metadata was only
        # written at the root.  The stage graph and stage manifest remain the
        # authority for the graph digest.
        root_manifest_path = onnx_dir / "manifest.json"
        try:
            root_manifest = json.loads(root_manifest_path.read_text())
        except Exception:
            root_manifest = {}
        patch = root_manifest.get("adapter_patch") if isinstance(root_manifest, Mapping) else None
    if not isinstance(patch, Mapping):
        error("staged T3 manifest has no adapter_patch provenance")
        patch = {}
    result["adapter_sha256"] = patch.get("adapter_sha256")
    result["adapter_scale"] = patch.get("scale")
    if not result["expected_adapter_sha256"]:
        error("Torch reference manifest has no adapter SHA-256")
    if not result["adapter_sha256"]:
        error("staged adapter patch has no adapter SHA-256")
    if result["adapter_sha256"] != result["expected_adapter_sha256"]:
        error("staged adapter SHA-256 does not match the Torch reference adapter")
    try:
        expected_scale = float(result["expected_adapter_scale"])
        actual_scale = float(result["adapter_scale"])
        if not math.isfinite(expected_scale) or not math.isfinite(actual_scale) or not math.isclose(expected_scale, actual_scale, rel_tol=0.0, abs_tol=1e-8):
            error("staged adapter scale does not match the Torch reference scale")
    except (TypeError, ValueError):
        error("staged or reference adapter scale is missing or non-numeric")
    if patch.get("patched_graph_sha256") and result["graph_sha256"] != patch.get("patched_graph_sha256"):
        error("staged graph SHA-256 does not match adapter_patch.patched_graph_sha256")

    checkpoint = stage_manifest.get("checkpoint")
    result["model_checkpoint_sha256"] = checkpoint.get("sha256") if isinstance(checkpoint, Mapping) else None
    if not result["expected_model_checkpoint_sha256"]:
        error("Torch reference manifest has no model checkpoint SHA-256")
    if not result["model_checkpoint_sha256"]:
        error("staged T3 manifest has no model checkpoint SHA-256")
    if result["model_checkpoint_sha256"] != result["expected_model_checkpoint_sha256"]:
        error("staged model checkpoint SHA-256 does not match the Torch reference")

    result["status"] = "passed" if not result["errors"] else "mismatch"
    return result


def _record_fitted_verification(
    manifest_path: Path,
    *,
    report_path: Path,
    report: Mapping[str, Any],
    provenance: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Append short-context fitted verification metadata without promotion."""

    stage_manifest = json.loads(manifest_path.read_text())
    before_sha = _sha256(manifest_path)
    cases = list(report.get("cases", {}))
    fitted = {
        "status": "verified",
        "experimental_t3": True,
        "promotion_status": "not_promoted",
        "production_status": "short_context_only",
        "scope": {
            "lengths": list(metadata.get("lengths", [])),
            "decode_steps": int(metadata.get("decode_steps", 1)),
            "cases": cases,
        },
        "numeric_gate": report.get("numeric_gate"),
        "graph_sha256": provenance.get("graph_sha256"),
        "stage_manifest_sha256_before_update": before_sha,
        "adapter_sha256": provenance.get("adapter_sha256"),
        "adapter_scale": provenance.get("adapter_scale"),
        "model_checkpoint_sha256": provenance.get("model_checkpoint_sha256"),
        "verification_report": str(report_path),
        "verification_report_sha256": _sha256(report_path),
        "context400_base_cache_gate": "unresolved",
        "disclosure": "L32/L128 prefill and one saved-cache decode passed; this does not promote the streaming graph while the known base L400 cache gate remains unresolved.",
    }
    stage_manifest["fitted_adapter_verification"] = fitted
    _write_json(manifest_path, stage_manifest)
    return {
        "field": "fitted_adapter_verification",
        "status": "verified",
        "manifest_sha256_before_update": before_sha,
        "manifest_sha256_after_update": _sha256(manifest_path),
        "report_sha256": fitted["verification_report_sha256"],
    }


def verify(args: argparse.Namespace) -> dict[str, Any]:
    """Compare saved adapted outputs with a fresh pure ORT session."""

    reference_dir = _resolve(args.reference_dir)
    metadata_path = reference_dir / "manifest.json"
    examples_path = reference_dir / "examples.npz"
    if not metadata_path.exists() or not examples_path.exists():
        raise FileNotFoundError(f"reference manifest/examples missing under {reference_dir}")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("format") != "nano_fitted_t3_onnx_reference_v1":
        raise ValueError("unsupported fitted T3 reference manifest")
    gate = float(args.gate)
    if gate not in {1e-4, 3e-4}:
        raise ValueError("--gate must be exactly 1e-4 or 3e-4")
    onnx_dir = _resolve(args.onnx_dir)
    provenance = _stage_provenance(metadata, onnx_dir)
    if provenance["status"] != "passed":
        # Persist a diagnostic even when the graph cannot be opened.  This is
        # deliberately a failed verification and never mutates the staged
        # manifest to make the strict runtime gate pass.
        report = {
            "format": "nano_fitted_t3_onnx_verification_v1",
            "stage": "t3",
            "status": "provenance_mismatch",
            "onnx_dir": str(onnx_dir),
            "reference_dir": str(reference_dir),
            "reference_manifest_sha256": _sha256(metadata_path),
            "examples_sha256": _sha256(examples_path),
            "model_checkpoint_sha256": metadata.get("model_checkpoint_sha256"),
            "adapter": metadata.get("adapter"),
            "adapter_scale": metadata.get("adapter_scale"),
            "provenance": provenance,
            "cases": {},
            "errors": [{"kind": "provenance", "message": value} for value in provenance["errors"]],
            "numeric_gate": {"atol": gate, "rtol": gate, "selection": "explicit_supported_gate", "posthoc_adjustment": False},
            "production_status": "not_promoted",
            "disclosure": "No ORT session was opened because staged graph, adapter, scale, or checkpoint provenance did not match the Torch reference.",
            "manifest_update": "none",
        }
        _write_json(reference_dir / "verification.json", report)
        return report
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    # Import ORT only in the verification command.  The reference process
    # imports Torch instead, so the two heavyweight runtimes never coexist.
    from onnx_staged_runtime import T3UnifiedOrtRuntime

    started = time.perf_counter()
    rss_before = _rss_bytes()
    runtime = T3UnifiedOrtRuntime(
        onnx_dir,
        intra_op_num_threads=int(args.intra_op_threads),
        inter_op_num_threads=int(args.inter_op_threads),
        ort_provider=args.ort_provider,
        cuda_device_id=int(args.cuda_device_id),
        gpu_mem_limit_mib=int(args.gpu_mem_limit_mib),
        # The regular inference path remains strict.  This deliberate escape
        # hatch is used only after _stage_provenance has checked the graph.
        verification_only=True,
    )
    cases_report: dict[str, Any] = {}
    errors: list[dict[str, Any]] = []
    try:
        with np.load(examples_path, allow_pickle=False) as archive:
            for case_name in metadata["cases"]:
                inputs, expected = _load_case(archive, metadata, case_name)
                if case_name.endswith(".prefill"):
                    got_logits, got_cache = runtime.prefill(inputs["inputs_embeds"], inputs["cache_position"])
                    actual = (got_logits, *got_cache)
                    # Prefill always uses the saved zero-length past tensors.
                    for index in range(T3_LAYERS * 2):
                        past = inputs[f"past_{index}"]
                        if past.shape != (1, T3_HEADS, 0, T3_HEAD_DIM) or past.size:
                            raise ValueError(f"{case_name} past_{index} is not an empty KV cache")
                elif case_name.endswith(".decode"):
                    past = tuple(inputs[f"past_{index}"] for index in range(T3_LAYERS * 2))
                    got_logits, got_cache = runtime.decode(inputs["inputs_embeds"], inputs["cache_position"], past)
                    actual = (got_logits, *got_cache)
                else:
                    raise ValueError(f"unsupported case name {case_name!r}")
                metrics = _output_metrics(actual, expected, atol=gate, rtol=gate)
                cases_report[case_name] = metrics
                if metrics["status"] != "passed":
                    errors.append({"case": case_name, "metrics": metrics})
    finally:
        runtime.close()
    report = {
        "format": "nano_fitted_t3_onnx_verification_v1",
        "stage": "t3",
        "status": "verified" if not errors else "mismatch",
        "onnx_dir": str(_resolve(args.onnx_dir)),
        "reference_dir": str(reference_dir),
        "reference_manifest_sha256": _sha256(metadata_path),
        "examples_sha256": _sha256(examples_path),
        "model_checkpoint_sha256": metadata.get("model_checkpoint_sha256"),
        "adapter": metadata.get("adapter"),
        "adapter_scale": metadata.get("adapter_scale"),
        "provenance": provenance,
        "cases": cases_report,
        "errors": errors[:16],
        "numeric_gate": {"atol": gate, "rtol": gate, "selection": "explicit_supported_gate", "posthoc_adjustment": False},
        "ort_provider": args.ort_provider,
        "intra_op_threads": int(args.intra_op_threads),
        "inter_op_threads": int(args.inter_op_threads),
        "runtime_info": runtime.info(),
        "rss_before_session_bytes": rss_before,
        "peak_rss_bytes": max(_peak_rss_bytes(), _rss_bytes()),
        "elapsed_seconds": time.perf_counter() - started,
        "production_status": "short_context_only; context400_base_cache_gate_unresolved",
        "disclosure": "A short L32/L128 adapter parity pass does not promote the streaming ONNX graph to production while the known base L400 cache gate remains unresolved.",
        "manifest_update": "fitted_adapter_verification" if not errors else "none",
    }
    report_path = reference_dir / "verification.json"
    _write_json(report_path, report)
    if report["status"] == "verified":
        report["manifest_update"] = _record_fitted_verification(
            onnx_dir / "t3" / "manifest.json",
            report_path=report_path,
            report=report,
            provenance=provenance,
            metadata=metadata,
        )
        # Keep the report self-describing.  Its digest is stored in the staged
        # fitted-verification field written above; rewriting it here would make
        # that digest stale.
    else:
        report["manifest_update"] = "none"
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    ref = sub.add_parser("reference", help="generate adapted Torch references and raw NPZ feeds")
    ref.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ref.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    ref.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ref.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ref.add_argument("--scale", type=float, default=1.0)
    ref.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ref.add_argument("--threads", type=int, default=2)
    ref.add_argument("--lengths", type=int, nargs="+", default=list(DEFAULT_LENGTHS))
    ver = sub.add_parser("verify", help="compare the saved references with pure ONNX Runtime")
    ver.add_argument("--reference-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ver.add_argument("--onnx-dir", type=Path, default=DEFAULT_ONNX_DIR)
    ver.add_argument("--gate", choices=("1e-4", "3e-4"), default="3e-4")
    ver.add_argument("--ort-provider", choices=("cpu", "cuda"), default="cpu")
    ver.add_argument("--intra-op-threads", type=int, default=1)
    ver.add_argument("--inter-op-threads", type=int, default=1)
    ver.add_argument("--cuda-device-id", type=int, default=0)
    ver.add_argument("--gpu-mem-limit-mib", type=int, default=2048)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if args.command == "reference":
        result = reference(args)
    elif args.command == "verify":
        result = verify(args)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") in {"reference_generated", "verified"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
