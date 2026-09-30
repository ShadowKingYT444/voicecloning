"""Generate and verify independent T3 references for the staged Nano graph.

``onnx_staged.py`` exports a hand-written GPT-2 attention wrapper.  That
wrapper must be checked against the upstream Transformers GPT-2 forward before
an ONNX result is useful.  This script keeps that check separate from ORT:

* generation imports PyTorch only, loads the exact Nano T3 through
  a minimal upstream GPT2/speech-head loader on ``meta``/``to_empty``, and compares the
  wrapper with ``model.tfmr`` for lengths 32, 128, and 400 plus two cache
  growth steps;
* verification is a fresh pure-ONNX-Runtime process.  It reads the saved
  upstream outputs, measures ORT latency and peak RSS, and exits non-zero on a
  shape or numerical mismatch.

The existing ``t3/examples.npz`` is never modified.  References are written
to ``t3/extended_examples.npz`` using the same ``case.<name>.input.*`` and
``case.<name>.output.*`` key convention, with metadata in
``t3/extended_examples.json``.

Run each mode through ``bounded_job.py``.  Do not run generation and ORT
verification in one process; that would keep both model copies resident.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
VENDOR_SRC = ROOT / "vendor" / "chatterbox" / "src"
DEFAULT_CHECKPOINT = ROOT / "models" / "chatterbox-nano"
DEFAULT_STAGE_DIR = ROOT / "artifacts" / "nano_lab" / "onnx_staged" / "t3"
DEFAULT_LENGTHS = (32, 128, 400)
T3_LAYERS = 12
T3_HEADS = 12
T3_HEAD_DIM = 64
T3_HIDDEN = 768


def _ensure_paths() -> None:
    for value in (SCRIPT_DIR, VENDOR_SRC):
        if str(value) not in sys.path:
            sys.path.insert(0, str(value))


def _rss_bytes() -> int:
    """Return current Linux RSS without importing psutil."""

    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, OSError, ValueError):
        pass
    return 0


def _peak_rss_bytes() -> int:
    # Linux ru_maxrss is KiB.  Keep the fallback portable for local review.
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return value
    return value * 1024


def _legacy_cache(cache: Any) -> tuple[Any, ...]:
    if hasattr(cache, "to_legacy_cache"):
        return tuple(cache.to_legacy_cache())
    return tuple(cache)


def _flat_cache(cache: Sequence[Any]) -> tuple[Any, ...]:
    values: list[Any] = []
    for pair in cache:
        if isinstance(pair, (tuple, list)) and len(pair) == 2:
            values.extend(pair)
        else:
            # A legacy cache is already a flat sequence only when this helper
            # is called with a pre-flattened object.  Keep that case explicit.
            values.append(pair)
    return tuple(values)


def _cache_pairs(cache: Sequence[Any]) -> tuple[Any, ...]:
    """Return a flat key/value cache regardless of HF cache representation."""

    if len(cache) == T3_LAYERS * 2:
        return tuple(cache)
    if len(cache) == T3_LAYERS and all(isinstance(item, (tuple, list)) and len(item) == 2 for item in cache):
        return _flat_cache(cache)
    raise RuntimeError(f"expected {T3_LAYERS * 2} KV tensors, got {len(cache)}")


def _tensor_stats(actual: Any, expected: Any, *, atol: float, rtol: float) -> dict[str, Any]:
    actual_np = np.asarray(actual.detach().float().cpu().numpy())
    expected_np = np.asarray(expected.detach().float().cpu().numpy())
    if actual_np.shape != expected_np.shape:
        return {
            "status": "shape_mismatch",
            "actual_shape": list(actual_np.shape),
            "expected_shape": list(expected_np.shape),
        }
    difference = np.abs(actual_np.astype(np.float64) - expected_np.astype(np.float64))
    max_abs = float(np.max(difference)) if difference.size else 0.0
    max_rel = float(np.max(difference / np.maximum(np.abs(expected_np.astype(np.float64)), 1e-5))) if difference.size else 0.0
    passed = bool(np.allclose(actual_np, expected_np, atol=atol, rtol=rtol))
    return {"status": "passed" if passed else "mismatch", "max_abs": max_abs, "max_relative": max_rel, "shape": list(actual_np.shape)}


def _put_npz_array(archive: zipfile.ZipFile, key: str, value: Any) -> dict[str, Any]:
    """Write one array to an NPZ archive without assembling a giant dict."""

    if hasattr(value, "detach"):
        array = value.detach().to(device="cpu").numpy()
    else:
        array = np.asarray(value)
    info = {"shape": list(array.shape), "dtype": str(array.dtype)}
    with archive.open(f"{key}.npy", "w") as member:
        np.lib.format.write_array(member, array, allow_pickle=False)
    del array
    return info


def _write_metadata(path: Path, metadata: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def _load_reference_core(checkpoint_dir, device_name, max_context):
    """Load only upstream GPT2 and the speech head used by these tests.

    Text/reference embeddings are not used with explicit inputs_embeds. Open
    one tensor mapping at a time so CPU destination weights do not coexist
    with a resident mapping of the entire checkpoint.
    """
    import importlib.util
    import torch
    from transformers import GPT2Config, GPT2Model
    from safetensors import safe_open
    config_path=VENDOR_SRC/"chatterbox/models/t3/llama_configs.py"
    spec=importlib.util.spec_from_file_location("nano_reference_config",config_path)
    config_module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    class Core(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.tfmr=GPT2Model(GPT2Config(**config_module.GPT2_SMALL_CONFIG))
            del self.tfmr.wte
            self.speech_head=torch.nn.Linear(768,6563)
        @property
        def device(self):
            return self.speech_head.weight.device

    device=torch.device(device_name)
    with torch.device("meta"):
        model=Core()
    model.to_empty(device=device)
    checkpoint=str(Path(checkpoint_dir)/"t3_nano_v1.safetensors")
    for name,target in model.named_parameters():
        with safe_open(checkpoint,framework="pt",device="cpu") as source:
            tensor=source.get_tensor(name)
            if tensor.shape!=target.shape:
                raise ValueError(f"Checkpoint shape mismatch: {name}")
            with torch.no_grad():target.copy_(tensor)
            del tensor
    shared_masks={}
    for layer in model.modules():
        bias=layer._buffers.get("bias")
        if torch.is_tensor(bias) and bias.dtype==torch.bool and bias.ndim==4:
            if max_context > bias.shape[-1]:
                raise ValueError("Reference sequence exceeds the checkpoint context limit")
            size=max_context
            if size not in shared_masks:
                shared_masks[size]=torch.ones(size,size,dtype=torch.bool,device=device).tril_().view(1,1,size,size)
            layer.bias=shared_masks[size]
        if "masked_bias" in layer._buffers:
            layer.masked_bias=torch.tensor(-1e4,device=device)
    return model.eval().requires_grad_(False)


def _torch_case(
    *,
    model: Any,
    wrapper: Any,
    length: int,
    decode_steps: int,
    seed: int,
    archive: zipfile.ZipFile,
    atol: float,
    rtol: float,
) -> tuple[dict[str, Any], bool]:
    """Compare one prefill plus cache-growth sequence and write upstream refs."""

    import torch
    from transformers import DynamicCache

    # Use a per-case seed.  Random hidden states exercise the complete
    # transformer path without requiring an encoder or a voice reference, and
    # lengths cover the dynamic range used by Nano's text/prompt inputs.
    torch.manual_seed(int(seed) + int(length))
    device = model.device
    embeds = torch.randn((1, int(length), T3_HIDDEN), device=device, dtype=torch.float32)
    positions = torch.arange(int(length), device=device, dtype=torch.long)
    empty_cache = tuple(
        torch.zeros((1, T3_HEADS, 0, T3_HEAD_DIM), device=device, dtype=torch.float32)
        for _ in range(T3_LAYERS * 2)
    )

    case_reports: dict[str, Any] = {}
    passed = True

    with torch.inference_mode():
        wrapper_prefill = wrapper(embeds, positions, *empty_cache)
        upstream_prefill = model.tfmr(inputs_embeds=embeds, use_cache=True, return_dict=True)
        upstream_prefill_cache = _cache_pairs(_legacy_cache(upstream_prefill.past_key_values))
        upstream_prefill_logits = model.speech_head(upstream_prefill.last_hidden_state[:, -1, :])

    prefill_logits = wrapper_prefill[0]
    report = {
        "logits": _tensor_stats(prefill_logits, upstream_prefill_logits, atol=atol, rtol=rtol),
        "cache": [],
    }
    for index, (actual, expected) in enumerate(zip(wrapper_prefill[1:], upstream_prefill_cache)):
        report["cache"].append({"index": index, **_tensor_stats(actual, expected, atol=atol, rtol=rtol)})
    case_reports[f"L{length}.prefill"] = report
    passed = passed and report["logits"]["status"] == "passed" and all(item["status"] == "passed" for item in report["cache"])

    prefill_inputs = {
        "inputs_embeds": embeds,
        "cache_position": positions,
        **{f"past_{index}": value for index, value in enumerate(empty_cache)},
    }
    prefill_outputs = (upstream_prefill_logits, *upstream_prefill_cache)
    prefill_meta = {
        "input_names": list(prefill_inputs),
        "output_names": [f"output_{index}" for index in range(len(prefill_outputs))],
        "inputs": {name: _put_npz_array(archive, f"case.L{length}.prefill.input.{name}", value) for name, value in prefill_inputs.items()},
        "outputs": {f"output_{index}": _put_npz_array(archive, f"case.L{length}.prefill.output.output_{index}", value) for index, value in enumerate(prefill_outputs)},
    }

    wrapper_cache = tuple(wrapper_prefill[1:])
    upstream_cache = tuple(upstream_prefill_cache)
    previous_embeds = embeds
    previous_position = int(length)
    for step in range(int(decode_steps)):
        next_embeds = torch.randn((1, 1, T3_HIDDEN), device=device, dtype=torch.float32)
        next_position = torch.tensor([previous_position], device=device, dtype=torch.long)
        with torch.inference_mode():
            wrapper_decode = wrapper(next_embeds, next_position, *wrapper_cache)
            upstream_decode = model.tfmr(
                inputs_embeds=next_embeds,
                past_key_values=DynamicCache.from_legacy_cache(tuple(zip(upstream_cache[::2], upstream_cache[1::2]))),
                use_cache=True,
                return_dict=True,
            )
            upstream_decode_cache = _cache_pairs(_legacy_cache(upstream_decode.past_key_values))
            upstream_decode_logits = model.speech_head(upstream_decode.last_hidden_state[:, -1, :])
        report = {
            "logits": _tensor_stats(wrapper_decode[0], upstream_decode_logits, atol=atol, rtol=rtol),
            "cache": [],
        }
        for index, (actual, expected) in enumerate(zip(wrapper_decode[1:], upstream_decode_cache)):
            report["cache"].append({"index": index, **_tensor_stats(actual, expected, atol=atol, rtol=rtol)})
        case_name = f"L{length}.decode{step}"
        case_reports[case_name] = report
        step_passed = report["logits"]["status"] == "passed" and all(item["status"] == "passed" for item in report["cache"])
        passed = passed and step_passed

        decode_inputs = {
            "inputs_embeds": next_embeds,
            "cache_position": next_position,
            **{f"past_{index}": value for index, value in enumerate(upstream_cache)},
        }
        decode_outputs = (upstream_decode_logits, *upstream_decode_cache)
        case_reports_meta = {
            "input_names": list(decode_inputs),
            "output_names": [f"output_{index}" for index in range(len(decode_outputs))],
            "inputs": {name: _put_npz_array(archive, f"case.{case_name}.input.{name}", value) for name, value in decode_inputs.items()},
            "outputs": {f"output_{index}": _put_npz_array(archive, f"case.{case_name}.output.output_{index}", value) for index, value in enumerate(decode_outputs)},
        }
        # The JSON metadata is assembled by the caller.  Attach only the
        # serialized layout here so no tensor remains referenced after this
        # step.
        case_reports[case_name]["examples"] = case_reports_meta
        wrapper_cache = tuple(wrapper_decode[1:])
        upstream_cache = tuple(upstream_decode_cache)
        previous_embeds = next_embeds
        previous_position += 1
        del wrapper_decode, upstream_decode, upstream_decode_cache, upstream_decode_logits

    case_reports[f"L{length}.prefill"]["examples"] = prefill_meta
    # Keep explicit references out of the returned object; reports only hold
    # JSON scalars and arrays metadata.
    del wrapper_prefill, upstream_prefill, upstream_prefill_cache, upstream_prefill_logits
    del embeds, positions, empty_cache, wrapper_cache, upstream_cache, previous_embeds
    gc.collect()
    return {"length": int(length), "cases": case_reports}, passed


def generate_references(
    *,
    stage_dir: Path,
    checkpoint_dir: Path,
    device_name: str,
    lengths: Sequence[int],
    decode_steps: int,
    seed: int,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    """Generate independent upstream references and wrapper diagnostics."""

    import torch

    _ensure_paths()
    from onnx_t3_core import NanoT3KV

    stage_dir.mkdir(parents=True, exist_ok=True)
    rss_before_load = _rss_bytes()
    started = time.perf_counter()
    model = _load_reference_core(checkpoint_dir,device_name,max(lengths)+decode_steps)
    wrapper = NanoT3KV(model).eval()
    load_seconds = time.perf_counter() - started
    npz_path = stage_dir / "extended_examples.npz"
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "stage": "t3",
        "source": "upstream Transformers GPT2 forward using exact Nano T3 checkpoint",
        "checkpoint_dir": str(Path(checkpoint_dir).resolve()),
        "device": str(device_name),
        "dtype": "torch.float32",
        "load_scope": "Upstream GPT2 and speech head only; explicit embedding inputs; tensor-at-a-time mappings; shared causal mask sized to the longest requested test context",
        "lengths": [int(value) for value in lengths],
        "decode_steps": int(decode_steps),
        "tolerances": {"atol": float(atol), "rtol": float(rtol)},
        "examples": {"path": npz_path.name, "cases": {}},
        "wrapper_comparison": {},
        "load_seconds": load_seconds,
        "rss_before_load_bytes": rss_before_load,
    }
    all_passed = True
    with zipfile.ZipFile(npz_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for length in lengths:
            report, passed = _torch_case(
                model=model,
                wrapper=wrapper,
                length=int(length),
                decode_steps=int(decode_steps),
                seed=int(seed),
                archive=archive,
                atol=float(atol),
                rtol=float(rtol),
            )
            all_passed = all_passed and passed
            metadata["wrapper_comparison"][f"L{int(length)}"] = report
            for case_name, case_report in report["cases"].items():
                metadata["examples"]["cases"][case_name] = case_report.pop("examples")
    metadata.update(
        {
            "status": "generated_and_wrapper_verified" if all_passed else "wrapper_mismatch",
            "npz_bytes": npz_path.stat().st_size,
            "rss_after_generation_bytes": _rss_bytes(),
            "peak_rss_bytes": _peak_rss_bytes(),
            "generated_unix": time.time(),
            "verification": {"status": "not_run", "command": "onnx_t3_reference.py --verify"},
        }
    )
    _write_metadata(stage_dir / "extended_examples.json", metadata)
    # Release the model before a caller starts ORT verification.  This function
    # itself never constructs an ORT session.
    del wrapper, model
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return metadata


def _load_npz_case(archive: Any, metadata: Mapping[str, Any], case_name: str) -> tuple[dict[str, np.ndarray], list[np.ndarray]]:
    case = metadata["examples"]["cases"][case_name]
    inputs = {
        name: np.asarray(archive[f"case.{case_name}.input.{name}"])
        for name in case["input_names"]
    }
    outputs = [
        np.asarray(archive[f"case.{case_name}.output.{name}"])
        for name in case["output_names"]
    ]
    return inputs, outputs


def verify_references(*, stage_dir: Path, intra_op_num_threads: int, atol: float, rtol: float, optimization: str = "all") -> dict[str, Any]:
    """Verify saved upstream cases in a pure ORT process."""

    import onnxruntime as ort

    metadata_path = stage_dir / "extended_examples.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"extended reference metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    npz_path = stage_dir / metadata["examples"]["path"]
    manifest_path = stage_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"staged T3 manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "exported":
        raise RuntimeError(f"staged T3 graph is not exported: {manifest.get('reason', manifest.get('status'))}")
    graph_path = Path(manifest["graph"]["path"])
    if not graph_path.is_absolute():
        graph_path = stage_dir / graph_path
    options = ort.SessionOptions()
    options.intra_op_num_threads = max(1, min(int(intra_op_num_threads), 2))
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = {
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "disabled": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    }[optimization]
    rss_before_session = _rss_bytes()
    session_started = time.perf_counter()
    session = ort.InferenceSession(str(graph_path), options, providers=["CPUExecutionProvider"])
    session_load_seconds = time.perf_counter() - session_started
    rss_after_session = _rss_bytes()
    feed_names = [value.name for value in session.get_inputs()]
    reports: dict[str, Any] = {}
    errors: list[dict[str, Any]] = []
    max_abs = 0.0
    max_rel = 0.0
    started = time.perf_counter()
    with np.load(npz_path, allow_pickle=False) as archive:
        for case_name in metadata["examples"]["cases"]:
            inputs, expected = _load_npz_case(archive, metadata, case_name)
            missing = sorted(set(feed_names) - set(inputs))
            if missing:
                errors.append({"case": case_name, "kind": "missing_inputs", "inputs": missing})
                continue
            call_started = time.perf_counter()
            actual = session.run(None, {name: inputs[name] for name in feed_names})
            elapsed = time.perf_counter() - call_started
            case_report = {"elapsed_seconds": elapsed, "outputs": len(actual), "errors": [], "output_metrics": []}
            if len(actual) != len(expected):
                case_report["errors"].append({"kind": "output_count", "actual": len(actual), "expected": len(expected)})
                errors.append({"case": case_name, **case_report["errors"][-1]})
                reports[case_name] = case_report
                continue
            for index, (got, want) in enumerate(zip(actual, expected)):
                got_array = np.asarray(got)
                want_array = np.asarray(want)
                if got_array.shape != want_array.shape:
                    detail = {"kind": "shape", "index": index, "actual": list(got_array.shape), "expected": list(want_array.shape)}
                    case_report["errors"].append(detail)
                    errors.append({"case": case_name, **detail})
                    continue
                difference = np.abs(got_array.astype(np.float64) - want_array.astype(np.float64))
                abs_error = float(np.max(difference)) if difference.size else 0.0
                rel_error = float(np.max(difference / np.maximum(np.abs(want_array.astype(np.float64)), 1e-5))) if difference.size else 0.0
                max_abs = max(max_abs, abs_error)
                max_rel = max(max_rel, rel_error)
                case_report["output_metrics"].append({
                    "index": index,
                    "max_abs": abs_error,
                    "relative_l2": float(np.linalg.norm(difference.ravel()) / max(np.linalg.norm(want_array.astype(np.float64).ravel()), 1e-12)),
                    "outside_tolerance_count": int(np.count_nonzero(~np.isclose(got_array,want_array,atol=atol,rtol=rtol))),
                    "elements": int(got_array.size),
                })
                if not np.allclose(got_array, want_array, atol=float(atol), rtol=float(rtol)):
                    detail = {"kind": "numeric", "index": index, "max_abs": abs_error, "max_relative": rel_error}
                    case_report["errors"].append(detail)
                    errors.append({"case": case_name, **detail})
            reports[case_name] = case_report
    result = {
        "stage": "t3",
        "status": "verified" if not errors else "mismatch",
        "provider": "CPUExecutionProvider",
        "optimization": optimization,
        "graph": str(graph_path.resolve()),
        "inputs": feed_names,
        "cases": reports,
        "errors": errors[:32],
        "max_abs_error": max_abs,
        "max_relative_error": max_rel,
        "session_load_seconds": session_load_seconds,
        "inference_seconds": time.perf_counter() - started,
        "rss_before_session_bytes": rss_before_session,
        "rss_after_session_bytes": rss_after_session,
        "peak_rss_bytes": max(_peak_rss_bytes(), _rss_bytes()),
        "method": "pure ORT CPU against upstream Transformers GPT2 outputs; no PyTorch model loaded",
        "tolerances": {"atol": float(atol), "rtol": float(rtol)},
    }
    metadata["verification"] = result
    _write_metadata(metadata_path, metadata)
    _write_metadata(stage_dir / "extended_verification.json", result)
    _write_metadata(stage_dir / f"extended_verification_{optimization}.json", result)
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="run pure ORT verification instead of PyTorch generation")
    parser.add_argument("--stage-dir", type=Path, default=DEFAULT_STAGE_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--lengths", type=int, nargs="+", default=list(DEFAULT_LENGTHS))
    parser.add_argument("--decode-steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--intra-op-num-threads", type=int, default=1)
    parser.add_argument("--optimization", choices=("all", "basic", "disabled"), default="all")
    parser.add_argument("--atol", type=float, default=3e-4)
    parser.add_argument("--rtol", type=float, default=3e-4)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.verify:
        result = verify_references(
            stage_dir=args.stage_dir,
            intra_op_num_threads=args.intra_op_num_threads,
            atol=args.atol,
            rtol=args.rtol,
            optimization=args.optimization,
        )
    else:
        result = generate_references(
            stage_dir=args.stage_dir,
            checkpoint_dir=args.checkpoint_dir,
            device_name=args.device,
            lengths=args.lengths,
            decode_steps=max(1, int(args.decode_steps)),
            seed=args.seed,
            atol=args.atol,
            rtol=args.rtol,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    if result.get("status") in {"mismatch", "wrapper_mismatch"}:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
