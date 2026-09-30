"""Low-RSS ONNX export with externalised parameter inputs.

The legacy TorchScript exporter normally holds a serialised copy of every
parameter while it builds an ONNX graph.  That transient copy pushed the
Nano flow encoder over the lab's 1,280 MiB export limit.  This module uses the
installed PyTorch 2.11 private TorchScript exporter with
``export_params=False``, ``keep_initializers_as_inputs=True``, disabled
constant folding, and ONNX shape inference. Parameters therefore
remain graph inputs during tracing.  After tracing, this helper attaches
external-data initializers one tensor at a time and removes those parameter
inputs from the final graph.

The private API is version-pinned and a safe fallback is deliberately disabled
because falling back to ``torch.onnx.export`` would reintroduce the memory
spike.  The helper fails with an explicit diagnostic if the signature or
parameter-name mapping changes.  It never substitutes another model or
silently drops an unbound parameter.

The module has no top-level PyTorch import.  ``export_external_parameters``
imports PyTorch only inside the export process.  ORT verification can run in a
fresh process through :func:`verify_onnx`, which uses only ONNX Runtime and
NumPy.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import resource
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "artifacts" / "nano_lab" / "onnx_external"
SUPPORTED_TORCH_PREFIX = "2.11."
PRIVATE_API = "torch.onnx._internal.torchscript_exporter.utils._export"
_REQUIRED_EXPORT_KWARGS = (
    "export_params",
    "keep_initializers_as_inputs",
    "onnx_shape_inference",
    "do_constant_folding",
)


def _rss_bytes() -> int:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return 0


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if os.name == "nt" else value * 1024


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _private_export(torch: Any):
    """Resolve and validate the pinned private exporter.

    Do not replace this with public ``torch.onnx.export``.  The public path is
    exactly the path that caused the observed export RSS failure.
    """

    version = str(torch.__version__).split("+")[0]
    if not version.startswith(SUPPORTED_TORCH_PREFIX):
        raise RuntimeError(
            f"low-RSS external export is pinned to torch {SUPPORTED_TORCH_PREFIX}*, found {torch.__version__!r}; safe fallback is disabled"
        )
    from torch.onnx._internal.torchscript_exporter import utils

    exporter = getattr(utils, "_export", None)
    if exporter is None:
        raise RuntimeError(f"{PRIVATE_API} is unavailable; safe fallback is disabled")
    signature = inspect.signature(exporter)
    missing = [name for name in _REQUIRED_EXPORT_KWARGS if name not in signature.parameters]
    if missing:
        raise RuntimeError(f"{PRIVATE_API} signature lacks {missing}; safe fallback is disabled")
    return exporter, signature


def _canonical_name(name: str) -> str:
    return str(name).replace("/", ".").lstrip(".")


def _state_tensors(module: Any) -> dict[str, Any]:
    """Return persistent parameters and buffers with exact state-dict names."""

    persistent = set(module.state_dict().keys())
    result: dict[str, Any] = {}
    result.update({name: value for name, value in module.named_parameters() if name in persistent})
    result.update({name: value for name, value in module.named_buffers() if name in persistent})
    return result


def _resolve_parameter_map(
    graph_input_names: Sequence[str],
    runtime_input_names: Sequence[str],
    state_tensors: Mapping[str, Any],
) -> tuple[dict[str, str], list[str], list[str]]:
    """Map graph parameter inputs to exact module state names.

    The exporter assigns ``state_dict`` names to traced parameter inputs.  We
    allow only exact canonical matches, not suffix guessing, because a guessed
    match could attach a wrong tensor while producing a plausible graph.
    """

    runtime = {_canonical_name(name) for name in runtime_input_names}
    state_names = {_canonical_name(name): name for name in state_tensors}
    mapping: dict[str, str] = {}
    unknown_graph: list[str] = []
    for graph_name in graph_input_names:
        canonical = _canonical_name(graph_name)
        if canonical in runtime:
            continue
        state_name = state_names.get(canonical)
        if state_name is None:
            unknown_graph.append(graph_name)
        else:
            mapping[graph_name] = state_name
    unbound_state = [name for canonical, name in state_names.items() if name not in mapping.values()]
    return mapping, unknown_graph, sorted(unbound_state)


def _stream_tensor_bytes(handle: Any, tensor: Any, *, chunk_bytes: int = 4 * 1024 * 1024) -> tuple[int, str, str]:
    """Write a CPU tensor through a memoryview without ``tobytes`` copies."""

    cpu_tensor = tensor.detach().to(device="cpu").contiguous()
    array = cpu_tensor.numpy()
    raw = memoryview(array).cast("B")
    digest = hashlib.sha256()
    total = 0
    for start in range(0, len(raw), int(chunk_bytes)):
        chunk = raw[start : start + int(chunk_bytes)]
        handle.write(chunk)
        digest.update(chunk)
        total += len(chunk)
    dtype = str(array.dtype)
    del raw, array, cpu_tensor
    return total, dtype, digest.hexdigest()


def _attach_external_initializers(
    *,
    graph_path: Path,
    weight_path: Path,
    module: Any,
    runtime_input_names: Sequence[str],
    expected_output_names: Sequence[str] | None,
) -> dict[str, Any]:
    """Attach streamed external initializers and remove parameter inputs."""

    import onnx

    model = onnx.load(str(graph_path), load_external_data=False)
    graph = model.graph
    state_tensors = _state_tensors(module)
    input_names = [value.name for value in graph.input]
    runtime_names = {_canonical_name(name) for name in runtime_input_names}
    missing_runtime = sorted(runtime_names - {_canonical_name(name) for name in input_names})
    if missing_runtime:
        raise RuntimeError(f"exported graph lost runtime inputs: {missing_runtime}")
    mapping, unknown_graph_inputs, unbound_state = _resolve_parameter_map(input_names, runtime_input_names, state_tensors)
    if unknown_graph_inputs:
        raise RuntimeError(
            "graph has non-runtime inputs that do not exactly match module state names: "
            + ", ".join(unknown_graph_inputs[:16])
        )
    if unbound_state:
        raise RuntimeError(
            "module state tensors were not represented as graph inputs; refusing to attach a partial model: "
            + ", ".join(unbound_state[:16])
        )
    if expected_output_names:
        graph_outputs = [value.name for value in graph.output]
        missing_outputs = sorted(set(expected_output_names) - set(graph_outputs))
        if missing_outputs:
            raise RuntimeError(f"exported graph lost requested outputs: {missing_outputs}")

    weight_path.parent.mkdir(parents=True, exist_ok=True)
    initializers: list[dict[str, Any]] = []
    with weight_path.open("wb") as handle:
        for graph_name, state_name in mapping.items():
            tensor = state_tensors[state_name]
            position = handle.tell()
            alignment = 64
            padding = (-position) % alignment
            if padding:
                handle.write(b"\0" * padding)
                position += padding
            byte_count, dtype, digest = _stream_tensor_bytes(handle, tensor)
            # Remove graph input only after the data is safely written.
            initializer = graph.initializer.add()
            initializer.name = graph_name
            initializer.data_type = onnx.helper.np_dtype_to_tensor_dtype(np.dtype(dtype))
            initializer.dims.extend(int(value) for value in tensor.shape)
            initializer.data_location = onnx.TensorProto.EXTERNAL
            initializer.ClearField("raw_data")
            fields = {
                "location": weight_path.name,
                "offset": str(position),
                "length": str(byte_count),
            }
            for key, value in fields.items():
                item = initializer.external_data.add()
                item.key = key
                item.value = value
            initializers.append(
                {
                    "graph_name": graph_name,
                    "state_name": state_name,
                    "shape": list(tensor.shape),
                    "dtype": dtype,
                    "offset": position,
                    "length": byte_count,
                    "sha256": digest,
                }
            )

    runtime_values = [value for value in graph.input if _canonical_name(value.name) in runtime_names]
    graph.ClearField("input")
    graph.input.extend(runtime_values)
    # Make sure every node input resolves to either a runtime input, an
    # initializer, or an ONNX graph value produced by an earlier node.
    known = set(value.name for value in graph.input) | {value.name for value in graph.initializer}
    produced = set(value for node in graph.node for value in node.output)
    unresolved: list[str] = []
    for node in graph.node:
        for name in node.input:
            if name and name not in known and name not in produced:
                unresolved.append(name)
    if unresolved:
        raise RuntimeError("graph contains unresolved node inputs after externalization: " + ", ".join(sorted(set(unresolved))[:16]))
    onnx.save(model, str(graph_path), save_as_external_data=False)
    # Pass the filename so relative external data resolves beside the graph.
    onnx.checker.check_model(str(graph_path))
    return {
        "weight_file": str(weight_path.name),
        "weight_bytes": weight_path.stat().st_size,
        "initializers": initializers,
        "runtime_inputs": [value.name for value in graph.input],
        "outputs": [value.name for value in graph.output],
        "unbound_state": unbound_state,
        "graph_inputs_before": input_names,
        "graph_inputs_after": [value.name for value in graph.input],
    }


def export_external_parameters(
    model: Any,
    args: Any,
    output_path: str | Path,
    *,
    runtime_input_names: Sequence[str],
    output_names: Sequence[str] | None = None,
    input_names: Sequence[str] | None = None,
    dynamic_axes: Mapping[str, Mapping[int, str]] | None = None,
    opset: int = 18,
    weight_path: str | Path | None = None,
) -> dict[str, Any]:
    """Export ``model`` with streamed external parameters.

    ``args`` must be the exact tensor tuple used by the model.  The returned
    report records the private API signature, final graph inputs, parameter
    offsets, and external-weight hashes.  The helper does not open ORT.
    """

    import torch
    import onnx

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    weight_path = Path(weight_path).expanduser().resolve() if weight_path is not None else output_path.with_suffix(".weights.bin")
    exporter, signature = _private_export(torch)
    started = time.perf_counter()
    rss_before = _rss_bytes()
    runtime_input_names = tuple(str(name) for name in runtime_input_names)
    input_names = list(input_names or runtime_input_names)
    if set(runtime_input_names) - set(input_names):
        raise ValueError("runtime_input_names must be included in input_names")
    # This is intentionally a private call.  Public torch.onnx.export is not a
    # safe fallback for the desktop memory budget.
    exporter(
        model,
        args,
        str(output_path),
        export_params=False,
        input_names=input_names,
        output_names=list(output_names) if output_names is not None else None,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
        opset_version=int(opset),
        do_constant_folding=False,
        dynamic_axes=dict(dynamic_axes or {}),
        keep_initializers_as_inputs=True,
        # Required for valid Pad lowering with constant integer pad lists.
        # Disabling it emits ConcatFromSequence fed by an integer tensor.
        onnx_shape_inference=True,
    )
    attach = _attach_external_initializers(
        graph_path=output_path,
        weight_path=weight_path,
        module=model,
        runtime_input_names=runtime_input_names,
        expected_output_names=output_names,
    )
    version = str(torch.__version__)
    report = {
        "status": "exported",
        "graph": str(output_path),
        "graph_bytes": output_path.stat().st_size,
        "torch_version": version,
        "private_api": PRIVATE_API,
        "private_api_signature": str(signature),
        "safe_fallback": "disabled",
        "export_options": {"export_params": False, "keep_initializers_as_inputs": True, "onnx_shape_inference": True, "do_constant_folding": False},
        "runtime_inputs": list(runtime_input_names),
        "output_names": list(output_names or []),
        "external": attach,
        "elapsed_seconds": time.perf_counter() - started,
        "rss_before_bytes": rss_before,
        "rss_after_bytes": _rss_bytes(),
        "peak_rss_bytes": _peak_rss_bytes(),
    }
    _write_json(output_path.with_suffix(".external.json"), report)
    return report


def verify_onnx(
    graph_path: str | Path,
    *,
    feeds: Mapping[str, np.ndarray],
    expected: Sequence[np.ndarray],
    intra_op_num_threads: int = 1,
    atol: float = 3e-4,
    rtol: float = 3e-4,
) -> dict[str, Any]:
    """Run pure ORT against caller-provided expected arrays."""

    import onnxruntime as ort

    graph_path = Path(graph_path).expanduser().resolve()
    options = ort.SessionOptions()
    options.intra_op_num_threads = max(1, min(int(intra_op_num_threads), 2))
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    rss_before = _rss_bytes()
    started = time.perf_counter()
    session = ort.InferenceSession(str(graph_path), options, providers=["CPUExecutionProvider"])
    load_seconds = time.perf_counter() - started
    names = [value.name for value in session.get_inputs()]
    missing = sorted(set(names) - set(feeds))
    if missing:
        raise ValueError(f"missing ORT feeds: {missing}")
    infer_started = time.perf_counter()
    actual = session.run(None, {name: np.asarray(feeds[name]) for name in names})
    infer_seconds = time.perf_counter() - infer_started
    errors: list[dict[str, Any]] = []
    max_abs = 0.0
    max_rel = 0.0
    if len(actual) != len(expected):
        errors.append({"kind": "output_count", "actual": len(actual), "expected": len(expected)})
    else:
        for index, (got, want) in enumerate(zip(actual, expected)):
            got = np.asarray(got)
            want = np.asarray(want)
            if got.shape != want.shape:
                errors.append({"kind": "shape", "index": index, "actual": list(got.shape), "expected": list(want.shape)})
                continue
            diff = np.abs(got.astype(np.float64) - want.astype(np.float64))
            abs_error = float(np.max(diff)) if diff.size else 0.0
            rel_error = float(np.max(diff / np.maximum(np.abs(want.astype(np.float64)), 1e-5))) if diff.size else 0.0
            max_abs = max(max_abs, abs_error)
            max_rel = max(max_rel, rel_error)
            if not np.allclose(got, want, atol=float(atol), rtol=float(rtol)):
                errors.append({"kind": "numeric", "index": index, "max_abs": abs_error, "max_relative": rel_error})
    return {
        "status": "verified" if not errors else "mismatch",
        "graph": str(graph_path),
        "provider": "CPUExecutionProvider",
        "inputs": names,
        "outputs": len(actual),
        "errors": errors,
        "max_abs_error": max_abs,
        "max_relative_error": max_rel,
        "session_load_seconds": load_seconds,
        "inference_seconds": infer_seconds,
        "rss_before_session_bytes": rss_before,
        "rss_after_session_bytes": _rss_bytes(),
        "peak_rss_bytes": max(_peak_rss_bytes(), _rss_bytes()),
        "tolerances": {"atol": float(atol), "rtol": float(rtol)},
    }


def _toy_export(output_dir: Path) -> dict[str, Any]:
    """Tiny smoke path for parent-side validation of external attachment."""

    import torch
    import torch.nn as nn

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 3)

        def forward(self, x):
            return self.linear(x)

    torch.manual_seed(7)
    model = Toy().eval()
    inputs = (torch.randn(2, 4),)
    report = export_external_parameters(
        model,
        inputs,
        output_dir / "toy.onnx",
        runtime_input_names=("x",),
        input_names=("x",),
        output_names=("y",),
        dynamic_axes={"x": {0: "batch"}, "y": {0: "batch"}},
    )
    with torch.inference_mode():
        expected = (model(*inputs),)
    np.savez(output_dir/"toy.examples.npz",x=inputs[0].detach().cpu().numpy(),y=expected[0].detach().cpu().numpy())
    _write_json(output_dir / "toy.report.json", report)
    return report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    toy = sub.add_parser("toy-export", help="export a tiny external-parameter graph; no ORT in this process")
    toy.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    ver = sub.add_parser("toy-verify",help="verify the tiny graph in a separate pure ORT process")
    ver.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "toy-export":
        report = _toy_export(args.output_dir)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report.get("status") == "exported" else 1
    if args.command == "toy-verify":
        with np.load(args.output_dir/"toy.examples.npz",allow_pickle=False) as data:
            result=verify_onnx(args.output_dir/"toy.onnx",feeds={"x":data["x"]},expected=[data["y"]])
        _write_json(args.output_dir/"toy.verification.json",result)
        print(json.dumps(result,indent=2))
        return 0 if result["status"]=="verified" else 1
    return 2


__all__ = ["export_external_parameters", "verify_onnx"]


if __name__ == "__main__":
    raise SystemExit(main())
