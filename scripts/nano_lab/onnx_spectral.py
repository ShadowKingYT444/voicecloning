"""Real-valued STFT/ISTFT helpers for the Nano HiFT vocoder.

The upstream HiFT implementation uses ``torch.stft`` and ``torch.istft`` with
complex tensors.  Those operators are a poor ONNX boundary on the installed
PyTorch/ORT versions.  This module provides mathematically equivalent real
convolution graphs for the fixed HiFT parameters:

* ``n_fft = win_length = 16``;
* periodic Hann window;
* hop length 4;
* centered STFT with reflect padding;
* unnormalised one-sided DFT and inverse DFT;
* overlap-add divided by the window-square envelope, then center trim.

The synthesis graph accepts the magnitude and phase produced by HiFT's final
convolution.  It clips magnitude at 100 before constructing the real spectrum,
matching ``HiFTGenerator._istft``.  Random source/phase generation remains an
explicit input to the vocoder wrapper; this file only replaces complex
spectral arithmetic.

The module has no top-level PyTorch import.  ``build_hift_spectral_stage`` and
the command-line exporter import PyTorch only when a bounded export/test job
asks for it.  ``--verify`` loads only ONNX Runtime and NumPy.

Examples::

    .venv-nano/bin/python scripts/nano_lab/onnx_spectral.py export \
      --output-dir artifacts/nano_lab/onnx_spectral --kind all
    .venv-nano/bin/python scripts/nano_lab/onnx_spectral.py verify \
      --output-dir artifacts/nano_lab/onnx_spectral --kind roundtrip
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "nano_lab" / "onnx_spectral"
KINDS = ("stft", "istft", "roundtrip")
N_FFT = 16
HOP_LENGTH = 4
N_FREQS = N_FFT // 2 + 1


def periodic_hann(*, n_fft: int = N_FFT, dtype: Any = None, device: Any = None) -> Any:
    """Return the periodic Hann used by ``HiFTGenerator``."""

    import torch

    kwargs: dict[str, Any] = {}
    if dtype is not None:
        kwargs["dtype"] = dtype
    if device is not None:
        kwargs["device"] = device
    return torch.hann_window(int(n_fft), periodic=True, **kwargs)


def analysis_kernel(*, n_fft: int = N_FFT, dtype: Any = None, device: Any = None) -> Any:
    """Build ``[real frequencies, imaginary frequencies]`` Conv1d weights."""

    import torch

    window = periodic_hann(n_fft=n_fft, dtype=dtype, device=device)
    frequency = torch.arange(0, int(n_fft) // 2 + 1, dtype=dtype, device=device).view(-1, 1)
    sample = torch.arange(0, int(n_fft), dtype=dtype, device=device).view(1, -1)
    angle = 2 * torch.pi * frequency * sample / float(n_fft)
    real = torch.cos(angle) * window.view(1, -1)
    imag = -torch.sin(angle) * window.view(1, -1)
    return torch.cat((real, imag), dim=0).unsqueeze(1)


def synthesis_kernel(*, n_fft: int = N_FFT, dtype: Any = None, device: Any = None) -> Any:
    """Build real ``ConvTranspose1d`` overlap-add weights for inverse DFT.

    For a one-sided spectrum, interior positive frequencies occur with their
    conjugates in the full spectrum and therefore have a factor of two.  DC
    and Nyquist have factor one.  The inverse DFT contributes ``1 / n_fft``.
    The same Hann window is folded into the kernel; the caller divides by the
    overlap window-square envelope after overlap-add.
    """

    import torch

    n_fft = int(n_fft)
    n_freqs = n_fft // 2 + 1
    window = periodic_hann(n_fft=n_fft, dtype=dtype, device=device)
    frequency = torch.arange(0, n_freqs, dtype=dtype, device=device).view(-1, 1)
    sample = torch.arange(0, n_fft, dtype=dtype, device=device).view(1, -1)
    angle = 2 * torch.pi * frequency * sample / float(n_fft)
    factor = torch.ones((n_freqs, 1), dtype=dtype, device=device)
    if n_freqs > 2:
        factor[1:-1] = 2
    real = factor * torch.cos(angle) / float(n_fft)
    imag = -factor * torch.sin(angle) / float(n_fft)
    weights = torch.cat((real, imag), dim=0) * window.view(1, -1)
    # ConvTranspose expects [in_channels, out_channels, kernel].  One output
    # channel sums all real and imaginary frequency contributions.
    return weights.unsqueeze(1).contiguous()


def real_stft(
    waveform: Any,
    *,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    window: Any = None,
    kernel: Any = None,
) -> tuple[Any, Any]:
    """Exact real Conv1d equivalent of the HiFT centered one-sided STFT.

    ``waveform`` has shape ``(B,T)`` or ``(B,1,T)``.  Centered reflect
    padding requires ``T > n_fft // 2``.  The result has separate real and
    imaginary tensors with shape ``(B,n_fft//2+1,frames)``.
    """

    import torch
    import torch.nn.functional as F

    value = waveform
    if value.ndim == 3:
        if value.shape[1] != 1:
            raise ValueError(f"waveform channel dimension must be 1, got {tuple(value.shape)}")
        value = value[:, 0]
    if value.ndim != 2:
        raise ValueError(f"waveform must have shape (B,T) or (B,1,T), got {tuple(value.shape)}")
    n_fft = int(n_fft)
    hop_length = int(hop_length)
    if value.shape[-1] <= n_fft // 2:
        raise ValueError("centered reflect STFT requires waveform length greater than n_fft//2")
    if kernel is None:
        kernel = analysis_kernel(n_fft=n_fft, dtype=value.dtype, device=value.device)
    padded = F.pad(value.unsqueeze(1), (n_fft // 2, n_fft // 2), mode="reflect")
    result = F.conv1d(padded, kernel.to(device=value.device, dtype=value.dtype), stride=hop_length)
    n_freqs = n_fft // 2 + 1
    return result[:, :n_freqs], result[:, n_freqs:]


def real_istft(
    magnitude: Any,
    phase: Any,
    *,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    window: Any = None,
    kernel: Any = None,
) -> Any:
    """Exact real ConvTranspose1d equivalent of HiFT's default ISTFT.

    The returned length is PyTorch's default centered ISTFT length,
    ``(frames - 1) * hop_length``.  This matches the upstream call where no
    explicit ``length=`` argument is supplied.
    """

    import torch
    import torch.nn.functional as F

    magnitude = torch.clamp(magnitude, max=1e2)
    if magnitude.ndim != 3 or phase.ndim != 3 or magnitude.shape != phase.shape:
        raise ValueError("magnitude and phase must have equal shape (B,n_freqs,frames)")
    n_fft = int(n_fft)
    hop_length = int(hop_length)
    n_freqs = n_fft // 2 + 1
    if magnitude.shape[1] != n_freqs:
        raise ValueError(f"expected {n_freqs} one-sided frequencies, got {magnitude.shape[1]}")
    if kernel is None:
        kernel = synthesis_kernel(n_fft=n_fft, dtype=magnitude.dtype, device=magnitude.device)
    real = magnitude * torch.cos(phase)
    imag = magnitude * torch.sin(phase)
    spectrum = torch.cat((real, imag), dim=1)
    overlap = F.conv_transpose1d(spectrum, kernel.to(device=magnitude.device, dtype=magnitude.dtype), stride=hop_length)
    window_value = periodic_hann(n_fft=n_fft, dtype=magnitude.dtype, device=magnitude.device) if window is None else window
    envelope_kernel = window_value.square().view(1, 1, n_fft)
    envelope = F.conv_transpose1d(
        torch.ones((magnitude.shape[0], 1, magnitude.shape[-1]), dtype=magnitude.dtype, device=magnitude.device),
        envelope_kernel,
        stride=hop_length,
    )
    # torch.istft's NOLA check is satisfied by the periodic Hann/hop=4 pair.
    # A tiny dtype epsilon avoids an accidental divide-by-zero in a malformed
    # caller input while leaving valid interior samples unchanged.
    overlap = overlap / envelope.clamp_min(torch.finfo(overlap.dtype).eps)
    trim = n_fft // 2
    return overlap[:, 0, trim:-trim]


def build_hift_spectral_stage(kind: str = "roundtrip", *, n_fft: int = N_FFT, hop_length: int = HOP_LENGTH) -> Any:
    """Build an exportable ``nn.Module`` for one spectral stage.

    ``stft`` returns real and imaginary tensors; ``istft`` accepts magnitude
    and phase; ``roundtrip`` performs real STFT, magnitude/phase conversion,
    and real ISTFT in one graph.  The function imports torch lazily.
    """

    import torch
    import torch.nn as nn

    kind = str(kind).lower()
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    n_fft = int(n_fft)
    hop_length = int(hop_length)

    class SpectralStage(nn.Module):
        def __init__(self):
            super().__init__()
            self.kind = kind
            self.n_fft = n_fft
            self.hop_length = hop_length
            self.register_buffer("analysis_weight", analysis_kernel(n_fft=n_fft, dtype=torch.float32), persistent=False)
            self.register_buffer("synthesis_weight", synthesis_kernel(n_fft=n_fft, dtype=torch.float32), persistent=False)
            self.register_buffer("window", periodic_hann(n_fft=n_fft, dtype=torch.float32), persistent=False)

        def forward(self, first, second=None):
            if self.kind == "stft":
                real, imag = real_stft(first, n_fft=self.n_fft, hop_length=self.hop_length, kernel=self.analysis_weight)
                return real, imag
            if self.kind == "istft":
                return real_istft(
                    first,
                    second,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    window=self.window,
                    kernel=self.synthesis_weight,
                )
            real, imag = real_stft(first, n_fft=self.n_fft, hop_length=self.hop_length, kernel=self.analysis_weight)
            magnitude = torch.sqrt(real * real + imag * imag)
            phase = torch.atan2(imag, real)
            return real_istft(
                magnitude,
                phase,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=self.window,
                kernel=self.synthesis_weight,
            )

    return SpectralStage().eval()


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
    return value if sys.platform == "darwin" else value * 1024


def _manifest_path(output_dir: Path, kind: str) -> Path:
    return output_dir / kind / "manifest.json"


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def export_kind(
    *,
    output_dir: Path,
    kind: str,
    length: int = 32,
    opset: int = 18,
    seed: int = 20260929,
) -> dict[str, Any]:
    """Export one tiny spectral graph and save its Torch reference."""

    import onnx
    import torch

    kind = str(kind).lower()
    stage_dir = output_dir / kind
    stage_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(seed))
    model = build_hift_spectral_stage(kind).eval()
    input_names: list[str]
    dynamic_axes: dict[str, dict[int, str]]
    if kind == "stft":
        first = torch.randn((1, int(length)), dtype=torch.float32)
        args = (first,)
        input_names = ["waveform"]
        dynamic_axes = {"waveform": {0: "batch", 1: "sample_length"}, "real": {0: "batch", 2: "frame_length"}, "imag": {0: "batch", 2: "frame_length"}}
    elif kind == "istft":
        frames = max(2, int(length) // HOP_LENGTH + 1)
        first = torch.rand((1, N_FREQS, frames), dtype=torch.float32)
        second = torch.randn((1, N_FREQS, frames), dtype=torch.float32)
        args = (first, second)
        input_names = ["magnitude", "phase"]
        dynamic_axes = {"magnitude": {0: "batch", 2: "frame_length"}, "phase": {0: "batch", 2: "frame_length"}, "audio": {0: "batch", 1: "sample_length"}}
    else:
        first = torch.randn((1, int(length)), dtype=torch.float32)
        args = (first,)
        input_names = ["waveform"]
        dynamic_axes = {"waveform": {0: "batch", 1: "sample_length"}, "audio": {0: "batch", 1: "sample_length"}}
    with torch.inference_mode():
        reference = model(*args)
    output_names = ["real", "imag"] if kind == "stft" else ["audio"]
    graph_path = stage_dir / f"nano_{kind}_real.onnx"
    started = time.perf_counter()
    torch.onnx.export(
        model,
        args,
        str(graph_path),
        opset_version=int(opset),
        dynamo=False,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        external_data=False,
    )
    graph = onnx.load(str(graph_path))
    onnx.checker.check_model(graph)
    arrays = {f"input_{index}": value.detach().cpu().numpy() for index, value in enumerate(args)}
    if isinstance(reference, tuple):
        arrays.update({f"output_{index}": value.detach().cpu().numpy() for index, value in enumerate(reference)})
    else:
        arrays["output_0"] = reference.detach().cpu().numpy()
    np.savez(stage_dir / "examples.npz", **arrays)
    manifest = {
        "schema_version": 1,
        "stage": kind,
        "status": "exported",
        "kind": kind,
        "graph": {"path": graph_path.name, "bytes": graph_path.stat().st_size, "nodes": len(graph.graph.node), "opset": int(opset), "inputs": [x.name for x in graph.graph.input], "outputs": [x.name for x in graph.graph.output]},
        "examples": {"path": "examples.npz", "inputs": len(args), "outputs": len(output_names)},
        "parameters": {"n_fft": N_FFT, "hop_length": HOP_LENGTH, "center": True, "pad_mode": "reflect", "periodic_hann": True, "one_sided": True, "normalization": "unnormalized DFT + overlap window-square envelope"},
        "export_seconds": time.perf_counter() - started,
        "torch_peak_rss_bytes": _peak_rss_bytes(),
        "verification": {"status": "not_run", "command": f"onnx_spectral.py verify --kind {kind}"},
    }
    _write_json(stage_dir / "manifest.json", manifest)
    del model, reference
    return manifest


def verify_kind(*, output_dir: Path, kind: str, intra_op_num_threads: int = 1, atol: float = 3e-5, rtol: float = 3e-5) -> dict[str, Any]:
    """Verify one graph with CPU ORT only."""

    import onnxruntime as ort

    kind = str(kind).lower()
    stage_dir = output_dir / kind
    manifest_path = stage_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    graph_path = stage_dir / manifest["graph"]["path"]
    examples_path = stage_dir / manifest["examples"]["path"]
    options = ort.SessionOptions()
    options.intra_op_num_threads = max(1, min(int(intra_op_num_threads), 2))
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    rss_before = _rss_bytes()
    started = time.perf_counter()
    session = ort.InferenceSession(str(graph_path), options, providers=["CPUExecutionProvider"])
    load_seconds = time.perf_counter() - started
    rss_after = _rss_bytes()
    with np.load(examples_path, allow_pickle=False) as data:
        inputs = {f"input_{index}": np.asarray(data[f"input_{index}"]) for index in range(int(manifest["examples"]["inputs"]))}
        expected = [np.asarray(data[f"output_{index}"]) for index in range(int(manifest["examples"]["outputs"]))]
    feed_names = [x.name for x in session.get_inputs()]
    # Export input names are semantic, while examples are index-stable.
    feeds = {name: inputs[f"input_{index}"] for index, name in enumerate(feed_names)}
    infer_started = time.perf_counter()
    actual = session.run(None, feeds)
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
    result = {"stage": kind, "status": "verified" if not errors else "mismatch", "provider": "CPUExecutionProvider", "graph": str(graph_path.resolve()), "inputs": feed_names, "outputs": len(actual), "errors": errors, "max_abs_error": max_abs, "max_relative_error": max_rel, "session_load_seconds": load_seconds, "inference_seconds": infer_seconds, "rss_before_session_bytes": rss_before, "rss_after_session_bytes": rss_after, "peak_rss_bytes": max(_peak_rss_bytes(), _rss_bytes()), "tolerances": {"atol": float(atol), "rtol": float(rtol)}, "method": "pure ORT CPU against Torch reference saved during export"}
    manifest["verification"] = result
    _write_json(manifest_path, manifest)
    _write_json(stage_dir / "verification.json", result)
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    exp = sub.add_parser("export")
    exp.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    exp.add_argument("--kind", choices=KINDS + ("all",), default="all")
    exp.add_argument("--length", type=int, default=32)
    exp.add_argument("--opset", type=int, default=18)
    exp.add_argument("--seed", type=int, default=20260929)
    ver = sub.add_parser("verify")
    ver.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ver.add_argument("--kind", choices=KINDS + ("all",), default="all")
    ver.add_argument("--intra-op-num-threads", type=int, default=1)
    ver.add_argument("--atol", type=float, default=3e-5)
    ver.add_argument("--rtol", type=float, default=3e-5)
    args = parser.parse_args(list(argv) if argv is not None else None)
    kinds = KINDS if args.kind == "all" else (args.kind,)
    results: dict[str, Any] = {}
    for kind in kinds:
        if args.command == "export":
            results[kind] = export_kind(output_dir=args.output_dir, kind=kind, length=args.length, opset=args.opset, seed=args.seed)
        else:
            results[kind] = verify_kind(output_dir=args.output_dir, kind=kind, intra_op_num_threads=args.intra_op_num_threads, atol=args.atol, rtol=args.rtol)
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0 if all(value.get("status") == ("exported" if args.command == "export" else "verified") for value in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
