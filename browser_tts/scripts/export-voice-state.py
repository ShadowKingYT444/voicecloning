#!/usr/bin/env python3
"""Export fixed conditioning tensors from the pinned public Nano encoder."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import wave
from pathlib import Path
from typing import Any


REVISION = "4a66d7dab72a9e98f24b515d49a1d7a81632df2e"
MODEL_REPOSITORY = "owensong/chatterbox-nano-ONNX"
SHA256SUMS_SHA256 = "ecd0b0ee1fde0f9342564fda72fe1ebb4907baa6c9d013c09052bcad5220dcf4"
ENCODER_GRAPH = "onnx/speech_encoder_q4f16.onnx"
ENCODER_GRAPH_SHA256 = "29e249f59eaf95015527588b955e5286c7ee4524e7bb54a2f8d589b838e8aed2"
ENCODER_WEIGHTS = "onnx/speech_encoder_q4f16.onnx_data"
ENCODER_WEIGHTS_SHA256 = "55d89bd87fd36be48b2e831c99c2e309d432169119049066f9f22fcfe517798d"
REFERENCE_RELATIVE_PATH = "browser_tts/public/voice/asmr_t3_seed47_fit.wav"
REFERENCE_SHA256 = "67f94a868976b22a47bce8fd00a873d4c5b7085ba6eedd698f8a898a26ce76c0"
SUPPORTED_ORT_VERSION = "1.29.0"
OUTPUT_NAMES = ("audio_features", "audio_tokens", "speaker_embeddings", "speaker_features")
TENSOR_SPECS = {
    "audio_features": {"type": "float32", "rank": 3, "last_dim": 768},
    "audio_tokens": {"type": "int64", "rank": 2},
    "speaker_embeddings": {"type": "float32", "rank": 2, "last_dim": 192},
    "speaker_features": {"type": "float32", "rank": 3, "last_dim": 80},
}
ORT_TYPE_STRINGS = {"float32": "tensor(float)", "int64": "tensor(int64)"}
DATA_FILE = "asmr-state.bin"
MANIFEST_FILE = "asmr-state.json"

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Local snapshot of owensong/chatterbox-nano-ONNX at the pinned revision.",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=REPO_ROOT / REFERENCE_RELATIVE_PATH,
        help="Tracked 24 kHz mono PCM reference WAV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "browser_tts/public/voice",
        help="Directory for asmr-state.json and asmr-state.bin.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: expected {expected}, received {actual}: {path}")


def read_reference(path: Path, np: Any) -> Any:
    require_hash(path, REFERENCE_SHA256, "reference WAV")
    with wave.open(str(path), "rb") as wav:
        if wav.getcomptype() != "NONE":
            raise ValueError("The reference WAV must use uncompressed PCM.")
        if wav.getnchannels() != 1 or wav.getframerate() != 24_000 or wav.getsampwidth() != 3:
            raise ValueError("The pinned reference must be mono 24 kHz PCM-24.")
        frames = wav.getnframes()
        raw = wav.readframes(frames)

    if len(raw) != frames * 3:
        raise ValueError("The reference WAV frame count does not match its payload.")
    packed = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
    values = packed[:, 0] | (packed[:, 1] << 8) | (packed[:, 2] << 16)
    signed = (values ^ 0x800000) - 0x800000
    samples = signed.astype(np.float32) / np.float32(8_388_608.0)
    return samples.reshape(1, -1)


def validate_outputs(output_metadata: list[Any], outputs: list[Any], np: Any) -> list[dict[str, Any]]:
    names = [item.name for item in output_metadata]
    if tuple(names) != OUTPUT_NAMES or len(outputs) != len(OUTPUT_NAMES):
        raise ValueError(f"Pinned speech encoder outputs changed: expected {OUTPUT_NAMES}, received {names}.")

    validated = []
    for metadata, output in zip(output_metadata, outputs, strict=True):
        spec = TENSOR_SPECS[metadata.name]
        if metadata.type != ORT_TYPE_STRINGS[spec["type"]]:
            raise TypeError(f"{metadata.name} graph metadata has type {metadata.type}; expected {ORT_TYPE_STRINGS[spec['type']]}.")
        graph_shape = metadata.shape
        if len(graph_shape) != spec["rank"] or (isinstance(graph_shape[-1], int) and graph_shape[-1] != spec.get("last_dim", graph_shape[-1])):
            raise ValueError(f"{metadata.name} graph metadata has unexpected shape {graph_shape}.")
        array = np.asarray(output)
        expected_dtype = np.dtype(spec["type"])
        if array.dtype != expected_dtype:
            raise TypeError(f"{metadata.name} has dtype {array.dtype}; expected {expected_dtype}.")
        if array.ndim != spec["rank"] or array.shape[0] != 1:
            raise ValueError(f"{metadata.name} has unexpected shape {array.shape}.")
        if "last_dim" in spec and array.shape[-1] != spec["last_dim"]:
            raise ValueError(f"{metadata.name} has unexpected trailing dimension {array.shape[-1]}.")
        if any(int(dim) <= 0 for dim in array.shape):
            raise ValueError(f"{metadata.name} contains an empty or invalid dimension: {array.shape}.")
        if spec["type"] == "float32" and not np.isfinite(array).all():
            raise ValueError(f"{metadata.name} contains non-finite values.")
        if metadata.name == "audio_tokens" and (np.any(array < 0) or np.any(array >= 6563)):
            raise ValueError("audio_tokens contains an ID outside the pinned Nano speech-token vocabulary.")
        validated.append({"metadata": metadata, "array": array})
    return validated


def tensor_bytes(array: Any, np: Any) -> bytes:
    if sys.byteorder != "little":
        raise RuntimeError("Voice-state export requires a little-endian host.")
    return np.ascontiguousarray(array).tobytes(order="C")


def verify_serialization_round_trip(array: Any, raw: bytes, np: Any, name: str) -> None:
    original = np.ascontiguousarray(array)
    restored = np.frombuffer(raw, dtype=original.dtype).reshape(original.shape)
    original_bytes = original.view(np.uint8).reshape(-1)
    restored_bytes = restored.view(np.uint8).reshape(-1)
    if not np.array_equal(original_bytes, restored_bytes):
        raise RuntimeError(f"{name} did not survive exact CPU-output serialization round-trip.")


def atomic_write(path: Path, contents: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(contents)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path and temporary_path.exists():
            temporary_path.unlink()


def export_state(model_dir: Path, reference_path: Path, output_dir: Path) -> dict[str, Any]:
    import numpy as np
    import onnxruntime as ort

    if ort.__version__ != SUPPORTED_ORT_VERSION:
        raise RuntimeError(
            f"Use ONNX Runtime {SUPPORTED_ORT_VERSION} from .venv-nano-cpu; found {ort.__version__}."
        )

    graph_path = model_dir / ENCODER_GRAPH
    weights_path = model_dir / ENCODER_WEIGHTS
    require_hash(graph_path, ENCODER_GRAPH_SHA256, "pinned encoder graph")
    require_hash(weights_path, ENCODER_WEIGHTS_SHA256, "pinned encoder external weights")
    audio_values = read_reference(reference_path, np)

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(graph_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    providers = session.get_providers()
    if providers != ["CPUExecutionProvider"]:
        raise RuntimeError(f"Expected CPU-only ONNX Runtime, received providers {providers}.")

    inputs = session.get_inputs()
    if len(inputs) != 1 or inputs[0].name != "audio_values" or inputs[0].type != "tensor(float)":
        raise RuntimeError("Pinned encoder input metadata changed; expected float32 audio_values only.")
    metadata = session.get_outputs()
    outputs = session.run(OUTPUT_NAMES, {"audio_values": audio_values})
    tensors = validate_outputs(metadata, outputs, np)

    output_dir.mkdir(parents=True, exist_ok=True)
    binary_path = output_dir / DATA_FILE
    manifest_path = output_dir / MANIFEST_FILE
    binary_digest = hashlib.sha256()
    tensor_entries = []
    offset = 0

    temporary_binary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=output_dir, prefix=f".{DATA_FILE}.", delete=False) as binary:
            temporary_binary = Path(binary.name)
            for tensor in tensors:
                meta = tensor["metadata"]
                array = tensor["array"]
                raw = tensor_bytes(array, np)
                verify_serialization_round_trip(array, raw, np, meta.name)
                binary.write(raw)
                binary_digest.update(raw)
                tensor_entries.append(
                    {
                        "role": meta.name,
                        "name": meta.name,
                        "type": str(array.dtype),
                        "dims": [int(dim) for dim in array.shape],
                        "offset": offset,
                        "byte_length": len(raw),
                        "serialization_round_trip_exact": True,
                    }
                )
                offset += len(raw)
            binary.flush()
            os.fsync(binary.fileno())

        if offset <= 0 or offset > 16 * 1024 * 1024:
            raise RuntimeError(f"Serialized voice state has an unexpected size: {offset} bytes.")

        manifest = {
            "schema_version": 1,
            "format": "chatterbox-nano-reference-state-v1",
            "model": {
                "repo_id": MODEL_REPOSITORY,
                "revision": REVISION,
                "sha256sums_sha256": SHA256SUMS_SHA256,
                "encoder_graph": {"path": ENCODER_GRAPH, "sha256": ENCODER_GRAPH_SHA256},
                "encoder_external_weights": {"path": ENCODER_WEIGHTS, "sha256": ENCODER_WEIGHTS_SHA256},
            },
            "reference": {
                "path": REFERENCE_RELATIVE_PATH,
                "sha256": REFERENCE_SHA256,
                "sample_rate": 24_000,
                "channels": 1,
                "wav_encoding": "PCM-24",
            },
            "conditioning": {
                "source": "generated_reference",
                "zero_shot": False,
                "speaker_specific": True,
                "fitted_adapter_applied": False,
                "description": (
                    "Generated-reference conditioning from the pinned public Q4F16 base conversion. "
                    "No experimental fitted adapter was applied. This state is specific to the exact reference audio."
                ),
            },
            "encoder_execution": {
                "provider": "CPUExecutionProvider",
                "onnxruntime_version": ort.__version__,
                "graph_optimization_level": "ORT_ENABLE_ALL",
            },
            "data": {
                "file": DATA_FILE,
                "size_bytes": offset,
                "sha256": binary_digest.hexdigest(),
            },
            "tensors": tensor_entries,
        }

        os.replace(temporary_binary, binary_path)
        temporary_binary = None
        manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
        atomic_write(manifest_path, manifest_bytes)
    finally:
        if temporary_binary and temporary_binary.exists():
            temporary_binary.unlink()

    return {
        "manifest": str(manifest_path),
        "binary": str(binary_path),
        "binary_sha256": manifest["data"]["sha256"],
        "binary_size_bytes": offset,
        "onnxruntime_version": ort.__version__,
        "provider": "CPUExecutionProvider",
        "tensors": [
            {"name": item["name"], "type": item["type"], "dims": item["dims"]}
            for item in tensor_entries
        ],
        "serialization_round_trip": "exact byte equality for each CPU output tensor",
    }


def main() -> None:
    args = parse_args()
    report = export_state(args.model_dir.resolve(), args.reference.resolve(), args.output_dir.resolve())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
