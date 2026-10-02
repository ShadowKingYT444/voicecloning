#!/usr/bin/env python3
"""Export an experimental native-T3 prefix with pinned public decoder state.

This writes a separate voice-state candidate. It does not change the default
ASMR state. Run it through scripts/nano_lab/bounded_job.py with the CPU Nano
environment. The script verifies all source pins and reproduces the supplied
native prefix byte for byte before it writes the candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PUBLIC_STATE_DIR = ROOT / "browser_tts" / "public" / "voice"
DEFAULT_DONOR_CACHE = ROOT / "artifacts" / "nano_lab" / "decoder_embedding_fit" / "conditionals.pt"
DEFAULT_CHECKPOINT_DIR = ROOT / "models" / "chatterbox-nano"
DEFAULT_PREFIX = ROOT / "artifacts" / "nano_lab" / "browser_measurements" / "local_asmr_cache_prefix_20261002" / "prefix.npy"
DEFAULT_PREFIX_SOURCE = DEFAULT_PREFIX.with_name("source.json")
DEFAULT_OUTPUT_DIR = PUBLIC_STATE_DIR / "native-asmr-donor"

PUBLIC_REPOSITORY = "owensong/chatterbox-nano-ONNX"
PUBLIC_REVISION = "4a66d7dab72a9e98f24b515d49a1d7a81632df2e"
PUBLIC_SHA256SUMS_SHA256 = "ecd0b0ee1fde0f9342564fda72fe1ebb4907baa6c9d013c09052bcad5220dcf4"
PUBLIC_ENCODER_GRAPH = "onnx/speech_encoder_q4f16.onnx"
PUBLIC_ENCODER_GRAPH_SHA256 = "29e249f59eaf95015527588b955e5286c7ee4524e7bb54a2f8d589b838e8aed2"
PUBLIC_ENCODER_WEIGHTS = "onnx/speech_encoder_q4f16.onnx_data"
PUBLIC_ENCODER_WEIGHTS_SHA256 = "55d89bd87fd36be48b2e831c99c2e309d432169119049066f9f22fcfe517798d"
PUBLIC_REFERENCE_PATH = "browser_tts/public/voice/asmr_t3_seed47_fit.wav"
PUBLIC_REFERENCE_SHA256 = "67f94a868976b22a47bce8fd00a873d4c5b7085ba6eedd698f8a898a26ce76c0"
PUBLIC_STATE_SHA256 = "6f56c0dd844ecc038e5b60debff22d1e16b44c9d13386d0f7a91156bfca1b38e"
PUBLIC_STATE_BYTES = 331_200

DONOR_CACHE_PATH = "artifacts/nano_lab/decoder_embedding_fit/conditionals.pt"
DONOR_CACHE_SHA256 = "3c78b8bedb9b5ac94d5aaf900d509162cb4c5274f59cf7fb525117498786c40c"
NATIVE_REPOSITORY = "ResembleAI/chatterbox-nano"
NATIVE_REVISION = "71ccd1d0081b430592cea481f4307e764e07bc64"
NATIVE_CHECKPOINT_PATH = "models/chatterbox-nano/t3_nano_v1.safetensors"
NATIVE_CHECKPOINT_SHA256 = "72b110185087d945dbdf54dee4e333848e1811bdd5fd6cb16ceb8da50006f0c9"
PREFIX_PATH = "artifacts/nano_lab/browser_measurements/local_asmr_cache_prefix_20261002/prefix.npy"
PREFIX_SHA256 = "6d27b0087b442abb8264d0fea19425be649f663519830e9ca28bdb57c80fe311"
PREFIX_FRAMES = 334
PROMPT_TOKEN_COUNT = PREFIX_FRAMES - 1

DATA_FILE = "asmr-state.bin"
MANIFEST_FILE = "asmr-state.json"
TENSOR_SPECS = (
    ("audio_features", "audio_features", "float32", (1, PREFIX_FRAMES, 768)),
    ("audio_tokens", "audio_tokens", "int64", (1, 88)),
    ("speaker_embeddings", "speaker_embeddings", "float32", (1, 192)),
    ("speaker_features", "speaker_features", "float32", (1, 176, 80)),
)
ELEMENT_BYTES = {"float32": 4, "int64": 8}
PUBLIC_ENCODER_SCOPE = (
    "three retained decoder-conditioning tensors copied bitwise from source_public_state; "
    "does not describe native T3 prefix"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-state-dir", type=Path, default=PUBLIC_STATE_DIR)
    parser.add_argument("--donor-cache", type=Path, default=ROOT / DONOR_CACHE_PATH)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--prefix", type=Path, default=DEFAULT_PREFIX)
    parser.add_argument("--prefix-source", type=Path, default=DEFAULT_PREFIX_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def require_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: expected {expected}, received {actual}.")


def read_public_state(state_dir: Path) -> tuple[dict[str, Any], bytes, dict[str, bytes]]:
    manifest_path = state_dir / "asmr-state.json"
    data_path = state_dir / "asmr-state.bin"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != 1
            or manifest.get("format") != "chatterbox-nano-reference-state-v1"):
        raise ValueError("Source public voice-state format is not supported.")

    model = manifest.get("model", {})
    if (model.get("repo_id") != PUBLIC_REPOSITORY
            or model.get("revision") != PUBLIC_REVISION
            or model.get("sha256sums_sha256") != PUBLIC_SHA256SUMS_SHA256):
        raise ValueError("Source public voice state does not match the pinned conversion.")
    graph = model.get("encoder_graph", {})
    weights = model.get("encoder_external_weights", {})
    if (graph.get("path") != PUBLIC_ENCODER_GRAPH
            or graph.get("sha256") != PUBLIC_ENCODER_GRAPH_SHA256
            or weights.get("path") != PUBLIC_ENCODER_WEIGHTS
            or weights.get("sha256") != PUBLIC_ENCODER_WEIGHTS_SHA256):
        raise ValueError("Source public encoder provenance differs from its pinned asset hashes.")

    reference = manifest.get("reference", {})
    if (reference.get("path") != PUBLIC_REFERENCE_PATH
            or reference.get("sha256") != PUBLIC_REFERENCE_SHA256
            or reference.get("sample_rate") != 24_000
            or reference.get("channels") != 1
            or reference.get("wav_encoding") != "PCM-24"):
        raise ValueError("Source public voice state has unexpected reference provenance.")
    require_hash(ROOT / PUBLIC_REFERENCE_PATH, PUBLIC_REFERENCE_SHA256, "pinned reference WAV")

    conditioning = manifest.get("conditioning", {})
    if (conditioning.get("source") != "generated_reference"
            or conditioning.get("speaker_specific") is not True
            or conditioning.get("zero_shot") is not False
            or conditioning.get("fitted_adapter_applied") is not False):
        raise ValueError("Source public state is not the pinned, unadapted reference state.")
    execution = manifest.get("encoder_execution", {})
    if (execution.get("provider") != "CPUExecutionProvider"
            or execution.get("onnxruntime_version") != "1.29.0"
            or execution.get("graph_optimization_level") != "ORT_ENABLE_ALL"):
        raise ValueError("Source decoder conditioning lacks the pinned CPU encoder execution record.")

    raw = data_path.read_bytes()
    if (len(raw) != PUBLIC_STATE_BYTES
            or manifest.get("data", {}).get("size_bytes") != PUBLIC_STATE_BYTES
            or manifest.get("data", {}).get("file") != DATA_FILE
            or manifest.get("data", {}).get("sha256") != PUBLIC_STATE_SHA256
            or hashlib.sha256(raw).hexdigest() != PUBLIC_STATE_SHA256):
        raise ValueError("Source public voice-state binary failed its pinned size/SHA-256 check.")

    expected_shapes = (
        ("audio_features", "float32", (1, 89, 768)),
        ("audio_tokens", "int64", (1, 88)),
        ("speaker_embeddings", "float32", (1, 192)),
        ("speaker_features", "float32", (1, 176, 80)),
    )
    tensors = manifest.get("tensors", [])
    if len(tensors) != len(expected_shapes):
        raise ValueError("Source public voice state must contain the four pinned tensors.")
    retained: dict[str, bytes] = {}
    offset = 0
    for item, (name, dtype, dims) in zip(tensors, expected_shapes, strict=True):
        byte_length = math.prod(dims) * ELEMENT_BYTES[dtype]
        if (item.get("name") != name or item.get("role") != name
                or item.get("type") != dtype or tuple(item.get("dims", ())) != dims
                or item.get("offset") != offset or item.get("byte_length") != byte_length
                or item.get("serialization_round_trip_exact") is not True):
            raise ValueError(f"Source public tensor contract differs for {name}.")
        retained[name] = raw[offset:offset + byte_length]
        offset += byte_length
    if offset != len(raw):
        raise ValueError("Source public tensor ranges do not cover the state binary.")

    audio_tokens = np.frombuffer(retained["audio_tokens"], dtype="<i8")
    if np.any(audio_tokens < 0) or np.any(audio_tokens >= 6561):
        raise ValueError("Source decoder audio tokens contain an invalid speech ID.")
    for name in ("speaker_embeddings", "speaker_features"):
        values = np.frombuffer(retained[name], dtype="<f4")
        if not np.isfinite(values).all():
            raise ValueError(f"Source public {name} contains non-finite values.")
    return manifest, raw, retained


def read_cache(cache_path: Path, torch: Any) -> tuple[np.ndarray, list[int]]:
    payload = torch.load(cache_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("t3"), dict):
        raise ValueError("The native donor cache must contain a `t3` tensor mapping.")
    t3 = payload["t3"]
    speaker_value = t3.get("speaker_emb")
    prompt_value = t3.get("cond_prompt_speech_tokens")
    if not torch.is_tensor(speaker_value) or not torch.is_tensor(prompt_value):
        raise ValueError("The native donor cache lacks speaker_emb or cond_prompt_speech_tokens.")
    speaker = speaker_value.detach().to(device="cpu", dtype=torch.float32).reshape(1, -1).numpy().copy()
    tokens_array = prompt_value.detach().to(device="cpu", dtype=torch.int64).reshape(-1).numpy().copy()
    if speaker.shape != (1, 256) or not np.isfinite(speaker).all():
        raise ValueError(f"Native T3 speaker embedding has unexpected shape or values: {speaker.shape}.")
    if tokens_array.shape != (PROMPT_TOKEN_COUNT,):
        raise ValueError(f"Native T3 prompt must contain {PROMPT_TOKEN_COUNT} tokens, got {tokens_array.size}.")
    if np.any(tokens_array < 0) or np.any(tokens_array >= 6561):
        raise ValueError("Native T3 prompt contains a token outside the speech embedding vocabulary.")
    return speaker, [int(value) for value in tokens_array]


def read_speech_embedding_rows(checkpoint: Any, token_ids: list[int]) -> np.ndarray:
    entry = checkpoint.header.get("speech_emb.weight")
    if (entry is None or entry.get("dtype") != "F32"
            or entry.get("shape") != [6563, 768]):
        raise ValueError("Native Nano speech embedding tensor differs from the pinned shape/dtype.")
    data_offset = entry["data_offsets"][0]
    row_bytes = 768 * 4
    rows: dict[int, np.ndarray] = {}
    for token_id in set(token_ids):
        checkpoint.file.seek(checkpoint.start + data_offset + token_id * row_bytes)
        raw = checkpoint.file.read(row_bytes)
        if len(raw) != row_bytes:
            raise ValueError(f"Could not read native speech embedding row {token_id}.")
        row = np.frombuffer(raw, dtype="<f4").copy()
        if not np.isfinite(row).all():
            raise ValueError(f"Native speech embedding row {token_id} contains non-finite values.")
        rows[token_id] = row
    return np.stack([rows[token_id] for token_id in token_ids]).astype(np.float32, copy=False)


def derive_native_prefix(cache_path: Path, checkpoint_path: Path, torch: Any) -> np.ndarray:
    sys.path.insert(0, str(ROOT / "scripts" / "nano_lab"))
    from checkpoint_reader import CheckpointReader

    speaker, token_ids = read_cache(cache_path, torch)
    with CheckpointReader(checkpoint_path) as checkpoint:
        weight_entry = checkpoint.header.get("cond_enc.spkr_enc.weight")
        bias_entry = checkpoint.header.get("cond_enc.spkr_enc.bias")
        if (weight_entry is None or weight_entry.get("dtype") != "F32"
                or weight_entry.get("shape") != [768, 256]
                or bias_entry is None or bias_entry.get("dtype") != "F32"
                or bias_entry.get("shape") != [768]):
            raise ValueError("Native Nano speaker projection differs from the pinned shape/dtype.")
        projection_weight = checkpoint.get_tensor("cond_enc.spkr_enc.weight").float().numpy().copy()
        projection_bias = checkpoint.get_tensor("cond_enc.spkr_enc.bias").float().numpy().copy()
        prompt_rows = read_speech_embedding_rows(checkpoint, token_ids)

    # Match the measured CPU control's projection and its exact float32 bytes.
    speaker_row = speaker @ projection_weight.T + projection_bias
    prefix = np.concatenate((speaker_row[:, None, :], prompt_rows[None, :, :]), axis=1)
    prefix = np.ascontiguousarray(prefix, dtype="<f4")
    if prefix.shape != (1, PREFIX_FRAMES, 768) or not np.isfinite(prefix).all():
        raise ValueError(f"Derived native T3 prefix has an invalid shape or values: {prefix.shape}.")
    return prefix


def verify_prefix_prototype(prefix_path: Path, source_path: Path, prefix: np.ndarray) -> None:
    require_hash(prefix_path, PREFIX_SHA256, "native T3 prefix prototype")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if (source.get("source") != str((ROOT / DONOR_CACHE_PATH).resolve())
            or source.get("sha256") != DONOR_CACHE_SHA256
            or source.get("promptTokens") != PROMPT_TOKEN_COUNT
            or source.get("prefixShape") != [1, PREFIX_FRAMES, 768]):
        raise ValueError("Native T3 prefix source metadata differs from the pinned donor cache.")
    prototype = np.load(prefix_path, allow_pickle=False)
    if (prototype.dtype != np.dtype("<f4")
            or prototype.shape != (1, PREFIX_FRAMES, 768)
            or not np.isfinite(prototype).all()):
        raise ValueError("Native T3 prefix prototype has an unexpected dtype, shape, or values.")
    derived_bytes = np.ascontiguousarray(prefix, dtype="<f4").tobytes(order="C")
    prototype_bytes = np.ascontiguousarray(prototype, dtype="<f4").tobytes(order="C")
    if derived_bytes != prototype_bytes:
        difference = np.abs(prefix.astype(np.float64) - prototype.astype(np.float64))
        raise ValueError(
            "Re-derived native T3 prefix is not bitwise identical to the pinned prototype "
            f"(max abs difference {float(np.max(difference))})."
        )


def build_candidate(
    *,
    source_manifest: dict[str, Any],
    retained: dict[str, bytes],
    prefix: np.ndarray,
    torch_version: str,
) -> tuple[dict[str, Any], bytes]:
    prefix_bytes = np.ascontiguousarray(prefix, dtype="<f4").tobytes(order="C")
    retained_bytes = b"".join(retained[name] for name in ("audio_tokens", "speaker_embeddings", "speaker_features"))
    binary = prefix_bytes + retained_bytes

    tensor_entries: list[dict[str, Any]] = []
    offset = 0
    for name, role, dtype, dims in TENSOR_SPECS:
        byte_length = math.prod(dims) * ELEMENT_BYTES[dtype]
        raw = binary[offset:offset + byte_length]
        if len(raw) != byte_length:
            raise ValueError(f"Candidate binary is truncated at tensor {name}.")
        numpy_dtype = "<f4" if dtype == "float32" else "<i8"
        restored = np.frombuffer(raw, dtype=numpy_dtype).reshape(dims)
        if restored.tobytes(order="C") != raw:
            raise ValueError(f"{name} failed its exact CPU serialization round trip.")
        tensor_entries.append({
            "byte_length": byte_length,
            "dims": list(dims),
            "name": name,
            "offset": offset,
            "role": role,
            "serialization_round_trip_exact": True,
            "type": dtype,
        })
        offset += byte_length
    if offset != len(binary):
        raise ValueError("Candidate tensor ranges do not cover the candidate binary.")

    donor_provenance = {
        "cache": {"path": DONOR_CACHE_PATH, "sha256": DONOR_CACHE_SHA256},
        "checkpoint": {
            "path": NATIVE_CHECKPOINT_PATH,
            "repo_id": NATIVE_REPOSITORY,
            "revision": NATIVE_REVISION,
            "sha256": NATIVE_CHECKPOINT_SHA256,
        },
        "execution": {
            "framework": "PyTorch",
            "provider": "CPU",
            "threads": 2,
            "version": torch_version,
        },
        "prefix": {
            "frames": PREFIX_FRAMES,
            "path": PREFIX_PATH,
            "processing": "cond_enc.spkr_enc(speaker_emb) concatenated with speech_emb(cond_prompt_speech_tokens)",
            "prompt_tokens": PROMPT_TOKEN_COUNT,
            "sha256": PREFIX_SHA256,
        },
        "profile": "clean_asmr_original",
    }
    manifest = {
        "conditioning": {
            "description": (
                "Experimental native T3 conditioning prefix from the clean ASMR original profile. "
                "The three decoder-conditioning tensors remain copied from the pinned public reference state. "
                "No fitted adapter was applied."
            ),
            "fitted_adapter_applied": False,
            "native_t3_donor": donor_provenance,
            "source": "native_t3_donor",
            "speaker_specific": True,
            "zero_shot": False,
        },
        "data": {
            "file": DATA_FILE,
            "sha256": hashlib.sha256(binary).hexdigest(),
            "size_bytes": len(binary),
        },
        "encoder_execution": {
            **source_manifest["encoder_execution"],
            "scope": PUBLIC_ENCODER_SCOPE,
        },
        "format": "chatterbox-nano-reference-state-v1",
        "model": source_manifest["model"],
        "reference": source_manifest["reference"],
        "schema_version": 1,
        "source_public_state": {
            "path": "browser_tts/public/voice/asmr-state.bin",
            "sha256": PUBLIC_STATE_SHA256,
        },
        "tensors": tensor_entries,
    }
    return manifest, binary


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
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def main() -> int:
    args = parse_args()
    source_dir = args.source_state_dir.resolve()
    cache_path = args.donor_cache.resolve()
    checkpoint_path = (args.checkpoint_dir / "t3_nano_v1.safetensors").resolve()
    output_dir = args.output_dir.resolve()
    if output_dir == source_dir:
        raise ValueError("Candidate output directory must differ from the original public voice-state directory.")
    if sys.byteorder != "little":
        raise RuntimeError("Voice-state export requires a little-endian host.")

    require_hash(cache_path, DONOR_CACHE_SHA256, "native T3 donor cache")
    require_hash(checkpoint_path, NATIVE_CHECKPOINT_SHA256, "native Nano T3 checkpoint")
    source_manifest, _source_raw, retained = read_public_state(source_dir)
    prefix_file = args.prefix.resolve()
    prefix_source_file = args.prefix_source.resolve()
    require_hash(prefix_file, PREFIX_SHA256, "native T3 prefix prototype")

    import torch

    torch.set_num_threads(2)
    prefix = derive_native_prefix(cache_path, checkpoint_path, torch)
    verify_prefix_prototype(prefix_file, prefix_source_file, prefix)
    torch_version = str(torch.__version__)
    if re.fullmatch(r"2\.\d+\.\d+\+cpu", torch_version) is None:
        raise RuntimeError(f"Use the bounded CPU PyTorch environment; found {torch_version}.")

    manifest, binary = build_candidate(
        source_manifest=source_manifest,
        retained=retained,
        prefix=prefix,
        torch_version=torch_version,
    )
    actual_audio_tokens_offset = math.prod(TENSOR_SPECS[0][3]) * ELEMENT_BYTES["float32"]
    if binary[actual_audio_tokens_offset:] != b"".join(
        retained[name] for name in ("audio_tokens", "speaker_embeddings", "speaker_features")
    ):
        raise RuntimeError("Candidate decoder-conditioning tensors differ from the pinned source state.")

    data_path = output_dir / DATA_FILE
    manifest_path = output_dir / MANIFEST_FILE
    atomic_write(data_path, binary)
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    atomic_write(manifest_path, manifest_bytes)
    print(json.dumps({
        "status": "experimental_candidate_exported",
        "manifest": str(manifest_path),
        "data": str(data_path),
        "data_bytes": len(binary),
        "data_sha256": manifest["data"]["sha256"],
        "audio_features_shape": [1, PREFIX_FRAMES, 768],
        "native_prefix_file_sha256": PREFIX_SHA256,
        "decoder_tensors_bitwise_preserved": True,
        "source_public_state_sha256": PUBLIC_STATE_SHA256,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
