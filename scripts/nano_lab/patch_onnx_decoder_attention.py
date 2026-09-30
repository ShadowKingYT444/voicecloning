"""Statically fold the Nano decoder-attention adapter into ONNX external weights.

The meanflow estimator export stores its parameters in
``nano_meanflow_estimator.weights.bin``.  This patcher changes only the 224
self-attention weight spans used by ``nano_decoder_attention_lora_v1``.  It
never loads the ONNX graph with an ONNX library and never constructs a model.
The graph bytes stay unchanged, so an ONNX Runtime process has no adapter
module or adapter computation at inference time.

The output is a new model directory.  The input directory and its external
weight file are never edited.  The output manifest deliberately clears the
ordinary numerical verification field and writes a pending
``decoder_attention_verification`` field.  A separate Torch-reference versus
ORT check must verify the adapted sidecar before the output is used.

Example::

    .venv-nano-cpu/bin/python scripts/nano_lab/patch_onnx_decoder_attention.py \
      --base-model-dir artifacts/nano_lab/onnx_staged \
      --adapter artifacts/nano_lab/decoder_attention_fit/best_attention.pt \
      --output-model-dir artifacts/nano_lab/onnx_staged_decoder_attention
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Mapping, Sequence

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[2]
DEFAULT_MODEL_DIR = ROOT / "artifacts" / "nano_lab" / "onnx_staged"
DEFAULT_ADAPTER = ROOT / "artifacts" / "nano_lab" / "decoder_attention_fit" / "best_attention.pt"
DEFAULT_INVENTORY = ROOT / "artifacts" / "nano_lab" / "decoder_attention_inventory.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "nano_lab" / "onnx_staged_decoder_attention"
MEANFLOW_STAGE = "meanflow_estimator"
CHECKPOINT_FORMAT = "nano_decoder_attention_lora_v1"
PATCH_FORMAT = "nano_decoder_attention_merged_v1"
TARGET_PREFIX = "flow.decoder.estimator."
ONNX_PREFIX = "estimator."
RANK = 2
ALPHA = 2.0
ADAPTER_SCALE = 1.0
TARGET_COUNT = 224
TARGET_PARAMETER_COUNT = 344064
ONNX_FLOAT = 1
ONNX_EXTERNAL = 1
COPY_CHUNK_BYTES = 4 * 1024 * 1024
MAX_METADATA_FIELD_BYTES = 1 << 20


class PatchError(RuntimeError):
    """Raised when a staged graph or adapter violates the patch contract."""


@dataclass(frozen=True)
class GraphTensor:
    """TensorProto metadata read without materialising tensor values."""

    name: str
    dims: tuple[int, ...]
    data_type: int
    external_data: Mapping[str, str]
    has_raw_data: bool
    data_location: int | None


@dataclass(frozen=True)
class GraphMetadata:
    path: Path
    bytes: int
    initializers: tuple[GraphTensor, ...]

    def by_name(self) -> dict[str, GraphTensor]:
        return {item.name: item for item in self.initializers}


@dataclass(frozen=True)
class ExternalInitializer:
    graph_name: str
    state_name: str
    shape: tuple[int, ...]
    dtype: str
    offset: int
    length: int
    sha256: str


@dataclass(frozen=True)
class ExternalLayout:
    graph_path: Path
    graph_metadata: GraphMetadata
    report_path: Path
    report: Mapping[str, Any]
    weights_path: Path
    weight_bytes: int
    initializers: tuple[ExternalInitializer, ...]

    def by_name(self) -> dict[str, ExternalInitializer]:
        return {item.graph_name: item for item in self.initializers}


@dataclass(frozen=True)
class FactorSpec:
    """One CPU fp32 factor pair and its required baseline hash."""

    native_name: str
    onnx_name: str
    shape: tuple[int, int]
    down: np.ndarray
    up: np.ndarray
    base_sha256: str


@dataclass(frozen=True)
class AttentionAdapter:
    path: Path
    sha256: str
    inventory_sha256: str
    model_checkpoint_sha256: str
    initial_conditionals_sha256: str
    prepared_conditionals_sha256: str
    factors: tuple[FactorSpec, ...]
    rank: int = RANK
    alpha: float = ALPHA
    checkpoint_scale: float = ADAPTER_SCALE

    def by_onnx_name(self) -> dict[str, FactorSpec]:
        return {item.onnx_name: item for item in self.factors}


@dataclass(frozen=True)
class WeightPatchResult:
    base_weights_sha256: str
    patched_weights_sha256: str
    weight_bytes: int
    changed_spans: tuple[Mapping[str, Any], ...]
    outside_bytes_identical: bool


# ---------------------------------------------------------------------------
# Small protobuf reader.  It reads ModelProto/GraphProto/TensorProto metadata
# only.  Raw tensor fields are skipped by seeking over their length.


def _read_varint(handle: BinaryIO, *, limit: int | None = None) -> int:
    start = handle.tell()
    value = 0
    shift = 0
    while True:
        byte = handle.read(1)
        if not byte:
            raise PatchError("truncated protobuf varint")
        digit = byte[0]
        value |= (digit & 0x7F) << shift
        if not digit & 0x80:
            if limit is not None and handle.tell() - start > limit:
                raise PatchError("protobuf varint crosses its message boundary")
            return value
        shift += 7
        if shift > 70:
            raise PatchError("protobuf varint is too long")


def _read_key(handle: BinaryIO, end: int) -> tuple[int, int]:
    if handle.tell() >= end:
        raise PatchError("protobuf field read crossed its message boundary")
    key = _read_varint(handle)
    number, wire_type = key >> 3, key & 0x07
    if number <= 0 or wire_type in (3, 4):
        raise PatchError(f"unsupported protobuf field key {key}")
    return number, wire_type


def _read_length(handle: BinaryIO, end: int) -> tuple[int, int, int]:
    length = _read_varint(handle)
    start = handle.tell()
    stop = start + length
    if stop > end:
        raise PatchError(f"protobuf length {length} crosses its message boundary")
    return length, start, stop


def _skip_field(handle: BinaryIO, wire_type: int, end: int) -> None:
    if wire_type == 0:
        _read_varint(handle, limit=end - handle.tell())
    elif wire_type == 1:
        if handle.tell() + 8 > end:
            raise PatchError("truncated fixed64 protobuf field")
        handle.seek(8, os.SEEK_CUR)
    elif wire_type == 2:
        _, _, stop = _read_length(handle, end)
        handle.seek(stop)
    elif wire_type == 5:
        if handle.tell() + 4 > end:
            raise PatchError("truncated fixed32 protobuf field")
        handle.seek(4, os.SEEK_CUR)
    else:
        raise PatchError(f"unsupported protobuf wire type {wire_type}")


def _read_small_bytes(handle: BinaryIO, end: int) -> bytes:
    length, start, stop = _read_length(handle, end)
    if length > MAX_METADATA_FIELD_BYTES:
        raise PatchError(f"protobuf metadata field is too large: {length} bytes")
    handle.seek(start)
    value = handle.read(length)
    if len(value) != length:
        raise PatchError("truncated protobuf bytes field")
    handle.seek(stop)
    return value


def _decode_signed_int64(value: int) -> int:
    return value - (1 << 64) if value >= (1 << 63) else value


def _decode_packed_int64(payload: bytes) -> list[int]:
    values: list[int] = []
    index = 0
    while index < len(payload):
        value = 0
        shift = 0
        while True:
            if index >= len(payload):
                raise PatchError("truncated packed int64 protobuf field")
            digit = payload[index]
            index += 1
            value |= (digit & 0x7F) << shift
            if not digit & 0x80:
                values.append(_decode_signed_int64(value))
                break
            shift += 7
            if shift > 70:
                raise PatchError("packed int64 varint is too long")
    return values


def _parse_external_entry(handle: BinaryIO, end: int) -> tuple[str, str]:
    key: str | None = None
    value: str | None = None
    while handle.tell() < end:
        number, wire_type = _read_key(handle, end)
        if number == 1 and wire_type == 2:
            key = _read_small_bytes(handle, end).decode("utf-8")
        elif number == 2 and wire_type == 2:
            value = _read_small_bytes(handle, end).decode("utf-8")
        else:
            _skip_field(handle, wire_type, end)
    if handle.tell() != end or key is None or value is None:
        raise PatchError("ONNX external_data entry is missing key or value")
    return key, value


def _parse_tensor(handle: BinaryIO, end: int) -> GraphTensor:
    dims: list[int] = []
    data_type: int | None = None
    name: str | None = None
    external: dict[str, str] = {}
    has_raw_data = False
    data_location: int | None = None
    while handle.tell() < end:
        number, wire_type = _read_key(handle, end)
        if number == 1:
            if wire_type == 0:
                dims.append(_decode_signed_int64(_read_varint(handle)))
            elif wire_type == 2:
                dims.extend(_decode_packed_int64(_read_small_bytes(handle, end)))
            else:
                _skip_field(handle, wire_type, end)
        elif number == 2 and wire_type == 0:
            data_type = int(_read_varint(handle))
        elif number == 8 and wire_type == 2:
            try:
                name = _read_small_bytes(handle, end).decode("utf-8")
            except UnicodeDecodeError as exc:
                raise PatchError("ONNX initializer name is not UTF-8") from exc
        elif number == 9 and wire_type == 2:
            _, _, stop = _read_length(handle, end)
            handle.seek(stop)
            has_raw_data = True
        elif number == 13 and wire_type == 2:
            _, start, stop = _read_length(handle, end)
            handle.seek(start)
            key, value = _parse_external_entry(handle, stop)
            if key in external:
                raise PatchError(f"ONNX initializer has duplicate external_data key {key!r}")
            external[key] = value
            handle.seek(stop)
        elif number == 14 and wire_type == 0:
            data_location = int(_read_varint(handle))
        else:
            _skip_field(handle, wire_type, end)
    if handle.tell() != end:
        raise PatchError("TensorProto parser did not consume its message")
    if name is None:
        raise PatchError("ONNX initializer has no name")
    if data_type is None:
        raise PatchError(f"ONNX initializer {name!r} has no data_type")
    return GraphTensor(name, tuple(dims), data_type, dict(external), has_raw_data, data_location)


def read_graph_metadata(path: Path | str) -> GraphMetadata:
    """Read graph initializer metadata without loading the ONNX graph."""

    graph_path = Path(path).expanduser().resolve()
    if not graph_path.is_file():
        raise FileNotFoundError(graph_path)
    file_size = graph_path.stat().st_size
    initializers: list[GraphTensor] = []
    graph_found = False
    with graph_path.open("rb") as handle:
        while handle.tell() < file_size:
            number, wire_type = _read_key(handle, file_size)
            if number == 7 and wire_type == 2:
                if graph_found:
                    raise PatchError("ModelProto contains multiple graphs")
                graph_found = True
                _, graph_start, graph_end = _read_length(handle, file_size)
                while handle.tell() < graph_end:
                    field_number, field_wire = _read_key(handle, graph_end)
                    if field_number == 5 and field_wire == 2:
                        _, tensor_start, tensor_end = _read_length(handle, graph_end)
                        handle.seek(tensor_start)
                        tensor = _parse_tensor(handle, tensor_end)
                        if any(item.name == tensor.name for item in initializers):
                            raise PatchError(f"duplicate ONNX initializer name {tensor.name!r}")
                        initializers.append(tensor)
                    else:
                        _skip_field(handle, field_wire, graph_end)
                if handle.tell() != graph_end:
                    raise PatchError("GraphProto parser did not consume its message")
            else:
                _skip_field(handle, wire_type, file_size)
        if handle.tell() != file_size:
            raise PatchError("ModelProto parser did not consume its file")
    if not graph_found:
        raise PatchError(f"ModelProto has no graph: {graph_path}")
    return GraphMetadata(graph_path, file_size, tuple(initializers))


# ---------------------------------------------------------------------------
# Hashing, JSON, and report validation.


def _file_identity(path: Path) -> tuple[int, int, int, int, int]:
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _sha256_file(path: Path, *, chunk_bytes: int = COPY_CHUNK_BYTES) -> str:
    before = _file_identity(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(block)
    after = _file_identity(path)
    if before != after:
        raise PatchError(f"file changed while hashing: {path}")
    return digest.hexdigest()


def _sha256_bytes(value: bytes | bytearray | memoryview) -> str:
    return hashlib.sha256(value).hexdigest()


def _hex(value: object, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise PatchError(f"{name} must be a 64-character hexadecimal SHA-256 string")
    result = value.lower()
    if any(char not in "0123456789abcdef" for char in result):
        raise PatchError(f"{name} must be a 64-character hexadecimal SHA-256 string")
    return result


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except Exception as exc:
        raise PatchError(f"could not read JSON manifest: {path}") from exc
    if not isinstance(value, dict):
        raise PatchError(f"JSON manifest must be an object: {path}")
    return value


def _safe_stage_file(stage_dir: Path, relative_name: str, *, label: str) -> Path:
    if not isinstance(relative_name, str) or not relative_name:
        raise PatchError(f"{label} must be a non-empty relative path")
    value = Path(relative_name)
    if value.is_absolute() or ".." in value.parts or len(value.parts) != 1:
        raise PatchError(f"{label} must be a stage-relative basename, got {relative_name!r}")
    path = (stage_dir / value).resolve()
    if path.parent != stage_dir.resolve():
        raise PatchError(f"{label} escapes the stage directory")
    return path


def _as_positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise PatchError(f"{name} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise PatchError(f"{name} must be a positive integer") from exc
    if result <= 0:
        raise PatchError(f"{name} must be a positive integer")
    return result


def _as_nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise PatchError(f"{name} must be a non-negative integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise PatchError(f"{name} must be a non-negative integer") from exc
    if result < 0:
        raise PatchError(f"{name} must be a non-negative integer")
    return result


def _validate_strength(value: object) -> float:
    if isinstance(value, bool):
        raise PatchError("strength must be finite and in [0,1]")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise PatchError("strength must be finite and in [0,1]") from exc
    if not math.isfinite(result) or result < 0.0 or result > 1.0:
        raise PatchError("strength must be finite and in [0,1]")
    return result


def _load_external_layout(stage_dir: Path | str) -> ExternalLayout:
    """Validate the graph, external report, and sidecar spans."""

    stage = Path(stage_dir).expanduser().resolve()
    manifest_path = stage / "manifest.json"
    graph_path = stage / "nano_meanflow_estimator.onnx"
    report_path = stage / "nano_meanflow_estimator.external.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"meanflow stage manifest not found: {manifest_path}")
    if not graph_path.is_file():
        raise FileNotFoundError(f"meanflow ONNX graph not found: {graph_path}")
    if not report_path.is_file():
        raise FileNotFoundError(f"meanflow external report not found: {report_path}")
    report = _load_json(report_path)
    if report.get("status") != "exported":
        raise PatchError(f"external report is not exported: {report.get('status')!r}")
    if report.get("graph_bytes") is not None and int(report["graph_bytes"]) != graph_path.stat().st_size:
        raise PatchError("external report graph_bytes does not match the graph")
    report_graph = report.get("graph")
    if report_graph and Path(str(report_graph)).name != graph_path.name:
        raise PatchError("external report refers to a different graph filename")
    external = report.get("external")
    if not isinstance(external, Mapping):
        raise PatchError("external report has no external object")
    weight_file = external.get("weight_file")
    weights_path = _safe_stage_file(stage, weight_file, label="external weight_file")
    if not weights_path.is_file():
        raise FileNotFoundError(f"meanflow external weights not found: {weights_path}")
    weight_bytes = weights_path.stat().st_size
    declared_weight_bytes = _as_nonnegative_int(external.get("weight_bytes"), name="external weight_bytes")
    if declared_weight_bytes != weight_bytes:
        raise PatchError("external report weight_bytes does not match the sidecar")
    raw_initializers = external.get("initializers")
    if not isinstance(raw_initializers, list) or not raw_initializers:
        raise PatchError("external report has no initializer records")
    records: list[ExternalInitializer] = []
    names: set[str] = set()
    for index, raw in enumerate(raw_initializers):
        if not isinstance(raw, Mapping):
            raise PatchError(f"external initializer {index} is not an object")
        graph_name = raw.get("graph_name")
        state_name = raw.get("state_name")
        if not isinstance(graph_name, str) or not graph_name:
            raise PatchError(f"external initializer {index} has no graph_name")
        if not isinstance(state_name, str) or state_name != graph_name:
            raise PatchError(f"external initializer {graph_name!r} state_name does not exactly match graph_name")
        if graph_name in names:
            raise PatchError(f"external report repeats initializer {graph_name!r}")
        names.add(graph_name)
        raw_shape = raw.get("shape")
        if not isinstance(raw_shape, (list, tuple)) or not raw_shape:
            raise PatchError(f"external initializer {graph_name!r} has no shape")
        shape = tuple(_as_positive_int(value, name=f"{graph_name} shape") for value in raw_shape)
        dtype = raw.get("dtype")
        if dtype != "float32":
            raise PatchError(f"external initializer {graph_name!r} dtype must be float32")
        offset = _as_nonnegative_int(raw.get("offset"), name=f"{graph_name} offset")
        length = _as_positive_int(raw.get("length"), name=f"{graph_name} length")
        expected_length = int(np.prod(shape, dtype=np.int64)) * 4
        if length != expected_length:
            raise PatchError(f"external initializer {graph_name!r} length does not match shape")
        if offset + length > weight_bytes:
            raise PatchError(f"external initializer {graph_name!r} exceeds sidecar bounds")
        digest = _hex(raw.get("sha256"), name=f"{graph_name} sha256")
        records.append(ExternalInitializer(graph_name, state_name, shape, dtype, offset, length, digest))
    ordered = sorted(records, key=lambda item: item.offset)
    cursor = 0
    for item in ordered:
        if item.offset < cursor:
            raise PatchError(f"external initializer spans overlap at {item.graph_name!r}")
        cursor = item.offset + item.length
    graph_metadata = read_graph_metadata(graph_path)
    graph_by_name = graph_metadata.by_name()
    if set(graph_by_name) != names:
        missing = sorted(names - set(graph_by_name))[:4]
        extra = sorted(set(graph_by_name) - names)[:4]
        raise PatchError(f"graph and external report initializer sets differ; missing={missing}, extra={extra}")
    for record in records:
        tensor = graph_by_name[record.graph_name]
        if tuple(tensor.dims) != record.shape:
            raise PatchError(f"graph shape differs from external report for {record.graph_name!r}")
        if tensor.data_type != ONNX_FLOAT:
            raise PatchError(f"graph initializer {record.graph_name!r} is not ONNX FLOAT")
        if tensor.has_raw_data or tensor.data_location != ONNX_EXTERNAL:
            raise PatchError(f"graph initializer {record.graph_name!r} is not external-only")
        if tensor.external_data.get("location") != weight_file:
            raise PatchError(f"graph external location differs for {record.graph_name!r}")
        try:
            graph_offset = int(tensor.external_data.get("offset", ""))
            graph_length = int(tensor.external_data.get("length", ""))
        except ValueError as exc:
            raise PatchError(f"graph external offset/length is invalid for {record.graph_name!r}") from exc
        if graph_offset != record.offset or graph_length != record.length:
            raise PatchError(f"graph external span differs for {record.graph_name!r}")
    return ExternalLayout(graph_path, graph_metadata, report_path, report, weights_path, weight_bytes, tuple(records))


# ---------------------------------------------------------------------------
# Adapter loading and the low-RSS sidecar patch core.


def _load_attention_adapter(
    adapter_path: Path | str,
    *,
    inventory_path: Path | str = DEFAULT_INVENTORY,
) -> AttentionAdapter:
    """Load the small factor checkpoint without loading a model."""

    path = Path(adapter_path).expanduser().resolve()
    inventory_file = Path(inventory_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"decoder attention adapter not found: {path}")
    if not inventory_file.is_file():
        raise FileNotFoundError(f"decoder attention inventory not found: {inventory_file}")
    try:
        import torch
        from decoder_attention import _load_checkpoint, _load_inventory
    except Exception as exc:
        raise PatchError("PyTorch and decoder_attention validators are required to read the adapter") from exc
    inventory, inventory_sha = _load_inventory(inventory_file)
    try:
        payload, raw_factors, _ = _load_checkpoint(torch, path, inventory)
    except Exception as exc:
        if isinstance(exc, PatchError):
            raise
        raise PatchError(f"decoder attention adapter validation failed: {exc}") from exc
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise PatchError(f"unsupported decoder attention format: {payload.get('format')!r}")
    if int(payload.get("rank", -1)) != RANK or not math.isclose(float(payload.get("alpha")), ALPHA, rel_tol=0.0, abs_tol=1e-8):
        raise PatchError("decoder attention adapter requires rank=2 and alpha=2")
    if not math.isclose(float(payload.get("scale")), ADAPTER_SCALE, rel_tol=0.0, abs_tol=1e-8):
        raise PatchError("decoder attention adapter requires scale=1")
    if payload.get("inventory_sha256") != inventory_sha:
        raise PatchError("decoder attention inventory SHA-256 does not match adapter metadata")
    adapter_sha = _sha256_file(path)
    factors: list[FactorSpec] = []
    for native_name in sorted(inventory):
        raw = raw_factors.get(native_name)
        if not isinstance(raw, Mapping):
            raise PatchError(f"decoder attention factor is missing: {native_name}")
        onnx_name = native_name.removeprefix("flow.decoder.")
        if not onnx_name.startswith(ONNX_PREFIX):
            raise PatchError(f"decoder attention target does not map to estimator state: {native_name}")
        shape = tuple(int(value) for value in inventory[native_name])
        try:
            down = np.ascontiguousarray(raw["down"].detach().cpu().numpy(), dtype="<f4")
            up = np.ascontiguousarray(raw["up"].detach().cpu().numpy(), dtype="<f4")
        except Exception as exc:
            raise PatchError(f"could not copy decoder attention factors for {native_name}") from exc
        if down.shape != (RANK, shape[1]) or up.shape != (shape[0], RANK):
            raise PatchError(f"decoder attention factor shapes are invalid for {native_name}")
        if not np.isfinite(down).all() or not np.isfinite(up).all():
            raise PatchError(f"decoder attention factors are non-finite for {native_name}")
        factors.append(
            FactorSpec(
                native_name=native_name,
                onnx_name=onnx_name,
                shape=(shape[0], shape[1]),
                down=down,
                up=up,
                base_sha256=str(raw["base_hash"]),
            )
        )
    if len(factors) != TARGET_COUNT:
        raise PatchError(f"decoder attention adapter must contain {TARGET_COUNT} factors")
    parameter_count = sum(RANK * (item.shape[0] + item.shape[1]) for item in factors)
    if parameter_count != TARGET_PARAMETER_COUNT:
        raise PatchError(f"decoder attention factor parameter count is {parameter_count}, expected {TARGET_PARAMETER_COUNT}")
    return AttentionAdapter(
        path=path,
        sha256=adapter_sha,
        inventory_sha256=inventory_sha,
        model_checkpoint_sha256=_hex(payload["model_checkpoint_sha256"], name="model_checkpoint_sha256"),
        initial_conditionals_sha256=_hex(payload["initial_conditionals_sha256"], name="initial_conditionals_sha256"),
        prepared_conditionals_sha256=_hex(payload["prepared_conditionals_sha256"], name="prepared_conditionals_sha256"),
        factors=tuple(factors),
    )


def _merge_weight(base_raw: bytes, factor: FactorSpec, *, strength: float) -> bytes:
    expected_length = int(np.prod(factor.shape, dtype=np.int64)) * 4
    if len(base_raw) != expected_length:
        raise PatchError(f"base span for {factor.onnx_name!r} has the wrong byte length")
    base = np.frombuffer(base_raw, dtype="<f4").copy().reshape(factor.shape)
    delta = np.matmul(factor.up, factor.down).astype(np.float32, copy=False)
    merged = base + np.float32((ALPHA / float(RANK)) * strength) * delta
    if not np.isfinite(merged).all():
        raise PatchError(f"merged decoder attention weight is non-finite: {factor.onnx_name}")
    return np.ascontiguousarray(merged, dtype="<f4").tobytes(order="C")


def _copy_digest_range(source: BinaryIO, destination: BinaryIO, start: int, stop: int, base_digest: Any, patched_digest: Any) -> None:
    if stop < start:
        raise PatchError(f"invalid sidecar copy range {start}:{stop}")
    source.seek(start)
    remaining = stop - start
    while remaining:
        block = source.read(min(COPY_CHUNK_BYTES, remaining))
        if not block:
            raise PatchError("external sidecar ended during copy")
        destination.write(block)
        base_digest.update(block)
        patched_digest.update(block)
        remaining -= len(block)


def _patch_external_weights(
    base_weights: Path | str,
    output_weights: Path | str,
    layout: ExternalLayout,
    factors: Mapping[str, FactorSpec],
    *,
    strength: float,
) -> WeightPatchResult:
    """Patch external spans while retaining only one span in memory.

    This helper accepts a small factor mapping so pure synthetic tests do not
    need to create the canonical 224-target checkpoint.  Production callers
    validate the exact canonical adapter before calling it.
    """

    value = _validate_strength(strength)
    source_path = Path(base_weights).expanduser().resolve()
    destination_path = Path(output_weights).expanduser().resolve()
    if source_path != layout.weights_path.resolve():
        raise PatchError("base sidecar does not match the validated external layout")
    if destination_path.exists():
        raise FileExistsError(f"refusing to overwrite existing sidecar: {destination_path}")
    report_by_name = layout.by_name()
    if set(factors) - set(report_by_name):
        unknown = sorted(set(factors) - set(report_by_name))[:4]
        raise PatchError(f"adapter targets are absent from the external graph: {unknown}")
    for name, factor in factors.items():
        record = report_by_name[name]
        if tuple(record.shape) != tuple(factor.shape):
            raise PatchError(f"external shape differs from adapter for {name!r}")
        if record.sha256 != _hex(factor.base_sha256, name=f"{name} base_sha256"):
            raise PatchError(f"external baseline hash differs from adapter for {name!r}")
    if source_path.stat().st_size != layout.weight_bytes:
        raise PatchError("external sidecar changed after layout validation")
    source_identity = _file_identity(source_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    changed: list[Mapping[str, Any]] = []
    base_digest = hashlib.sha256()
    patched_digest = hashlib.sha256()
    ordered = sorted(layout.initializers, key=lambda item: item.offset)
    cursor = 0
    try:
        with source_path.open("rb") as source, destination_path.open("xb") as destination:
            for record in ordered:
                _copy_digest_range(source, destination, cursor, record.offset, base_digest, patched_digest)
                source.seek(record.offset)
                original = source.read(record.length)
                if len(original) != record.length:
                    raise PatchError(f"external sidecar ended inside {record.graph_name!r}")
                before_sha = _sha256_bytes(original)
                if before_sha != record.sha256:
                    raise PatchError(f"external baseline bytes do not match declared hash for {record.graph_name!r}")
                factor = factors.get(record.graph_name)
                if factor is None:
                    merged = original
                else:
                    if before_sha != record.sha256 or before_sha != factor.base_sha256:
                        raise PatchError(f"external baseline bytes do not match declared hash for {record.graph_name!r}")
                    merged = _merge_weight(original, factor, strength=value)
                    changed.append(
                        {
                            "name": record.graph_name,
                            "state_name": record.state_name,
                            "native_name": factor.native_name,
                            "shape": list(record.shape),
                            "dtype": record.dtype,
                            "offset": record.offset,
                            "length": record.length,
                            "before_sha256": before_sha,
                            "after_sha256": _sha256_bytes(merged),
                            "strength": value,
                        }
                    )
                destination.write(merged)
                base_digest.update(original)
                patched_digest.update(merged)
                cursor = record.offset + record.length
            _copy_digest_range(source, destination, cursor, layout.weight_bytes, base_digest, patched_digest)
        if _file_identity(source_path) != source_identity:
            raise PatchError("external sidecar changed while it was being patched")
    except Exception:
        destination_path.unlink(missing_ok=True)
        raise
    if destination_path.stat().st_size != layout.weight_bytes:
        destination_path.unlink(missing_ok=True)
        raise PatchError("patched external sidecar changed size")
    return WeightPatchResult(
        base_weights_sha256=base_digest.hexdigest(),
        patched_weights_sha256=patched_digest.hexdigest(),
        weight_bytes=layout.weight_bytes,
        changed_spans=tuple(changed),
        outside_bytes_identical=True,
    )


# ---------------------------------------------------------------------------
# Model-directory orchestration and provenance.


def _relative_symlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(os.path.relpath(source, destination.parent), target_is_directory=source.is_dir())


def _copy_unaffected_stages(base_dir: Path, output_dir: Path) -> None:
    for child in base_dir.iterdir():
        if child.name in {MEANFLOW_STAGE, "manifest.json"}:
            continue
        destination = output_dir / child.name
        if child.is_dir():
            _relative_symlink(child, destination)
        elif child.is_file() or child.is_symlink():
            shutil.copy2(child, destination, follow_symlinks=False)


def _copy_meanflow_stage(base_dir: Path, output_dir: Path, *, weights_name: str) -> Path:
    source = base_dir / MEANFLOW_STAGE
    destination = output_dir / MEANFLOW_STAGE
    # The sidecar is streamed directly into the destination below. Avoid a
    # redundant full copy and its transient file-cache charge.
    shutil.copytree(source, destination, symlinks=True, ignore=shutil.ignore_patterns(weights_name))
    return destination


def _stage_checkpoint_sha(manifest: Mapping[str, Any]) -> str:
    checkpoint = manifest.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise PatchError("meanflow stage manifest has no checkpoint provenance")
    return _hex(checkpoint.get("sha256"), name="stage checkpoint sha256")


def _validate_base_stage(stage_dir: Path) -> tuple[dict[str, Any], str]:
    manifest_path = stage_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"meanflow stage manifest not found: {manifest_path}")
    manifest = _load_json(manifest_path)
    if manifest.get("stage") != MEANFLOW_STAGE or manifest.get("status") != "exported":
        raise PatchError("meanflow stage is not an exported stage")
    if manifest.get("adapter_patch") is not None or manifest.get("decoder_attention_patch") is not None:
        raise PatchError("meanflow stage already contains a decoder adapter patch")
    if manifest.get("decoder_attention_verification") is not None:
        raise PatchError("meanflow stage already contains decoder attention verification metadata")
    return manifest, _sha256_file(manifest_path)


def _make_patch_metadata(
    *,
    adapter: AttentionAdapter,
    layout: ExternalLayout,
    result: WeightPatchResult,
    strength: float,
    graph_sha256: str,
    external_report_sha256: str,
    base_external_report_sha256: str,
    base_stage_manifest_sha256: str,
    base_model_manifest_sha256: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "format": PATCH_FORMAT,
        "mode": "merged_lora",
        "adapter_path": str(adapter.path),
        "adapter_sha256": adapter.sha256,
        "inventory_sha256": adapter.inventory_sha256,
        "model_checkpoint_sha256": adapter.model_checkpoint_sha256,
        "initial_conditionals_sha256": adapter.initial_conditionals_sha256,
        "prepared_conditionals_sha256": adapter.prepared_conditionals_sha256,
        "rank": RANK,
        "alpha": ALPHA,
        "adapter_scale": ADAPTER_SCALE,
        "strength": strength,
        "target_prefix": TARGET_PREFIX,
        "target_count": len(adapter.factors),
        "target_parameter_count": sum(RANK * (item.shape[0] + item.shape[1]) for item in adapter.factors),
        "base_weights_sha256": result.base_weights_sha256,
        "patched_weights_sha256": result.patched_weights_sha256,
        "weights_file": layout.weights_path.name,
        "graph_sha256": graph_sha256,
        "external_report_file": layout.report_path.name,
        "external_report_sha256": external_report_sha256,
        "base_external_report_sha256": base_external_report_sha256,
        "base_stage_manifest_sha256": base_stage_manifest_sha256,
        "base_model_manifest_sha256": base_model_manifest_sha256,
        "changed_count": len(result.changed_spans),
        "changed_initializers": list(result.changed_spans),
        "outside_bytes_identical": result.outside_bytes_identical,
        "storage": "external_weights_stream_patch",
        "graph_unchanged": True,
        "verification_required": True,
        "verification_status": "pending",
        "source_patcher": str(SCRIPT_PATH),
        "source_patcher_sha256": _sha256_file(SCRIPT_PATH),
    }


def _updated_external_report(
    payload: Mapping[str, Any],
    *,
    graph_path: Path,
    result: WeightPatchResult,
    changed_spans: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    value = dict(payload)
    value["graph"] = graph_path.name
    external = dict(value.get("external") or {})
    external["weight_bytes"] = result.weight_bytes
    updates = {str(item["name"]): str(item["after_sha256"]) for item in changed_spans}
    initializers: list[dict[str, Any]] = []
    for raw in external.get("initializers", []):
        if not isinstance(raw, Mapping):
            raise PatchError("external report initializer is not an object during update")
        item = dict(raw)
        name = str(item.get("graph_name"))
        if name in updates:
            item["sha256"] = updates[name]
        initializers.append(item)
    external["initializers"] = initializers
    value["external"] = external
    return value


def _update_stage_manifest(
    base_manifest: Mapping[str, Any],
    *,
    patch: Mapping[str, Any],
    updated_external_report: Mapping[str, Any],
    graph_sha256: str,
    graph_bytes: int,
    patched_weights_sha256: str,
) -> dict[str, Any]:
    manifest = dict(base_manifest)
    old_verification = manifest.get("verification")
    manifest["base_verification"] = old_verification
    manifest["verification"] = {
        "status": "pending",
        "reason": "decoder attention external weights changed; run the adapted Torch versus ORT verifier",
        "base_status": old_verification.get("status") if isinstance(old_verification, Mapping) else None,
    }
    manifest["decoder_attention_verification"] = {
        "status": "pending",
        "promotion_status": "not_promoted",
        "graph_sha256": graph_sha256,
        "weights_sha256": patched_weights_sha256,
        "adapter_sha256": patch.get("adapter_sha256"),
        "strength": patch.get("strength"),
        "disclosure": "Static folding has not established numerical parity or speech quality. A separate verifier must compare adapted Torch and ORT outputs.",
    }
    graph = dict(manifest.get("graph") or {})
    graph["path"] = Path(str(graph.get("path") or "nano_meanflow_estimator.onnx")).name
    graph["bytes"] = int(graph_bytes)
    graph["sha256"] = graph_sha256
    graph["external_weights_file"] = patch.get("weights_file")
    graph["external_weights_sha256"] = patched_weights_sha256
    graph["base_external_weights_sha256"] = patch.get("base_weights_sha256")
    graph["validation"] = "pending decoder attention numerical verification"
    graph["external_parameters"] = dict(updated_external_report)
    manifest["graph"] = graph
    manifest["adapter_patch"] = dict(patch)
    return manifest


def _update_root_manifest(
    base_root: Mapping[str, Any] | None,
    *,
    output_dir: Path,
    base_dir: Path,
    stage_manifest: Mapping[str, Any],
    patch: Mapping[str, Any],
) -> dict[str, Any]:
    root: dict[str, Any] = dict(base_root or {})
    root["schema_version"] = int(root.get("schema_version", 1))
    root["model"] = root.get("model", "ResembleAI/chatterbox-nano")
    root["source"] = root.get("source", "local Nano checkpoints; no Turbo substitution")
    root["output_dir"] = str(output_dir)
    root["source_model_dir"] = str(base_dir)
    stages = dict(root.get("stages") or {})
    summary = dict(stages.get(MEANFLOW_STAGE) or {})
    summary.update(
        {
            "status": stage_manifest.get("status"),
            "graph": stage_manifest.get("graph"),
            "reason": stage_manifest.get("reason"),
            "verification": stage_manifest.get("verification"),
            "decoder_attention_verification": stage_manifest.get("decoder_attention_verification"),
            "adapter_patch": stage_manifest.get("adapter_patch"),
        }
    )
    stages[MEANFLOW_STAGE] = summary
    root["stages"] = stages
    root["decoder_attention_patch"] = dict(patch)
    root["decoder_attention_verification"] = stage_manifest.get("decoder_attention_verification")
    root["updated_unix"] = time.time()
    return root


def patch_model_dir(
    base_model_dir: Path | str,
    adapter_path: Path | str,
    output_model_dir: Path | str,
    *,
    strength: float = 1.0,
    inventory_path: Path | str = DEFAULT_INVENTORY,
) -> dict[str, Any]:
    """Create a new model root containing statically folded meanflow weights."""

    value = _validate_strength(strength)
    base_dir = Path(base_model_dir).expanduser().resolve()
    output_dir = Path(output_model_dir).expanduser().resolve()
    if not base_dir.is_dir():
        raise FileNotFoundError(f"base model directory not found: {base_dir}")
    if output_dir.exists():
        raise FileExistsError(f"output model directory already exists: {output_dir}")
    try:
        output_dir.relative_to(base_dir)
    except ValueError:
        pass
    else:
        raise PatchError("output model directory must be outside the baseline directory")
    base_stage = base_dir / MEANFLOW_STAGE
    base_stage_manifest, base_stage_manifest_sha = _validate_base_stage(base_stage)
    base_model_manifest_path = base_dir / "manifest.json"
    base_model_manifest = _load_json(base_model_manifest_path) if base_model_manifest_path.is_file() else None
    base_model_manifest_sha = _sha256_file(base_model_manifest_path) if base_model_manifest_path.is_file() else None
    adapter = _load_attention_adapter(adapter_path, inventory_path=inventory_path)
    if adapter.model_checkpoint_sha256 != _stage_checkpoint_sha(base_stage_manifest):
        raise PatchError("adapter model checkpoint SHA-256 does not match the meanflow stage checkpoint")
    layout = _load_external_layout(base_stage)
    if len(adapter.factors) != TARGET_COUNT:
        raise PatchError(f"decoder attention adapter must contain exactly {TARGET_COUNT} targets")
    factor_map = adapter.by_onnx_name()
    layout_map = layout.by_name()
    if set(factor_map) != set(item.graph_name for item in layout.initializers if item.graph_name in factor_map):
        missing = sorted(set(factor_map) - set(layout_map))[:4]
        raise PatchError(f"meanflow external graph is missing decoder attention targets: {missing}")
    for name, factor in factor_map.items():
        record = layout_map.get(name)
        if record is None:
            raise PatchError(f"meanflow external graph has no target {name!r}")
        if record.shape != factor.shape:
            raise PatchError(f"meanflow external shape differs from adapter for {name!r}")
        if record.sha256 != factor.base_sha256:
            raise PatchError(f"meanflow external baseline hash differs from adapter for {name!r}")
    graph_sha256 = _sha256_file(layout.graph_path)
    base_external_report_sha = _sha256_file(layout.report_path)
    temporary_dir = output_dir.with_name(f".{output_dir.name}.{os.getpid()}.tmp")
    if temporary_dir.exists():
        raise FileExistsError(f"temporary output already exists: {temporary_dir}")
    temporary_dir.mkdir(parents=True)
    try:
        _copy_unaffected_stages(base_dir, temporary_dir)
        output_stage = _copy_meanflow_stage(base_dir, temporary_dir, weights_name=layout.weights_path.name)
        if _sha256_file(output_stage / layout.graph_path.name) != graph_sha256:
            raise PatchError('graph bytes changed during stage copy')
        output_weights = output_stage / layout.weights_path.name
        output_weights.unlink(missing_ok=True)
        result = _patch_external_weights(layout.weights_path, output_weights, layout, factor_map, strength=value)
        output_report = output_stage / layout.report_path.name
        output_report.unlink(missing_ok=True)
        updated_report = _updated_external_report(
            layout.report,
            graph_path=output_stage / layout.graph_path.name,
            result=result,
            changed_spans=result.changed_spans,
        )
        _write_json(output_report, updated_report)
        external_report_sha = _sha256_file(output_report)
        patch = _make_patch_metadata(
            adapter=adapter,
            layout=layout,
            result=result,
            strength=value,
            graph_sha256=graph_sha256,
            external_report_sha256=external_report_sha,
            base_external_report_sha256=base_external_report_sha,
            base_stage_manifest_sha256=base_stage_manifest_sha,
            base_model_manifest_sha256=base_model_manifest_sha,
        )
        updated_stage_manifest = _update_stage_manifest(
            base_stage_manifest,
            patch=patch,
            updated_external_report=updated_report,
            graph_sha256=graph_sha256,
            graph_bytes=layout.graph_metadata.bytes,
            patched_weights_sha256=result.patched_weights_sha256,
        )
        output_manifest_path = output_stage / "manifest.json"
        output_manifest_path.unlink(missing_ok=True)
        _write_json(output_manifest_path, updated_stage_manifest)
        root = _update_root_manifest(
            base_model_manifest,
            output_dir=output_dir,
            base_dir=base_dir,
            stage_manifest=updated_stage_manifest,
            patch=patch,
        )
        _write_json(temporary_dir / "manifest.json", root)
        os.replace(temporary_dir, output_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    return {
        "output_model_dir": str(output_dir),
        "meanflow_stage_manifest": str(output_dir / MEANFLOW_STAGE / "manifest.json"),
        "graph": str(output_dir / MEANFLOW_STAGE / layout.graph_path.name),
        "weights": str(output_dir / MEANFLOW_STAGE / layout.weights_path.name),
        "graph_sha256": graph_sha256,
        "base_weights_sha256": result.base_weights_sha256,
        "patched_weights_sha256": result.patched_weights_sha256,
        "adapter_sha256": adapter.sha256,
        "inventory_sha256": adapter.inventory_sha256,
        "strength": value,
        "changed_count": len(result.changed_spans),
        "verification": "pending",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--output-model-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    report = patch_model_dir(
        args.base_model_dir,
        args.adapter,
        args.output_model_dir,
        strength=args.strength,
        inventory_path=args.inventory,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


__all__ = [
    "AttentionAdapter",
    "ExternalInitializer",
    "ExternalLayout",
    "FactorSpec",
    "GraphMetadata",
    "GraphTensor",
    "PatchError",
    "WeightPatchResult",
    "_load_external_layout",
    "_load_attention_adapter",
    "_merge_weight",
    "_patch_external_weights",
    "patch_model_dir",
    "read_graph_metadata",
]


if __name__ == "__main__":
    raise SystemExit(main())
