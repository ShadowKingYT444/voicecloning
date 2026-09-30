"""Stream a Nano T3 LoRA adapter into an existing ONNX graph.

The staged Nano T3 graph stores its GPT-2 ``Conv1D`` weights as inline
``TensorProto.raw_data`` fields.  This module merges a small LoRA checkpoint
into those fields without loading the graph protobuf into memory.  Every byte
outside the selected raw tensor spans is copied unchanged.

The patch is static.  Runtime processes load the same ONNX graph shape and do
not load PyTorch or LoRA factors.  The command therefore keeps the resident
set of the inference process unchanged.  It accepts the checkpoint format
written by :mod:`adaptation` and rejects unknown modules, external target
weights, dtype changes, and shape mismatches.

The real model/export graph is intentionally not loaded at module import.  A
PyTorch import occurs only in ``load_adapter_checkpoint`` when the CLI reads
the small adapter checkpoint on CPU.  The protobuf reader and merge helpers
are pure Python/NumPy and are used by the focused synthetic tests.

Example::

    .venv-nano-cpu/bin/python scripts/nano_lab/patch_onnx_adapter.py \
      --base-model-dir artifacts/nano_lab/onnx_staged \
      --adapter artifacts/nano_lab/adapter_aligned_all_attn.pt \
      --scale 1 \
      --output-model-dir artifacts/nano_lab/onnx_staged_adapter_aligned_all_attn

The output is a new model directory.  The baseline directory is never edited.
The output T3 manifest has ``adapter_patch`` provenance and leaves verification
pending until a separate ORT comparison is run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Mapping, Sequence

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
ONNX_FLOAT = 1
ONNX_EXTERNAL = 1
MAX_METADATA_FIELD_BYTES = 1 << 20
COPY_CHUNK_BYTES = 4 << 20
MODULE_RE = re.compile(r"^h\.(?P<layer>[0-9]+)\.attn\.(?P<projection>c_attn|c_proj)$")


class PatchError(RuntimeError):
    """Raised when a graph or adapter does not satisfy the patch contract."""


@dataclass(frozen=True)
class TensorSpan:
    """Metadata for one ONNX initializer, without loading its raw bytes."""

    name: str
    dims: tuple[int, ...]
    data_type: int
    raw_offset: int | None
    raw_length: int | None
    data_location: int | None
    has_external_data: bool

    @property
    def is_external(self) -> bool:
        return self.has_external_data or self.data_location == ONNX_EXTERNAL


@dataclass(frozen=True)
class GraphMetadata:
    """Streaming metadata for an ONNX graph."""

    path: Path
    bytes: int
    initializers: tuple[TensorSpan, ...]

    def by_name(self) -> dict[str, TensorSpan]:
        return {item.name: item for item in self.initializers}


@dataclass(frozen=True)
class AdapterSpec:
    """CPU NumPy representation of one validated adaptation checkpoint."""

    path: Path
    format: str
    rank: int
    alpha: float
    dropout: float
    modules: tuple[str, ...]
    factors: Mapping[str, tuple[np.ndarray, np.ndarray]]
    module_shapes: Mapping[str, tuple[int, int]]


@dataclass(frozen=True)
class PatchResult:
    """Patch report used to build the output manifest."""

    graph_path: Path
    graph_bytes: int
    graph_sha256: str
    base_graph_sha256: str
    changed_spans: tuple[Mapping[str, Any], ...]
    outside_bytes_identical: bool


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
        raise PatchError("protobuf field read crossed message boundary")
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
        raise PatchError(f"protobuf length {length} crosses message boundary")
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
        raise PatchError(f"metadata field is too large: {length} bytes")
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
                raise PatchError("truncated packed int64 field")
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


def _parse_tensor(handle: BinaryIO, end: int) -> TensorSpan:
    dims: list[int] = []
    data_type: int | None = None
    name: str | None = None
    raw_offset: int | None = None
    raw_length: int | None = None
    data_location: int | None = None
    has_external_data = False

    while handle.tell() < end:
        number, wire_type = _read_key(handle, end)
        if number == 1:
            if wire_type == 0:
                dims.append(_decode_signed_int64(_read_varint(handle)))
            elif wire_type == 2:
                payload = _read_small_bytes(handle, end)
                dims.extend(_decode_packed_int64(payload))
            else:
                _skip_field(handle, wire_type, end)
        elif number == 2 and wire_type == 0:
            data_type = int(_read_varint(handle))
        elif number == 8 and wire_type == 2:
            try:
                name = _read_small_bytes(handle, end).decode("utf-8")
            except UnicodeDecodeError as exc:
                raise PatchError("initializer name is not UTF-8") from exc
        elif number == 9 and wire_type == 2:
            length, start, stop = _read_length(handle, end)
            if raw_offset is not None:
                raise PatchError(f"initializer {name!r} has duplicate raw_data fields")
            raw_offset, raw_length = start, length
            handle.seek(stop)
        elif number == 13 and wire_type == 2:
            has_external_data = True
            _skip_field(handle, wire_type, end)
        elif number == 14 and wire_type == 0:
            data_location = int(_read_varint(handle))
        else:
            _skip_field(handle, wire_type, end)

    if handle.tell() != end:
        raise PatchError("tensor parser did not consume its protobuf message")
    if name is None:
        raise PatchError("initializer is missing a name")
    if data_type is None:
        raise PatchError(f"initializer {name!r} is missing data_type")
    return TensorSpan(
        name=name,
        dims=tuple(dims),
        data_type=data_type,
        raw_offset=raw_offset,
        raw_length=raw_length,
        data_location=data_location,
        has_external_data=has_external_data,
    )


def read_graph_metadata(path: Path | str) -> GraphMetadata:
    """Read initializer names, shapes, and byte spans without loading weights.

    ``load_external_data=False`` is not sufficient for a low-RSS patcher when
    the graph stores inline raw bytes.  This reader seeks over every raw field
    and retains only small protobuf metadata.
    """

    graph_path = Path(path).expanduser().resolve()
    if not graph_path.is_file():
        raise FileNotFoundError(graph_path)
    file_size = graph_path.stat().st_size
    initializers: list[TensorSpan] = []
    graph_found = False
    with graph_path.open("rb") as handle:
        while handle.tell() < file_size:
            number, wire_type = _read_key(handle, file_size)
            if number == 7 and wire_type == 2:
                if graph_found:
                    raise PatchError("ONNX ModelProto contains multiple graph fields")
                graph_found = True
                _, graph_start, graph_end = _read_length(handle, file_size)
                while handle.tell() < graph_end:
                    field_number, field_wire = _read_key(handle, graph_end)
                    if field_number == 5 and field_wire == 2:
                        _, tensor_start, tensor_end = _read_length(handle, graph_end)
                        handle.seek(tensor_start)
                        tensor = _parse_tensor(handle, tensor_end)
                        if any(item.name == tensor.name for item in initializers):
                            raise PatchError(f"duplicate initializer name {tensor.name!r}")
                        initializers.append(tensor)
                    else:
                        _skip_field(handle, field_wire, graph_end)
                if handle.tell() != graph_end:
                    raise PatchError("graph parser did not consume its protobuf message")
            else:
                _skip_field(handle, wire_type, file_size)
        if handle.tell() != file_size:
            raise PatchError("ModelProto parser did not consume the file")
    if not graph_found:
        raise PatchError(f"ONNX ModelProto has no graph: {graph_path}")
    return GraphMetadata(path=graph_path, bytes=file_size, initializers=tuple(initializers))


def _sha256_file(path: Path, *, chunk_bytes: int = COPY_CHUNK_BYTES) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(value: bytes | bytearray | memoryview) -> str:
    return hashlib.sha256(value).hexdigest()


def _expected_graph_name(module: str) -> str:
    return f"t3.tfmr.{module}.weight"


def _validate_module_name(module: str) -> tuple[int, str]:
    match = MODULE_RE.fullmatch(module)
    if match is None:
        raise PatchError(
            f"unsupported adapter module {module!r}; only h.<layer>.attn.c_attn/c_proj is supported"
        )
    return int(match.group("layer")), match.group("projection")


def _as_factor_array(value: Any, *, name: str) -> np.ndarray:
    if not hasattr(value, "detach") or not callable(value.detach):
        raise PatchError(f"adapter_state[{name!r}] is not a tensor")
    try:
        array = np.asarray(value.detach().cpu().numpy())
    except Exception as exc:
        raise PatchError(f"could not copy adapter factor {name!r} to CPU NumPy") from exc
    if array.ndim != 2 or not np.issubdtype(array.dtype, np.floating):
        raise PatchError(f"adapter factor {name!r} must be a 2-D floating tensor; got {array.shape} {array.dtype}")
    result = np.ascontiguousarray(array, dtype=np.float32)
    if not np.isfinite(result).all():
        raise PatchError(f"adapter factor {name!r} contains NaN or infinity")
    return result


def load_adapter_checkpoint(path: Path | str) -> AdapterSpec:
    """Load and validate the small adaptation checkpoint on CPU.

    This is the only function in this module that imports PyTorch.  It is not
    called by the pure graph metadata reader or by the ONNX runtime.
    """

    adapter_path = Path(path).expanduser().resolve()
    if not adapter_path.is_file():
        raise FileNotFoundError(adapter_path)
    try:
        import torch  # type: ignore
    except Exception as exc:
        raise PatchError("PyTorch is required only to read the adapter checkpoint") from exc
    try:
        payload = torch.load(adapter_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise PatchError(
            "adapter checkpoint could not be read with weights_only=True; refusing unsafe pickle fallback"
        ) from exc
    if not isinstance(payload, Mapping):
        raise PatchError(f"adapter checkpoint must be a mapping, got {type(payload).__name__}")
    if payload.get("format") != "nano_t3_lora_checkpoint_v1":
        raise PatchError(f"unsupported adapter checkpoint format: {payload.get('format')!r}")
    config = payload.get("config")
    state = payload.get("adapter_state")
    if not isinstance(config, Mapping) or not isinstance(state, Mapping):
        raise PatchError("adapter checkpoint must contain mapping fields config and adapter_state")
    try:
        rank = int(config["rank"])
        alpha = float(config["alpha"])
        dropout = float(config.get("dropout", 0.0))
        modules_value = config["modules"]
    except (KeyError, TypeError, ValueError) as exc:
        raise PatchError("adapter config is missing rank, alpha, or modules") from exc
    if rank < 1 or not math.isfinite(alpha) or alpha <= 0 or not math.isfinite(dropout) or not 0 <= dropout < 1:
        raise PatchError(f"invalid adapter config rank={rank}, alpha={alpha}, dropout={dropout}")
    if not isinstance(modules_value, (list, tuple)) or not modules_value:
        raise PatchError("adapter config modules must be a non-empty list")
    modules = tuple(str(item) for item in modules_value)
    if len(set(modules)) != len(modules):
        raise PatchError("adapter config modules contain duplicates")
    for module in modules:
        _validate_module_name(module)
    state_names = {str(name) for name in state}
    if state_names != set(modules):
        missing = sorted(set(modules) - state_names)
        extra = sorted(state_names - set(modules))
        raise PatchError(f"adapter state/module mismatch; missing={missing}, extra={extra}")
    shape_config = config.get("module_shapes")
    if not isinstance(shape_config, Mapping):
        raise PatchError("adapter config is missing module_shapes")

    factors: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    module_shapes: dict[str, tuple[int, int]] = {}
    for module in modules:
        values = state[module]
        if not isinstance(values, Mapping) or set(values) != {"lora_A", "lora_B"}:
            raise PatchError(f"adapter state for {module!r} must contain only lora_A and lora_B")
        factor_a = _as_factor_array(values["lora_A"], name=f"{module}.lora_A")
        factor_b = _as_factor_array(values["lora_B"], name=f"{module}.lora_B")
        configured = shape_config.get(module)
        if not isinstance(configured, Mapping):
            raise PatchError(f"adapter config has no module_shapes entry for {module!r}")
        try:
            in_features = int(configured["in_features"])
            out_features = int(configured["out_features"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PatchError(f"invalid module_shapes entry for {module!r}") from exc
        if factor_a.shape != (rank, in_features) or factor_b.shape != (out_features, rank):
            raise PatchError(
                f"adapter factor shapes for {module!r} are {factor_a.shape}/{factor_b.shape}; "
                f"expected {(rank, in_features)}/{(out_features, rank)}"
            )
        factors[module] = (factor_a, factor_b)
        module_shapes[module] = (in_features, out_features)
    del payload
    return AdapterSpec(
        path=adapter_path,
        format="nano_t3_lora_checkpoint_v1",
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        modules=modules,
        factors=factors,
        module_shapes=module_shapes,
    )


def merge_factors(
    base_raw: bytes | bytearray | memoryview,
    factor_a: np.ndarray,
    factor_b: np.ndarray,
    *,
    effective_scale: float,
    shape: tuple[int, int],
) -> bytes:
    """Merge one HF Conv1D weight and return same-length little-endian bytes."""

    in_features, out_features = map(int, shape)
    expected_bytes = in_features * out_features * 4
    if len(base_raw) != expected_bytes:
        raise PatchError(f"base raw tensor has {len(base_raw)} bytes; expected {expected_bytes}")
    a = np.asarray(factor_a, dtype=np.float32)
    b = np.asarray(factor_b, dtype=np.float32)
    if (
        a.ndim != 2
        or b.ndim != 2
        or a.shape[1:] != (in_features,)
        or b.shape[0:1] != (out_features,)
        or a.shape[0] != b.shape[1]
    ):
        raise PatchError(f"factor shapes {a.shape}/{b.shape} do not match Conv1D shape {shape}")
    if not math.isfinite(float(effective_scale)):
        raise PatchError(f"non-finite adapter scale: {effective_scale}")
    base = np.frombuffer(base_raw, dtype="<f4").copy().reshape(shape)
    delta = np.matmul(a.T, b.T, dtype=np.float32)
    merged = base + np.float32(effective_scale) * delta
    if not np.isfinite(merged).all():
        raise PatchError("merged ONNX weight contains NaN or infinity")
    return np.ascontiguousarray(merged, dtype="<f4").tobytes(order="C")


def _copy_range(source: BinaryIO, destination: BinaryIO, start: int, stop: int) -> None:
    if stop < start:
        raise PatchError(f"invalid copy range {start}:{stop}")
    source.seek(start)
    remaining = stop - start
    while remaining:
        block = source.read(min(COPY_CHUNK_BYTES, remaining))
        if not block:
            raise PatchError("source graph ended during streaming copy")
        destination.write(block)
        remaining -= len(block)


def _outside_spans_identical(left: Path, right: Path, spans: Sequence[TensorSpan]) -> bool:
    """Compare two equal-size files while skipping changed raw payloads."""

    left_size = left.stat().st_size
    right_size = right.stat().st_size
    if left_size != right_size:
        return False
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for span in sorted(spans, key=lambda item: int(item.raw_offset) if item.raw_offset is not None else -1):
        if span.raw_offset is None or span.raw_length is None:
            return False
        if span.raw_offset < cursor or span.raw_offset + span.raw_length > left_size:
            return False
        ranges.append((cursor, span.raw_offset))
        cursor = span.raw_offset + span.raw_length
    ranges.append((cursor, left_size))
    with left.open("rb") as first, right.open("rb") as second:
        for start, stop in ranges:
            first.seek(start)
            second.seek(start)
            remaining = stop - start
            while remaining:
                size = min(COPY_CHUNK_BYTES, remaining)
                if first.read(size) != second.read(size):
                    return False
                remaining -= size
    return True


def patch_graph(
    base_graph: Path | str,
    output_graph: Path | str,
    adapter: AdapterSpec,
    *,
    scale: float = 1.0,
) -> PatchResult:
    """Stream-merge adapter weights into a copy of one ONNX graph."""

    source_path = Path(base_graph).expanduser().resolve()
    destination_path = Path(output_graph).expanduser().resolve()
    if source_path == destination_path:
        raise PatchError("refusing to patch the baseline graph in place")
    if not math.isfinite(float(scale)):
        raise PatchError(f"adapter scale must be finite, got {scale}")
    metadata = read_graph_metadata(source_path)
    by_name = metadata.by_name()
    selected: list[tuple[str, TensorSpan, np.ndarray, np.ndarray, tuple[int, int]]] = []
    for module in adapter.modules:
        _validate_module_name(module)
        graph_name = _expected_graph_name(module)
        span = by_name.get(graph_name)
        if span is None:
            raise PatchError(f"baseline graph has no expected initializer {graph_name!r}")
        if span.is_external or span.raw_offset is None or span.raw_length is None:
            raise PatchError(f"target initializer {graph_name!r} is external or has no inline raw_data")
        shape = adapter.module_shapes[module]
        if span.data_type != ONNX_FLOAT:
            raise PatchError(f"target initializer {graph_name!r} has ONNX data_type={span.data_type}, expected FLOAT=1")
        if tuple(span.dims) != tuple(shape):
            raise PatchError(f"target initializer {graph_name!r} dims {span.dims} do not match adapter shape {shape}")
        expected_length = int(np.prod(shape, dtype=np.int64)) * 4
        if span.raw_length != expected_length:
            raise PatchError(f"target initializer {graph_name!r} raw length {span.raw_length} != {expected_length}")
        factor_a, factor_b = adapter.factors[module]
        selected.append((module, span, factor_a, factor_b, shape))
    selected.sort(key=lambda item: int(item[1].raw_offset) if item[1].raw_offset is not None else -1)
    spans = [item[1] for item in selected]
    if len({span.raw_offset for span in spans}) != len(spans):
        raise PatchError("adapter modules map to duplicate raw_data spans")

    effective_scale = float(adapter.alpha) / float(adapter.rank) * float(scale)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination_path.with_name(f".{destination_path.name}.{os.getpid()}.tmp")
    if temporary_path.exists() or destination_path.exists():
        raise FileExistsError(destination_path)
    changed: list[Mapping[str, Any]] = []
    try:
        with source_path.open("rb") as source, temporary_path.open("wb") as destination:
            cursor = 0
            for module, span, factor_a, factor_b, shape in selected:
                assert span.raw_offset is not None and span.raw_length is not None
                _copy_range(source, destination, cursor, span.raw_offset)
                source.seek(span.raw_offset)
                original = source.read(span.raw_length)
                if len(original) != span.raw_length:
                    raise PatchError(f"source graph ended inside {span.name!r} raw_data")
                merged = merge_factors(
                    original,
                    factor_a,
                    factor_b,
                    effective_scale=effective_scale,
                    shape=shape,
                )
                if len(merged) != len(original):
                    raise PatchError(f"merge changed raw byte length for {span.name!r}")
                destination.write(merged)
                changed.append(
                    {
                        "module": module,
                        "name": span.name,
                        "shape": list(shape),
                        "dtype": "float32",
                        "raw_offset": span.raw_offset,
                        "raw_length": span.raw_length,
                        "before_sha256": _sha256_bytes(original),
                        "after_sha256": _sha256_bytes(merged),
                    }
                )
                cursor = span.raw_offset + span.raw_length
            _copy_range(source, destination, cursor, metadata.bytes)
        os.replace(temporary_path, destination_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    output_metadata = read_graph_metadata(destination_path)
    output_by_name = output_metadata.by_name()
    for item in changed:
        output_span = output_by_name.get(str(item["name"]))
        if output_span is None or output_span.raw_offset != item["raw_offset"] or output_span.raw_length != item["raw_length"]:
            raise PatchError(f"patched graph metadata changed span for {item['name']!r}")
    outside_identical = _outside_spans_identical(source_path, destination_path, spans)
    if not outside_identical:
        destination_path.unlink(missing_ok=True)
        raise PatchError("patched graph differs outside declared adapter raw_data spans")
    return PatchResult(
        graph_path=destination_path,
        graph_bytes=destination_path.stat().st_size,
        graph_sha256=_sha256_file(destination_path),
        base_graph_sha256=_sha256_file(source_path),
        changed_spans=tuple(changed),
        outside_bytes_identical=outside_identical,
    )


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _relative_symlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(os.path.relpath(source, destination.parent), target_is_directory=source.is_dir())


def _copy_stage_links(base_dir: Path, output_dir: Path) -> None:
    """Reuse non-T3 stage directories through read-only relative symlinks."""

    for name in ("flow_encoder", "meanflow_estimator", "vocoder"):
        source = base_dir / name
        if source.exists():
            _relative_symlink(source, output_dir / name)


def _copy_t3_support_links(base_stage: Path, output_stage: Path) -> None:
    output_stage.mkdir(parents=True, exist_ok=True)
    for source in base_stage.iterdir():
        if source.name in {"nano_t3_kv.onnx", "manifest.json"}:
            continue
        destination = output_stage / source.name
        if source.is_file() or source.is_symlink():
            _relative_symlink(source, destination)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise PatchError(f"JSON manifest must be an object: {path}")
    return value


def _source_patcher_hash() -> str:
    return _sha256_file(SCRIPT_PATH)


def patch_model_dir(
    base_model_dir: Path | str,
    adapter_path: Path | str,
    output_model_dir: Path | str,
    *,
    scale: float = 1.0,
) -> dict[str, Any]:
    """Create a patched model directory while preserving the baseline."""

    base_dir = Path(base_model_dir).expanduser().resolve()
    adapter_file = Path(adapter_path).expanduser().resolve()
    output_dir = Path(output_model_dir).expanduser().resolve()
    if base_dir == output_dir:
        raise PatchError("output model directory must differ from the baseline")
    try:
        output_dir.relative_to(base_dir)
    except ValueError:
        pass
    else:
        raise PatchError("output model directory must be outside the baseline directory")
    if output_dir.exists():
        raise FileExistsError(f"output model directory already exists: {output_dir}")
    base_stage = base_dir / "t3"
    base_graph = base_stage / "nano_t3_kv.onnx"
    base_manifest_path = base_stage / "manifest.json"
    root_manifest_path = base_dir / "manifest.json"
    if not base_graph.is_file() or not base_manifest_path.is_file():
        raise FileNotFoundError(f"baseline staged T3 graph/manifest missing under {base_stage}")
    if root_manifest_path.exists() and not root_manifest_path.is_file():
        raise PatchError(f"baseline root manifest is not a regular file: {root_manifest_path}")
    base_stage_manifest = _load_json(base_manifest_path)
    if base_stage_manifest.get("adapter_patch") is not None:
        raise PatchError("baseline T3 manifest already contains adapter_patch; patch only an unmodified base graph")
    base_manifest_sha = _sha256_file(base_manifest_path)
    root_manifest = _load_json(root_manifest_path) if root_manifest_path.exists() else None
    adapter = load_adapter_checkpoint(adapter_file)

    temporary_dir = output_dir.with_name(f".{output_dir.name}.{os.getpid()}.tmp")
    if temporary_dir.exists():
        raise FileExistsError(temporary_dir)
    temporary_dir.mkdir(parents=True)
    try:
        _copy_stage_links(base_dir, temporary_dir)
        temporary_t3 = temporary_dir / "t3"
        base_t3 = base_dir / "t3"
        _copy_t3_support_links(base_t3, temporary_t3)
        patched_graph = temporary_t3 / "nano_t3_kv.onnx"
        result = patch_graph(base_graph, patched_graph, adapter, scale=scale)

        manifest = dict(base_stage_manifest)
        manifest["status"] = "exported"
        manifest["verification"] = {
            "status": "not_run",
            "command": "run fresh ORT verification against a Torch LoRA reference before use",
        }
        graph = dict(manifest.get("graph") or {})
        graph.update(
            {
                "path": "nano_t3_kv.onnx",
                "bytes": result.graph_bytes,
                "sha256": result.graph_sha256,
                "base_sha256": result.base_graph_sha256,
                "validation": "pending separate ORT verification of merged adapter",
            }
        )
        manifest["graph"] = graph
        manifest["base_graph_sha256"] = result.base_graph_sha256
        manifest["adapter_patch"] = {
            "schema_version": 1,
            "format": "nano_t3_lora_merged_v1",
            "mode": "merged_lora",
            "path": str(adapter.path),
            "adapter_path": str(adapter.path),
            "sha256": _sha256_file(adapter.path),
            "adapter_sha256": _sha256_file(adapter.path),
            "base_graph_sha256": result.base_graph_sha256,
            "patched_graph_sha256": result.graph_sha256,
            "base_model_manifest_sha256": base_manifest_sha,
            "rank": adapter.rank,
            "alpha": adapter.alpha,
            "checkpoint_scale": float(scale),
            "scale": float(scale),
            "effective_scale": float(adapter.alpha) / float(adapter.rank) * float(scale),
            "dropout": adapter.dropout,
            "modules": list(adapter.modules),
            "changed_count": len(result.changed_spans),
            "changed_initializers": list(result.changed_spans),
            "changed_spans": list(result.changed_spans),
            "outside_bytes_identical": result.outside_bytes_identical,
            "storage": "inline_raw_data_stream_patch",
            "graph_size_bytes": result.graph_bytes,
            "source_patcher": str(SCRIPT_PATH),
            "source_patcher_sha256": _source_patcher_hash(),
            "verification_required": True,
        }
        _write_json(temporary_t3 / "manifest.json", manifest)

        if root_manifest is not None:
            root = dict(root_manifest)
            root["output_dir"] = str(output_dir)
            stages = dict(root.get("stages") or {})
            stages["t3"] = manifest
            root["stages"] = stages
            root["adapter_patch"] = manifest["adapter_patch"]
            root["source_model_dir"] = str(base_dir)
            _write_json(temporary_dir / "manifest.json", root)
        os.replace(temporary_dir, output_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    return {
        "output_model_dir": str(output_dir),
        "t3_manifest": str(output_dir / "t3" / "manifest.json"),
        "graph": str(output_dir / "t3" / "nano_t3_kv.onnx"),
        "graph_sha256": result.graph_sha256,
        "base_graph_sha256": result.base_graph_sha256,
        "adapter_sha256": _sha256_file(adapter.path),
        "changed_count": len(result.changed_spans),
        "changed_spans": list(result.changed_spans),
        "verification": "pending",
    }


def _inspect_graph(path: Path) -> dict[str, Any]:
    metadata = read_graph_metadata(path)
    return {
        "path": str(metadata.path),
        "bytes": metadata.bytes,
        "initializer_count": len(metadata.initializers),
        "inline_count": sum(item.raw_offset is not None and not item.is_external for item in metadata.initializers),
        "external_count": sum(item.is_external for item in metadata.initializers),
        "initializers": [
            {
                "name": item.name,
                "dims": list(item.dims),
                "data_type": item.data_type,
                "raw_offset": item.raw_offset,
                "raw_length": item.raw_length,
                "external": item.is_external,
            }
            for item in metadata.initializers
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model-dir", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, help="nano_t3_lora_checkpoint_v1 .pt file")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--output-model-dir", type=Path)
    parser.add_argument("--inspect", action="store_true", help="print graph initializer metadata and exit")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    base_dir = args.base_model_dir.expanduser().resolve()
    if args.inspect:
        print(json.dumps(_inspect_graph(base_dir / "t3" / "nano_t3_kv.onnx"), indent=2, sort_keys=True))
        return 0
    if args.adapter is None or args.output_model_dir is None:
        raise SystemExit("--adapter and --output-model-dir are required unless --inspect is used")
    report = patch_model_dir(base_dir, args.adapter, args.output_model_dir, scale=args.scale)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
