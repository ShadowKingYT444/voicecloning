#!/usr/bin/env python3
"""Build an offline, unpromoted Q4 Gather candidate for Nano token embeddings.

Run this tool through scripts/nano_lab/bounded_job.py. The tool accepts only
the pinned public conversion files after their SHA256SUMS entries match.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import random
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Iterable


PINNED_REVISION = "4a66d7dab72a9e98f24b515d49a1d7a81632df2e"
SOURCE_REPO_ID = "owensong/chatterbox-nano-ONNX"
SHA256SUMS_SHA256 = "ecd0b0ee1fde0f9342564fda72fe1ebb4907baa6c9d013c09052bcad5220dcf4"
SOURCE_FILES = {
    "onnx/embed_tokens_fp16.onnx": "019d257243774091d78c2ad91c2c0f61e4e442740cb7b3b00b5a89109417b18d",
    "onnx/embed_tokens_fp16.onnx_data": "bcd7b35ae4f206932e2491cb60b42ebb80f6d8facfdb53ba7d7449ad00a3237b",
}
MANIFEST_SCHEMA = "browser_tts.embedding-q4-candidate/v1"

GRAPH_RELATIVE_PATH = Path("onnx/embed_tokens_fp16.onnx")
EXTERNAL_DATA_RELATIVE_PATH = Path("onnx/embed_tokens_fp16.onnx_data")
TARGET_GRAPH_NAME = "embed_tokens_gather_q4.onnx"
TARGET_EXTERNAL_DATA_NAME = TARGET_GRAPH_NAME + ".data"
TARGET_MANIFEST_NAME = "manifest.json"

BLOCK_SIZE = 128
QUANTIZE_AXIS = 1
GATHER_AXIS = 0
EMBEDDING_DIM = 768
TEXT_VOCAB_SIZE = 50276
SPEECH_VOCAB_SIZE = 6563
SPEECH_SENTINEL_ID = 50256
START_SPEECH_ID = 6561
PROBE_SEED = 20261001
STREAM_BATCH_SIZE = 32
HASH_CHUNK_BYTES = 1024 * 1024


class CandidateError(RuntimeError):
    """Raised when a source or candidate does not match the frozen contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, relative_path: str) -> dict[str, Any]:
    return {
        "path": relative_path,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _parse_checksums(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        match = re.fullmatch(r"([0-9a-fA-F]{64})\s+\*?(.+?)\s*", line)
        if match is None:
            raise CandidateError(f"Invalid SHA256SUMS row {line_number}: {line!r}")
        digest, filename = match.groups()
        if filename in rows:
            raise CandidateError(f"Duplicate SHA256SUMS entry for {filename!r}")
        rows[filename] = digest.lower()
    return rows


def _external_data_metadata(tensor: Any) -> dict[str, str]:
    return {entry.key: entry.value for entry in tensor.external_data}


def _node_attributes(onnx: Any, node: Any) -> dict[str, Any]:
    return {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}


def verify_source(source_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify file hashes and inspect only the small ONNX graph header."""
    try:
        import onnx
    except ImportError as exc:
        raise CandidateError("The selected Python environment needs the 'onnx' package.") from exc

    source_dir = source_dir.expanduser().resolve()
    sums_path = source_dir / "SHA256SUMS"
    if not sums_path.is_file():
        raise CandidateError(f"Missing pinned source checksum file: {sums_path}")
    sums_digest = sha256_file(sums_path)
    if sums_digest != SHA256SUMS_SHA256:
        raise CandidateError(
            "SHA256SUMS does not match the pinned conversion revision. "
            f"Expected {SHA256SUMS_SHA256}; found {sums_digest}."
        )

    checksums = _parse_checksums(sums_path)
    source_file_records: dict[str, dict[str, Any]] = {}
    for relative_name, expected_hash in SOURCE_FILES.items():
        if checksums.get(relative_name) != expected_hash:
            raise CandidateError(
                f"Pinned SHA256SUMS entry for {relative_name} is missing or changed. "
                f"Expected {expected_hash}; found {checksums.get(relative_name)!r}."
            )
        source_path = source_dir / relative_name
        if not source_path.is_file():
            raise CandidateError(f"Missing pinned source file: {source_path}")
        record = file_record(source_path, relative_name)
        if record["sha256"] != expected_hash:
            raise CandidateError(
                f"Hash mismatch for {relative_name}. Expected {expected_hash}; found {record['sha256']}."
            )
        source_file_records[relative_name] = record

    graph_path = source_dir / GRAPH_RELATIVE_PATH
    # The source model graph is 1.5 KB. Do not load its external weight file
    # during this header inspection.
    model = onnx.load_model(str(graph_path), load_external_data=False)
    graph = model.graph
    initializers = {tensor.name: tensor for tensor in graph.initializer}
    gather_nodes = [node for node in graph.node if node.op_type == "Gather"]
    expected_table_names = {"text_emb.weight", "speech_emb.weight"}
    if len(gather_nodes) != 2:
        raise CandidateError(f"Expected two embedding Gather nodes; found {len(gather_nodes)}")

    gather_tables: list[dict[str, Any]] = []
    for node in gather_nodes:
        if not node.input or node.input[0] not in initializers:
            raise CandidateError(f"Gather node {node.name!r} does not use a constant initializer")
        initializer = initializers[node.input[0]]
        if initializer.name not in expected_table_names:
            raise CandidateError(f"Unexpected embedding initializer {initializer.name!r}")
        if initializer.data_type != onnx.TensorProto.FLOAT16:
            dtype = onnx.TensorProto.DataType.Name(initializer.data_type)
            raise CandidateError(
                f"Expected FP16 embedding data at {initializer.name}; found {dtype}."
            )
        if len(initializer.dims) != 2 or initializer.dims[1] != EMBEDDING_DIM:
            raise CandidateError(
                f"Unexpected shape for {initializer.name}: {list(initializer.dims)}"
            )
        attributes = _node_attributes(onnx, node)
        gather_axis = int(attributes.get("axis", 0))
        if gather_axis != GATHER_AXIS:
            raise CandidateError(
                f"Expected source Gather axis {GATHER_AXIS} for {initializer.name}; got {gather_axis}."
            )
        external = _external_data_metadata(initializer)
        if external.get("location") != EXTERNAL_DATA_RELATIVE_PATH.name:
            raise CandidateError(
                f"Unexpected external-data location for {initializer.name}: {external.get('location')!r}"
            )
        expected_bytes = int(initializer.dims[0]) * int(initializer.dims[1]) * 2
        if int(external.get("length", "-1")) != expected_bytes:
            raise CandidateError(
                f"Unexpected external-data length for {initializer.name}: "
                f"expected {expected_bytes}, found {external.get('length')!r}."
            )
        offset = int(external.get("offset", "-1"))
        if offset < 0:
            raise CandidateError(f"Missing external-data offset for {initializer.name}")
        gather_tables.append(
            {
                "initializer": initializer.name,
                "shape": [int(value) for value in initializer.dims],
                "dtype": "float16",
                "gather_axis": gather_axis,
                "external_offset": offset,
                "external_length": expected_bytes,
                "node_name": node.name,
                "output_name": node.output[0],
            }
        )

    gather_tables.sort(key=lambda item: item["initializer"])
    if {item["initializer"] for item in gather_tables} != expected_table_names:
        raise CandidateError("The pinned graph does not contain the expected text and speech tables")
    table_sizes = {item["initializer"]: item["shape"][0] for item in gather_tables}
    if table_sizes != {"text_emb.weight": TEXT_VOCAB_SIZE, "speech_emb.weight": SPEECH_VOCAB_SIZE}:
        raise CandidateError(f"Unexpected embedding vocabulary sizes: {table_sizes}")

    node_by_name = {node.name: node for node in graph.node}
    text_slice = node_by_name.get("node_Slice_25")
    speech_slice = node_by_name.get("node_Slice_47")
    equal_node = node_by_name.get("node_Equal_49")
    where_node = node_by_name.get("node_Where_54")
    expand_node = node_by_name.get("node_Expand_53")
    if not all((text_slice, speech_slice, equal_node, where_node, expand_node)):
        raise CandidateError("The pinned graph is missing its expected hybrid text/speech tail nodes")
    if text_slice.input[2] not in initializers or speech_slice.input[1] not in initializers:
        raise CandidateError("Could not inspect the pinned graph's hybrid tail slice constants")
    if expand_node.input[0] not in initializers:
        raise CandidateError("Could not inspect the pinned graph's speech start token constant")
    from onnx import numpy_helper

    text_end = int(numpy_helper.to_array(initializers[text_slice.input[2]]).item())
    speech_start = int(numpy_helper.to_array(initializers[speech_slice.input[1]]).item())
    marker = int(numpy_helper.to_array(initializers[equal_node.input[1]]).item())
    replacement = int(numpy_helper.to_array(initializers[expand_node.input[0]]).item())
    hybrid_tail_contract = {
        "text_slice_end_exclusive": text_end,
        "speech_slice_start": speech_start,
        "sentinel_token_id": marker,
        "speech_replacement_token_id": replacement,
    }
    expected_tail = {
        "text_slice_end_exclusive": -2,
        "speech_slice_start": -2,
        "sentinel_token_id": SPEECH_SENTINEL_ID,
        "speech_replacement_token_id": START_SPEECH_ID,
    }
    if hybrid_tail_contract != expected_tail:
        raise CandidateError(
            f"Unexpected hybrid text/speech tail contract: {hybrid_tail_contract}; expected {expected_tail}."
        )

    opsets = {item.domain or "ai.onnx": int(item.version) for item in model.opset_import}
    graph_input = [value for value in graph.input if value.name == "input_ids"]
    graph_output = [value for value in graph.output if value.name == "inputs_embeds"]
    if len(graph_input) != 1 or len(graph_output) != 1:
        raise CandidateError("The pinned graph input/output names do not match the browser contract")
    input_type = graph_input[0].type.tensor_type.elem_type
    output_type = graph_output[0].type.tensor_type.elem_type
    if input_type != onnx.TensorProto.INT64 or output_type != onnx.TensorProto.FLOAT:
        raise CandidateError("The pinned graph input/output element types do not match the browser contract")

    source_metadata = {
        "repo_id": SOURCE_REPO_ID,
        "revision": PINNED_REVISION,
        "sha256sums": {"path": "SHA256SUMS", "sha256": sums_digest},
        "files": source_file_records,
        "graph": {
            "input_name": graph_input[0].name,
            "output_name": graph_output[0].name,
            "input_type": onnx.TensorProto.DataType.Name(input_type).lower(),
            "output_type": onnx.TensorProto.DataType.Name(output_type).lower(),
            "embedding_dim": EMBEDDING_DIM,
            "opsets": opsets,
            "gather_tables": gather_tables,
            "hybrid_tail_contract": hybrid_tail_contract,
        },
    }
    return source_metadata, model


def _ort_imports() -> tuple[Any, Any, Any, str, str]:
    """Import the verified installed weight-only Gather quantizer API."""
    try:
        import onnx
        import onnxruntime
        from onnxruntime.quantization.matmul_nbits_quantizer import (
            DefaultWeightOnlyQuantConfig,
            MatMulNBitsQuantizer,
        )
        from onnxruntime.quantization.quant_utils import QuantFormat
    except ImportError as exc:
        raise CandidateError(
            "The active Python environment must provide ONNX, ONNX Runtime, "
            "MatMulNBitsQuantizer, DefaultWeightOnlyQuantConfig, and QuantFormat. "
            "The repository's .venv-nano-cpu environment exposes the installed ORT API."
        ) from exc

    try:
        ort_version = importlib.metadata.version("onnxruntime")
    except importlib.metadata.PackageNotFoundError as exc:
        raise CandidateError("The imported ONNX Runtime package has no installed distribution metadata") from exc
    numbers = [int(part) for part in re.findall(r"\d+", ort_version)[:2]]
    if len(numbers) < 2 or tuple(numbers) < (1, 20):
        raise CandidateError(
            f"ONNX Runtime {ort_version} is too old for GatherBlockQuantized; use ORT 1.20 or newer."
        )
    return onnx, onnxruntime, (MatMulNBitsQuantizer, DefaultWeightOnlyQuantConfig, QuantFormat), ort_version, onnx.__version__


def _validate_candidate_graph(
    onnx: Any,
    target_graph: Path,
    target_external_data: Path,
) -> None:
    candidate = onnx.load_model(str(target_graph), load_external_data=False)
    nodes = [node for node in candidate.graph.node if node.op_type == "GatherBlockQuantized"]
    if len(nodes) != 2:
        raise CandidateError(f"Expected two quantized Gather nodes; found {len(nodes)}")
    if any(node.domain != "com.microsoft" for node in nodes):
        raise CandidateError("A Q4 Gather node does not use the com.microsoft domain")

    initializers = {tensor.name: tensor for tensor in candidate.graph.initializer}
    expected_outputs = {"embedding", "embedding_1"}
    if {node.output[0] for node in nodes} != expected_outputs:
        raise CandidateError("Quantized Gather outputs do not preserve the pinned graph contract")
    expected_qweights = {
        "embedding": ("text_emb.weight_Q4", [TEXT_VOCAB_SIZE, EMBEDDING_DIM]),
        "embedding_1": ("speech_emb.weight_Q4", [SPEECH_VOCAB_SIZE, EMBEDDING_DIM]),
    }
    for node in nodes:
        attributes = _node_attributes(onnx, node)
        expected_attributes = {
            "gather_axis": GATHER_AXIS,
            "quantize_axis": QUANTIZE_AXIS,
            "block_size": BLOCK_SIZE,
        }
        for key, value in expected_attributes.items():
            if int(attributes.get(key, -999999)) != value:
                raise CandidateError(
                    f"Unexpected {key} for quantized Gather node {node.name}: {attributes.get(key)!r}"
                )
        if len(node.input) != 4:
            raise CandidateError(f"Expected explicit UINT4 zero points on {node.name}")
        qweight = initializers.get(node.input[0])
        if qweight is None or qweight.data_type != onnx.TensorProto.UINT4:
            raise CandidateError(f"Quantized weight for {node.name} is not a UINT4 initializer")
        expected_name, expected_shape = expected_qweights[node.output[0]]
        if qweight.name != expected_name or list(qweight.dims) != expected_shape:
            raise CandidateError(
                f"Quantized table for {node.name} has unexpected identity or shape: "
                f"{qweight.name!r} {list(qweight.dims)}."
            )
        scales = initializers.get(node.input[2])
        zero_points = initializers.get(node.input[3])
        if scales is None or scales.data_type != onnx.TensorProto.FLOAT16:
            raise CandidateError(f"Quantization scales for {node.name} are not FP16 initializers")
        if zero_points is None or zero_points.data_type != onnx.TensorProto.UINT4:
            raise CandidateError(f"Quantization zero points for {node.name} are not UINT4 initializers")

    if not target_external_data.is_file() or target_external_data.stat().st_size == 0:
        raise CandidateError("ONNX Runtime did not save the quantized tensors to external data")
    referenced_locations: set[str] = set()
    for tensor in candidate.graph.initializer:
        metadata = _external_data_metadata(tensor)
        if metadata.get("location"):
            referenced_locations.add(metadata["location"])
    if referenced_locations != {target_external_data.name}:
        raise CandidateError(
            f"Candidate graph external-data references are {sorted(referenced_locations)}, "
            f"expected only {target_external_data.name!r}."
        )
    opsets = {item.domain or "ai.onnx": int(item.version) for item in candidate.opset_import}
    if opsets.get("ai.onnx", 0) < 21 or opsets.get("com.microsoft") != 1:
        raise CandidateError(f"Unexpected candidate opset imports: {opsets}")


@dataclass
class ErrorStats:
    count: int = 0
    sum_absolute: float = 0.0
    sum_squared: float = 0.0
    maximum_absolute: float = 0.0

    def update(self, actual: Any, expected: Any) -> None:
        import numpy as np

        actual_array = np.asarray(actual, dtype=np.float64)
        expected_array = np.asarray(expected, dtype=np.float64)
        if actual_array.shape != expected_array.shape:
            raise CandidateError(
                f"CPU component output shape differs: {actual_array.shape} vs {expected_array.shape}"
            )
        if not np.isfinite(actual_array).all() or not np.isfinite(expected_array).all():
            raise CandidateError("CPU component lookup returned a non-finite value")
        difference = actual_array - expected_array
        absolute = np.abs(difference)
        self.count += int(difference.size)
        self.sum_absolute += float(absolute.sum(dtype=np.float64))
        self.sum_squared += float(np.square(difference).sum(dtype=np.float64))
        if absolute.size:
            self.maximum_absolute = max(self.maximum_absolute, float(absolute.max()))

    def record(self) -> dict[str, Any]:
        if self.count == 0:
            return {
                "elements_compared": 0,
                "max_absolute_error": None,
                "mean_absolute_error": None,
                "root_mean_square_error": None,
            }
        return {
            "elements_compared": self.count,
            "max_absolute_error": self.maximum_absolute,
            "mean_absolute_error": self.sum_absolute / self.count,
            "root_mean_square_error": (self.sum_squared / self.count) ** 0.5,
        }


def _text_probe_ids(vocabulary_size: int, rng: random.Random) -> list[int]:
    boundaries = {0, 1, 2, vocabulary_size - 3, vocabulary_size - 2, vocabulary_size - 1}
    random_ids = rng.sample(range(vocabulary_size), min(24, vocabulary_size))
    return sorted(boundaries.union(random_ids))


def _speech_probe_ids(vocabulary_size: int, rng: random.Random) -> list[int]:
    boundaries = {0, 1, 2, vocabulary_size - 3, vocabulary_size - 2, vocabulary_size - 1}
    random_ids = rng.sample(range(vocabulary_size), min(24, vocabulary_size))
    return sorted(boundaries.union(random_ids))


def _create_cpu_session(onnxruntime: Any, model_path: Path) -> Any:
    options = onnxruntime.SessionOptions()
    options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    session = onnxruntime.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    providers = session.get_providers()
    if providers != ["CPUExecutionProvider"]:
        raise CandidateError(f"CPU component check selected unexpected providers: {providers}")
    if [item.name for item in session.get_inputs()] != ["input_ids"]:
        raise CandidateError("CPU component check found an unexpected input contract")
    if [item.name for item in session.get_outputs()] != ["inputs_embeds"]:
        raise CandidateError("CPU component check found an unexpected output contract")
    return session


def _output(session: Any, token_ids: Any, expected_rows: int) -> Any:
    import numpy as np

    value = session.run(["inputs_embeds"], {"input_ids": token_ids})[0]
    if value.ndim != 3 or value.shape[1] != expected_rows or value.shape[2] != EMBEDDING_DIM:
        raise CandidateError(
            f"Unexpected CPU component output shape {value.shape}; "
            f"expected [batch,{expected_rows},{EMBEDDING_DIM}]."
        )
    if value.dtype != np.float32:
        raise CandidateError(f"Expected FP32 graph output; found {value.dtype}")
    return value


def _source_stream(
    session: Any,
    reference_file: BinaryIO,
    text_probe_ids: set[int],
    speech_probe_ids: set[int],
) -> Any:
    """Run the FP16 source graph in bounded chunks and stage only output rows."""
    import numpy as np

    seen_text_probes: set[int] = set()
    seen_speech_probes: set[int] = set()
    for start in range(0, TEXT_VOCAB_SIZE, STREAM_BATCH_SIZE):
        end = min(start + STREAM_BATCH_SIZE, TEXT_VOCAB_SIZE)
        token_ids = np.arange(start, end, dtype=np.int64)
        input_ids = np.concatenate(
            (token_ids, np.asarray([SPEECH_SENTINEL_ID, 0], dtype=np.int64))
        )[None, :]
        embeddings = _output(session, input_ids, len(token_ids) + 2)[0]
        text_rows = embeddings[: len(token_ids)]
        text_rows.tofile(reference_file)
        for token_id in token_ids.tolist():
            if token_id in text_probe_ids:
                seen_text_probes.add(token_id)

    for start in range(0, SPEECH_VOCAB_SIZE, STREAM_BATCH_SIZE):
        end = min(start + STREAM_BATCH_SIZE, SPEECH_VOCAB_SIZE)
        token_ids = np.arange(start, end, dtype=np.int64)
        input_ids = np.empty((len(token_ids), 2), dtype=np.int64)
        input_ids[:, 0] = SPEECH_SENTINEL_ID
        input_ids[:, 1] = token_ids
        embeddings = _output(session, input_ids, 2)
        speech_rows = embeddings[:, 1, :]
        speech_rows.tofile(reference_file)
        for token_id in token_ids.tolist():
            if token_id in speech_probe_ids:
                seen_speech_probes.add(token_id)

    initial_tail = _output(
        session,
        np.asarray([[SPEECH_SENTINEL_ID, SPEECH_SENTINEL_ID]], dtype=np.int64),
        2,
    ).copy()
    expected_text = set(text_probe_ids)
    expected_speech = set(speech_probe_ids)
    if seen_text_probes != expected_text or seen_speech_probes != expected_speech:
        raise CandidateError("The streamed source lookup did not cover every fixed probe token")
    return initial_tail


def _candidate_stream(
    session: Any,
    reference_file: BinaryIO,
    text_probe_ids: set[int],
    speech_probe_ids: set[int],
    initial_tail_reference: Any,
    initial_tail_stats: ErrorStats,
) -> dict[str, Any]:
    """Compare candidate lookup rows while keeping working arrays bounded."""
    import numpy as np

    text_stats = ErrorStats()
    speech_stats = ErrorStats()
    text_probe_stats = ErrorStats()
    speech_probe_stats = ErrorStats()

    for start in range(0, TEXT_VOCAB_SIZE, STREAM_BATCH_SIZE):
        end = min(start + STREAM_BATCH_SIZE, TEXT_VOCAB_SIZE)
        token_ids = np.arange(start, end, dtype=np.int64)
        input_ids = np.concatenate(
            (token_ids, np.asarray([SPEECH_SENTINEL_ID, 0], dtype=np.int64))
        )[None, :]
        embeddings = _output(session, input_ids, len(token_ids) + 2)[0]
        actual = embeddings[: len(token_ids)]
        expected = np.fromfile(reference_file, dtype=np.float32, count=actual.size)
        if expected.size != actual.size:
            raise CandidateError("The staged source lookup ended during the text table comparison")
        expected = expected.reshape(actual.shape)
        text_stats.update(actual, expected)
        for index, token_id in enumerate(token_ids.tolist()):
            if token_id in text_probe_ids:
                text_probe_stats.update(actual[index], expected[index])

    for start in range(0, SPEECH_VOCAB_SIZE, STREAM_BATCH_SIZE):
        end = min(start + STREAM_BATCH_SIZE, SPEECH_VOCAB_SIZE)
        token_ids = np.arange(start, end, dtype=np.int64)
        input_ids = np.empty((len(token_ids), 2), dtype=np.int64)
        input_ids[:, 0] = SPEECH_SENTINEL_ID
        input_ids[:, 1] = token_ids
        embeddings = _output(session, input_ids, 2)
        actual = embeddings[:, 1, :]
        expected = np.fromfile(reference_file, dtype=np.float32, count=actual.size)
        if expected.size != actual.size:
            raise CandidateError("The staged source lookup ended during the speech table comparison")
        expected = expected.reshape(actual.shape)
        speech_stats.update(actual, expected)
        for index, token_id in enumerate(token_ids.tolist()):
            if token_id in speech_probe_ids:
                speech_probe_stats.update(actual[index], expected[index])

    tail_candidate = _output(
        session,
        np.asarray([[SPEECH_SENTINEL_ID, SPEECH_SENTINEL_ID]], dtype=np.int64),
        2,
    )
    initial_tail_stats.update(tail_candidate, initial_tail_reference)
    expected_text_count = TEXT_VOCAB_SIZE * EMBEDDING_DIM
    expected_speech_count = SPEECH_VOCAB_SIZE * EMBEDDING_DIM
    if text_stats.count != expected_text_count or speech_stats.count != expected_speech_count:
        raise CandidateError(
            "Full lookup comparison did not cover exactly the two embedding tables: "
            f"text={text_stats.count}/{expected_text_count}, "
            f"speech={speech_stats.count}/{expected_speech_count}."
        )
    if reference_file.read(1):
        raise CandidateError("The staged source lookup has unexpected trailing bytes")
    return {
        "text": text_stats.record(),
        "speech": speech_stats.record(),
        "fixed_text": text_probe_stats.record(),
        "fixed_speech": speech_probe_stats.record(),
    }


def run_cpu_component_check(
    onnxruntime: Any,
    source_graph: Path,
    candidate_graph: Path,
    staging_dir: Path,
) -> dict[str, Any]:
    """Compare all table rows on CPU, with a small fixed diagnostic subset."""
    rng = random.Random(PROBE_SEED)
    text_probe_ids = set(_text_probe_ids(TEXT_VOCAB_SIZE, rng))
    speech_probe_ids = set(_speech_probe_ids(SPEECH_VOCAB_SIZE, rng))
    # All tensors are streamed in batches. The scratch file holds only the
    # source lookup output on disk; it is deleted automatically when closed.
    with tempfile.TemporaryFile(mode="w+b", dir=staging_dir) as reference_file:
        source_session = _create_cpu_session(onnxruntime, source_graph)
        initial_tail_reference = _source_stream(
            source_session,
            reference_file,
            text_probe_ids,
            speech_probe_ids,
        )
        del source_session
        gc.collect()

        reference_file.flush()
        reference_file.seek(0)
        candidate_session = _create_cpu_session(onnxruntime, candidate_graph)
        stats = ErrorStats()
        full_metrics = _candidate_stream(
            candidate_session,
            reference_file,
            text_probe_ids,
            speech_probe_ids,
            initial_tail_reference,
            stats,
        )
        del candidate_session
        gc.collect()

    def probe_record(ids: set[int], full_key: str) -> dict[str, Any]:
        # Fixed probes are already included in the full sweep. These values
        # report their isolated errors without retaining a full-table diff.
        return {
            "token_ids": sorted(ids),
            **full_metrics[full_key],
        }

    return {
        "status": "completed",
        "provider": "CPUExecutionProvider",
        "method": "sequential_streamed_full_lookup",
        "acceptance": "descriptive_only_no_gate",
        "fixed_probe_seed": PROBE_SEED,
        "batch_size": STREAM_BATCH_SIZE,
        "full_lookup": {
            "text_emb": {
                "token_count": TEXT_VOCAB_SIZE,
                **full_metrics["text"],
            },
            "speech_emb": {
                "token_count": SPEECH_VOCAB_SIZE,
                **full_metrics["speech"],
            },
        },
        "fixed_probes": {
            "text": probe_record(text_probe_ids, "fixed_text"),
            "speech": probe_record(speech_probe_ids, "fixed_speech"),
            "initial_tail": {
                "token_ids": [SPEECH_SENTINEL_ID, SPEECH_SENTINEL_ID],
                **stats.record(),
            },
        },
        "quality_or_promotion_gate": False,
    }


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_candidate(
    source_graph: Path,
    target_graph: Path,
    quantizer_api: tuple[Any, Any, Any],
) -> None:
    MatMulNBitsQuantizer, DefaultWeightOnlyQuantConfig, QuantFormat = quantizer_api
    config = DefaultWeightOnlyQuantConfig(
        block_size=BLOCK_SIZE,
        is_symmetric=False,
        quant_format=QuantFormat.QOperator,
        op_types_to_quantize=("Gather",),
        quant_axes=(("Gather", QUANTIZE_AXIS),),
        bits=4,
    )
    quantizer = MatMulNBitsQuantizer(
        model=str(source_graph),
        algo_config=config,
    )
    quantizer.process()
    quantizer.model.save_model_to_file(str(target_graph), use_external_data_format=True)
    del quantizer
    gc.collect()


def _write_manifest(
    out_dir: Path,
    source: dict[str, Any],
    source_unchanged: bool,
    ort_version: str,
    onnx_version: str,
    cpu_component_check: dict[str, Any],
) -> dict[str, Any]:
    target_graph = out_dir / TARGET_GRAPH_NAME
    target_external_data = out_dir / TARGET_EXTERNAL_DATA_NAME
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "candidate_status": "experimental_unpromoted",
        "production_gate": False,
        "source": source,
        "quantization": {
            "api": "onnxruntime.quantization.matmul_nbits_quantizer.MatMulNBitsQuantizer",
            "ort_version": ort_version,
            "onnx_version": onnx_version,
            "algorithm": "DEFAULT",
            "bits": 4,
            "block_size": BLOCK_SIZE,
            "is_symmetric": False,
            "quantized_dtype": "UINT4",
            "quant_format": "QOperator",
            "op_types_to_quantize": ["Gather"],
            "quant_axes": {"Gather": QUANTIZE_AXIS},
            "source_gather_axis": GATHER_AXIS,
            "target_operator": "com.microsoft::GatherBlockQuantized",
        },
        "target": {
            "graph": file_record(target_graph, TARGET_GRAPH_NAME),
            "external_data": file_record(target_external_data, TARGET_EXTERNAL_DATA_NAME),
        },
        "source_integrity_after_export": {
            "unchanged": source_unchanged,
            "checked_files": sorted(SOURCE_FILES),
        },
        "cpu_component_check": cpu_component_check,
        "limitations": [
            "This artifact is an experimental embedding-only Q4 candidate.",
            "The CPU lookup report is descriptive and does not create a numerical acceptance gate.",
            "No browser inference, WebGPU provider, generated tokens, audio, quality, RSS, or latency is verified by this tool.",
            "Matched browser token and audio comparisons, host GPU behavior, memory, latency, and listening remain required before promotion.",
        ],
    }
    _write_json(out_dir / TARGET_MANIFEST_NAME, manifest)
    return manifest


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Local root of the pinned owensong/chatterbox-nano-ONNX source tree.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New, empty target directory for the Q4 graph, external data, and manifest.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    source_dir = args.source_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir == source_dir or source_dir in output_dir.parents:
        raise CandidateError("The output directory must be outside the pinned source tree")
    if output_dir.exists():
        raise CandidateError(f"Output directory already exists; use a new path: {output_dir}")

    source, _ = verify_source(source_dir)
    onnx, onnxruntime, quantizer_api, ort_version, onnx_version = _ort_imports()

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
    try:
        source_graph = source_dir / GRAPH_RELATIVE_PATH
        target_graph = staging_dir / TARGET_GRAPH_NAME
        target_external_data = staging_dir / TARGET_EXTERNAL_DATA_NAME

        print(
            "Quantizing only the two pinned FP16 embedding Gather tables: "
            f"revision={PINNED_REVISION}, ort={ort_version}, Gather axis={GATHER_AXIS}, "
            f"Q4 block axis={QUANTIZE_AXIS}, block_size={BLOCK_SIZE}.",
            flush=True,
        )
        _build_candidate(source_graph, target_graph, quantizer_api)
        _validate_candidate_graph(onnx, target_graph, target_external_data)

        cpu_check: dict[str, Any]
        try:
            print("Comparing the full text and speech embedding tables on CPU in small batches.", flush=True)
            cpu_check = run_cpu_component_check(
                onnxruntime,
                source_graph,
                target_graph,
                staging_dir,
            )
        except Exception as exc:  # The Q4 files remain an unpromoted artifact if CPU checks cannot run.
            cpu_check = {
                "status": "failed",
                "provider": "CPUExecutionProvider",
                "method": "sequential_streamed_full_lookup",
                "acceptance": "descriptive_only_no_gate",
                "quality_or_promotion_gate": False,
                "failure_reason": f"{type(exc).__name__}: {exc}",
            }

        source_after = {}
        for relative_name, expected_hash in SOURCE_FILES.items():
            path = source_dir / relative_name
            actual_hash = sha256_file(path)
            if actual_hash != expected_hash:
                raise CandidateError(
                    f"Pinned source changed during export/check: {relative_name}; "
                    f"expected {expected_hash}, found {actual_hash}."
                )
            source_after[relative_name] = actual_hash

        source_unchanged = all(source_after[name] == record["sha256"] for name, record in source["files"].items())
        _write_manifest(
            staging_dir,
            source,
            source_unchanged,
            ort_version,
            onnx_version,
            cpu_check,
        )
        shutil.move(str(staging_dir), str(output_dir))
    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        raise

    manifest_path = output_dir / TARGET_MANIFEST_NAME
    print(f"Wrote unpromoted candidate manifest: {manifest_path}")
    print(f"Candidate graph: {output_dir / TARGET_GRAPH_NAME}")
    print(f"Candidate external data: {output_dir / TARGET_EXTERNAL_DATA_NAME}")
    if cpu_check["status"] != "completed":
        print(f"CPU component check did not complete: {cpu_check.get('failure_reason', 'unknown error')}", file=sys.stderr)
        return 2
    print("CPU component errors are descriptive only. This manifest is not a quality or promotion gate.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CandidateError as exc:
        raise SystemExit(f"quantize-embedding.py: {exc}") from exc
