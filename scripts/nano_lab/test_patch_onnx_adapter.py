"""Pure synthetic tests for the streaming ONNX LoRA patcher.

These tests construct the small protobuf messages needed by the metadata
reader.  They do not read Nano checkpoints, import PyTorch, or create an ORT
session.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from patch_onnx_adapter import (
    AdapterSpec,
    PatchError,
    _expected_graph_name,
    merge_factors,
    patch_graph,
    read_graph_metadata,
)


def _varint(value: int) -> bytes:
    value = int(value)
    if value < 0:
        value &= (1 << 64) - 1
    out = bytearray()
    while value >= 0x80:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def _field(number: int, wire_type: int, payload: bytes | int) -> bytes:
    value = _varint((number << 3) | wire_type)
    if wire_type == 0:
        return value + _varint(int(payload))
    if wire_type == 2:
        raw = bytes(payload)
        return value + _varint(len(raw)) + raw
    raise AssertionError(wire_type)


def _tensor(name: str, dims: tuple[int, ...], raw: bytes, *, dtype: int = 1, external: bool = False) -> bytes:
    message = bytearray()
    for dim in dims:
        message += _field(1, 0, dim)
    message += _field(2, 0, dtype)
    message += _field(8, 2, name.encode("utf-8"))
    if external:
        # StringStringEntryProto payload. The parser only needs to detect the
        # outer field and must not materialize the entry.
        message += _field(13, 2, _field(1, 2, b"location") + _field(2, 2, b"weights.bin"))
        message += _field(14, 0, 1)
    else:
        message += _field(9, 2, raw)
    return _field(5, 2, bytes(message))


def _graph(initializers: list[bytes], *, extra: bytes = b"") -> bytes:
    # GraphProto initializer is field 5. The extra field is deliberately
    # unknown metadata, used to prove byte preservation around raw spans.
    graph = b"".join(initializers) + extra
    return _field(7, 2, graph)


def _adapter(module: str, a: np.ndarray, b: np.ndarray, *, rank: int | None = None, alpha: float = 1.0) -> AdapterSpec:
    rank_value = int(rank or a.shape[0])
    shape = (int(a.shape[1]), int(b.shape[0]))
    return AdapterSpec(
        path=Path("adapter.pt"),
        format="nano_t3_lora_checkpoint_v1",
        rank=rank_value,
        alpha=float(alpha),
        dropout=0.05,
        modules=(module,),
        factors={module: (np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32))},
        module_shapes={module: shape},
    )


class PatcherTests(unittest.TestCase):
    def test_merge_orientation_matches_hf_conv1d(self) -> None:
        base = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32).tobytes()
        a = np.asarray([[1.0, 2.0]], dtype=np.float32)
        b = np.asarray([[5.0], [7.0]], dtype=np.float32)
        merged = np.frombuffer(
            merge_factors(base, a, b, effective_scale=0.5, shape=(2, 2)), dtype="<f4"
        ).reshape(2, 2)
        # x @ (A.T @ B.T), with Conv1D's [in,out] storage.
        np.testing.assert_allclose(merged, [[3.5, 5.5], [8.0, 11.0]], rtol=0, atol=0)

    def test_reader_exposes_inline_spans_without_tensor_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.onnx"
            raw = np.arange(6, dtype="<f4").tobytes()
            path.write_bytes(_graph([_tensor("t3.tfmr.h.0.attn.c_proj.weight", (2, 3), raw)]))
            metadata = read_graph_metadata(path)
            self.assertEqual(metadata.bytes, path.stat().st_size)
            self.assertEqual(len(metadata.initializers), 1)
            item = metadata.initializers[0]
            self.assertEqual(item.name, "t3.tfmr.h.0.attn.c_proj.weight")
            self.assertEqual(item.dims, (2, 3))
            self.assertEqual(item.raw_length, len(raw))
            self.assertFalse(item.is_external)

    def test_patch_preserves_every_byte_outside_target_span(self) -> None:
        module = "h.0.attn.c_proj"
        graph_name = _expected_graph_name(module)
        base_values = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        prefix = b"model-prefix"
        suffix = b"model-suffix"
        # Keep prefix/suffix as valid unknown ModelProto fields so the
        # streaming parser must skip them and the preservation check has real
        # protobuf bytes on both sides of the graph.
        graph_bytes = _field(2, 2, prefix) + _graph([_tensor(graph_name, (2, 2), base_values.tobytes())], extra=_field(2, 2, b"g")) + _field(3, 2, suffix)
        a = np.asarray([[1.0, 2.0]], dtype=np.float32)
        b = np.asarray([[5.0], [7.0]], dtype=np.float32)
        adapter = _adapter(module, a, b, alpha=2.0)
        with tempfile.TemporaryDirectory() as directory:
            base_path = Path(directory) / "base.onnx"
            output_path = Path(directory) / "patched.onnx"
            base_path.write_bytes(graph_bytes)
            result = patch_graph(base_path, output_path, adapter, scale=0.5)
            self.assertTrue(result.outside_bytes_identical)
            self.assertEqual(result.graph_bytes, len(graph_bytes))
            self.assertEqual(len(result.changed_spans), 1)
            span = result.changed_spans[0]
            self.assertNotEqual(span["before_sha256"], span["after_sha256"])
            metadata = read_graph_metadata(output_path)
            target = metadata.initializers[0]
            with output_path.open("rb") as handle:
                handle.seek(int(target.raw_offset))
                merged = np.frombuffer(handle.read(int(target.raw_length)), dtype="<f4").reshape(2, 2)
            # alpha/rank=2, CLI scale=.5, so effective scale is 1.
            np.testing.assert_allclose(merged, [[6.0, 9.0], [13.0, 18.0]], rtol=0, atol=0)
            original = bytearray(graph_bytes)
            patched = output_path.read_bytes()
            changed = [i for i, (left, right) in enumerate(zip(original, patched)) if left != right]
            self.assertTrue(changed)
            self.assertGreaterEqual(min(changed), int(target.raw_offset))
            self.assertLess(max(changed), int(target.raw_offset) + int(target.raw_length))

    def test_external_target_is_rejected(self) -> None:
        module = "h.0.attn.c_proj"
        graph_name = _expected_graph_name(module)
        adapter = _adapter(
            module,
            np.ones((1, 2), dtype=np.float32),
            np.ones((2, 1), dtype=np.float32),
        )
        with tempfile.TemporaryDirectory() as directory:
            base_path = Path(directory) / "base.onnx"
            output_path = Path(directory) / "patched.onnx"
            base_path.write_bytes(_graph([_tensor(graph_name, (2, 2), b"", external=True)]))
            with self.assertRaisesRegex(PatchError, "external"):
                patch_graph(base_path, output_path, adapter)

    def test_dtype_and_shape_fail_closed(self) -> None:
        module = "h.0.attn.c_proj"
        graph_name = _expected_graph_name(module)
        adapter = _adapter(
            module,
            np.ones((1, 2), dtype=np.float32),
            np.ones((2, 1), dtype=np.float32),
        )
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            base_path = directory_path / "base.onnx"
            output_path = directory_path / "patched.onnx"
            base_path.write_bytes(_graph([_tensor(graph_name, (3, 3), np.zeros(9, dtype="<f4").tobytes())]))
            with self.assertRaisesRegex(PatchError, "dims"):
                patch_graph(base_path, output_path, adapter)
            base_path.write_bytes(_graph([_tensor(graph_name, (2, 2), np.zeros(4, dtype="<f4").tobytes(), dtype=6)]))
            with self.assertRaisesRegex(PatchError, "data_type"):
                patch_graph(base_path, output_path, adapter)


if __name__ == "__main__":
    unittest.main()
