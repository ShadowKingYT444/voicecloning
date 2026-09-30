"""Pure synthetic tests for the meanflow external-weight patcher.

Most tests build tiny protobuf metadata and sidecars. One integration check
reads the small actual factor checkpoint with CPU Torch and compares its names
and hashes with the external report. No Nano model or ORT session is loaded.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import patch_onnx_decoder_attention as patcher


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
    result = _varint((number << 3) | wire_type)
    if wire_type == 0:
        return result + _varint(int(payload))
    if wire_type == 2:
        value = bytes(payload)
        return result + _varint(len(value)) + value
    raise AssertionError(wire_type)


def _tensor(name: str, shape: tuple[int, ...], *, location: str, offset: int, length: int, dtype: int = 1) -> bytes:
    message = bytearray()
    for value in shape:
        message += _field(1, 0, value)
    message += _field(2, 0, dtype)
    message += _field(8, 2, name.encode())
    message += _field(13, 2, _field(1, 2, b"location") + _field(2, 2, location.encode()))
    message += _field(13, 2, _field(1, 2, b"offset") + _field(2, 2, str(offset).encode()))
    message += _field(13, 2, _field(1, 2, b"length") + _field(2, 2, str(length).encode()))
    message += _field(14, 0, patcher.ONNX_EXTERNAL)
    return _field(5, 2, bytes(message))


def _graph(records: list[tuple[str, tuple[int, ...], int, int, int]], *, location: str = "nano_meanflow_estimator.weights.bin") -> bytes:
    payload = b"".join(
        _tensor(name, shape, location=location, offset=offset, length=length, dtype=dtype)
        for name, shape, offset, length, dtype in records
    )
    return _field(7, 2, payload)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _fixture(directory: Path, *, report_overrides: dict | None = None, graph_records: list | None = None) -> tuple[patcher.ExternalLayout, bytes, list[dict]]:
    # The production loader requires a stage manifest before it reads the
    # graph/report.  Keep the synthetic manifest minimal because these tests
    # exercise sidecar validation only.
    (directory / "manifest.json").write_text(json.dumps({"stage": "meanflow_estimator", "status": "exported"}))
    graph_path = directory / "nano_meanflow_estimator.onnx"
    report_path = directory / "nano_meanflow_estimator.external.json"
    weights_path = directory / "nano_meanflow_estimator.weights.bin"
    target_a = "estimator.down_blocks.0.1.0.attn1.to_q.weight"
    target_b = "estimator.down_blocks.0.1.0.attn1.to_out.0.weight"
    first = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype="<f4").tobytes()
    second = np.asarray([[5.0, 6.0], [7.0, 8.0]], dtype="<f4").tobytes()
    prefix = b"prefix!"
    gap = b"gap-unchanged"
    suffix = b"suffix-unchanged"
    first_offset = len(prefix)
    second_offset = first_offset + len(first) + len(gap)
    weight_bytes = prefix + first + gap + second + suffix
    weights_path.write_bytes(weight_bytes)
    records = [
        (target_a, (2, 2), first_offset, len(first), 1),
        (target_b, (2, 2), second_offset, len(second), 1),
    ] if graph_records is None else graph_records
    graph_path.write_bytes(_graph(records))
    by_offset = {offset: payload for offset, payload in ((first_offset, first), (second_offset, second))}
    initializers = []
    for name, shape, offset, length, dtype in records:
        raw = by_offset.get(offset, b"\0" * length)
        initializers.append({
            "graph_name": name,
            "state_name": name,
            "shape": list(shape),
            "dtype": "float32" if dtype == 1 else "float16",
            "offset": offset,
            "length": length,
            "sha256": _sha(raw),
        })
    external = {"weight_file": weights_path.name, "weight_bytes": len(weight_bytes), "initializers": initializers}
    report = {
        "status": "exported",
        "graph": graph_path.name,
        "graph_bytes": graph_path.stat().st_size,
        "external": external,
    }
    if report_overrides:
        for key, value in report_overrides.items():
            if key == "external":
                external.update(value)
            else:
                report[key] = value
    report_path.write_text(json.dumps(report))
    layout = patcher._load_external_layout(directory)
    return layout, first + gap + second, initializers


def _factors(layout: patcher.ExternalLayout, *, bad_hash: bool = False) -> dict[str, patcher.FactorSpec]:
    result = {}
    for index, record in enumerate(layout.initializers):
        down = np.asarray([[2, -1], [.5, 1]], dtype="<f4")
        up = np.asarray([[1, 2], [3, 4]], dtype="<f4") * np.float32(index + 1)
        result[record.graph_name] = patcher.FactorSpec(
            native_name="flow.decoder." + record.graph_name,
            onnx_name=record.graph_name,
            shape=(2, 2),
            down=down,
            up=up,
            base_sha256=("0" * 64 if bad_hash else record.sha256),
        )
    return result


class ExternalPatcherTests(unittest.TestCase):
    def test_real_checkpoint_maps_to_real_external_inventory(self) -> None:
        # Read the small actual factors and graph metadata, not a model. This
        # checks the production native-to-ONNX namespace boundary which the
        # synthetic sidecar tests intentionally bypass.
        adapter = patcher._load_attention_adapter(patcher.DEFAULT_ADAPTER)
        layout = patcher._load_external_layout(patcher.DEFAULT_MODEL_DIR / 'meanflow_estimator')
        self.assertEqual(len(adapter.factors), 224)
        records = layout.by_name()
        for factor in adapter.factors:
            self.assertIn(factor.onnx_name, records)
            self.assertEqual(factor.shape, records[factor.onnx_name].shape)
            self.assertEqual(factor.base_sha256, records[factor.onnx_name].sha256)

    def test_strength_zero_and_one_and_non_target_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            layout, _, _ = _fixture(root)
            factors = _factors(layout)
            source = layout.weights_path.read_bytes()
            zero_path = root / "zero.bin"
            zero = patcher._patch_external_weights(layout.weights_path, zero_path, layout, factors, strength=0.0)
            self.assertEqual(zero_path.read_bytes(), source)
            self.assertEqual(zero.base_weights_sha256, zero.patched_weights_sha256)
            self.assertTrue(all(item["before_sha256"] == item["after_sha256"] for item in zero.changed_spans))
            one_path = root / "one.bin"
            one = patcher._patch_external_weights(layout.weights_path, one_path, layout, factors, strength=1.0)
            self.assertNotEqual(one_path.read_bytes(), source)
            self.assertEqual(one.weight_bytes, len(source))
            changed_ranges = [(item["offset"], item["offset"] + item["length"]) for item in one.changed_spans]
            for index, (before, after) in enumerate(zip(source, one_path.read_bytes())):
                inside = any(start <= index < stop for start, stop in changed_ranges)
                if not inside:
                    self.assertEqual(before, after, index)
            for item in one.changed_spans:
                start = int(item["offset"])
                length = int(item["length"])
                original = np.frombuffer(source[start : start + length], dtype="<f4").reshape(2, 2)
                # alpha/rank = 2/2 = 1; the non-symmetric product also catches
                # an accidental transpose in the actual sidecar write.
                expected = original + np.asarray([[3, 1], [8, 1]], dtype="<f4") * (1 if start < changed_ranges[-1][0] else 2)
                actual = np.frombuffer(one_path.read_bytes()[start : start + length], dtype="<f4").reshape(2, 2)
                np.testing.assert_array_equal(actual, expected)

    def test_wrong_factor_hash_is_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            layout, _, _ = _fixture(root)
            output = root / "bad.bin"
            with self.assertRaisesRegex(patcher.PatchError, "baseline hash"):
                patcher._patch_external_weights(layout.weights_path, output, layout, _factors(layout, bad_hash=True), strength=1)
            self.assertFalse(output.exists())

    def test_layout_rejects_transposed_missing_dtype_location_and_offset(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            # Transposed report shape differs from the graph and sidecar length.
            report_layout, _, _ = _fixture(root)
            payload = json.loads((root / "nano_meanflow_estimator.external.json").read_text())
            payload["external"]["initializers"][0]["shape"] = [2, 3]
            payload["external"]["initializers"][0]["length"] = 24
            (root / "nano_meanflow_estimator.external.json").write_text(json.dumps(payload))
            with self.assertRaises(patcher.PatchError):
                patcher._load_external_layout(root)
            # Missing initializer record.
            root2 = Path(raw) / "missing"
            root2.mkdir()
            _fixture(root2)
            payload = json.loads((root2 / "nano_meanflow_estimator.external.json").read_text())
            payload["external"]["initializers"].pop()
            (root2 / "nano_meanflow_estimator.external.json").write_text(json.dumps(payload))
            with self.assertRaisesRegex(patcher.PatchError, "sets differ"):
                patcher._load_external_layout(root2)
            # Wrong dtype.
            root3 = Path(raw) / "dtype"
            root3.mkdir()
            _fixture(root3)
            payload = json.loads((root3 / "nano_meanflow_estimator.external.json").read_text())
            payload["external"]["initializers"][0]["dtype"] = "float16"
            (root3 / "nano_meanflow_estimator.external.json").write_text(json.dumps(payload))
            with self.assertRaisesRegex(patcher.PatchError, "dtype"):
                patcher._load_external_layout(root3)
            # Wrong location in the graph.
            root4 = Path(raw) / "location"
            root4.mkdir()
            _fixture(root4)
            graph = _graph([
                ("estimator.down_blocks.0.1.0.attn1.to_q.weight", (2, 2), 7, 16, 1),
                ("estimator.down_blocks.0.1.0.attn1.to_out.0.weight", (2, 2), 36, 16, 1),
            ], location="wrong.bin")
            (root4 / "nano_meanflow_estimator.onnx").write_bytes(graph)
            payload = json.loads((root4 / "nano_meanflow_estimator.external.json").read_text())
            payload['graph_bytes'] = len(graph)
            (root4 / "nano_meanflow_estimator.external.json").write_text(json.dumps(payload))
            with self.assertRaisesRegex(patcher.PatchError, "location"):
                patcher._load_external_layout(root4)
            # Out-of-bounds offset in the report.
            root5 = Path(raw) / "offset"
            root5.mkdir()
            _fixture(root5)
            payload = json.loads((root5 / "nano_meanflow_estimator.external.json").read_text())
            payload["external"]["initializers"][1]["offset"] = 10_000
            (root5 / "nano_meanflow_estimator.external.json").write_text(json.dumps(payload))
            with self.assertRaisesRegex(patcher.PatchError, "exceeds sidecar bounds"):
                patcher._load_external_layout(root5)

    def test_untargeted_tensor_corruption_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            layout, _, _ = _fixture(root)
            factors = _factors(layout)
            untouched = layout.initializers[-1]
            factors.pop(untouched.graph_name)
            data = bytearray(layout.weights_path.read_bytes())
            data[untouched.offset] ^= 1
            layout.weights_path.write_bytes(data)
            output = root / 'corrupted.bin'
            with self.assertRaisesRegex(patcher.PatchError, 'baseline bytes'):
                patcher._patch_external_weights(layout.weights_path, output, layout, factors, strength=1)
            self.assertFalse(output.exists())

    def test_patch_model_dir_failure_does_not_publish_and_repatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            base = root / "base"
            stage = base / "meanflow_estimator"
            stage.mkdir(parents=True)
            layout, _, _ = _fixture(stage)
            checkpoint_sha = "a" * 64
            stage_manifest = {
                "stage": "meanflow_estimator",
                "status": "exported",
                "checkpoint": {"sha256": checkpoint_sha},
                "graph": {"path": layout.graph_path.name},
                "verification": {"status": "verified"},
            }
            (stage / "manifest.json").write_text(json.dumps(stage_manifest))
            (base / "manifest.json").write_text(json.dumps({"schema_version": 1, "stages": {}}))
            adapter = patcher.AttentionAdapter(
                path=root / "adapter.pt",
                sha256="b" * 64,
                inventory_sha256="c" * 64,
                model_checkpoint_sha256=checkpoint_sha,
                initial_conditionals_sha256="d" * 64,
                prepared_conditionals_sha256="e" * 64,
                factors=tuple(_factors(layout).values()),
            )
            adapter_path = root / "adapter.pt"
            adapter_path.write_bytes(b"adapter")
            with patch.object(patcher, "_load_attention_adapter", return_value=adapter):
                # The synthetic adapter is deliberately incomplete, so the
                # production canonical-count gate must fail before publication.
                output = root / "published"
                with self.assertRaisesRegex(patcher.PatchError, "exactly 224"):
                    patcher.patch_model_dir(base, adapter_path, output)
                self.assertFalse(output.exists())

    def test_graph_bytes_and_manifest_pending_after_synthetic_publication(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            base = root / "base"
            stage = base / "meanflow_estimator"
            stage.mkdir(parents=True)
            layout, _, _ = _fixture(stage)
            checkpoint_sha = "a" * 64
            manifest = {
                "stage": "meanflow_estimator",
                "status": "exported",
                "checkpoint": {"sha256": checkpoint_sha},
                "graph": {"path": layout.graph_path.name},
                "verification": {"status": "verified"},
            }
            (stage / "manifest.json").write_text(json.dumps(manifest))
            (base / "manifest.json").write_text(json.dumps({"schema_version": 1, "stages": {}}))
            adapter_path = root / "adapter.pt"
            adapter_path.write_bytes(b"adapter")
            adapter = patcher.AttentionAdapter(
                path=adapter_path,
                sha256=patcher._sha256_file(adapter_path),
                inventory_sha256="c" * 64,
                model_checkpoint_sha256=checkpoint_sha,
                initial_conditionals_sha256="d" * 64,
                prepared_conditionals_sha256="e" * 64,
                factors=tuple(_factors(layout).values()),
            )
            output = root / "published"
            with patch.object(patcher, "_load_attention_adapter", return_value=adapter), patch.object(patcher, "TARGET_COUNT", 2), patch.object(patcher, "TARGET_PARAMETER_COUNT", 16):
                report = patcher.patch_model_dir(base, adapter_path, output, strength=1)
            self.assertTrue(output.exists())
            self.assertEqual((output / "meanflow_estimator" / layout.graph_path.name).read_bytes(), layout.graph_path.read_bytes())
            stage_output = json.loads((output / "meanflow_estimator" / "manifest.json").read_text())
            self.assertEqual(stage_output["verification"]["status"], "pending")
            self.assertEqual(stage_output["decoder_attention_verification"]["status"], "pending")
            self.assertEqual(stage_output["adapter_patch"]["format"], patcher.PATCH_FORMAT)
            second = root / "second"
            with patch.object(patcher, "_load_attention_adapter", return_value=adapter), patch.object(patcher, "TARGET_COUNT", 2), patch.object(patcher, "TARGET_PARAMETER_COUNT", 16):
                with self.assertRaisesRegex(patcher.PatchError, "already contains"):
                    patcher.patch_model_dir(output, adapter_path, second, strength=1)
            self.assertFalse(second.exists())


if __name__ == "__main__":
    unittest.main()
