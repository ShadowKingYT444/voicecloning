"""Pure checks for the fitted-T3 ONNX verification helper."""

from __future__ import annotations

import numpy as np
import json

from verify_fitted_onnx import (
    T3_HEADS,
    T3_HEAD_DIM,
    T3_LAYERS,
    _baseline_effect,
    _empty_cache,
    _output_metrics,
    build_parser,
    compare_array,
)
from onnx_staged_runtime import _read_manifest


def test_compare_array_reports_gate_and_relative_metrics() -> None:
    expected = np.ones((2, 3), dtype=np.float32)
    passed = compare_array(expected, expected.copy(), atol=1e-4, rtol=1e-4)
    assert passed["status"] == "passed"
    changed = compare_array(expected + 2e-4, expected, atol=1e-4, rtol=1e-4)
    assert changed["status"] == "mismatch"
    assert np.isclose(changed["max_abs"], 2e-4, rtol=0.0, atol=1e-7)
    assert changed["outside_tolerance_count"] == expected.size


def test_output_metrics_separate_logits_and_cache() -> None:
    outputs = [np.zeros((1, 4), dtype=np.float32)] + [
        np.zeros((1, T3_HEADS, 2, T3_HEAD_DIM), dtype=np.float32)
        for _ in range(T3_LAYERS * 2)
    ]
    changed = [array.copy() for array in outputs]
    changed[0][0, 1] = 1e-2
    report = _output_metrics(changed, outputs, atol=1e-4, rtol=1e-4)
    assert report["status"] == "mismatch"
    assert report["logits"]["status"] == "mismatch"
    assert all(metric["status"] == "passed" for metric in report["cache"])


def test_baseline_effect_and_empty_cache_contract() -> None:
    cache = _empty_cache()
    assert len(cache) == T3_LAYERS * 2
    assert all(item.shape == (1, T3_HEADS, 0, T3_HEAD_DIM) for item in cache)
    base = [np.zeros((1, 2), dtype=np.float32)]
    adapted = [np.ones((1, 2), dtype=np.float32)]
    effect = _baseline_effect(adapted, base)
    assert effect["status"] == "nonzero"
    assert effect["changed_elements_1e-8"] == 2


def test_gate_is_explicit_and_limited_to_supported_values() -> None:
    parser = build_parser()
    args = parser.parse_args(["verify", "--gate", "1e-4"])
    assert args.gate == "1e-4"
    args = parser.parse_args(["verify", "--gate", "3e-4"])
    assert args.gate == "3e-4"


def test_staged_manifest_is_strict_by_default_but_explicitly_readable_for_verification(tmp_path) -> None:
    """The verifier may inspect an exported graph without promoting it."""

    stage_dir = tmp_path / "t3"
    stage_dir.mkdir()
    graph = stage_dir / "graph.onnx"
    graph.write_bytes(b"test graph")
    (stage_dir / "manifest.json").write_text(
        json.dumps(
            {
                "stage": "t3",
                "status": "exported",
                "graph": {"path": graph.name},
                "verification": {"status": "not_run"},
            }
        )
    )
    with np.testing.assert_raises(RuntimeError):
        _read_manifest(tmp_path, "t3")
    loaded = _read_manifest(tmp_path, "t3", require_verified=False)
    assert loaded["_graph_path"] == str(graph)


def test_fitted_provenance_rejects_modified_graph_or_adapter_scale(tmp_path):
    import hashlib
    from verify_fitted_onnx import _stage_provenance
    stage = tmp_path / 't3'
    stage.mkdir()
    graph = stage / 'model.onnx'
    graph.write_bytes(b'graph')
    digest = hashlib.sha256(b'graph').hexdigest()
    metadata = {'adapter': {'sha256': 'a' * 64}, 'adapter_scale': 1.0,
                'model_checkpoint_sha256': 'b' * 64}
    manifest = {'stage': 't3', 'status': 'exported',
                'graph': {'path': graph.name, 'sha256': digest},
                'adapter_patch': {'adapter_sha256': 'a' * 64, 'scale': 1.0,
                                  'patched_graph_sha256': digest},
                'checkpoint': {'sha256': 'b' * 64}}
    path = stage / 'manifest.json'
    path.write_text(json.dumps(manifest))
    assert _stage_provenance(metadata, tmp_path)['status'] == 'passed'
    graph.write_bytes(b'changed')
    assert _stage_provenance(metadata, tmp_path)['status'] == 'mismatch'
    graph.write_bytes(b'graph')
    manifest['adapter_patch']['scale'] = .5
    path.write_text(json.dumps(manifest))
    assert _stage_provenance(metadata, tmp_path)['status'] == 'mismatch'


def test_cuda_preserves_full_precision_reference_contract():
    from onnx_staged_runtime import _cuda_provider_options
    assert _cuda_provider_options()['use_tf32'] == 0
