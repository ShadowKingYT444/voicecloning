"""A runtime failure must preserve evidence without authorizing a provider."""
import json
from types import SimpleNamespace

import numpy as np

import onnx_staged_runtime
import verify_decoder_attention_onnx as verifier


def test_failure_after_one_case_preserves_previous_cpu_gate(tmp_path, monkeypatch):
    stage = tmp_path / 'model' / 'meanflow_estimator'
    stage.mkdir(parents=True)
    refdir = tmp_path / 'reference'
    refdir.mkdir()
    (stage / 'graph.onnx').write_bytes(b'fake graph')
    (stage / 'weights.bin').write_bytes(b'fake weights')
    patch = dict(format='nano_decoder_attention_merged_v1', strength=1.0,
                 weights_file='weights.bin', graph_sha256=verifier.digest(stage/'graph.onnx'),
                 patched_weights_sha256=verifier.digest(stage/'weights.bin'))
    for key in ('adapter_sha256', 'inventory_sha256', 'model_checkpoint_sha256', 'initial_conditionals_sha256'):
        patch[key] = 'a' * 64
    manifest = dict(adapter_patch=patch, graph=dict(path='graph.onnx'),
                    decoder_attention_verification=dict(status='verified', provider_reports={'cpu': 'prior_cpu.json'}))
    path = stage / 'manifest.json'
    path.write_text(json.dumps(manifest))
    before = path.read_bytes()
    arrays = {}
    for case in ('first', 'second'):
        arrays[f'{case}.adapted'] = np.zeros(1, dtype=np.float32)
        for name in verifier.NAMES:
            arrays[f'{case}.input.{name}'] = np.zeros(1, dtype=np.float32)
    np.savez(refdir/'reference.npz', **arrays)
    (refdir/'reference.json').write_text(json.dumps(dict(status='reference_complete',
        reference_sha256=verifier.digest(refdir/'reference.npz'), runtime=patch,
        cases=['first', 'second'], adapter_effect_max_abs={'first': .1, 'second': .2})))

    class BrokenRuntime:
        calls = 0
        def __init__(self, *args, **kwargs):
            self.session = SimpleNamespace(get_provider_options=lambda: {'CUDAExecutionProvider': {'use_tf32': '0'}})
        def estimate(self, **inputs):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError('injected cuDNN plan failure')
            return np.zeros(1, dtype=np.float32)

    monkeypatch.setattr(onnx_staged_runtime, 'MeanflowEstimatorOrtRuntime', BrokenRuntime)
    args = SimpleNamespace(model_dir=stage.parent, reference_dir=refdir, ort_provider='cuda', out=tmp_path/'result.json')
    assert verifier.verify(args) == 1
    result = json.loads(args.out.read_text())
    assert result['status'] == 'runtime_failed'
    assert result['runtime_error']['case'] == 'second'
    assert result['checks'][0]['status'] == 'passed'
    assert len(result['checks']) == 1
    assert path.read_bytes() == before
