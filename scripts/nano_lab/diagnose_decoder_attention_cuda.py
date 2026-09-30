"""Bounded CUDA option experiment. Never publishes a verification gate."""
import argparse
import json
import os
import resource
import time
from pathlib import Path

import numpy as np

from onnx_staged_runtime import _read_manifest, _provider_request
from verify_decoder_attention_onnx import NAMES, digest, write
from verify_fitted_onnx import compare_array


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--reference-dir', type=Path, required=True)
    p.add_argument('--algo', choices=['HEURISTIC', 'DEFAULT', 'EXHAUSTIVE'], default='DEFAULT')
    p.add_argument('--pad-nc1d', type=int, choices=[0, 1], default=0)
    p.add_argument('--disable-optimization', action='store_true')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    import onnxruntime as ort
    ref = json.loads((a.reference_dir / 'reference.json').read_text())
    if digest(a.reference_dir / 'reference.npz') != ref['reference_sha256']:
        raise ValueError('Reference hash mismatch')
    manifest = _read_manifest(a.model_dir, 'meanflow_estimator', require_verified=False)
    patch = manifest['adapter_patch']
    for key in ('adapter_sha256', 'inventory_sha256', 'model_checkpoint_sha256', 'initial_conditionals_sha256', 'strength'):
        if patch[key] != ref['runtime'][key]:
            raise ValueError('Reference mismatch: ' + key)
    providers, metadata = _provider_request(ort_provider='cuda', cuda_device_id=0,
        gpu_mem_limit_mib=2048, arena_extend_strategy='kSameAsRequested',
        cudnn_conv_algo_search=a.algo, do_copy_in_default_stream=True)
    providers[0][1]['cudnn_conv1d_pad_to_nc1d'] = a.pad_nc1d
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.enable_mem_pattern = False
    options.graph_optimization_level = (ort.GraphOptimizationLevel.ORT_DISABLE_ALL if a.disable_optimization
                                         else ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
    started = time.perf_counter()
    report = dict(status='pending', diagnostic_only=True, publishes_gate=False,
        ort_version=ort.__version__, requested_providers=providers,
        nvidia_tf32_override=os.environ.get('NVIDIA_TF32_OVERRIDE'),
        optimization_disabled=a.disable_optimization, reference_sha256=ref['reference_sha256'],
        patch=patch, checks=[])
    case = None
    try:
        session = ort.InferenceSession(manifest['_graph_path'], options, providers=providers)
        session.disable_fallback()
        if session.get_providers()[0] != 'CUDAExecutionProvider':
            raise RuntimeError('CUDA provider was not selected')
        report['actual_provider_options'] = session.get_provider_options()
        with np.load(a.reference_dir / 'reference.npz', allow_pickle=False) as arrays:
            for case in ref['cases']:
                actual = session.run(None, {k: arrays[f'{case}.input.{k}'] for k in NAMES})[0]
                check = compare_array(actual, arrays[f'{case}.adapted'], atol=3e-4, rtol=3e-4)
                report['checks'].append(dict(case=case, **check))
        report['status'] = 'passed' if all(c['status'] == 'passed' for c in report['checks']) else 'failed'
    except Exception as exc:
        report.update(status='runtime_failed', error=dict(case=case, type=type(exc).__name__, message=str(exc)))
    report.update(elapsed_seconds=time.perf_counter()-started,
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    write(a.out, report)
    print(json.dumps({k: v for k, v in report.items() if k not in ('patch', 'error')}, indent=2))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
