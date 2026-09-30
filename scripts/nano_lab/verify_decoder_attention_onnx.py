"""Separate Torch reference and pure ORT checks for folded acoustic adapters.

Only deterministic estimator calls are verified here. Full waveform and T3
parity require separate end-to-end measurements.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
NAMES = ('x', 'mask', 'mu', 't', 'speaker_embedding', 'cond', 'r')


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)


def peak():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def reference(args):
    import torch
    from onnx_staged import _load_meanflow_estimator, _load_case_examples
    from decoder_attention import configure_decoder_attention

    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError('Reference output must be new: ' + str(output))
    stage = args.base_model_dir.resolve() / 'meanflow_estimator'
    manifest = json.loads((stage / 'manifest.json').read_text())
    if manifest.get('adapter_patch'):
        raise ValueError('Reference input must be a base meanflow stage')
    cases = {name: inputs for name, (inputs, _) in _load_case_examples(stage, manifest).items()}
    rng = np.random.default_rng(20260929)
    for length in (512, 768):
        inputs = {name: rng.normal(size=(1, 80, length)).astype(np.float32) for name in ('x', 'mu', 'cond')}
        inputs.update(mask=np.ones((1, 1, length), dtype=np.float32),
                      speaker_embedding=rng.normal(size=(1, 192)).astype(np.float32),
                      t=np.array([.25], dtype=np.float32), r=np.array([.75], dtype=np.float32))
        cases[f'mel{length}'] = inputs
    started = time.perf_counter()
    device = torch.device(args.device)
    model, _ = _load_meanflow_estimator(ROOT / 'models/chatterbox-nano', device)
    model.eval()
    wrapper = SimpleNamespace(s3gen=SimpleNamespace(flow=SimpleNamespace(decoder=SimpleNamespace(estimator=model.estimator))))
    arrays = {}
    def collect(label):
        with torch.inference_mode():
            for name, inputs in cases.items():
                tensors = [torch.from_numpy(inputs[key]).to(device) for key in NAMES]
                prediction = model(*tensors).detach().cpu().numpy()
                if not np.isfinite(prediction).all():
                    raise ValueError('Nonfinite Torch estimator output')
                arrays[f'{name}.{label}'] = prediction
                for key in NAMES:
                    arrays[f'{name}.input.{key}'] = inputs[key]
    collect('base')
    runtime = configure_decoder_attention(wrapper, args.adapter, conditionals_path=args.conditionals)
    collect('adapted')
    effects = {name: float(np.max(np.abs(arrays[f'{name}.adapted'] - arrays[f'{name}.base']))) for name in cases}
    if not all(value > 1e-6 for value in effects.values()):
        raise ValueError('Adapter has no measurable effect in a reference case')
    output.mkdir(parents=True)
    np.savez(output / 'reference.npz', **arrays)
    report = dict(format='nano_decoder_attention_onnx_reference_v1', status='reference_complete',
                  cases=list(cases), inputs=list(NAMES), adapter_effect_max_abs=effects,
                  base_stage_manifest_sha256=digest(stage / 'manifest.json'),
                  reference_sha256=digest(output / 'reference.npz'), runtime=runtime,
                  precision=dict(matmul_tf32=False, cudnn_tf32=False), device=str(device),
                  peak_rss_bytes=peak(), elapsed_seconds=time.perf_counter() - started)
    write(output / 'reference.json', report)
    print(json.dumps(report, indent=2))


def verify(args):
    import onnxruntime
    from onnx_staged_runtime import MeanflowEstimatorOrtRuntime
    from verify_fitted_onnx import compare_array

    reference_dir = args.reference_dir.resolve()
    ref = json.loads((reference_dir / 'reference.json').read_text())
    if ref.get('status') != 'reference_complete' or digest(reference_dir / 'reference.npz') != ref['reference_sha256']:
        raise ValueError('Reference status/hash mismatch')
    stage = args.model_dir.resolve() / 'meanflow_estimator'
    manifest_path = stage / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    patch = manifest.get('adapter_patch') or {}
    if patch.get('format') != 'nano_decoder_attention_merged_v1':
        raise ValueError('Expected folded acoustic attention stage')
    for key in ('adapter_sha256', 'inventory_sha256', 'model_checkpoint_sha256', 'initial_conditionals_sha256'):
        if patch.get(key) != ref['runtime'].get(key):
            raise ValueError('Reference/stage mismatch: ' + key)
    if patch.get('strength') != ref['runtime'].get('strength'):
        raise ValueError('Reference/stage strength mismatch')
    graph = Path(manifest['graph']['path'])
    if not graph.is_absolute():
        graph = stage / graph
    weights = stage / patch['weights_file']
    graph_sha, weights_sha = digest(graph), digest(weights)
    if graph_sha != patch['graph_sha256'] or weights_sha != patch['patched_weights_sha256']:
        raise ValueError('Patched stage bytes differ from provenance')
    started = time.perf_counter()
    runtime = MeanflowEstimatorOrtRuntime(args.model_dir, intra_op_num_threads=2,
                                         ort_provider=args.ort_provider, verification_only=True)
    checks = []
    runtime_error = None
    with np.load(reference_dir / 'reference.npz', allow_pickle=False) as arrays:
        for name in ref['cases']:
            inputs = {key: arrays[f'{name}.input.{key}'] for key in NAMES}
            try:
                actual = runtime.estimate(**inputs)
            except Exception as exc:
                runtime_error = dict(case=name, type=type(exc).__name__, message=str(exc))
                break
            check = compare_array(actual, arrays[f'{name}.adapted'], atol=3e-4, rtol=3e-4)
            check.update(case=name, adapter_effect_max_abs=ref['adapter_effect_max_abs'][name])
            checks.append(check)
    passed = runtime_error is None and len(checks) == len(ref['cases']) and all(row['status'] == 'passed' for row in checks)
    # Recheck immutable graph weights before publishing the verification gate.
    if digest(graph) != graph_sha or digest(weights) != weights_sha:
        raise ValueError('Graph or weights changed during verification')
    report = dict(status='runtime_failed' if runtime_error else ('verified' if passed else 'failed'), experimental=True,
                  scope='Deterministic estimator calls at mel lengths 8,64,256,512,768; no waveform or T3 claim',
                  graph_sha256=graph_sha, weights_sha256=weights_sha,
                  adapter_sha256=patch['adapter_sha256'], strength=patch['strength'],
                  reference_sha256=ref['reference_sha256'], atol=3e-4, rtol=3e-4,
                  ort_provider=args.ort_provider, onnxruntime_version=onnxruntime.__version__,
                  nvidia_tf32_override=os.environ.get('NVIDIA_TF32_OVERRIDE'),
                  provider_options=runtime.session.get_provider_options(), checks=checks, runtime_error=runtime_error,
                  peak_rss_bytes=peak(), elapsed_seconds=time.perf_counter() - started)
    write(args.out, report)
    if passed:
        previous = manifest.get('decoder_attention_verification', {})
        reports = previous.get('provider_reports', {}) if previous.get('weights_sha256') == weights_sha else {}
        reports = {**reports, args.ort_provider: str(args.out.resolve())}
        manifest['decoder_attention_verification'] = {**report, 'provider_reports': reports}
        write(manifest_path, manifest)
        root_path = args.model_dir / 'manifest.json'
        if root_path.exists():
            root_manifest = json.loads(root_path.read_text())
            root_manifest.setdefault('stages', {}).setdefault('meanflow_estimator', {})['decoder_attention_verification'] = manifest['decoder_attention_verification']
            root_manifest['decoder_attention_verification'] = manifest['decoder_attention_verification']
            write(root_path, root_manifest)
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    ref = sub.add_parser('reference')
    ref.add_argument('--adapter', type=Path, required=True)
    ref.add_argument('--conditionals', type=Path, default=ROOT / 'artifacts/nano_lab/decoder_embedding_fit/conditionals.pt')
    ref.add_argument('--base-model-dir', type=Path, default=ROOT / 'artifacts/nano_lab/onnx_staged')
    ref.add_argument('--output-dir', type=Path, required=True)
    ref.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    ort = sub.add_parser('verify')
    ort.add_argument('--model-dir', type=Path, required=True)
    ort.add_argument('--reference-dir', type=Path, required=True)
    ort.add_argument('--out', type=Path, required=True)
    ort.add_argument('--ort-provider', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    return reference(args) if args.command == 'reference' else verify(args)


if __name__ == '__main__':
    raise SystemExit(main())
