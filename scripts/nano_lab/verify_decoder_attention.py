"""Compare factorized and folded attention adapters without two model copies."""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import fit_decoder_attention as fit
import fit_decoder_embedding as emb
import fit_decoder_projection as projection
from decoder_attention import configure_decoder_attention


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--prepare-dir', type=Path, default=fit.DEFAULT_PREPARE_DIR)
    parser.add_argument('--conditionals', type=Path, default=fit.DEFAULT_INITIAL_CONDITIONALS)
    parser.add_argument('--strict-fp32', action='store_true')
    args = parser.parse_args()
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    precision_before = dict(matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                            cudnn_tf32=torch.backends.cudnn.allow_tf32)
    if args.strict_fp32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    started = time.perf_counter()
    manifest, rows, prepared_path, _, _ = emb._validate_prepare_inputs(args.prepare_dir.resolve())
    prepared, initial, _ = projection._load_conditionals_pair(torch, prepared_path, args.conditionals.resolve())
    inventory, _ = fit._inventory(fit.DEFAULT_INVENTORY)
    checkpoint = torch.load(args.adapter, weights_only=True, map_location='cpu')
    fit.validate_attention_checkpoint(checkpoint, inventory)
    device = torch.device('cuda')
    encoder, estimator, _ = emb._stream_s3gen_modules(torch, fit.DEFAULT_MODEL_DIR, device)
    selected = [next(r for r in rows if r['split'] == 'train')] + [r for r in rows if r['split'] == 'valid']
    states = [emb._prepare_flow_row(torch, encoder, prepared, row, device) for row in selected]
    noise = {state['id']: emb._fixed_noise(torch, state, seed=state['seed']) for state in states}
    embedding = initial['gen']['embedding'].to(device)
    baseline = fit._collect_predictions(torch, estimator, states, embedding, noise, steps=2)
    adapters, _ = fit._install_attention_adapters(torch, estimator, inventory)
    with torch.no_grad():
        for name, adapter in adapters.items():
            adapter.down.weight.copy_(checkpoint['targets'][name]['down_weight'])
            adapter.up.weight.copy_(checkpoint['targets'][name]['up_weight'])
    factorized = fit._collect_predictions(torch, estimator, states, embedding, noise, steps=2)
    for name, adapter in adapters.items():
        local = 'estimator.' + name[len(fit.TARGET_PREFIX):-len('.weight')]
        parent, leaf = fit._locate_parent(estimator, local)
        fit._set_child(parent, leaf, adapter.base)
    del adapters
    model = SimpleNamespace(s3gen=SimpleNamespace(flow=SimpleNamespace(decoder=SimpleNamespace(estimator=estimator.estimator))))
    runtime = configure_decoder_attention(model, args.adapter, conditionals_path=args.conditionals)
    folded = fit._collect_predictions(torch, estimator, states, embedding, noise, steps=2)
    checks = []
    for name in factorized:
        delta = (factorized[name] - folded[name]).float()
        checks.append(dict(id=name, max_abs=float(delta.abs().max()), rmse=float(delta.square().mean().sqrt())))
    repeated = configure_decoder_attention(model, args.adapter, conditionals_path=args.conditionals)
    twice = fit._collect_predictions(torch, estimator, states, embedding, noise, steps=2)
    repeat_exact = all(torch.equal(folded[name], twice[name]) for name in folded)
    restored = configure_decoder_attention(model, None)
    restored_outputs = fit._collect_predictions(torch, estimator, states, embedding, noise, steps=2)
    restore_exact = all(torch.equal(baseline[name], restored_outputs[name]) for name in baseline)
    passed = all(r['max_abs'] <= 1e-4 and r['rmse'] <= 1e-5 for r in checks) and repeat_exact and restore_exact
    report = dict(status='passed' if passed else 'failed', scope='Three source-token flow reconstructions; no T3 or vocoder',
                  precision_before=precision_before, strict_fp32=args.strict_fp32,
                  precision_used=dict(matmul_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32),
                  adapter_sha256=fit._sha256(args.adapter), checks=checks, atol=1e-4, rmse_limit=1e-5,
                  repeated_fold_exact=repeat_exact, restore_exact=restore_exact, runtime=runtime,
                  idempotent_metadata=repeated, restore_metadata=restored, peak_rss_bytes=fit._peak_rss_bytes(),
                  elapsed_seconds=time.perf_counter()-started)
    fit._write_json(args.out, report)
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
