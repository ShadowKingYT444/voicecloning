"""Exact, tensor-by-tensor parity against the official safetensors reader.

Run separately through bounded_job.py. No model modules or state dict copies
are allocated. This verifies checkpoint I/O, not synthesis or ONNX parity.
"""
import argparse
import hashlib
import json
from pathlib import Path
import resource


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    import torch
    from safetensors import safe_open
    from checkpoint_reader import CheckpointReader
    torch.set_num_threads(2)
    report = {'role': 'exact_checkpoint_reader_parity', 'checkpoints': []}
    for name in ('ve.safetensors', 't3_nano_v1.safetensors', 's3gen_meanflow.safetensors'):
        path = args.checkpoint_dir/name
        count = 0
        with safe_open(path, framework='pt', device='cpu') as official, CheckpointReader(path) as streamed:
            if set(official.keys()) != set(streamed.keys()):
                raise ValueError('Checkpoint key mismatch')
            for key in official.keys():
                left, right = official.get_tensor(key), streamed.get_tensor(key)
                if left.shape != right.shape or left.dtype != right.dtype or not torch.equal(left, right):
                    raise ValueError('Checkpoint tensor mismatch: '+key)
                count += 1
                del left, right
        with path.open('rb') as file:
            digest = hashlib.file_digest(file, 'sha256').hexdigest()
        report['checkpoints'].append({'file': name, 'sha256': digest, 'tensors': count, 'exact': True})
        report['peak_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(json.dumps(report['checkpoints'][-1]), flush=True)


if __name__ == '__main__':
    main()
