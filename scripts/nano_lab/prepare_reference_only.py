"""Prepare native Nano reference conditions without T3/flow/vocoder weights.

Uses upstream prepare_conditionals and embed_ref unchanged. Keep this stage
separate from synthesis on hosts where the combined working set cannot fit.
Run through bounded_job.py; outputs are speaker-specific, not fitted adapters.
"""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time
from types import SimpleNamespace


def load_reference_model(checkpoint):
    """Return the native reference API and its strict component load report."""
    import torch
    import runtime
    from runtime import (S3Gen, VoiceEncoder, ChatterboxTurboTTS, _nano_hp,
                         _stream_load_safetensors, _repair_runtime_buffers)
    torch.set_num_threads(2)
    checkpoint = Path(checkpoint)
    class ReferenceS3Gen(S3Gen):
        @property
        def dtype(self):
            return torch.float32
    device = torch.device('cpu')
    with torch.device('meta'):
        ve = VoiceEncoder()
        s3gen = ReferenceS3Gen(meanflow=True)
    s3gen.flow = None
    s3gen.mel2wav = None
    ve.to_empty(device=device).eval()
    s3gen.to_empty(device=device).eval()
    ve_report = _stream_load_safetensors(ve, checkpoint/'ve.safetensors', device=device)
    s3_report = _stream_load_safetensors(s3gen, checkpoint/'s3gen_meanflow.safetensors',
                                       device=device, skip_prefixes=('flow', 'mel2wav'))
    _repair_runtime_buffers(s3gen, device, torch.float32, s3_report['ignored_missing_tensors'])
    # No synthesis model or watermarker is needed to call the unchanged
    # reference-conditioning method. Only t3.hp is accessed by that method.
    model = object.__new__(ChatterboxTurboTTS)
    model.t3 = SimpleNamespace(hp=_nano_hp())
    model.s3gen = s3gen
    model.ve = ve
    model.device = 'cpu'
    model.conds = None
    return model, {'ve': ve_report, 's3gen': s3_report}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    import torch
    import runtime
    config = json.loads(args.config.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out/'manifest.json').exists():
        raise SystemExit('Reference preparation already exists; use a fresh directory')
    def sha(path):
        with Path(path).open('rb') as file:
            return hashlib.file_digest(file, 'sha256').hexdigest()
    model, load_report = load_reference_model(config['checkpoint'])
    report = {'role': 'native_reference_conditioning_only',
              'upstream_conditioning_sha256': sha(runtime.ChatterboxTurboTTS.prepare_conditionals.__code__.co_filename),
              'implementation_sha256': sha(__file__), 'runtime_sha256': sha(runtime.__file__),
              'load_report': load_report, 'references': []}
    for reference in config['references']:
        path = Path(reference['path'])
        if sha(path) != reference['sha256']:
            raise ValueError('Reference hash mismatch: '+str(path))
        started = time.perf_counter()
        with torch.inference_mode():
            model.prepare_conditionals(str(path), exaggeration=0., norm_loudness=True)
        output = args.out/(reference['id']+'.conds.pt')
        model.conds.save(output)
        row = {**reference, 'conditionals_path': str(output.resolve()),
               'conditionals_sha256': sha(output), 'seconds': time.perf_counter()-started}
        report['references'].append(row)
        report['peak_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        (args.out/'manifest.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
