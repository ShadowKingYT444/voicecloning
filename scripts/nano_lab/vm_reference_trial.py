"""Fresh, serial CPU reference experiment; never reuse an existing run.

Invoke through bounded_job.py. Save raw watermarked outputs and exact input
hashes; evaluate content/listening separately before selecting a voice.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import time
import traceback


def sha(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def memory():
    status = Path('/proc/self/status').read_text().splitlines()
    return {'rusage_peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            'proc_status': {line.split(':')[0]: line.split(':', 1)[1].strip()
                            for line in status if line.startswith(('VmRSS:', 'VmHWM:', 'VmSize:'))}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report_path = args.out/'report.json'
    if report_path.exists():
        raise SystemExit('Run already exists; choose a fresh output directory')
    config = json.loads(args.config.read_text())
    report = {'config': config, 'config_sha256': sha(args.config), 'runs': [],
              'status': 'started', 'measurements': [], 'implementation_sha256': sha(__file__)}
    def save():
        temporary = report_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2))
        temporary.replace(report_path)
    started = time.perf_counter()
    save()
    try:
        import soundfile as sf
        import torch
        import runtime
        from runtime import NanoEngine
        report['torch_version'] = torch.__version__
        report['runtime_sha256'] = sha(runtime.__file__)
        # Log allocations around each component to diagnose strict AS limits.
        loader = runtime._stream_load_safetensors
        def measured_loader(module, checkpoint, **kwargs):
            report['measurements'].append({'stage': 'before '+checkpoint.name, **memory()})
            save()
            result = loader(module, checkpoint, **kwargs)
            report['measurements'].append({'stage': 'after '+checkpoint.name, **memory()})
            save()
            return result
        runtime._stream_load_safetensors = measured_loader
        engine = NanoEngine.from_pretrained(config['checkpoint'], device='cpu',
                 dtype='fp32', optimized=True, cpu_threads=2,
                 conditionals_path=config.get('conditionals_path'))
        report['load_report'] = engine.load_report
        adapters = {}
        if config.get('adapter_path'):
            if sha(config['adapter_path']) != config['adapter_sha256']:
                raise ValueError('Adapter hash mismatch')
            from adaptation import load_adapter
            adapters = load_adapter(engine.model.t3, config['adapter_path'])
            adapter_scales = {name: module.scaling for name, module in adapters.items()}
        report['measurements'].append({'stage': 'loaded', **memory()})
        conditions = {}
        for reference in config['references']:
            path = Path(reference['path'])
            if sha(path) != reference['sha256']:
                raise ValueError('Reference hash mismatch: '+str(path))
            reference_started = time.perf_counter()
            if reference.get('conditionals_path'):
                if sha(reference['conditionals_path']) != reference['conditionals_sha256']:
                    raise ValueError('Conditionals hash mismatch')
                conditions[reference['id']] = runtime.Conditionals.load(reference['conditionals_path'], map_location='cpu')
            else:
                conditions[reference['id']] = engine.prepare_conditionals(path)
                engine.save_conditionals(args.out/(reference['id']+'.conds.pt'))
            report['measurements'].append({'stage': 'conditioned '+reference['id'], **memory(),
                                           'seconds': time.perf_counter()-reference_started})
            save()
        engine.unload_voice_encoder()
        engine.unload_decoder_reference_encoder()
        engine.unload_tokenizer()
        for trial in config['trials']:
            engine.model.conds = conditions[trial['reference_id']]
            if adapters:
                scale = float(trial['adapter_scale'])
                if not 0 <= scale <= 1:
                    raise ValueError('Trial adapter scale must be between zero and one')
                for name, module in adapters.items():
                    module.scaling = adapter_scales[name]*scale
            result = engine.generate(trial['text'], seed=trial['seed'],
                acoustic_seed=trial['acoustic_seed'], n_cfm_steps=trial['n_cfm_steps'],
                temperature=trial.get('temperature', .8), return_result=True)
            path = args.out/(trial['id']+'.wav')
            sf.write(path, result.audio.squeeze().numpy(), engine.sr, subtype='PCM_24')
            row = {**trial, 'path': str(path.resolve()), 'sha256': sha(path),
                   'duration_s': result.audio.numel()/engine.sr,
                   'elapsed_s': result.elapsed_seconds, 'speech_tokens': result.speech_tokens.numel(),
                   'sample_rate': engine.sr, 'memory': memory()}
            row['speech_tokens_sha256'] = hashlib.sha256(result.speech_tokens.numpy().tobytes()).hexdigest()
            row['speech_token_values'] = result.speech_tokens.tolist()
            report['runs'].append(row)
            print(json.dumps({key: row[key] for key in ('id', 'duration_s', 'elapsed_s', 'speech_tokens')}), flush=True)
            save()
        report['status'] = 'generated_pending_evaluation'
    except BaseException:
        report['status'] = 'failed'
        report['error'] = traceback.format_exc()
        raise
    finally:
        report['elapsed_s'] = time.perf_counter()-started
        report['final_memory'] = memory()
        save()


if __name__ == '__main__':
    main()
