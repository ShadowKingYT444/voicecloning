#!/usr/bin/env python3
"""Download and hash-check the pinned browser conversion with bounded buffers."""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
import tempfile
import urllib.request

REVISION = '4a66d7dab72a9e98f24b515d49a1d7a81632df2e'
ROOT = f'https://huggingface.co/owensong/chatterbox-nano-ONNX/resolve/{REVISION}/'
SUMS_SHA256 = 'ecd0b0ee1fde0f9342564fda72fe1ebb4907baa6c9d013c09052bcad5220dcf4'
REPO = Path(__file__).resolve().parents[2]


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while data := stream.read(1024 * 1024):
            digest.update(data)
    return digest.hexdigest()


def download(path, expected, destination):
    if destination.is_file() and file_hash(destination) == expected:
        return {'path': path, 'sha256': expected, 'bytes': destination.stat().st_size, 'reused': True}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        digest = hashlib.sha256()
        with urllib.request.urlopen(ROOT + path, timeout=90) as response, tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as output:
            temporary = Path(output.name)
            while data := response.read(1024 * 1024):
                digest.update(data); output.write(data)
            output.flush(); os.fsync(output.fileno())
        if digest.hexdigest() != expected:
            raise ValueError(f'Hash mismatch for {path}')
        os.replace(temporary, destination)
        print(f'{path}: verified {destination.stat().st_size} bytes', flush=True)
        return {'path': path, 'sha256': expected, 'bytes': destination.stat().st_size, 'reused': False}
    finally:
        if temporary and temporary.exists():
            temporary.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, default=REPO / 'models/chatterbox-nano-browser')
    parser.add_argument('--stage-runtime', action='store_true', help='Stage verified browser runtime assets under public/models/chatterbox-nano-browser.')
    subset = parser.add_mutually_exclusive_group()
    subset.add_argument('--encoder-only', action='store_true')
    subset.add_argument('--embedding-only', action='store_true')
    subset.add_argument('--runtime-only', action='store_true', help='Three browser graphs and tokenizer/config files; omit the offline encoder.')
    args = parser.parse_args()
    if args.stage_runtime and (args.encoder_only or args.embedding_only):
        parser.error('--stage-runtime requires the full or --runtime-only download.')
    with urllib.request.urlopen(ROOT + 'SHA256SUMS', timeout=30) as response:
        sums = response.read(65537)
    if len(sums) > 65536 or hashlib.sha256(sums).hexdigest() != SUMS_SHA256:
        raise ValueError('Pinned SHA256SUMS verification failed')
    expected = {}
    for line in sums.decode().splitlines():
        digest, path = line.split(maxsplit=1)
        expected[path.lstrip('*')] = digest
    names = ['speech_encoder_q4f16'] if args.encoder_only else ['embed_tokens_fp16'] if args.embedding_only else ['embed_tokens_fp16', 'language_model_q4f16', 'conditional_decoder_q4'] if args.runtime_only else ['embed_tokens_fp16', 'speech_encoder_q4f16', 'language_model_q4f16', 'conditional_decoder_q4']
    paths = [f'onnx/{name}.onnx{suffix}' for name in names for suffix in ['', '_data']]
    if not args.encoder_only and not args.embedding_only:
        paths += ['tokenizer.json', 'tokenizer_config.json', 'config.json', 'generation_config.json']
    files = [download(path, expected[path], args.model_dir / path) for path in paths]
    args.model_dir.mkdir(parents=True, exist_ok=True)
    (args.model_dir / 'SHA256SUMS').write_bytes(sums)
    (args.model_dir / 'download_manifest.json').write_text(json.dumps({'repo_id': 'owensong/chatterbox-nano-ONNX', 'revision': REVISION, 'sha256sums_sha256': SUMS_SHA256, 'files': files}, indent=2) + '\n')
    if args.stage_runtime:
        destination_root = REPO / 'browser_tts/public/models/chatterbox-nano-browser'
        staged = []
        for item in files:
            if 'speech_encoder_' in item['path']:
                continue
            source = args.model_dir / item['path']
            target = destination_root / item['path']
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as output:
                temporary = Path(output.name)
            try:
                temporary.unlink()
                try:
                    os.link(source, temporary)
                except OSError:
                    shutil.copyfile(source, temporary)
                if file_hash(temporary) != item['sha256']:
                    raise ValueError(f'Staged asset hash mismatch: {item["path"]}')
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
            staged.append(item)
        (destination_root / 'download_manifest.json').write_text(json.dumps({'repo_id': 'owensong/chatterbox-nano-ONNX', 'revision': REVISION, 'files': staged}, indent=2) + '\n')
        print(f'Staged {len(staged)} verified runtime assets at {destination_root}', flush=True)


if __name__ == '__main__':
    main()
