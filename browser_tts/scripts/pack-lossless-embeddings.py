#!/usr/bin/env python3
"""Pack hash-pinned FP16 Gather tables into independently verified 64-row shards.

No ML library, model load, quantization, or objective change. The audited table
offsets/contract apply only to the exact pinned 1520-byte ONNX graph digest.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import os

ROOT = Path(__file__).resolve().parents[2]
REVISION = '4a66d7dab72a9e98f24b515d49a1d7a81632df2e'
GRAPH_SHA = '019d257243774091d78c2ad91c2c0f61e4e442740cb7b3b00b5a89109417b18d'
DATA_SHA = 'bcd7b35ae4f206932e2491cb60b42ebb80f6d8facfdb53ba7d7449ad00a3237b'
TABLES = (('text', 50276, 0), ('speech', 6563, 77223936))
ROW_BYTES = 768 * 2
ROWS_PER_SHARD = 64


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def pack(source_dir, output_dir):
    graph = source_dir / 'onnx/embed_tokens_fp16.onnx'
    data = source_dir / 'onnx/embed_tokens_fp16.onnx_data'
    if graph.stat().st_size != 1520 or file_hash(graph) != GRAPH_SHA:
        raise ValueError('The embedding graph is not the exact audited pinned graph.')
    if data.stat().st_size != 87304704 or file_hash(data) != DATA_SHA:
        raise ValueError('Embedding table size/hash differs from the pinned source.')
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = dict(schema='voice-study.lossless-embedding-shards/v1',
                    revision=REVISION, source_graph_sha256=GRAPH_SHA,
                    source_data_sha256=DATA_SHA, source_data_bytes=87304704,
                    dtype='float16-le', embedding_dim=768, rows_per_shard=ROWS_PER_SHARD,
                    contract=dict(text_tail_excluded=2, sentinel_token=50256, speech_start_token=6561),
                    tables={}, inference_or_quality_verified=False)
    with data.open('rb') as source:
        for name, rows, offset in TABLES:
            source.seek(offset)
            shards = []
            for start in range(0, rows, ROWS_PER_SHARD):
                count = min(ROWS_PER_SHARD, rows - start)
                block = source.read(count * ROW_BYTES)
                if len(block) != count * ROW_BYTES:
                    raise ValueError('Truncated source table.')
                filename = f'{name}-{start:05d}.fp16'
                path = output_dir / filename
                if path.exists() and path.read_bytes() != block:
                    raise ValueError(f'Refusing to replace a different shard: {path}')
                if not path.exists():
                    with tempfile.NamedTemporaryFile(dir=output_dir, delete=False) as target:
                        temporary = Path(target.name)
                        target.write(block); target.flush(); os.fsync(target.fileno())
                    os.replace(temporary, path)
                shards.append(dict(file=filename, first_row=start, rows=count, bytes=len(block),
                                   sha256=hashlib.sha256(block).hexdigest()))
            manifest['tables'][name] = dict(rows=rows, source_offset=offset, shards=shards)
    # Canonical, timestamp-free manifest: its hash binds every row to the source.
    encoded = (json.dumps(manifest, sort_keys=True, separators=(',', ':')) + '\n').encode()
    (output_dir / 'manifest.json').write_bytes(encoded)
    return dict(manifest_sha256=hashlib.sha256(encoded).hexdigest(),
                shard_bytes=sum(s['bytes'] for t in manifest['tables'].values() for s in t['shards']),
                manifest_bytes=len(encoded), shards=sum(len(t['shards']) for t in manifest['tables'].values()),
                source_data_sha256=DATA_SHA, bitwise_copy=True, inference_or_quality_verified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=ROOT / 'models/chatterbox-nano-browser')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'browser_tts/public/experiments/embedding_lossless')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    report = pack(args.source_dir, args.output_dir)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
