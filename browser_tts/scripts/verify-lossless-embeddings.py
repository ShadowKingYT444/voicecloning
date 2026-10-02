#!/usr/bin/env python3
"""Compare every packed FP16 row to the pinned CPU ONNX Gather/Cast output."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from importlib.util import spec_from_file_location, module_from_spec
spec = spec_from_file_location('pack', Path(__file__).with_name('pack-lossless-embeddings.py'))
pack = module_from_spec(spec); spec.loader.exec_module(pack)

EXPECTED_MANIFEST_SHA256 = 'a244131d73e89e0b39e06522535eb1267340ff7c27306290f7ec4cca890198da'
EXPECTED_ROWS = {'text': 50276, 'speech': 6563}
EXPECTED_VALUES = {'text': 38611968, 'speech': 5040384}
EXPECTED_TOTAL_VALUES = 43652352


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=pack.ROOT / 'models/chatterbox-nano-browser')
    parser.add_argument('--shard-dir', type=Path, default=pack.ROOT / 'browser_tts/public/experiments/embedding_lossless')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import onnxruntime as ort
    if ort.__version__ != '1.29.0': raise ValueError('Use the pinned CPU ORT 1.29.0.')
    graph = args.source_dir/'onnx/embed_tokens_fp16.onnx'
    data = graph.with_name('embed_tokens_fp16.onnx_data')
    if pack.file_hash(graph) != pack.GRAPH_SHA or pack.file_hash(data) != pack.DATA_SHA:
        raise ValueError('Source embedding hashes changed.')
    manifest_bytes = (args.shard_dir/'manifest.json').read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != EXPECTED_MANIFEST_SHA256:
        raise ValueError('Lossless embedding manifest SHA-256 differs from the pinned export.')
    manifest = json.loads(manifest_bytes)
    options = ort.SessionOptions(); options.intra_op_num_threads = 1; options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    started = time.monotonic()
    session = ort.InferenceSession(str(graph), sess_options=options, providers=['CPUExecutionProvider'])
    compared = {}; byte_equal = True
    for table in ('text', 'speech'):
        if manifest['tables'][table]['rows'] != EXPECTED_ROWS[table]:
            raise ValueError(f'Lossless {table} table row count differs from the pinned export.')
        count = 0
        for shard in manifest['tables'][table]['shards']:
            raw = (args.shard_dir/shard['file']).read_bytes()
            if len(raw) != shard['bytes'] or hashlib.sha256(raw).hexdigest() != shard['sha256']:
                raise ValueError('Packed shard bytes changed.')
            expected = np.frombuffer(raw, dtype='<f2').reshape(shard['rows'], 768).astype(np.float32)
            if not np.isfinite(expected).all(): raise ValueError('Source embeddings contain non-finite values.')
            first = shard['first_row']
            if table == 'text':
                ids = np.asarray([*range(first, first+shard['rows']), 50256, 50256], dtype=np.int64)[None]
                actual = session.run(None, {'input_ids': ids})[0][0, :-2]
            else:
                # The final two IDs are speech-table lookups in the hybrid graph.
                actual = np.empty_like(expected)
                for row in range(0, shard['rows'], 2):
                    ids = np.asarray(list(range(first+row, first+min(row+2, shard['rows']))), dtype=np.int64)[None]
                    actual[row:row+ids.shape[1]] = session.run(None, {'input_ids': ids})[0][0]
            equal = np.array_equal(actual.view(np.uint32), expected.view(np.uint32))
            byte_equal = byte_equal and bool(equal); count += expected.size
            if not equal: raise ValueError(f'CPU bitwise lookup mismatch in {shard["file"]}.')
        if count != EXPECTED_VALUES[table]:
            raise ValueError(f'Lossless {table} lookup compared {count} values; expected {EXPECTED_VALUES[table]}.')
        compared[table] = count
    total_values = sum(compared.values())
    if total_values != EXPECTED_TOTAL_VALUES:
        raise ValueError(f'Lossless lookup compared {total_values} values; expected {EXPECTED_TOTAL_VALUES}.')
    result = dict(provider='CPUExecutionProvider', onnxruntime_version=ort.__version__,
                  source_graph_sha256=pack.GRAPH_SHA, source_data_sha256=pack.DATA_SHA,
                  compared_float32_values=compared, total_values=total_values,
                  bitwise_equal=byte_equal, elapsed_seconds=time.monotonic()-started,
                  browser_or_audio_equivalence_verified=False, voice_quality_verified=False)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
