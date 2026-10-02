"""CPU safetensors reader with no whole-file mapping or state dictionary.

Used explicitly on hosts with strict address-space limits. Tensor bytes are
read directly into one writable buffer; torch views that buffer without a copy.
Only little-endian hosts and the checkpoint dtypes used by Nano are supported.
"""
from __future__ import annotations

import json
import math
import struct
import sys
from pathlib import Path

DTYPES = {'F32': ('float32', 4), 'I64': ('int64', 8),
          'F16': ('float16', 2), 'BF16': ('bfloat16', 2), 'BOOL': ('bool', 1)}


class CheckpointReader:
    def __init__(self, path):
        if sys.byteorder != 'little':
            raise ValueError('CheckpointReader requires a little-endian host')
        self.path = Path(path)
        self.file = self.path.open('rb', buffering=0)
        try:
            prefix = self.file.read(8)
            if len(prefix) != 8:
                raise ValueError('Truncated safetensors header length')
            length, = struct.unpack('<Q', prefix)
            if not 2 <= length <= 16 * 2**20:
                raise ValueError('Invalid or oversized safetensors header')
            header = self.file.read(length)
            if len(header) != length:
                raise ValueError('Truncated safetensors header')
            def unique(items):
                result = {}
                for key, value in items:
                    if key in result:
                        raise ValueError('Duplicate safetensors header key')
                    result[key] = value
                return result
            self.header = json.loads(header, object_pairs_hook=unique)
            self.start = 8 + length
            size = self.path.stat().st_size - self.start
            regions = []
            for name, entry in self.header.items():
                if name == '__metadata__':
                    continue
                dtype, shape, offsets = entry['dtype'], entry['shape'], entry['data_offsets']
                if dtype not in DTYPES:
                    raise ValueError(f'Unsupported checkpoint dtype {dtype}')
                if not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape):
                    raise ValueError(f'Invalid shape for {name}')
                if len(offsets) != 2 or any(type(n) is not int for n in offsets):
                    raise ValueError(f'Invalid offsets for {name}')
                lo, hi = offsets
                if not 0 <= lo <= hi <= size or hi-lo != math.prod(shape)*DTYPES[dtype][1]:
                    raise ValueError(f'Invalid tensor region for {name}')
                regions.append((lo, hi))
            end = 0
            for lo, hi in sorted(regions):
                if lo != end:
                    raise ValueError('Checkpoint tensor regions overlap or contain gaps')
                end = hi
            if end != size:
                raise ValueError('Checkpoint contains unclaimed tensor data')
        except BaseException:
            self.file.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.file.close()

    def keys(self):
        return sorted(k for k in self.header if k != '__metadata__')

    def get_tensor(self, name):
        import torch
        entry = self.header[name]
        lo, hi = entry['data_offsets']
        dtype = getattr(torch, DTYPES[entry['dtype']][0])
        if hi == lo:
            return torch.empty(entry['shape'], dtype=dtype)
        storage = bytearray(hi-lo)
        self.file.seek(self.start+lo)
        view = memoryview(storage)
        position = 0
        while position < len(storage):
            count = self.file.readinto(view[position:])
            if not count:
                raise ValueError(f'Truncated tensor {name}')
            position += count
        return torch.frombuffer(storage, dtype=dtype).reshape(entry['shape'])
