"""Exact reader parity against safetensors, plus damaged-file rejection."""
import json
import struct
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file, load_file
from checkpoint_reader import CheckpointReader


class ReaderTests(unittest.TestCase):
    def test_all_supported_dtypes_shapes_and_storage_lifetime(self):
        tensors = {'matrix': torch.randn(31, 23), 'scalar': torch.tensor(1.75),
                   'integer': torch.tensor([-900, 0, 4321]),
                   'half': torch.randn(20).half(), 'bfloat': torch.randn(20).bfloat16(),
                   'bool': torch.tensor([True, False]), 'empty': torch.empty(0, 3)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'fixture.safetensors'
            save_file(tensors, path, metadata={'role': 'reader parity'})
            expected = load_file(path)
            with CheckpointReader(path) as reader:
                actual = {name: reader.get_tensor(name) for name in reader.keys()}
            self.assertEqual(set(actual), set(expected))
            for name in actual:
                self.assertEqual(actual[name].dtype, expected[name].dtype)
                self.assertEqual(actual[name].shape, expected[name].shape)
                self.assertTrue(torch.equal(actual[name], expected[name]), name)

    def test_truncation_and_overlapping_regions_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'broken.safetensors'
            save_file({'x': torch.ones(3)}, path)
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaises(ValueError):
                CheckpointReader(path)
            header = json.dumps({'x': {'dtype': 'F32', 'shape': [2], 'data_offsets': [0, 8]},
                                 'y': {'dtype': 'F32', 'shape': [2], 'data_offsets': [4, 12]}}).encode()
            path.write_bytes(struct.pack('<Q', len(header))+header+b'\0'*12)
            with self.assertRaises(ValueError):
                CheckpointReader(path)


if __name__ == '__main__':
    unittest.main()
