"""Pure checks for the bounded graph fingerprint cache."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import file_fingerprint as fingerprints


class FileFingerprintTests(unittest.TestCase):
    def setUp(self) -> None:
        fingerprints.clear_cache()

    def tearDown(self) -> None:
        fingerprints.clear_cache()

    def test_unchanged_file_uses_cached_digest_without_second_stream(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "graph.onnx"
            path.write_bytes(b"graph bytes")
            with patch.object(fingerprints, "_stream_sha256", wraps=fingerprints._stream_sha256) as stream:
                first = fingerprints.fingerprint_file(path)
                second = fingerprints.fingerprint_file(path)
            self.assertEqual(first, second)
            stream.assert_called_once()
            self.assertEqual(fingerprints.cache_info(), {"size": 1, "max_entries": 64})

    def test_metadata_change_invalidates_digest_and_restreams(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "graph.onnx"
            path.write_bytes(b"graph bytes")
            with patch.object(fingerprints, "_stream_sha256", wraps=fingerprints._stream_sha256) as stream:
                first = fingerprints.fingerprint_file(path)
                path.write_bytes(b"changed bytes")
                second = fingerprints.fingerprint_file(path)
            self.assertNotEqual(first, second)
            self.assertEqual(stream.call_count, 2)
            self.assertEqual(fingerprints.cache_info()["size"], 2)

    def test_metadata_change_during_read_is_rejected_and_not_cached(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "graph.onnx"
            path.write_bytes(b"graph bytes")
            stable = fingerprints._stat_signature(path)
            changed = replace(stable, size=stable.size + 1)
            with patch.object(fingerprints, "_stat_signature", side_effect=[stable, changed]):
                with self.assertRaises(fingerprints.FileChangedDuringHashError):
                    fingerprints.fingerprint_file(path)
            self.assertEqual(fingerprints.cache_info()["size"], 0)

    def test_cache_is_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for index in range(fingerprints.MAX_CACHE_ENTRIES + 3):
                path = Path(tmp) / f"graph-{index}.onnx"
                path.write_bytes(str(index).encode("ascii"))
                fingerprints.fingerprint_file(path)
            self.assertEqual(fingerprints.cache_info()["size"], fingerprints.MAX_CACHE_ENTRIES)


if __name__ == "__main__":
    unittest.main()
