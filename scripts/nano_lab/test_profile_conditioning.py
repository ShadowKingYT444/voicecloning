"""Pure cache-integrity tests.  These tests do not import Torch or ORT."""

from pathlib import Path
import hashlib
import tempfile
import unittest

from profile_conditioning import resolve_conditioning_cache


class ConditioningCacheTests(unittest.TestCase):
    def test_required_missing_cache_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError, "required conditioning cache not found"):
                resolve_conditioning_cache(
                    {
                        "conditioning_cache": "missing.conds.pt",
                        "conditioning_cache_required": True,
                    },
                    root=tmp,
                )

    def test_expected_hash_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "voice.conds.pt"
            path.write_bytes(b"cache-v1")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                resolve_conditioning_cache(
                    {
                        "conditioning_cache": str(path),
                        "conditioning_cache_sha256": "0" * 64,
                    },
                    root=tmp,
                )

    def test_expected_hash_accepts_matching_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "voice.conds.pt"
            path.write_bytes(b"cache-v1")
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            result = resolve_conditioning_cache(
                {"conditioning_cache": path.name, "conditioning_cache_sha256": expected},
                root=tmp,
            )
        self.assertEqual(result["path"], path.resolve())
        self.assertTrue(result["exists"])
        self.assertEqual(result["sha256"], expected)
        self.assertFalse(result["legacy_fallback"])

    def test_legacy_native_profile_can_auto_prepare_missing_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = resolve_conditioning_cache(
                {"conditioning_cache": "future.conds.pt"},
                root=tmp,
            )
        self.assertEqual(result["path"], (Path(tmp) / "future.conds.pt").resolve())
        self.assertFalse(result["exists"])
        self.assertTrue(result["legacy_fallback"])
        self.assertFalse(result["required"])


if __name__ == "__main__":
    unittest.main()
