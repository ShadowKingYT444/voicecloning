"""Pure tests for the score-free prosody diagnostic helpers."""

import json
import tempfile
import unittest
from pathlib import Path

from prosody_audit import (
    DEFAULT_PITCH_CEILING_HZ,
    DEFAULT_PITCH_FLOOR_HZ,
    _distribution,
    _harmonicity_stats,
    _load_rows,
    _parser,
    _pitch_stats,
    _run_durations,
)


class ProsodyAuditTests(unittest.TestCase):
    def test_distribution_reports_robust_quantiles_without_score(self):
        result = _distribution([1, 2, None, float("nan"), 4])
        self.assertEqual(result["count"], 3)
        self.assertEqual(result["median"], 2.0)
        self.assertIsNone(result.get("acceptance_score"))

    def test_pitch_stats_exposes_voiced_and_jump_coverage(self):
        result = _pitch_stats([100.0, 105.0, 0.0, float("nan"), 220.0], 60.0, 500.0)
        self.assertEqual(result["frame_count"], 5)
        self.assertEqual(result["voiced_frame_count"], 3)
        self.assertEqual(result["finite_frame_count"], 4)
        self.assertIsNotNone(result["large_jump_fraction_abs_gt_4st"])

    def test_run_durations_uses_frame_step(self):
        self.assertEqual(_run_durations([False, True, True, False, True], 0.01), [0.02, 0.01])

    def test_harmonicity_excludes_finite_praat_undefined_sentinel(self):
        result = _harmonicity_stats([-200.0, -35.0, float("nan"), 4.0])
        self.assertEqual(result["raw_finite_frame_count"], 3)
        self.assertEqual(result["undefined_sentinel_frame_count"], 1)
        self.assertEqual(result["defined_frame_count"], 2)
        self.assertEqual(result["harmonicity_db"]["count"], 2)
        self.assertEqual(result["undefined_sentinel_db"], -200.0)

    def test_manifest_requires_provenance_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            path.write_text(json.dumps([{"id": "x", "path": "x.wav", "text": "hello", "source_group": "source"}]), encoding="utf-8")
            rows = _load_rows(path)
            self.assertEqual(rows[0]["source_group"], "source")
            path.write_text(json.dumps([{"id": "x", "path": "x.wav", "text": "hello"}]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source_group"):
                _load_rows(path)

    def test_cli_defaults_document_praat_range(self):
        args = _parser().parse_args(["manifest.json", "--out", "report.json"])
        self.assertEqual(args.pitch_floor_hz, DEFAULT_PITCH_FLOOR_HZ)
        self.assertEqual(args.pitch_ceiling_hz, DEFAULT_PITCH_CEILING_HZ)


if __name__ == "__main__":
    unittest.main()
