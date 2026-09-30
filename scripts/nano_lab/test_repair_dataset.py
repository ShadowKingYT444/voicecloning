"""Candidate-boundary and admission regression tests without model loading."""
import hashlib
import json
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace

from repair_dataset import (
    ASMR_FAMILY,
    DECODE_ALIGNMENT_TOLERANCE_S,
    DEFAULT_REFERENCES,
    _candidate_windows,
    _parser,
    _protected_intervals,
    _reference_overlap,
    _source_from_references,
    _validate_decode_source,
    rebase_proposal,
    stage_stage,
)


def segment(start, end, text):
    return {"start_s": start, "end_s": end, "text": text,
            "words": [{"start_s": start, "end_s": end, "word": text, "probability": .9}],
            "avg_logprob": -.2, "no_speech_prob": .01}


class RepairTests(unittest.TestCase):
    @staticmethod
    def _write_wav(path: Path, frames: int = 800) -> None:
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(8000)
            handle.writeframes(b"\0\0" * frames)

    def test_protected_interval_excluded(self):
        transcript = {"segments": [segment(10, 14, "This is protected speech.")]}
        accepted, rejected = _candidate_windows(transcript, [{"start_s": 13, "end_s": 16}], 30, max_candidates=40)
        self.assertEqual(accepted, [])
        self.assertEqual(rejected[0]["reason"], "protected_interval_overlap")

    def test_short_speech_is_retained_when_merging(self):
        transcript = {"segments": [segment(0, 4, "First complete sentence here."),
                       segment(4.5, 5.0, "Oops."), segment(5.5, 9.5, "Another complete sentence here.")]}
        accepted, _ = _candidate_windows(transcript, [], 30, max_candidates=40)
        self.assertEqual(len(accepted), 2)
        self.assertLess(accepted[0]["end_s"], 4.5)
        self.assertIn("Oops.", accepted[1]["text"])
        self.assertLess(accepted[1]["start_s"], 4.5)

    def test_sentence_spans_asr_segments(self):
        transcript = {"segments": [segment(1,3,"We might"), segment(3,7,"as well use them.")]}
        accepted, _ = _candidate_windows(transcript, [], 30, max_candidates=40)
        self.assertEqual(accepted[0]["text"], "We might as well use them.")

    def test_dangling_fragment_rejected(self):
        accepted, _ = _candidate_windows({"segments":[segment(1,7,"This ends abruptly")]}, [], 30, max_candidates=40)
        self.assertEqual(accepted, [])

    def test_relaxed_word_error_admission_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "exact normalized"):
            stage_stage(SimpleNamespace(max_consensus_wer=.2))

    def test_family_defaults_preserve_asmr_cli(self):
        propose = _parser().parse_args(["propose"])
        self.assertEqual(propose.family, ASMR_FAMILY)
        self.assertEqual(propose.references_manifest, DEFAULT_REFERENCES)
        stage = _parser().parse_args(["stage", "--proposal", "proposal.json", "--audit", "audit.json"])
        self.assertEqual(stage.family, ASMR_FAMILY)
        self.assertIsNone(stage.reference_profile)
        self.assertIsNone(propose.decode_audio)
        self.assertIsNone(propose.decode_provenance)

    def test_family_selects_matching_raw_source_and_protections(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            asmr = root / "asmr.mp3"
            harvey = root / "harvey.mp3"
            asmr.write_bytes(b"asmr source")
            harvey.write_bytes(b"harvey source")
            asmr_hash = hashlib.sha256(asmr.read_bytes()).hexdigest()
            harvey_hash = hashlib.sha256(harvey.read_bytes()).hexdigest()
            manifest = {
                "sources": [
                    {"path": asmr.name, "sha256": asmr_hash},
                    {"path": harvey.name, "sha256": harvey_hash},
                ],
                "references": [
                    {"id": "asmr-ref", "family": "asmr7", "source": {"path": asmr.name, "sha256": asmr_hash, "start_s": 1.0, "end_s": 2.0}},
                    {"id": "harvey-ref", "family": "harvey", "source": {"path": harvey.name, "sha256": harvey_hash, "start_s": 9.0, "end_s": 16.1}},
                ],
                "heldout": [
                    {"id": "harvey-heldout", "family": "harvey", "source": {"path": harvey.name, "sha256": harvey_hash, "start_s": 34.0, "end_s": 40.8}},
                ],
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            selected_path, selected_meta = _source_from_references(manifest_path, "harvey")
            self.assertEqual(selected_path, harvey.resolve())
            self.assertEqual(selected_meta["source_id"], harvey_hash)
            protected = _protected_intervals(manifest, selected_path, harvey_hash, "harvey")
            self.assertEqual([row["id"] for row in protected], ["harvey-ref", "harvey-heldout"])

    def test_isolated_profile_overlap_uses_raw_source_identity(self):
        source_hash = "raw-source-hash"
        overlap = _reference_overlap(
            {"start_s": 10.0, "end_s": 12.0},
            {"origin": {"source_sha256": source_hash, "source_id": "different-isolated-file", "start_s": 9.0, "end_s": 16.1}},
            "harvey",
            source_hash,
        )
        self.assertTrue(overlap["same_raw_source"])
        self.assertTrue(overlap["overlaps"])
        self.assertEqual(overlap["overlap_s"], 2.0)

    def test_aligned_decode_requires_and_records_raw_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw.wav"
            decoded = root / "decoded.wav"
            self._write_wav(raw)
            self._write_wav(decoded)
            provenance = root / "separation.json"
            provenance.write_text(json.dumps({"source": str(raw), "method": "test"}), encoding="utf-8")
            raw_hash = hashlib.sha256(raw.read_bytes()).hexdigest()
            metadata = _validate_decode_source(raw, raw_hash, decoded, provenance)
            self.assertTrue(metadata["source_path_match"])
            self.assertTrue(metadata["source_hash_match"])
            self.assertTrue(metadata["alignment_asserted"])
            self.assertLessEqual(metadata["alignment"]["delta_s"], DECODE_ALIGNMENT_TOLERANCE_S)
            self.assertEqual(metadata["sha256"], hashlib.sha256(decoded.read_bytes()).hexdigest())

    def test_aligned_decode_rejects_wrong_provenance_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw.wav"
            decoded = root / "decoded.wav"
            other = root / "other.wav"
            self._write_wav(raw)
            self._write_wav(decoded)
            self._write_wav(other, frames=1600)
            provenance = root / "separation.json"
            provenance.write_text(json.dumps({"source": str(other)}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not identify"):
                _validate_decode_source(raw, hashlib.sha256(raw.read_bytes()).hexdigest(), decoded, provenance)

    def test_rebase_uses_clip_tiny_text_and_retains_original_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            clip_paths = []
            candidates = []
            for index in range(2):
                clip = root / f"clip_{index}.wav"
                self._write_wav(clip, frames=800 + index * 40)
                digest = hashlib.sha256(clip.read_bytes()).hexdigest()
                clip_paths.append(clip)
                candidates.append(
                    {
                        "id": f"harvey_repair_{index + 1:03d}",
                        "path": str(clip),
                        "audio_sha256": digest,
                        "text": "Original source label.",
                        "family": "harvey",
                        "source_id": "raw-source-hash",
                        "source_audio_sha256": "raw-source-hash",
                        "start_s": float(index * 5),
                        "end_s": float(index * 5 + 4),
                        "avg_logprob": -0.25,
                        "no_speech_prob": 0.01,
                        "mean_word_probability": 0.8,
                    }
                )
            proposal = root / "proposal.json"
            protected = [{"id": "heldout", "start_s": 30.0, "end_s": 35.0, "source_id": "raw-source-hash"}]
            proposal.write_text(json.dumps({"status": "proposed", "family": "harvey", "source": {"actual_sha256": "raw-source-hash"}, "protected_intervals": protected, "candidates": candidates}), encoding="utf-8")

            def tiny_row(candidate, text):
                return {
                    "id": candidate["id"],
                    "path": candidate["path"],
                    "audio_sha256": candidate["audio_sha256"],
                    "transcript": text,
                    "segments": [{
                        "text": text,
                        "avg_logprob": -0.1,
                        "no_speech_prob": 0.01,
                        "compression_ratio": 1.1,
                        "words": [
                            # Existing admission uses mean word confidence.
                            {"word": "I", "start_s": 0.0, "end_s": 0.2, "probability": 0.20},
                            {"word": "am", "start_s": 0.2, "end_s": 0.4, "probability": 0.94},
                            {"word": "ready.", "start_s": 0.4, "end_s": 0.8, "probability": 0.93},
                        ],
                    }],
                }

            tiny = root / "tiny.json"
            tiny.write_text(json.dumps({"model": "models/tiny.en", "word_timestamps": True, "inputs": [tiny_row(candidates[0], "I am ready."), tiny_row(candidates[1], "I am ready.")]}), encoding="utf-8")
            small = root / "small.json"
            small.write_text(json.dumps({"model": "models/faster-whisper-small", "inputs": [
                {"id": candidates[0]["id"], "path": candidates[0]["path"], "audio_sha256": candidates[0]["audio_sha256"], "transcript": "I am ready.", "segments": []},
                {"id": candidates[1]["id"], "path": candidates[1]["path"], "audio_sha256": candidates[1]["audio_sha256"], "transcript": "They are ready.", "segments": []},
            ]}), encoding="utf-8")
            output = root / "rebased.json"
            result = rebase_proposal(SimpleNamespace(proposal=proposal, tiny_audit=tiny, small_audit=small, output_dir=root, output_proposal=output))
            self.assertEqual(result["exact_count"], 1)
            rebased = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(rebased["protected_intervals"], protected)
            self.assertEqual(rebased["candidates"][0]["original_text"], "Original source label.")
            self.assertEqual(rebased["candidates"][0]["text"], "I am ready.")
            self.assertEqual(rebased["candidates"][0]["avg_logprob"], -0.1)
            self.assertEqual(rebased["candidates"][0]["original_confidence"]["avg_logprob"], -0.25)
            self.assertFalse(rebased["candidates"][1]["clip_tiny_small_consensus"]["exact"])

    def test_rebase_marks_missing_tiny_word_timestamps_per_row(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            clip = root / "clip.wav"
            self._write_wav(clip)
            digest = hashlib.sha256(clip.read_bytes()).hexdigest()
            proposal = root / "proposal.json"
            candidate = {"id": "c1", "path": str(clip), "audio_sha256": digest, "text": "Old label."}
            proposal.write_text(json.dumps({"candidates": [candidate]}), encoding="utf-8")
            tiny = root / "tiny.json"
            tiny.write_text(json.dumps({"model": "models/tiny.en", "word_timestamps": True, "inputs": [
                {"id": "c1", "path": str(clip), "audio_sha256": digest, "transcript": "I am ready.", "segments": [{"avg_logprob": -0.1, "no_speech_prob": 0.01, "compression_ratio": 1.1}]},
            ]}), encoding="utf-8")
            small = root / "small.json"
            small.write_text(json.dumps({"model": "models/faster-whisper-small", "inputs": [
                {"id": "c1", "path": str(clip), "audio_sha256": digest, "transcript": "I am ready.", "segments": []},
            ]}), encoding="utf-8")
            result = rebase_proposal(SimpleNamespace(proposal=proposal, tiny_audit=tiny, small_audit=small, output_dir=root, output_proposal=None))
            self.assertEqual(result["exact_count"], 0)
            rebased = json.loads((root / "proposal_rebased.json").read_text(encoding="utf-8"))
            self.assertIn("missing_word_timestamps", rebased["candidates"][0]["clip_tiny_audit"]["quality_reasons"])


if __name__ == "__main__": unittest.main()
