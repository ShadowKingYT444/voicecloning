import hashlib
from pathlib import Path
import tempfile
import unittest

from dataset_contract import require_audited_rows, require_inference_conditioning


class DatasetContractTests(unittest.TestCase):
    def test_old_conditioning_and_changed_reference_rejected(self):
        with self.assertRaisesRegex(ValueError, "Inference-compatible"):
            require_inference_conditioning([{"split": "train"}])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "reference.wav"
            path.write_bytes(b"reference")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            row = {"split": "train", "reference_audio_path": str(path), "reference_sha256": digest,
                   "conditioning_provenance": {"path": str(path), "sha256": digest, "preprocessing": {
                       "entrypoint": "ChatterboxTurboTTS.prepare_conditionals", "norm_loudness": True,
                       "prompt_tokens_unmodified": True, "speaker_embedding_unmodified": True}}}
            require_inference_conditioning([row])
            path.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "reference audio changed"):
                require_inference_conditioning([row])

    def test_missing_audit_rejected(self):
        with self.assertRaisesRegex(ValueError, "Transcript audit required"):
            require_audited_rows([{"id": "old", "split": "train"}])

    def test_changed_audio_or_text_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "clip.wav"
            source.write_bytes(b"original audio")
            row = {"id": "test", "split": "train", "audio_path": str(source), "text": "Hello."}
            row["transcript_audit"] = {"status": "accepted", "method": "test fixture",
                "text_sha256": hashlib.sha256(b"Hello.").hexdigest(),
                "audio_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
            require_audited_rows([row])
            row["text"] = "Wrong words."
            with self.assertRaisesRegex(ValueError, "transcript changed"):
                require_audited_rows([row])
            row["text"] = "Hello."
            source.write_bytes(b"different audio")
            with self.assertRaisesRegex(ValueError, "source audio changed"):
                require_audited_rows([row])


if __name__ == "__main__":
    unittest.main()
