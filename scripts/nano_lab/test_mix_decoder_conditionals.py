"""Pure and file-contract tests for decoder conditionals mixing."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import mix_decoder_conditionals as mixer


class MixMathTests(unittest.TestCase):
    def test_embedding_endpoints_midpoint_and_no_alias(self) -> None:
        base = np.zeros((1, 192), dtype=np.float32)
        donor = np.zeros((1, 192), dtype=np.float32)
        base[0, :2] = (3.0, 4.0)
        donor[0, 2:4] = (6.0, 8.0)
        zero = mixer.interpolate_embedding_numpy(base, donor, 0.0)
        one = mixer.interpolate_embedding_numpy(base, donor, 1.0)
        midpoint = mixer.interpolate_embedding_numpy(base, donor, 0.5)
        np.testing.assert_array_equal(zero, base)
        np.testing.assert_array_equal(one, donor)
        self.assertAlmostEqual(float(np.linalg.norm(midpoint)), 5.0, places=6)
        self.assertGreater(float(midpoint[0, 0]), 0.0)
        self.assertGreater(float(midpoint[0, 2]), 0.0)
        zero[0, 0] = 99.0
        one[0, 2] = 99.0
        self.assertEqual(float(base[0, 0]), 3.0)
        self.assertEqual(float(donor[0, 2]), 6.0)

    def test_nearly_opposed_vectors_are_rejected(self) -> None:
        base = np.zeros((1, 192), dtype=np.float32)
        donor = np.zeros((1, 192), dtype=np.float32)
        base[0, 0] = 1.0
        donor[0, 0] = -1.0
        with self.assertRaisesRegex(ValueError, "opposed"):
            mixer.interpolate_embedding_numpy(base, donor, 0.5)


class MixFileTests(unittest.TestCase):
    @staticmethod
    def _state(torch, marker: int):
        return {
            "t3": {
                "speaker_emb": torch.arange(256, dtype=torch.float32).view(1, 256) + marker,
                "clap_emb": None,
                "cond_prompt_speech_tokens": torch.arange(5, dtype=torch.long).view(1, 5),
                "cond_prompt_speech_emb": None,
                "emotion_adv": torch.zeros(1, 1, 1),
            },
            "gen": {
                "prompt_token": torch.arange(4, dtype=torch.long).view(1, 4) + marker,
                "prompt_token_len": torch.tensor([4], dtype=torch.long),
                "prompt_feat": torch.full((1, 8, 80), float(marker)),
                "prompt_feat_len": torch.tensor([8], dtype=torch.long),
                "embedding": torch.nn.functional.normalize(
                    torch.arange(192, dtype=torch.float32).view(1, 192) + 1.0 + marker, dim=1
                ),
            },
        }

    def test_prompt_mode_copies_prompt_and_both_lengths_keeps_base_t3_embedding(self) -> None:
        import torch

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_path, donor_path = root / "base.conds.pt", root / "donor.conds.pt"
            output_path, report_path = root / "prompt.conds.pt", root / "prompt.json"
            base, donor = self._state(torch, 1), self._state(torch, 7)
            torch.save(base, base_path)
            torch.save(donor, donor_path)
            payload = mixer.mix_conditionals(
                base_path,
                donor_path,
                output_path,
                mode="prompt",
                strength=1.0,
                report=report_path,
            )
            result = torch.load(output_path, map_location="cpu", weights_only=True)
            self.assertTrue(torch.equal(result["t3"]["speaker_emb"], base["t3"]["speaker_emb"]))
            self.assertTrue(torch.equal(result["gen"]["embedding"], base["gen"]["embedding"]))
            for key in mixer.PROMPT_KEYS:
                self.assertTrue(torch.equal(result["gen"][key], donor["gen"][key]), key)
            self.assertTrue(payload["same_t3_proof"]["output_matches_base"])
            self.assertFalse(payload["same_t3_proof"]["base_vs_donor"])
            report = json.loads(report_path.read_text())
            self.assertEqual(report["mode"], "prompt")
            self.assertFalse(report["fit"])
            self.assertEqual(report["tensor_hashes"]["output"]["children"].keys(), {"gen", "t3"})

    def test_embedding_endpoint_and_non_alias_output(self) -> None:
        import torch

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_path, donor_path = root / "base.conds.pt", root / "donor.conds.pt"
            output_path = root / "embedding.conds.pt"
            base, donor = self._state(torch, 1), self._state(torch, 7)
            torch.save(base, base_path)
            torch.save(donor, donor_path)
            mixer.mix_conditionals(base_path, donor_path, output_path, mode="embedding", strength=1.0)
            result = torch.load(output_path, map_location="cpu", weights_only=True)
            self.assertTrue(torch.equal(result["gen"]["embedding"], donor["gen"]["embedding"]))
            for key in mixer.PROMPT_KEYS:
                self.assertTrue(torch.equal(result["gen"][key], base["gen"][key]), key)
            result["gen"]["prompt_feat"][0, 0, 0] = 999.0
            self.assertNotEqual(float(base["gen"]["prompt_feat"][0, 0, 0]), 999.0)

    def test_source_overwrite_is_rejected(self) -> None:
        import torch

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_path, donor_path = root / "base.conds.pt", root / "donor.conds.pt"
            torch.save(self._state(torch, 1), base_path)
            torch.save(self._state(torch, 7), donor_path)
            with self.assertRaisesRegex(ValueError, "new path"):
                mixer.mix_conditionals(base_path, donor_path, base_path, strength=0.5)


if __name__ == "__main__":
    unittest.main()
