"""Pure final-projection folding tests with a tiny synthetic model."""

from pathlib import Path
import hashlib
import tempfile
import types
import unittest

import torch

from decoder_projection import configure_decoder_projection


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model(weight: torch.Tensor) -> object:
    projection = torch.nn.Conv1d(256, 80, 1, bias=False)
    with torch.no_grad():
        projection.weight.copy_(weight)
    return types.SimpleNamespace(
        s3gen=types.SimpleNamespace(
            flow=types.SimpleNamespace(
                decoder=types.SimpleNamespace(
                    estimator=types.SimpleNamespace(final_proj=projection)
                )
            )
        )
    )


def _adapter(path: Path, checkpoint: Path, conditionals: Path, base: torch.Tensor, *, malformed: str | None = None) -> None:
    rank, alpha = 2, 2.0
    down = torch.arange(rank * 256, dtype=torch.float32).reshape(rank, 256, 1) / 1000.0
    up = torch.arange(80 * rank, dtype=torch.float32).reshape(80, rank, 1) / 100.0
    delta = (alpha / rank) * torch.matmul(up[:, :, 0], down[:, :, 0]).unsqueeze(-1)
    payload = {
        "format": "nano_decoder_projection_lora_v1",
        "down_weight": down,
        "up_weight": up,
        "merged_delta_weight": delta,
        "base_target_weight": base.detach().clone(),
        "target_weight_key": "flow.decoder.estimator.final_proj.weight",
        "target_shape": [80, 256, 1],
        "rank": rank,
        "alpha": alpha,
        "scale": 1.0,
        "base_target_weight_sha256": hashlib.sha256(base.detach().float().contiguous().numpy().tobytes()).hexdigest(),
        "metadata": {
            "model_checkpoint_sha256": _sha(checkpoint),
            "conditionals_sha256": _sha(conditionals),
        },
    }
    if malformed == "nan":
        payload["up_weight"][0, 0, 0] = float("nan")
    elif malformed == "delta":
        payload["merged_delta_weight"] = delta + 0.001
    elif malformed == "base":
        payload["base_target_weight_sha256"] = "0" * 64
    torch.save(payload, path)


class DecoderProjectionTests(unittest.TestCase):
    def test_fold_restore_switch_and_idempotence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "s3gen_meanflow.safetensors"
            checkpoint.write_bytes(b"checkpoint")
            conditionals = root / "conditionals.pt"
            conditionals.write_bytes(b"conditionals")
            base = torch.arange(80 * 256, dtype=torch.float32).reshape(80, 256, 1) / 1000.0
            adapter = root / "projection.pt"
            _adapter(adapter, checkpoint, conditionals, base)
            model = _model(base)
            first = configure_decoder_projection(model, adapter, conditionals_path=conditionals, model_checkpoint_path=checkpoint, strength=0.5)
            folded = model.s3gen.flow.decoder.estimator.final_proj.weight.detach().clone()
            self.assertTrue(first["enabled"])
            self.assertFalse(torch.equal(folded, base))
            second = configure_decoder_projection(model, adapter, conditionals_path=conditionals, model_checkpoint_path=checkpoint, strength=0.5)
            torch.testing.assert_close(model.s3gen.flow.decoder.estimator.final_proj.weight, folded, rtol=0, atol=0)
            self.assertEqual(second["base_target_weight_sha256"], first["base_target_weight_sha256"])
            restored = configure_decoder_projection(model, strength=0.0)
            self.assertTrue(restored["restored"])
            torch.testing.assert_close(model.s3gen.flow.decoder.estimator.final_proj.weight, base, rtol=0, atol=0)

    def test_no_path_is_an_exact_reset_and_initial_cache_hash_is_authoritative(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "s3gen_meanflow.safetensors"; checkpoint.write_bytes(b"checkpoint")
            original_cache = root / "conditionals.pt"; original_cache.write_bytes(b"conditionals")
            fitted_cache = root / "fitted.conds.pt"; fitted_cache.write_bytes(b"fitted-conditionals")
            base = torch.zeros(80, 256, 1)
            path = root / "projection.pt"
            _adapter(path, checkpoint, original_cache, base)
            payload = torch.load(path, map_location="cpu", weights_only=True)
            payload["metadata"]["initial_conditionals"] = {"sha256": _sha(fitted_cache), "provided": True}
            torch.save(payload, path)
            model = _model(base)
            with self.assertRaisesRegex(ValueError, "initial_conditionals_sha256"):
                configure_decoder_projection(model, path, conditionals_path=original_cache, model_checkpoint_path=checkpoint)
            # No path restores even with the default strength argument.
            result = configure_decoder_projection(model)
            self.assertFalse(result["enabled"])

    def test_invalid_factor_delta_nan_and_base_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "s3gen_meanflow.safetensors"; checkpoint.write_bytes(b"checkpoint")
            conditionals = root / "conditionals.pt"; conditionals.write_bytes(b"conditionals")
            base = torch.zeros(80, 256, 1)
            model = _model(base)
            for malformed, message in (("nan", "non-finite"), ("delta", "does not match"), ("base", "base_target")):
                path = root / f"{malformed}.pt"
                _adapter(path, checkpoint, conditionals, base, malformed=malformed)
                with self.assertRaisesRegex(ValueError, message):
                    configure_decoder_projection(model, path, conditionals_path=conditionals, model_checkpoint_path=checkpoint)

    def test_missing_provenance_and_conditionals_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "s3gen_meanflow.safetensors"; checkpoint.write_bytes(b"checkpoint")
            conditionals = root / "conditionals.pt"; conditionals.write_bytes(b"conditionals")
            base = torch.zeros(80, 256, 1)
            path = root / "projection.pt"
            _adapter(path, checkpoint, conditionals, base)
            model = _model(base)
            with self.assertRaisesRegex(ValueError, "conditionals_path"):
                configure_decoder_projection(model, path, model_checkpoint_path=checkpoint)
            with self.assertRaisesRegex(ValueError, "strength"):
                configure_decoder_projection(model, path, conditionals_path=conditionals, model_checkpoint_path=checkpoint, strength=1.1)

    def test_conditionals_and_checkpoint_hashes_are_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "s3gen_meanflow.safetensors"; checkpoint.write_bytes(b"checkpoint")
            conditionals = root / "conditionals.pt"; conditionals.write_bytes(b"conditionals")
            wrong_cache = root / "wrong.conds.pt"; wrong_cache.write_bytes(b"wrong")
            wrong_checkpoint = root / "wrong.safetensors"; wrong_checkpoint.write_bytes(b"wrong-checkpoint")
            base = torch.zeros(80, 256, 1)
            path = root / "projection.pt"
            _adapter(path, checkpoint, conditionals, base)
            model = _model(base)
            with self.assertRaisesRegex(ValueError, "conditionals_sha256"):
                configure_decoder_projection(model, path, conditionals_path=wrong_cache, model_checkpoint_path=checkpoint)
            with self.assertRaisesRegex(ValueError, "model_checkpoint_sha256"):
                configure_decoder_projection(model, path, conditionals_path=conditionals, model_checkpoint_path=wrong_checkpoint)


if __name__ == "__main__":
    unittest.main()
