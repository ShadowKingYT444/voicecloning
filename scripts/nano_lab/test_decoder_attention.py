"""Small factor-folding tests for decoder_attention.

The fake model has one target and uses a tiny safetensors file.  Production
validation still requires the canonical 224-target inventory; the test patches
that inventory loader so it does not allocate a decoder-sized fixture.
"""

from pathlib import Path
import hashlib
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file

import decoder_attention as attention


TARGET = "flow.decoder.estimator.down_blocks.0.1.0.attn1.to_q.weight"
INVENTORY = {TARGET: (2, 2)}
INVENTORY_SHA = "a" * 64


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_sha(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().float().contiguous().numpy().tobytes()).hexdigest()


def _fake_model(base: torch.Tensor) -> object:
    target = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        target.weight.copy_(base)
    leaf = types.SimpleNamespace(attn1=types.SimpleNamespace(to_q=target))
    estimator = types.SimpleNamespace(down_blocks=[[None, [leaf]]])
    return types.SimpleNamespace(
        s3gen=types.SimpleNamespace(
            flow=types.SimpleNamespace(
                decoder=types.SimpleNamespace(estimator=estimator)
            )
        )
    )


def _write_adapter(path: Path, checkpoint: Path, conditionals: Path, base: torch.Tensor, *, bad: str | None = None) -> None:
    down = torch.tensor([[0.2, -0.3], [0.4, 0.1]], dtype=torch.float32)
    up = torch.tensor([[0.5, 0.2], [-0.1, 0.3]], dtype=torch.float32)
    payload = {
        "format": attention.FORMAT,
        "rank": 2,
        "alpha": 2.0,
        "scale": 1.0,
        "target_parameter_count": 4,
        "target_weight_count": 1,
        "targets": {
            TARGET: {
                "shape": [2, 2],
                "down_weight": down,
                "up_weight": up,
                "base_weight_sha256": _tensor_sha(base),
            }
        },
        "model_checkpoint_sha256": _file_sha(checkpoint),
        "initial_conditionals_sha256": _file_sha(conditionals),
        "prepared_conditionals_sha256": _file_sha(conditionals),
        "cache_sha256": _file_sha(conditionals),
        "inventory_sha256": INVENTORY_SHA,
        "metadata": {
            "model_checkpoint_sha256": _file_sha(checkpoint),
            "conditionals_sha256": _file_sha(conditionals),
            "cache_sha256": _file_sha(conditionals),
            "inventory_sha256": INVENTORY_SHA,
            "initial_conditionals": {"sha256": _file_sha(conditionals)},
        },
    }
    if bad == "nan":
        payload["targets"][TARGET]["up_weight"][0, 0] = float("nan")
    elif bad == "base":
        payload["targets"][TARGET]["base_weight_sha256"] = "0" * 64
    torch.save(payload, path)


class DecoderAttentionTests(unittest.TestCase):
    def test_real_inventory_counts_factor_parameters(self):
        inventory, digest = attention._load_inventory(attention.DEFAULT_INVENTORY)
        self.assertEqual(len(inventory), 224)
        self.assertEqual(sum(2 * sum(shape) for shape in inventory.values()), 344064)
        self.assertEqual(len(digest), 64)

    def _paths(self, root: Path, base: torch.Tensor) -> tuple[Path, Path, Path]:
        checkpoint = root / "s3gen_meanflow.safetensors"
        save_file({TARGET: base.contiguous()}, checkpoint)
        conditionals = root / "conditionals.pt"
        conditionals.write_bytes(b"conditionals")
        adapter = root / "attention.pt"
        _write_adapter(adapter, checkpoint, conditionals, base)
        return checkpoint, conditionals, adapter

    def test_fold_switch_restore_and_no_dense_base_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
            checkpoint, conditionals, adapter = self._paths(root, base)
            model = _fake_model(base)
            with patch.object(attention, "TARGET_WEIGHT_COUNT", 1), patch.object(attention, "TARGET_PARAMETER_COUNT", 4), patch.object(attention, "_load_inventory", return_value=(INVENTORY, INVENTORY_SHA)):
                result = attention.configure_decoder_attention(model, adapter, conditionals_path=conditionals, model_checkpoint_path=checkpoint)
                folded = model.s3gen.flow.decoder.estimator.down_blocks[0][1][0].attn1.to_q.weight.detach().clone()
                repeat = attention.configure_decoder_attention(model, adapter, conditionals_path=conditionals, model_checkpoint_path=checkpoint)
                self.assertTrue(repeat["idempotent"])
                torch.testing.assert_close(model.s3gen.flow.decoder.estimator.down_blocks[0][1][0].attn1.to_q.weight, folded, rtol=0, atol=0)
                state = model.s3gen.flow.decoder.estimator._nano_decoder_attention_state
                self.assertNotIn("base_weights", state)
                self.assertTrue(all(set(spec) == {'shape', 'base_hash'} for spec in state['factors'].values()))
                self.assertTrue(result["enabled"])
                restored = attention.configure_decoder_attention(model, path=None)
                self.assertTrue(restored["restored"])
                torch.testing.assert_close(model.s3gen.flow.decoder.estimator.down_blocks[0][1][0].attn1.to_q.weight, base, rtol=0, atol=0)
                # Simulate an interrupted fold: state exists, but no adapter
                # is marked active and live weights contain a partial change.
                weight = model.s3gen.flow.decoder.estimator.down_blocks[0][1][0].attn1.to_q.weight
                with torch.no_grad():
                    weight.add_(99)
                attention.configure_decoder_attention(model, adapter, conditionals_path=conditionals, model_checkpoint_path=checkpoint)
                torch.testing.assert_close(weight, folded, rtol=0, atol=0)
                attention.configure_decoder_attention(model, adapter, strength=.5, conditionals_path=conditionals, model_checkpoint_path=checkpoint)
                torch.testing.assert_close(weight, base + .5 * (folded - base), rtol=1e-6, atol=1e-7)
                attention.configure_decoder_attention(model, path=None)
                wrong = root / 'wrong_after_reset.pt'
                _write_adapter(wrong, checkpoint, conditionals, base, bad='base')
                with self.assertRaisesRegex(ValueError, 'base weight hash'):
                    attention.configure_decoder_attention(model, wrong, conditionals_path=conditionals, model_checkpoint_path=checkpoint)
                torch.testing.assert_close(weight, base, rtol=0, atol=0)

    def test_factor_and_base_hashes_fail_before_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = torch.eye(2, dtype=torch.float32)
            checkpoint, conditionals, _ = self._paths(root, base)
            model = _fake_model(base)
            for bad, message in (("nan", "NaN"), ("base", "base weight hash")):
                adapter = root / f"{bad}.pt"
                _write_adapter(adapter, checkpoint, conditionals, base, bad=bad)
                with patch.object(attention, "TARGET_WEIGHT_COUNT", 1), patch.object(attention, "TARGET_PARAMETER_COUNT", 4), patch.object(attention, "_load_inventory", return_value=(INVENTORY, INVENTORY_SHA)):
                    with self.assertRaisesRegex(ValueError, message):
                        attention.configure_decoder_attention(model, adapter, conditionals_path=conditionals, model_checkpoint_path=checkpoint)
                torch.testing.assert_close(model.s3gen.flow.decoder.estimator.down_blocks[0][1][0].attn1.to_q.weight, base, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
