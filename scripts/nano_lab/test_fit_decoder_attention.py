"""Pure and tiny Torch tests for the attention LoRA experiment.

The tests do not load Nano checkpoints or start a model job.  The Torch cases
construct one small Linear layer only.  Full inventory/model checks run later
under the bounded job owned by the parent agent.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from fit_decoder_attention import (
    ALPHA,
    FORMAT,
    RANK,
    _apply_lora_numpy,
    _build_attention_adapter,
    _factor_delta_numpy,
    _fold_weight_numpy,
    _inventory,
    _validate_factor_payload,
    build_parser,
    validate_attention_checkpoint,
)


def test_inventory_has_exact_target_set_and_parameter_count() -> None:
    inventory, digest = _inventory(__import__("fit_decoder_attention").DEFAULT_INVENTORY)
    assert len(inventory) == 224
    assert digest and len(digest) == 64
    assert sum(2 * (shape[0] + shape[1]) for shape in inventory.values()) == 344064
    assert all(name.startswith("flow.decoder.estimator.") and ".attn1." in name for name in inventory)


def test_zero_up_is_exact_numpy_noop() -> None:
    rng = np.random.default_rng(4)
    shape = (512, 256)
    base = rng.normal(size=shape).astype(np.float32)
    hidden = rng.normal(size=(3, 7, shape[1])).astype(np.float32)
    down = rng.normal(size=(RANK, shape[1])).astype(np.float32)
    up = np.zeros((shape[0], RANK), dtype=np.float32)
    folded = _fold_weight_numpy(base, up, down, shape)
    np.testing.assert_array_equal(folded, base)
    np.testing.assert_array_equal(_apply_lora_numpy(base, hidden, up, down, shape), np.matmul(hidden, base.T))


def test_nonzero_factor_fold_matches_factorized_output() -> None:
    rng = np.random.default_rng(9)
    shape = (256, 512)
    # Match fan-in-scaled Linear weights. Unit-variance 512-wide weights make
    # reassociated FP32 dot products exceed the absolute test tolerance alone.
    base = rng.normal(0.0, 1 / np.sqrt(shape[1]), size=shape).astype(np.float32)
    hidden = rng.normal(size=(2, 5, shape[1])).astype(np.float32)
    down = rng.normal(0.0, 0.03, size=(RANK, shape[1])).astype(np.float32)
    up = rng.normal(0.0, 0.03, size=(shape[0], RANK)).astype(np.float32)
    expected = np.matmul(hidden, _fold_weight_numpy(base, up, down, shape).T)
    actual = np.matmul(hidden, base.T) + np.matmul(hidden, _factor_delta_numpy(up, down, shape).T)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-5)


def test_torch_adapter_zero_output_and_frozen_base() -> None:
    torch = pytest.importorskip("torch")
    base = torch.nn.Linear(6, 4, bias=True)
    adapter = _build_attention_adapter(torch, base, init_seed=2701)
    hidden = torch.randn(2, 3, 6)
    with torch.no_grad():
        expected = base(hidden)
        actual = adapter(hidden)
    assert torch.equal(expected, actual)
    assert tuple(adapter.down.weight.shape) == (RANK, 6)
    assert tuple(adapter.up.weight.shape) == (4, RANK)
    assert all(not parameter.requires_grad for parameter in adapter.base.parameters())
    assert all(torch.count_nonzero(parameter).item() == 0 for parameter in (adapter.up.weight,))


def test_installation_preserves_rng_and_repeats_seeded_factors() -> None:
    import copy
    torch = pytest.importorskip("torch")
    import fit_decoder_attention as module
    root = torch.nn.Module()
    root.estimator = torch.nn.Module()
    root.estimator.target = torch.nn.Linear(6, 4)
    twin = copy.deepcopy(root)
    inventory = {module.TARGET_PREFIX + 'target.weight': (4, 6)}
    before = torch.random.get_rng_state().clone()
    first, _ = module._install_attention_adapters(torch, root, inventory, init_seed=701)
    assert torch.equal(before, torch.random.get_rng_state())
    second, _ = module._install_attention_adapters(torch, twin, inventory, init_seed=701)
    assert torch.equal(before, torch.random.get_rng_state())
    key = next(iter(inventory))
    assert torch.equal(first[key].down.weight, second[key].down.weight)


def test_checkpoint_rejects_dense_or_malformed_factor_payload() -> None:
    inventory, inventory_sha = _inventory(__import__("fit_decoder_attention").DEFAULT_INVENTORY)
    payload_targets = {}
    for name, shape in inventory.items():
        payload_targets[name] = {
            "shape": list(shape),
            "down_weight": np.zeros((RANK, shape[1]), dtype=np.float32),
            "up_weight": np.zeros((shape[0], RANK), dtype=np.float32),
            "base_weight_sha256": "a" * 64,
        }
    payload = {
        "format": FORMAT,
        "rank": RANK,
        "alpha": ALPHA,
        "scale": 1.0,
        "target_weight_count": 224,
        "target_parameter_count": 344064,
        "targets": payload_targets,
        "model_checkpoint_sha256": "b" * 64,
        "initial_conditionals_sha256": "c" * 64,
        "prepared_conditionals_sha256": "d" * 64,
        "cache_sha256": "e" * 64,
        "inventory_sha256": inventory_sha,
    }
    assert validate_attention_checkpoint(payload, inventory)["format"] == FORMAT
    bad = dict(payload)
    bad["targets"] = dict(payload_targets)
    bad["targets"][next(iter(inventory))] = dict(bad["targets"][next(iter(inventory))], merged_delta_weight=np.zeros(inventory[next(iter(inventory))], dtype=np.float32))
    with pytest.raises(ValueError, match="forbidden"):
        validate_attention_checkpoint(bad, inventory)
    malformed = dict(payload)
    malformed["targets"] = dict(payload_targets)
    key = next(iter(inventory))
    malformed["targets"][key] = dict(malformed["targets"][key], down_weight=np.zeros((RANK + 1, inventory[key][1]), dtype=np.float32))
    with pytest.raises(ValueError, match="wrong shape"):
        validate_attention_checkpoint(malformed, inventory)


def test_factor_validator_rejects_nonfinite_and_hash_metadata() -> None:
    with pytest.raises(ValueError, match="hash"):
        _validate_factor_payload("target", {"shape": [4, 6], "down_weight": np.zeros((RANK, 6)), "up_weight": np.zeros((4, RANK)), "base_weight_sha256": "bad"}, (4, 6))
    with pytest.raises(ValueError, match="NaN"):
        _validate_factor_payload("target", {"shape": [4, 6], "down_weight": np.full((RANK, 6), np.nan), "up_weight": np.zeros((4, RANK)), "base_weight_sha256": "a" * 64}, (4, 6))


def test_parser_is_bounded() -> None:
    args = build_parser().parse_args([])
    assert args.max_steps == 30
    assert args.patience == 3
    assert args.lr == pytest.approx(1e-4)
    assert args.distill_weight == pytest.approx(0.5)


def test_evaluate_uses_no_grad_one_pass(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    import fit_decoder_attention as module

    calls = []

    def predict(_torch, _estimator, state, *_args, **_kwargs):
        assert not torch.is_grad_enabled()
        calls.append(state["id"])
        return torch.zeros(1, 2, 3)

    monkeypatch.setattr(module, "_predict", predict)
    monkeypatch.setattr(module, "_attention_loss", lambda *args, **kwargs: (torch.tensor(0.0), {key: 0.0 for key in ("total", "raw_mse", "envelope_mse", "baseline_distill_mse", "adapter_l2", "regularization")}))
    states = [{"id": "a", "target": torch.zeros(1, 2, 3)}, {"id": "b", "target": torch.zeros(1, 2, 3)}]
    anchors = {state["id"]: state["target"] for state in states}
    noises = {state["id"]: None for state in states}
    result = module._evaluate(torch, None, states, torch.zeros(1, 2), noises, anchors, {}, steps=2, envelope_weight=.25, distill_weight=.5, regularizer=1e-4)
    assert calls == ["a", "b"]
    assert result["delta"]["max_abs"] == 0.0
