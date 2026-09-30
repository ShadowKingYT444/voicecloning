"""Pure contract tests for the decoder final-projection LoRA experiment.

The tests construct only tiny tensors.  They do not load a checkpoint, build a
S3Gen module, or start a model job.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from fit_decoder_projection import (
    ALPHA,
    CHECKPOINT_FORMAT,
    INPUT_CHANNELS,
    OUTPUT_CHANNELS,
    RANK,
    TARGET_WEIGHT_KEY,
    _build_projection_adapter,
    _checkpoint_payload,
    _compare_conditionals_except_embedding,
    _factor_delta_numpy,
    _fold_projection_weight_numpy,
    _projection_output_numpy,
    build_parser,
    validate_projection_checkpoint,
)


def _valid_numpy_checkpoint() -> dict[str, object]:
    rng = np.random.default_rng(7)
    down = rng.normal(0.0, 0.02, (RANK, INPUT_CHANNELS, 1)).astype(np.float32)
    up = rng.normal(0.0, 0.02, (OUTPUT_CHANNELS, RANK, 1)).astype(np.float32)
    delta = _factor_delta_numpy(up, down)
    return {
        "format": CHECKPOINT_FORMAT,
        "target_weight_key": TARGET_WEIGHT_KEY,
        "target_shape": [OUTPUT_CHANNELS, INPUT_CHANNELS, 1],
        "rank": RANK,
        "alpha": ALPHA,
        "down_weight": down,
        "up_weight": up,
        "merged_delta_weight": delta,
    }


def test_zero_delta_fold_and_output_are_exact() -> None:
    rng = np.random.default_rng(11)
    base = rng.normal(size=(OUTPUT_CHANNELS, INPUT_CHANNELS, 1)).astype(np.float32)
    hidden = rng.normal(size=(2, INPUT_CHANNELS, 9)).astype(np.float32)
    down = rng.normal(size=(RANK, INPUT_CHANNELS, 1)).astype(np.float32)
    up = np.zeros((OUTPUT_CHANNELS, RANK, 1), dtype=np.float32)
    folded = _fold_projection_weight_numpy(base, up, down)
    np.testing.assert_array_equal(folded, base)
    delta_output = _projection_output_numpy(base, hidden, up, down)
    np.testing.assert_array_equal(delta_output, np.zeros((2, OUTPUT_CHANNELS, 9), dtype=np.float32))


def test_nonzero_fold_matches_factorized_output() -> None:
    rng = np.random.default_rng(13)
    base = rng.normal(size=(OUTPUT_CHANNELS, INPUT_CHANNELS, 1)).astype(np.float32)
    hidden = rng.normal(size=(1, INPUT_CHANNELS, 5)).astype(np.float32)
    down = rng.normal(0.0, 0.03, (RANK, INPUT_CHANNELS, 1)).astype(np.float32)
    up = rng.normal(0.0, 0.03, (OUTPUT_CHANNELS, RANK, 1)).astype(np.float32)
    folded = _fold_projection_weight_numpy(base, up, down)
    direct = np.einsum("oc,bct->bot", base[:, :, 0], hidden) + _projection_output_numpy(base, hidden, up, down)
    merged = np.einsum("oc,bct->bot", folded[:, :, 0], hidden)
    np.testing.assert_allclose(merged, direct, rtol=0.0, atol=2e-6)


def test_projection_module_is_zero_at_initialization() -> None:
    torch = pytest.importorskip("torch")
    base = torch.nn.Conv1d(INPUT_CHANNELS, OUTPUT_CHANNELS, 1)
    adapter = _build_projection_adapter(torch, base)
    hidden = torch.randn(1, INPUT_CHANNELS, 7)
    with torch.no_grad():
        expected = base(hidden)
        actual = adapter(hidden)
    assert torch.equal(actual, expected)
    assert sum(parameter.numel() for parameter in (adapter.down.weight, adapter.up.weight)) == 672
    assert all(not parameter.requires_grad for parameter in adapter.base.parameters())


def test_projection_factors_follow_base_device_and_dtype() -> None:
    torch = pytest.importorskip("torch")
    base = torch.nn.Conv1d(INPUT_CHANNELS, OUTPUT_CHANNELS, 1, dtype=torch.float64)
    before = torch.random.get_rng_state().clone()
    adapter = _build_projection_adapter(torch, base, init_seed=23)
    after = torch.random.get_rng_state()
    assert torch.equal(before, after)
    assert adapter.down.weight.device == base.weight.device
    assert adapter.up.weight.device == base.weight.device
    assert adapter.down.weight.dtype == base.weight.dtype
    assert adapter.up.weight.dtype == base.weight.dtype
    assert adapter.init_seed == 23


def test_validator_rejects_malformed_factor_and_weight_payloads() -> None:
    payload = _valid_numpy_checkpoint()
    bad_shape = dict(payload)
    bad_shape["up_weight"] = np.zeros((OUTPUT_CHANNELS, RANK + 1, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="up_weight"):
        validate_projection_checkpoint(bad_shape)

    bad_delta = dict(payload)
    bad_delta["merged_delta_weight"] = np.asarray(payload["merged_delta_weight"]).copy()
    bad_delta["merged_delta_weight"][0, 0, 0] += 1.0
    with pytest.raises(ValueError, match="merged delta"):
        validate_projection_checkpoint(bad_delta)

    bad_weight = dict(payload)
    bad_weight["base_target_weight"] = np.full((OUTPUT_CHANNELS, INPUT_CHANNELS, 1), np.nan, dtype=np.float32)
    with pytest.raises(ValueError, match="base_target_weight"):
        validate_projection_checkpoint(bad_weight)


def test_conditionals_reject_non_embedding_drift() -> None:
    torch = pytest.importorskip("torch")
    base = {
        "t3": {"speaker_emb": torch.ones(1, 4)},
        "gen": {
            "prompt_token": torch.ones(1, 2, dtype=torch.long),
            "prompt_token_len": torch.tensor([2]),
            "prompt_feat": torch.zeros(1, 4, 80),
            "prompt_feat_len": None,
            "embedding": torch.ones(1, 192),
        },
    }
    candidate = {
        "t3": {"speaker_emb": base["t3"]["speaker_emb"].clone()},
        "gen": {
            "prompt_token": base["gen"]["prompt_token"].clone(),
            "prompt_token_len": base["gen"]["prompt_token_len"].clone(),
            "prompt_feat": base["gen"]["prompt_feat"].clone(),
            "prompt_feat_len": None,
            "embedding": torch.zeros(1, 192),
        },
    }
    _compare_conditionals_except_embedding(torch, base, candidate)
    candidate["gen"]["prompt_feat"][0, 0, 0] = 1.0
    with pytest.raises(ValueError, match="prompt_feat"):
        _compare_conditionals_except_embedding(torch, base, candidate)


def test_epoch_zero_checkpoint_is_serialized() -> None:
    torch = pytest.importorskip("torch")

    class _FakeAdapter:
        def __init__(self) -> None:
            self.down = SimpleNamespace(weight=torch.zeros(RANK, INPUT_CHANNELS, 1))
            self.up = SimpleNamespace(weight=torch.zeros(OUTPUT_CHANNELS, RANK, 1))

        def merged_delta_weight(self):
            return torch.zeros(OUTPUT_CHANNELS, INPUT_CHANNELS, 1)

    adapter = _FakeAdapter()
    base = torch.ones(OUTPUT_CHANNELS, INPUT_CHANNELS, 1)
    payload = _checkpoint_payload(
        torch,
        adapter,
        base,
        metadata={"train_ids": ["train-a"], "valid_ids": ["valid-a"]},
        step=0,
        valid={"total": 1.0},
    )
    assert payload["step"] == 0
    assert payload["format"] == CHECKPOINT_FORMAT
    assert tuple(payload["merged_delta_weight"].shape) == (OUTPUT_CHANNELS, INPUT_CHANNELS, 1)
    assert payload["base_target_weight_sha256"]


def test_parser_has_bounded_defaults() -> None:
    args = build_parser().parse_args([])
    assert args.max_steps <= 40
    assert args.patience == 3
    assert args.lr == pytest.approx(1e-3)
    assert args.distill_weight == pytest.approx(0.5)


def test_evaluation_never_builds_grad_graphs_or_repeats_inference(monkeypatch):
    import torch
    import fit_decoder_projection as module
    calls = []
    def predict(_torch, _estimator, state, *_args, **_kwargs):
        assert not torch.is_grad_enabled()
        calls.append(state["id"])
        return torch.zeros(1, 80, 3)
    monkeypatch.setattr(module, "_predict", predict)
    keys = ("total", "raw_mse", "envelope_mse", "baseline_distill_mse", "adapter_l2", "regularization")
    monkeypatch.setattr(module, "_projection_loss", lambda *a, **k: (None, dict.fromkeys(keys, 0.0)))
    states = [{"id": "a", "target": torch.zeros(1, 80, 3)}, {"id": "b", "target": torch.zeros(1, 80, 3)}]
    anchors = {s["id"]: s["target"] for s in states}
    result = module._evaluate(torch, None, states, None, {"a": None, "b": None}, anchors,
                              steps=2, envelope_weight=.25, distill_weight=.5, regularizer=.0001, adapter=None)
    assert calls == ["a", "b"]
    assert result["delta"]["max_abs"] == 0.0
