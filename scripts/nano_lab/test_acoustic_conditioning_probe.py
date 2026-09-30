"""Pure tests for the source-conditioning ablation helper."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from acoustic_conditioning_probe import conditionals_record, mix_conditionals, target_token_record


def _fake_conditionals() -> SimpleNamespace:
    return SimpleNamespace(
        t3=SimpleNamespace(
            speaker_emb=np.array([[1.0, 2.0]], dtype=np.float32),
            cond_prompt_speech_tokens=np.array([[3, 4]], dtype=np.int64),
            emotion_adv=np.array([[[0.5]]], dtype=np.float32),
        ),
        gen={
            "prompt_token": np.array([[5, 6]], dtype=np.int64),
            "prompt_token_len": np.array([2], dtype=np.int64),
            "prompt_feat": np.ones((1, 80, 2), dtype=np.float32),
            "prompt_feat_len": np.array([2], dtype=np.int64),
            "embedding": np.array([[7.0, 8.0]], dtype=np.float32),
        },
    )


def test_embedding_variant_replaces_only_gen_speaker_embedding() -> None:
    baseline = _fake_conditionals()
    source = _fake_conditionals()
    source.t3.speaker_emb[:] = 11.0
    source.gen["embedding"][:] = 12.0
    source.gen["prompt_token"][:] = 13

    mixed = mix_conditionals(baseline, source, "baseline_self_embedding")

    np.testing.assert_array_equal(mixed.t3.speaker_emb, baseline.t3.speaker_emb)
    np.testing.assert_array_equal(mixed.gen["embedding"], source.gen["embedding"])
    np.testing.assert_array_equal(mixed.gen["prompt_token"], baseline.gen["prompt_token"])
    baseline.t3.speaker_emb[0, 0] = -99.0
    assert mixed.t3.speaker_emb[0, 0] == 1.0


def test_prompt_variant_replaces_prompt_pair_and_lengths_only() -> None:
    baseline = _fake_conditionals()
    source = _fake_conditionals()
    source.gen["prompt_token"][:] = 21
    source.gen["prompt_feat"][:] = 22.0
    source.gen["prompt_token_len"][:] = 1
    source.gen["prompt_feat_len"][:] = 1
    source.gen["embedding"][:] = 23.0

    mixed = mix_conditionals(baseline, source, "baseline_self_prompt")

    for key in ("prompt_token", "prompt_token_len", "prompt_feat", "prompt_feat_len"):
        np.testing.assert_array_equal(mixed.gen[key], source.gen[key])
    np.testing.assert_array_equal(mixed.gen["embedding"], baseline.gen["embedding"])
    np.testing.assert_array_equal(mixed.t3.speaker_emb, baseline.t3.speaker_emb)


def test_prompt_variant_requires_associated_lengths() -> None:
    baseline = _fake_conditionals()
    source = _fake_conditionals()
    del source.gen["prompt_feat_len"]
    with pytest.raises(ValueError, match="prompt fields"):
        mix_conditionals(baseline, source, "baseline_self_prompt")


def test_conditioning_and_target_records_are_stable_and_complete() -> None:
    conditionals = _fake_conditionals()
    record = conditionals_record(conditionals)
    assert set(record["gen"]) == set(conditionals.gen)
    assert record["gen"]["prompt_feat"]["shape"] == [1, 80, 2]
    tokens = np.array([4, 5, 4299, 4299, 4299], dtype=np.int64)
    token_record = target_token_record(tokens)
    assert token_record["count"] == 5
    assert token_record["dtype"] == "int64"
    assert len(token_record["sha256"]) == 64

