"""Pure contract checks for the decoder speaker-embedding fit."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fit_decoder_embedding import (
    _aligned_prediction,
    _append_silence_tokens,
    _normalise_tokens,
    _project_embedding_to_cap,
    build_parser,
)


@pytest.mark.parametrize("feature_frames", [4, 5])
def test_native_optional_prompt_feature_length(tmp_path: Path, feature_frames: int) -> None:
    import torch
    from fit_decoder_embedding import _load_conditionals_payload

    path = tmp_path / "native.pt"
    payload = {"t3": {}, "gen": {
        "embedding": torch.ones(1, 192),
        "prompt_token": torch.ones(1, 2, dtype=torch.long),
        "prompt_token_len": torch.tensor([2]),
        "prompt_feat": torch.zeros(1, feature_frames, 80),
        "prompt_feat_len": None,
    }}
    torch.save(payload, path)
    assert _load_conditionals_payload(torch, path)["gen"]["prompt_feat_len"] is None
    payload["gen"]["prompt_feat"] = torch.zeros(1, 6, 80)
    torch.save(payload, path)
    with pytest.raises(ValueError, match="not paired"):
        _load_conditionals_payload(torch, path)


def test_noise_matches_native_target_then_full_draw_order() -> None:
    import torch
    from fit_decoder_embedding import _fixed_noise

    mu = torch.empty(1, 80, 16)
    actual = _fixed_noise(torch, {"mu": mu, "prompt_len": 6}, seed=31)
    torch.manual_seed(31)
    target = torch.randn(1, 80, 10)
    expected = torch.randn_like(mu)
    expected[:, :, 6:] = target
    assert torch.equal(actual, expected)
    torch.manual_seed(31)
    assert not torch.equal(actual, torch.randn_like(mu))


def test_alignment_trims_only_silence_and_optional_single_frame() -> None:
    state = {"id": "row", "source_frames": 9}
    prediction = np.zeros((1, 80, 15), dtype=np.float32)
    assert _aligned_prediction(prediction, state).shape == (1, 80, 9)
    state["source_frames"] = 8
    assert _aligned_prediction(prediction, state).shape == (1, 80, 8)
    state["source_frames"] = 7
    with pytest.raises(ValueError, match="frame trim"):
        _aligned_prediction(prediction, state)


def test_odd_reference_noise_splices_by_token_length_not_feature_length() -> None:
    import torch
    from fit_decoder_embedding import _fixed_noise
    mu = torch.empty(1, 80, 16)
    actual = _fixed_noise(torch, {"mu": mu, "prompt_len": 7, "target_token_count": 5}, seed=31)
    torch.manual_seed(31)
    target = torch.randn(1, 80, 10)
    expected = torch.randn_like(mu)
    expected[:, :, -10:] = target
    assert torch.equal(actual, expected)


def test_token_contract_rejects_special_or_nonintegral_ids() -> None:
    np.testing.assert_array_equal(_normalise_tokens([1, 2, 3], row_id="a"), np.array([1, 2, 3], dtype=np.int64))
    with pytest.raises(ValueError, match="out-of-range"):
        _normalise_tokens([1, 6561], row_id="b")
    with pytest.raises(ValueError, match="not integer"):
        _normalise_tokens([1.5, 2.0], row_id="c")


def test_odd_reference_alignment_uses_exact_native_frame_offset() -> None:
    state = {"source_frames": 10, "source_token_count": 5, "prompt_frame_offset": 1}
    prediction = np.zeros((1, 80, 15), dtype=np.float32)
    assert _aligned_prediction(prediction, state).shape == (1, 80, 10)
    state["source_frames"] = 9
    assert _aligned_prediction(prediction, state).shape == (1, 80, 9)
    with pytest.raises(ValueError, match="exact native geometry"):
        _aligned_prediction(np.zeros((1, 80, 14)), state)
    state["prompt_frame_offset"] = 2
    with pytest.raises(ValueError, match="frame trim"):
        _aligned_prediction(prediction, state)


def test_append_silence_adds_exactly_three_native_ids() -> None:
    values = _append_silence_tokens(np.array([1, 2], dtype=np.int64))
    np.testing.assert_array_equal(values, np.array([1, 2, 4299, 4299, 4299], dtype=np.int64))


def test_embedding_projection_preserves_norm_and_cosine_floor() -> None:
    torch = pytest.importorskip("torch")
    baseline = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    candidate = torch.tensor([[-1.0, 0.0, 0.0]], dtype=torch.float32)
    baseline_unit = torch.nn.functional.normalize(baseline, dim=1)
    cosine, projected = _project_embedding_to_cap(torch, candidate, baseline_unit, 1.0, 0.98)
    assert projected is True
    assert cosine >= 0.98 - 1e-6
    assert abs(float(torch.linalg.vector_norm(candidate)) - 1.0) < 1e-6


def test_parser_defaults_to_bounded_two_step_fit() -> None:
    args = build_parser().parse_args([])
    assert args.steps == 2
    assert args.max_steps <= 60
    assert args.min_cosine >= 0.98


def test_prepare_join_rejects_wrong_source_frame_geometry(tmp_path: Path) -> None:
    import fit_decoder_embedding as fit

    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    arrays = tmp_path / "row.mel.npz"
    # N=3 requires source frames 6 or 5.  Seven is a deliberate mismatch.
    np.savez(arrays, source_mel=np.zeros((80, 7), dtype=np.float32), reconstructed_mel=np.zeros((80, 12), dtype=np.float32))
    cache = {
        "format": "nano_t3_adaptation_cache_v1",
        "rows": [{"id": "row", "split": "train", "audio_path": str(source), "speech_tokens": [1, 2, 3], "seed": 31}],
    }
    cache_path = tmp_path / "cache.json"
    cache_path.write_text(json.dumps(cache))
    manifest = {
        "format": "nano_mel_calibration_prepare_v1",
        "cache": str(cache_path),
        "cache_sha256": fit._sha256(cache_path),
        "conditionals_path": str(tmp_path / "conds.pt"),
        "rows": [{
            "id": "row", "split": "train", "audio_path": str(source),
            "source_sha256": fit._sha256(source), "speech_token_count": 3,
            "seed": 31,
            "mel_path": str(arrays),
        }],
    }
    (tmp_path / "conds.pt").write_bytes(b"conds")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="expected 2N or 2N-1"):
        fit._validate_prepare_inputs(tmp_path)
