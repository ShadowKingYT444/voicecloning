"""Pure NumPy tests for the mel calibration hypothesis.

These tests do not import Torch or start a model.  They cover the split gate,
bounded zero-mean fitting, unequal frame handling, and the reversible HiFT
wrapper contract.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from mel_calibration import (
    MEL_BANDS,
    configure_mel_calibration,
    envelope_mse,
    fit_delta,
    relative_spectral_envelope,
    reset_mel_calibration,
)


def _record(row_id: str, split: str, difference: np.ndarray) -> dict[str, object]:
    """Build a deterministic pair with a time-independent band difference."""

    reconstructed = np.tile(np.linspace(-1.0, 1.0, MEL_BANDS, dtype=np.float64)[:, None], (1, 3))
    source = reconstructed + np.asarray(difference, dtype=np.float64)[:, None]
    return {
        "id": row_id,
        "split": split,
        "source_mel": source,
        "reconstructed_mel": reconstructed,
    }


def test_fit_uses_train_only_and_is_bounded_zero_mean() -> None:
    train_difference = np.sin(np.linspace(0.0, 4.0 * np.pi, MEL_BANDS)) * 0.7
    train_rows = [
        _record("train-a", "train", train_difference),
        _record("train-b", "train", train_difference * 0.9),
    ]
    rows_without_valid = list(train_rows)
    rows_with_valid = rows_without_valid + [_record("heldout", "valid", np.full(MEL_BANDS, 99.0))]

    delta_without_valid = fit_delta(rows_without_valid, sigma=2.0, max_delta=0.35)
    delta_with_valid = fit_delta(rows_with_valid, sigma=2.0, max_delta=0.35)

    # The valid row is intentionally extreme.  It must not affect the fitted
    # correction, which is estimated from train rows only.
    np.testing.assert_allclose(delta_with_valid, delta_without_valid, rtol=0.0, atol=1e-12)
    assert delta_with_valid.shape == (MEL_BANDS,)
    assert float(np.max(np.abs(delta_with_valid))) <= 0.35 + 1e-12
    assert abs(float(np.mean(delta_with_valid))) <= 1e-12


def test_envelope_mse_does_not_require_frame_alignment() -> None:
    source = np.arange(MEL_BANDS * 5, dtype=np.float64).reshape(MEL_BANDS, 5)
    reconstructed = np.arange(MEL_BANDS * 3, dtype=np.float64).reshape(MEL_BANDS, 3)
    zero = np.zeros(MEL_BANDS, dtype=np.float64)
    value = envelope_mse(source, reconstructed, zero, strength=1.0)
    assert np.isfinite(value)
    assert value >= 0.0
    assert relative_spectral_envelope(source).shape == (MEL_BANDS,)


class _FakeMel2Wav:
    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, object]] = []

    def inference(self, speech_feat: np.ndarray, cache_source: object = None) -> np.ndarray:
        self.calls.append((np.array(speech_feat, copy=True), cache_source))
        return speech_feat


class _FakeS3Gen:
    def __init__(self) -> None:
        self.mel2wav = _FakeMel2Wav()


class _FakeModel:
    def __init__(self) -> None:
        self.s3gen = _FakeS3Gen()


def test_wrapper_zero_bypass_and_reset_restore_original() -> None:
    model = _FakeModel()
    input_mel = np.zeros((1, MEL_BANDS, 2), dtype=np.float32)
    input_mel[:, 3, :] = 2.0

    # Zero strength and no path must leave the original call unchanged.
    original_result = model.s3gen.mel2wav.inference(speech_feat=input_mel, cache_source="cache")
    np.testing.assert_array_equal(original_result, input_mel)
    original_call = model.s3gen.mel2wav.calls[-1]
    assert original_call[1] == "cache"

    with tempfile.TemporaryDirectory() as temporary:
        delta_path = Path(temporary) / "delta.npy"
        delta = np.zeros(MEL_BANDS, dtype=np.float32)
        delta[3] = 0.2
        np.save(delta_path, delta)

        configure_mel_calibration(model, delta_path, strength=0.5)
        adjusted = model.s3gen.mel2wav.inference(speech_feat=input_mel, cache_source="cache")
        expected = input_mel.copy()
        expected[:, 3, :] += 0.1
        np.testing.assert_allclose(adjusted, expected, rtol=0.0, atol=1e-7)
        assert model.s3gen.mel2wav.calls[-1][1] == "cache"

        # Reconfiguring with zero strength must restore the exact original
        # implementation path.  A subsequent call must not retain the delta.
        configure_mel_calibration(model, delta_path, strength=0.0)
        restored = model.s3gen.mel2wav.inference(speech_feat=input_mel, cache_source="cache")
        np.testing.assert_array_equal(restored, input_mel)

        # Explicit reset is idempotent and leaves the unwrapped method usable.
        reset_mel_calibration(model)
        restored_again = model.s3gen.mel2wav.inference(input_mel, "cache")
        np.testing.assert_array_equal(restored_again, input_mel)

