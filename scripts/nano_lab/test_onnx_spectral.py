"""Numerical tests for the real HiFT spectral replacement.

These tests are intentionally independent of Chatterbox checkpoints.  They
exercise the fixed n_fft=16/hop=4 arithmetic at several waveform lengths,
including a non-multiple of the hop, and compare both individual transforms
and a full magnitude/phase round trip against PyTorch's native complex
operators.

Run through the lab's bounded job wrapper when the desktop is under memory
pressure::

    .venv-nano/bin/python -m pytest scripts/nano_lab/test_onnx_spectral.py -q
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from onnx_spectral import (
    HOP_LENGTH,
    N_FFT,
    build_hift_spectral_stage,
    periodic_hann,
    real_istft,
    real_stft,
)


def _native_stft(waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    native = torch.stft(
        waveform,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=N_FFT,
        window=periodic_hann(),
        center=True,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    return native.real, native.imag


@pytest.mark.parametrize("length", [9, 16, 17, 31, 32, 41, 64])
def test_real_stft_matches_native(length: int) -> None:
    torch.manual_seed(1000 + length)
    waveform = torch.randn(2, length, dtype=torch.float32)
    expected_real, expected_imag = _native_stft(waveform)
    actual_real, actual_imag = real_stft(waveform)
    assert actual_real.shape == expected_real.shape
    assert actual_imag.shape == expected_imag.shape
    torch.testing.assert_close(actual_real, expected_real, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(actual_imag, expected_imag, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("frames", [2, 3, 5, 9, 17])
def test_real_istft_matches_native(frames: int) -> None:
    torch.manual_seed(2000 + frames)
    magnitude = torch.rand(2, N_FFT // 2 + 1, frames, dtype=torch.float32) * 3
    phase = torch.randn_like(magnitude)
    spectrum = torch.complex(magnitude.clamp(max=100), torch.zeros_like(magnitude))
    # Construct the same complex spectrum as the upstream _istft call.
    spectrum = torch.polar(magnitude.clamp(max=100), phase)
    expected = torch.istft(
        spectrum,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=N_FFT,
        window=periodic_hann(),
        center=True,
        normalized=False,
        onesided=True,
        return_complex=False,
    )
    actual = real_istft(magnitude, phase)
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)


@pytest.mark.parametrize("length", [16, 17, 32, 41, 64])
def test_roundtrip_matches_native(length: int) -> None:
    torch.manual_seed(3000 + length)
    waveform = torch.randn(2, length, dtype=torch.float32)
    expected_real, expected_imag = _native_stft(waveform)
    expected_mag = torch.sqrt(expected_real.square() + expected_imag.square())
    expected_phase = torch.atan2(expected_imag, expected_real)
    expected = torch.istft(
        torch.polar(expected_mag.clamp(max=100), expected_phase),
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=N_FFT,
        window=periodic_hann(),
        center=True,
        normalized=False,
        onesided=True,
        return_complex=False,
    )
    actual = build_hift_spectral_stage("roundtrip")(waveform)
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, atol=5e-5, rtol=5e-5)


def test_magnitude_clip_matches_upstream() -> None:
    torch.manual_seed(4000)
    magnitude = torch.full((1, N_FFT // 2 + 1, 5), 150.0)
    phase = torch.randn_like(magnitude)
    expected = torch.istft(
        torch.polar(magnitude.clamp(max=100), phase),
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=N_FFT,
        window=periodic_hann(),
        center=True,
        normalized=False,
        onesided=True,
        return_complex=False,
    )
    actual = real_istft(magnitude, phase)
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)

