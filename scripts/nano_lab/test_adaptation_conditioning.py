"""Focused, model-free regressions for Nano adaptation conditioning.

Run with ``.venv-nano/bin/python -m pytest scripts/nano_lab/test_adaptation_conditioning.py``.
The fake model below deliberately exposes values that would change if the
adapter code reimplemented the reference path, normalized the speaker vector,
filtered prompt IDs, or accidentally passed the target path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adaptation  # noqa: E402


class _FakeT3:
    def __init__(self) -> None:
        self.speaker_emb = torch.tensor(
            [[0.125, -0.25, 0.375] + [0.0] * 253], dtype=torch.float32
        )
        # Include IDs outside the ordinary S3 codebook range.  The production
        # helper may return special markers; exact caching must retain them.
        self.cond_prompt_speech_tokens = torch.tensor([[7, 6561, 13, 6562]], dtype=torch.long)


class _FakeConds:
    def __init__(self) -> None:
        self.t3 = _FakeT3()


class _FakeModel:
    def __init__(self) -> None:
        self.conds = None
        self.calls: list[tuple[str, bool]] = []

    def prepare_conditionals(self, path: str, *, norm_loudness: bool) -> None:
        self.calls.append((path, norm_loudness))
        self.conds = _FakeConds()


def test_extract_conditioning_uses_upstream_values_and_reference_path(tmp_path: Path) -> None:
    reference = tmp_path / "reference.wav"
    target = tmp_path / "target.wav"
    reference.write_bytes(b"reference bytes")
    target.write_bytes(b"target bytes")
    model = _FakeModel()

    speaker, prompt, provenance = adaptation.extract_conditioning(model, reference)

    assert model.calls == [(str(reference.resolve()), True)]
    assert str(target.resolve()) not in {call[0] for call in model.calls}
    np.testing.assert_array_equal(speaker[:3], np.array([0.125, -0.25, 0.375], dtype=np.float32))
    assert prompt == [7, 6561, 13, 6562]
    assert provenance["path"] == str(reference.resolve())
    assert provenance["sha256"] == adaptation.file_sha256(reference)
    assert provenance["preprocessing"]["entrypoint"] == "ChatterboxTurboTTS.prepare_conditionals"
    assert provenance["preprocessing"]["prompt_tokens_unmodified"] is True


def test_choose_reference_rejects_target_path(tmp_path: Path) -> None:
    target = tmp_path / "target.wav"
    target.write_bytes(b"target")
    row = {
        "audio_path": str(target),
        "reference_audio_path": str(target),
        "text": "A transcript.",
        "split": "train",
        "speaker_id": "voice-a",
        "_manifest_index": 0,
    }
    with pytest.raises(ValueError, match="target as conditioning"):
        adaptation.choose_reference_path(
            row,
            [row],
            manifest_path=tmp_path / "manifest.json",
            global_reference=None,
            allow_self_reference=False,
        )


def test_distinct_same_source_intervals_cannot_cross_splits() -> None:
    rows = [
        {
            "audio_path": "/tmp/train.wav",
            "split": "train",
            "source_id": "source-a",
            "start_s": 10.0,
            "end_s": 20.0,
        },
        {
            "audio_path": "/tmp/valid.wav",
            "split": "valid",
            "source_id": "source-a",
            "start_s": 19.0,
            "end_s": 25.0,
        },
    ]
    with pytest.raises(ValueError, match="source intervals overlap"):
        adaptation.validate_target_splits(rows)
