"""Pure contract tests for T3 donor-field copying."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from t3_conditioning_donor import copy_t3_conditioning


def _conditionals(torch, *, marker: float, prompt_dtype=None):
    prompt_dtype = prompt_dtype or torch.int64
    t3 = SimpleNamespace(
        speaker_emb=torch.full((1, 256), marker, dtype=torch.float32),
        cond_prompt_speech_tokens=torch.tensor([[1, 2, 3]], dtype=prompt_dtype),
        cond_prompt_speech_emb=torch.full((1, 3, 4), marker, dtype=torch.float32),
    )
    # Use a mutable object so the test can verify that the helper never edits
    # any destination.gen fields.
    return SimpleNamespace(t3=t3, gen={"embedding": torch.full((1, 192), marker)})


def test_prompt_mode_copies_only_prompt_and_invalidates_derived_embedding():
    torch = pytest.importorskip("torch")
    donor = _conditionals(torch, marker=7.0)
    donor.t3.cond_prompt_speech_tokens = torch.tensor([[511, 1024]], dtype=torch.int64)
    destination = _conditionals(torch, marker=2.0)
    gen_identity = destination.gen
    gen_before = destination.gen["embedding"].clone()
    speaker_before = destination.t3.speaker_emb.clone()
    result = copy_t3_conditioning(donor, destination, "prompt")
    assert result["selected_fields"] == ["t3.cond_prompt_speech_tokens"]
    assert destination.gen is gen_identity
    assert torch.equal(destination.gen["embedding"], gen_before)
    assert torch.equal(destination.t3.speaker_emb, speaker_before)
    assert torch.equal(destination.t3.cond_prompt_speech_tokens, donor.t3.cond_prompt_speech_tokens)
    assert destination.t3.cond_prompt_speech_tokens.data_ptr() != donor.t3.cond_prompt_speech_tokens.data_ptr()
    assert destination.t3.cond_prompt_speech_emb is None


def test_speaker_mode_copies_only_speaker_and_preserves_prompt():
    torch = pytest.importorskip("torch")
    donor = _conditionals(torch, marker=9.0)
    destination = _conditionals(torch, marker=2.0)
    prompt_before = destination.t3.cond_prompt_speech_tokens.clone()
    prompt_emb_before = destination.t3.cond_prompt_speech_emb.clone()
    result = copy_t3_conditioning(donor, destination, "speaker")
    assert result["selected_fields"] == ["t3.speaker_emb"]
    assert torch.equal(destination.t3.speaker_emb, donor.t3.speaker_emb)
    assert destination.t3.speaker_emb.data_ptr() != donor.t3.speaker_emb.data_ptr()
    assert torch.equal(destination.t3.cond_prompt_speech_tokens, prompt_before)
    assert torch.equal(destination.t3.cond_prompt_speech_emb, prompt_emb_before)


def test_prompt_and_speaker_mode_copies_both():
    torch = pytest.importorskip("torch")
    donor = _conditionals(torch, marker=11.0)
    destination = _conditionals(torch, marker=1.0)
    result = copy_t3_conditioning(donor, destination, "prompt_and_speaker")
    assert result["selected_fields"] == ["t3.cond_prompt_speech_tokens", "t3.speaker_emb"]
    assert torch.equal(destination.t3.speaker_emb, donor.t3.speaker_emb)
    assert torch.equal(destination.t3.cond_prompt_speech_tokens, donor.t3.cond_prompt_speech_tokens)
    assert destination.t3.cond_prompt_speech_emb is None


def test_invalid_combined_donor_fails_before_any_mutation():
    torch = pytest.importorskip("torch")
    donor = _conditionals(torch, marker=7.0)
    donor.t3.speaker_emb[0, 0] = float("nan")
    destination = _conditionals(torch, marker=2.0)
    before_prompt = destination.t3.cond_prompt_speech_tokens.clone()
    before_speaker = destination.t3.speaker_emb.clone()
    before_prompt_emb = destination.t3.cond_prompt_speech_emb.clone()
    with pytest.raises(ValueError, match="speaker_emb"):
        copy_t3_conditioning(donor, destination, "prompt_and_speaker")
    assert torch.equal(destination.t3.cond_prompt_speech_tokens, before_prompt)
    assert torch.equal(destination.t3.speaker_emb, before_speaker)
    assert torch.equal(destination.t3.cond_prompt_speech_emb, before_prompt_emb)


def test_invalid_prompt_mode_and_ids_fail_without_mutation():
    torch = pytest.importorskip("torch")
    donor = _conditionals(torch, marker=7.0)
    donor.t3.cond_prompt_speech_tokens[0, 1] = 6561
    destination = _conditionals(torch, marker=2.0)
    before = destination.t3.speaker_emb.clone()
    with pytest.raises(ValueError, match="out-of-range"):
        copy_t3_conditioning(donor, destination, "prompt")
    assert torch.equal(destination.t3.speaker_emb, before)
    with pytest.raises(ValueError, match="invalid T3 donor mode"):
        copy_t3_conditioning(donor, destination, "bad")


def test_prompt_requires_int64_and_bounded_length():
    torch = pytest.importorskip("torch")
    donor = _conditionals(torch, marker=7.0, prompt_dtype=torch.int32)
    destination = _conditionals(torch, marker=2.0)
    with pytest.raises(ValueError, match="int64"):
        copy_t3_conditioning(donor, destination, "prompt")
    donor = _conditionals(torch, marker=7.0)
    donor.t3.cond_prompt_speech_tokens = torch.zeros((1, 376), dtype=torch.int64)
    with pytest.raises(ValueError, match="shape"):
        copy_t3_conditioning(donor, destination, "prompt")
    donor.t3.cond_prompt_speech_tokens = torch.zeros((1, 375), dtype=torch.int64)
    copy_t3_conditioning(donor, destination, 'prompt')
    assert destination.t3.cond_prompt_speech_tokens.shape == (1, 375)


def test_destination_dtype_and_device_are_used_for_clones():
    torch = pytest.importorskip("torch")
    donor = _conditionals(torch, marker=7.0)
    destination = _conditionals(torch, marker=2.0)
    destination.t3.speaker_emb = destination.t3.speaker_emb.double()
    result = copy_t3_conditioning(donor, destination, "speaker")
    assert result["speaker_shape"] == [1, 256]
    assert destination.t3.speaker_emb.dtype == torch.float64
    assert torch.equal(destination.t3.speaker_emb.float(), donor.t3.speaker_emb)


def test_narrowing_conversion_fails_before_mutation():
    torch = pytest.importorskip('torch')
    donor = _conditionals(torch, marker=1e10)
    destination = _conditionals(torch, marker=2.0)
    original_prompt = destination.t3.cond_prompt_speech_tokens
    destination.t3.speaker_emb = destination.t3.speaker_emb.half()
    with pytest.raises(ValueError, match='overflowed'):
        copy_t3_conditioning(donor, destination, 'prompt_and_speaker')
    assert destination.t3.cond_prompt_speech_tokens is original_prompt
    destination.t3.cond_prompt_speech_tokens = original_prompt.to(torch.int8)
    with pytest.raises(ValueError, match='int64'):
        copy_t3_conditioning(donor, destination, 'prompt')
