"""Pure contract tests for :mod:`prepare_acoustic_data`.

These tests use tiny temporary files and NumPy arrays.  They do not load a
checkpoint, import Torch, decode audio, or start a model job.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from prepare_acoustic_data import (
    EXPECTED_CONDITIONALS_SHA256,
    MEL_BANDS,
    SAMPLE_RATE_16K,
    SAMPLE_RATE_24K,
    TOKEN_RATE_HZ,
    _json_sha256,
    SPEECH_VOCAB,
    _normalize_tokens,
    _validate_aligned_overlap,
    _validate_source_manifest,
    _source_rows_for_resume,
    validate_acoustic_cache,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_source_manifest(tmp_path: Path, *, with_text: bool = False) -> tuple[Path, dict[str, object]]:
    clip = tmp_path / "clip.wav"
    source = tmp_path / "source.mp3"
    proposal = tmp_path / "proposal.json"
    clip.write_bytes(b"recorded-waveform")
    source.write_bytes(b"recorded-source")
    protected = [{"id": "protected", "section": "heldout", "start_s": 1.0, "end_s": 3.0}]
    proposal.write_text(json.dumps({
        "source": {"absolute_path": str(source), "actual_sha256": _sha(source)},
        "protected_intervals": protected,
    }) + "\n")
    row: dict[str, object] = {
        "id": "clip-a",
        "path": str(clip),
        "audio_sha256": _sha(clip),
        "source_audio": str(source),
        "source_audio_sha256": _sha(source),
        "start_s": 10.0,
        "end_s": 12.0,
        "duration_s": 2.0,
        "voice": "fixture",
        "synthetic_tts": False,
        "split": "train",
    }
    if with_text:
        row["text"] = "must be rejected"
    payload: dict[str, object] = {
        "format": "nano_acoustic_source_manifest_v1",
        "status": "verified",
        "transcript_labels_used": False,
        "source_proposal": str(proposal),
        "source_proposal_sha256": _sha(proposal),
        "protected_interval_buffer_s": 2.0,
        "protected_intervals": protected,
        "valid_ids": [],
        "counts": {"train": 1, "valid": 0},
        "rows": [row],
    }
    path = tmp_path / "source_manifest.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path, payload


def test_source_manifest_accepts_audio_only_rows(tmp_path: Path) -> None:
    path, _ = _write_source_manifest(tmp_path)
    normalized, rows, contract = _validate_source_manifest(path)
    assert normalized["transcript_labels_used"] is False
    assert rows[0]["audio_path"].endswith("clip.wav")
    assert contract["protected_interval_buffer_s"] == pytest.approx(2.0)


def test_source_manifest_rejects_transcript_fields(tmp_path: Path) -> None:
    path, _ = _write_source_manifest(tmp_path, with_text=True)
    with pytest.raises(ValueError, match="transcript label"):
        _validate_source_manifest(path)


def test_token_normalization_matches_filter_and_rejects_invalid_values() -> None:
    values = _normalize_tokens(np.array([7000, 9, 10, 6561], dtype=np.int64), row_id="row")
    np.testing.assert_array_equal(values, np.array([9, 10], dtype=np.int64))
    with pytest.raises(ValueError, match="non-integer"):
        _normalize_tokens(np.array([1.5, 2.0], dtype=np.float32), row_id="row")
    with pytest.raises(ValueError, match="fewer than two"):
        _normalize_tokens(np.array([SPEECH_VOCAB, 4], dtype=np.int64), row_id="row")


def test_aligned_overlap_requires_exact_tokens_and_audio_hash(tmp_path: Path) -> None:
    aligned = tmp_path / "aligned.json"
    aligned_payload = {"format": "nano_t3_adaptation_cache_v1", "rows": [{
        "id": "clip-a",
        "audio_path": str(tmp_path / "clip.wav"),
        "speech_tokens": [1, 2, 3],
        "transcript_audit": {"audio_sha256": "audio-sha"},
    }]}
    aligned.write_text(json.dumps(aligned_payload))
    source_rows = [{"id": "clip-a", "audio_path": str(tmp_path / "clip.wav"), "audio_sha256": "audio-sha", "speech_tokens": [1, 2, 3]}]
    result = _validate_aligned_overlap(source_rows, aligned)
    assert result["status"] == "passed"
    assert result["token_equality"] == "exact"
    assert result["overlap_ids"] == ["clip-a"]
    source_rows[0]["speech_tokens"] = [1, 2, 4]
    with pytest.raises(ValueError, match="speech tokens differ"):
        _validate_aligned_overlap(source_rows, aligned)
    with pytest.raises(ValueError, match="no matching source rows"):
        _validate_aligned_overlap([], aligned)


def test_validate_acoustic_cache_expands_npz_without_text(tmp_path: Path) -> None:
    source_path, _ = _write_source_manifest(tmp_path)
    source_manifest, source_rows, contract = _validate_source_manifest(source_path)
    output = tmp_path / "cache"
    output.mkdir()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    checkpoint = model_dir / "s3gen_meanflow.safetensors"
    checkpoint.write_bytes(b"checkpoint-fixture")
    artifact = output / "clip-a.npz"
    tokens = np.array([4, 5, 6], dtype=np.int64)
    mel = np.zeros((80, 6), dtype=np.float32)
    with artifact.open("wb") as handle:
        np.savez(handle, speech_tokens=tokens, source_mel=mel)
    cache_manifest = {
        "format": "nano_acoustic_cache_v1",
        "status": "ready",
        "transcript_labels_used": False,
        "synthetic_audio": False,
        "source_manifest": str(source_path),
        "source_manifest_sha256": _sha(source_path),
        "source_proposal": str(tmp_path / "proposal.json"),
        "source_proposal_sha256": contract["source_proposal_sha256"],
        "protected_interval_buffer_s": 2.0,
        "protected_intervals_sha256": contract["protected_intervals_sha256"],
        "source_contract_sha256": _json_sha256({
            "sample_rate_16k": SAMPLE_RATE_16K,
            "sample_rate_24k": SAMPLE_RATE_24K,
            "token_rate_hz": TOKEN_RATE_HZ,
            "mel_bands": MEL_BANDS,
            "source_contract": contract,
        }),
        "model_dir": str(model_dir),
        "model_checkpoint": str(checkpoint),
        "model_checkpoint_sha256": _sha(checkpoint),
        "target_sample_rate": SAMPLE_RATE_16K,
        "source_sample_rate": SAMPLE_RATE_24K,
        "target_token_rate_hz": TOKEN_RATE_HZ,
        "mel_bands": MEL_BANDS,
        "aligned_overlap": {"status": "passed", "token_equality": "exact", "overlap_ids": ["clip-a"], "overlap_count": 1},
        "rows": [{
            "id": "clip-a",
            "split": "train",
            "audio_path": source_rows[0]["audio_path"],
            "audio_sha256": source_rows[0]["audio_sha256"],
            "source_audio_sha256": source_rows[0]["source_audio_sha256"],
            "speech_token_count": 3,
            "source_mel_shape": [80, 6],
            "npz_path": str(artifact),
            "npz_sha256": _sha(artifact),
        }],
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(cache_manifest, indent=2) + "\n")
    expanded = validate_acoustic_cache(manifest_path)
    assert expanded["rows"][0]["audio_path"] == source_rows[0]["audio_path"]
    assert expanded["rows"][0]["speech_tokens"] == [4, 5, 6]
    assert "text" not in expanded["rows"][0]


def test_partial_cache_is_not_accepted(tmp_path: Path) -> None:
    source_path, _ = _write_source_manifest(tmp_path)
    output = tmp_path / "cache"
    output.mkdir()
    path = output / "manifest.json"
    path.write_text(json.dumps({
        "format": "nano_acoustic_cache_v1",
        "status": "partial",
        "transcript_labels_used": False,
        "source_manifest": str(source_path),
    }))
    with pytest.raises(ValueError, match="not ready"):
        validate_acoustic_cache(path)


def test_ready_cache_rejects_float_or_out_of_range_persisted_tokens(tmp_path: Path) -> None:
    source_path, _ = _write_source_manifest(tmp_path)
    source_manifest, source_rows, contract = _validate_source_manifest(source_path)
    output = tmp_path / "cache"
    output.mkdir()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    checkpoint = model_dir / "s3gen_meanflow.safetensors"
    checkpoint.write_bytes(b"checkpoint-fixture")
    artifact = output / "clip-a.npz"
    with artifact.open("wb") as handle:
        np.savez(handle, speech_tokens=np.array([4.0, 5.0, 6.0], dtype=np.float32), source_mel=np.zeros((80, 6), dtype=np.float32))
    cache = {
        "format": "nano_acoustic_cache_v1", "status": "ready", "transcript_labels_used": False, "synthetic_audio": False,
        "source_manifest": str(source_path), "source_manifest_sha256": _sha(source_path),
        "source_proposal": str(tmp_path / "proposal.json"), "source_proposal_sha256": contract["source_proposal_sha256"],
        "protected_interval_buffer_s": 2.0, "protected_intervals_sha256": contract["protected_intervals_sha256"],
        "source_contract_sha256": _json_sha256({"sample_rate_16k": SAMPLE_RATE_16K, "sample_rate_24k": SAMPLE_RATE_24K, "token_rate_hz": TOKEN_RATE_HZ, "mel_bands": MEL_BANDS, "source_contract": contract}),
        "model_dir": str(model_dir), "model_checkpoint": str(checkpoint), "model_checkpoint_sha256": _sha(checkpoint),
        "target_sample_rate": SAMPLE_RATE_16K, "source_sample_rate": SAMPLE_RATE_24K, "target_token_rate_hz": TOKEN_RATE_HZ, "mel_bands": MEL_BANDS,
        "aligned_overlap": {"status": "passed", "token_equality": "exact", "overlap_ids": ["clip-a"], "overlap_count": 1},
        "rows": [{"id": "clip-a", "split": "train", "audio_path": source_rows[0]["audio_path"], "audio_sha256": source_rows[0]["audio_sha256"], "source_audio_sha256": source_rows[0]["source_audio_sha256"], "speech_token_count": 3, "source_mel_shape": [80, 6], "npz_path": str(artifact), "npz_sha256": _sha(artifact)}],
    }
    path = output / "manifest.json"
    path.write_text(json.dumps(cache))
    with pytest.raises(ValueError, match="1-D int64"):
        validate_acoustic_cache(path)


def test_conditionals_hash_constant_is_explicit() -> None:
    assert len(EXPECTED_CONDITIONALS_SHA256) == 64
    assert EXPECTED_CONDITIONALS_SHA256 == EXPECTED_CONDITIONALS_SHA256.lower()


def test_source_identity_cannot_change_with_a_rehashed_manifest(tmp_path: Path) -> None:
    path, payload = _write_source_manifest(tmp_path)
    substitute = tmp_path / "substitute.mp3"
    substitute.write_bytes(b"different source")
    payload["rows"][0]["source_audio"] = str(substitute)
    payload["rows"][0]["source_audio_sha256"] = _sha(substitute)
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="proposal"):
        _validate_source_manifest(path)


def test_resume_rejects_stale_split_even_when_artifact_hash_matches(tmp_path: Path) -> None:
    source_path, _ = _write_source_manifest(tmp_path)
    _, rows, contract = _validate_source_manifest(source_path)
    artifact = tmp_path / "tokens.npz"
    np.savez(artifact, speech_tokens=np.array([4, 5, 6], dtype=np.int64),
             source_mel=np.zeros((80, 6), dtype=np.float32))
    record = {**rows[0], "npz_path": str(artifact), "npz_sha256": _sha(artifact),
              "speech_token_count": 3, "source_mel_shape": [80, 6]}
    manifest = tmp_path / "cache.json"
    payload = {"format": "nano_acoustic_cache_v1", "status": "partial",
               "source_manifest_sha256": contract["source_manifest_sha256"],
               "model_checkpoint_sha256": "model", "source_contract_sha256": "config",
               "rows": [record]}
    manifest.write_text(json.dumps(payload))
    assert _source_rows_for_resume(manifest, contract, rows, "model", "config")
    record["split"] = "valid"
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="metadata differs for split"):
        _source_rows_for_resume(manifest, contract, rows, "model", "config")
