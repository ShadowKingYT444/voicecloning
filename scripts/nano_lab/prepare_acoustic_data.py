"""Prepare an audit-safe acoustic cache for Chatterbox Nano.

This module has two separate stages.  ``tokens`` loads only the S3 tokenizer,
extracts speech tokens from the recorded 16 kHz waveform, and writes a token
and normalized 24 kHz mel cache.  ``flow`` loads only the S3Gen flow encoder
and meanflow estimator and writes native flow reconstructions.  The stages
run in separate processes so the tokenizer and flow weights are not resident
at the same time.

The cache deliberately has no transcript field.  It is source-audio
supervision only.  It cannot replace the audited text/T3 adaptation cache.
Every source interval, file hash, split, protected interval, and model or
conditioning hash is checked before a ready status is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
VENDOR_SRC = ROOT / "vendor" / "chatterbox" / "src"
DEFAULT_SOURCE_MANIFEST = ROOT / "artifacts" / "nano_lab" / "acoustic_only_asmr" / "manifest.json"
DEFAULT_TOKEN_OUTPUT = ROOT / "artifacts" / "nano_lab" / "acoustic_only_asmr" / "cache"
DEFAULT_FLOW_OUTPUT = ROOT / "artifacts" / "nano_lab" / "acoustic_only_asmr" / "flow"
DEFAULT_MODEL_DIR = ROOT / "models" / "chatterbox-nano"
DEFAULT_CONDITIONALS = ROOT / "artifacts" / "nano_lab" / "sweep_round2" / "asmr_conversational_morning_31.conds.pt"
DEFAULT_REFERENCE_PREPARE = ROOT / "artifacts" / "nano_lab" / "mel_calibration_asmr"
DEFAULT_ALIGNED_CACHE = ROOT / "artifacts" / "nano_lab" / "adaptation_asmr_aligned_cache.json"
EXPECTED_CONDITIONALS_SHA256 = "92e644466f79befe5ef937f819c921729ba00f29790efb81684cc3e88ef18c52"
S3GEN_CHECKPOINT_NAME = "s3gen_meanflow.safetensors"
S3GEN_SIL = 4299
SPEECH_VOCAB = 6561
SAMPLE_RATE_16K = 16_000
SAMPLE_RATE_24K = 24_000
TOKEN_RATE_HZ = 25
MEL_BANDS = 80
FLOW_STEPS = 2
PROTECTED_BUFFER_DEFAULT = 2.0
FORBIDDEN_LABEL_KEYS = {
    "text",
    "source_text",
    "transcript",
    "transcript_audit",
    "text_tokens",
    "asr",
    "hypothesis",
}


def _resolve(value: str | os.PathLike[str], *, base: Path = ROOT) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _atomic_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **{key: np.asarray(value) for key, value in arrays.items()})
    os.replace(temporary, path)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def _rss_bytes() -> int:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, OSError, ValueError):
        pass
    return 0


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _configure_torch_threads(torch: Any) -> None:
    """Keep each bounded stage at the two-core project limit."""

    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(2)
    except RuntimeError:
        # A caller may have configured inter-op workers before importing this
        # module.  The intra-op limit above still applies, and a new bounded
        # process sets both values before any work starts.
        pass


def _finite_number(value: Any, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _forbidden_label_keys(value: Any, *, path: str = "row") -> list[str]:
    """Return transcript-like keys, including nested audit payloads."""

    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            if key_text in FORBIDDEN_LABEL_KEYS:
                found.append(f"{path}.{key_text}")
            found.extend(_forbidden_label_keys(child, path=f"{path}.{key_text}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_forbidden_label_keys(child, path=f"{path}[{index}]"))
    return found


def _normalize_tokens(values: Any, *, row_id: str) -> np.ndarray:
    """Apply the same post-processing as adaptation.extract_target_tokens."""

    array = np.asarray(values)
    if array.ndim != 1:
        array = array.reshape(-1)
    if array.size == 0:
        raise ValueError(f"row {row_id} produced no speech tokens")
    if not np.issubdtype(array.dtype, np.integer):
        if not np.isfinite(array).all() or not np.equal(array, np.floor(array)).all():
            raise ValueError(f"row {row_id} produced non-integer speech IDs")
    array = array.astype(np.int64, copy=False)
    array = array[array < SPEECH_VOCAB]
    if array.size < 2:
        raise ValueError(f"row {row_id} produced fewer than two valid speech tokens")
    if np.any(array < 0) or np.any(array >= SPEECH_VOCAB):
        raise ValueError(f"row {row_id} produced invalid speech IDs")
    return np.array(array, dtype=np.int64, copy=True)


def _validate_persisted_tokens(values: Any, *, row_id: str, expected_count: int | None = None) -> np.ndarray:
    """Validate an on-disk token array without applying extraction filters.

    The tokenizer extraction path intentionally removes IDs outside the speech
    vocabulary.  A persisted cache must already contain only valid IDs.  A
    malformed artifact must fail closed instead of being silently repaired.
    """

    array = np.asarray(values)
    if array.ndim != 1 or array.dtype != np.dtype(np.int64):
        raise ValueError(f"row {row_id} persisted speech_tokens must be a 1-D int64 array")
    if array.size < 2 or np.any(array < 0) or np.any(array >= SPEECH_VOCAB):
        raise ValueError(f"row {row_id} persisted speech_tokens contain invalid IDs")
    if expected_count is not None and int(array.size) != int(expected_count):
        raise ValueError(f"row {row_id} persisted speech token count does not match the manifest")
    return np.array(array, dtype=np.int64, copy=True)


def _validate_mel(value: Any, *, row_id: str, name: str = "source_mel") -> np.ndarray:
    mel = np.asarray(value, dtype=np.float32)
    if mel.ndim == 3 and mel.shape[0] == 1:
        mel = mel[0]
    if mel.ndim != 2 or mel.shape[0] != MEL_BANDS or mel.shape[1] < 1:
        raise ValueError(f"row {row_id} {name} must have shape (80,T), got {mel.shape}")
    if not np.isfinite(mel).all():
        raise ValueError(f"row {row_id} {name} contains NaN or infinity")
    return np.array(mel, dtype=np.float32, copy=True)


def _validate_source_manifest(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Validate the immutable source proposal and return canonical rows."""

    path = path.resolve()
    payload = json.loads(path.read_text())
    if not isinstance(payload, Mapping) or payload.get("format") != "nano_acoustic_source_manifest_v1":
        raise ValueError("source manifest must use nano_acoustic_source_manifest_v1")
    if payload.get("transcript_labels_used") is not False:
        raise ValueError("acoustic source manifest must declare transcript_labels_used=false")
    rows_value = payload.get("rows")
    if not isinstance(rows_value, list) or not rows_value:
        raise ValueError("source manifest has no rows")
    proposal = _resolve(str(payload.get("source_proposal")), base=path.parent)
    if not proposal.exists():
        raise FileNotFoundError(proposal)
    proposal_sha = _sha256(proposal)
    if payload.get("source_proposal_sha256") != proposal_sha:
        raise ValueError("source proposal SHA-256 does not match source manifest")
    proposal_payload = json.loads(proposal.read_text())
    proposal_source = proposal_payload.get("source") if isinstance(proposal_payload, Mapping) else None
    buffer_s = _finite_number(payload.get("protected_interval_buffer_s", PROTECTED_BUFFER_DEFAULT), field="protected_interval_buffer_s")
    if buffer_s < 0 or buffer_s > 60:
        raise ValueError("protected interval buffer is outside the allowed range")
    protected_value = payload.get("protected_intervals")
    if not isinstance(protected_value, list):
        raise ValueError("source manifest has no protected intervals")
    proposal_protected = proposal_payload.get("protected_intervals") if isinstance(proposal_payload, Mapping) else None
    if proposal_protected is not None:
        def _interval_signature(items: Any) -> list[dict[str, Any]]:
            if not isinstance(items, list):
                raise ValueError("protected intervals in source proposal are not a list")
            signature: list[dict[str, Any]] = []
            for item in items:
                if not isinstance(item, Mapping):
                    raise ValueError("protected interval in source proposal is not an object")
                signature.append(
                    {
                        "id": str(item.get("id", "")),
                        "section": str(item.get("section", "")),
                        "start_s": _finite_number(item.get("start_s"), field="proposal protected start_s"),
                        "end_s": _finite_number(item.get("end_s"), field="proposal protected end_s"),
                    }
                )
            return signature

        if _interval_signature(protected_value) != _interval_signature(proposal_protected):
            raise ValueError("source manifest protected intervals differ from the source proposal")
    protected: list[tuple[float, float, str]] = []
    for item in protected_value:
        if not isinstance(item, Mapping):
            raise ValueError("protected interval is not an object")
        start = _finite_number(item.get("start_s"), field="protected start_s")
        end = _finite_number(item.get("end_s"), field="protected end_s")
        if end <= start:
            raise ValueError("protected interval end must be greater than start")
        protected.append((start, end, str(item.get("id", ""))))
    canonical: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    intervals: list[tuple[float, float, str, str]] = []
    path_hashes: dict[str, str] = {}
    for raw in rows_value:
        if not isinstance(raw, Mapping):
            raise ValueError("source row is not an object")
        forbidden = _forbidden_label_keys(raw)
        if forbidden:
            raise ValueError("source manifest contains transcript label fields: " + ", ".join(forbidden[:4]))
        row = dict(raw)
        row_id = str(row.get("id", ""))
        if not row_id or row_id in seen_ids:
            raise ValueError(f"duplicate or empty source row id: {row_id!r}")
        seen_ids.add(row_id)
        if row.get("synthetic_tts") is not False or row.get("synthetic_audio") is True:
            raise ValueError(f"source row {row_id} is marked synthetic")
        split = str(row.get("split", ""))
        if split not in {"train", "valid"}:
            raise ValueError(f"source row {row_id} has invalid split {split!r}")
        audio_path = _resolve(str(row.get("path", row.get("audio_path", ""))), base=path.parent)
        source_audio = _resolve(str(row.get("source_audio", "")), base=path.parent)
        if not audio_path.exists() or not source_audio.exists():
            raise FileNotFoundError(f"source row {row_id} audio path is missing")
        audio_sha = _sha256(audio_path)
        source_sha = _sha256(source_audio)
        if str(row.get("audio_sha256")) != audio_sha:
            raise ValueError(f"audio SHA-256 mismatch for source row {row_id}")
        if str(row.get("source_audio_sha256")) != source_sha:
            raise ValueError(f"source MP3 SHA-256 mismatch for source row {row_id}")
        if str(source_audio) in path_hashes and path_hashes[str(source_audio)] != source_sha:
            raise ValueError("source audio hash changed across rows")
        path_hashes[str(source_audio)] = source_sha
        start = _finite_number(row.get("start_s"), field=f"{row_id}.start_s")
        end = _finite_number(row.get("end_s"), field=f"{row_id}.end_s")
        duration = _finite_number(row.get("duration_s"), field=f"{row_id}.duration_s")
        if end <= start or duration <= 0 or abs((end - start) - duration) > 0.03:
            raise ValueError(f"source row {row_id} has inconsistent interval duration")
        for protected_start, protected_end, protected_id in protected:
            if start < protected_end + buffer_s and end > protected_start - buffer_s:
                raise ValueError(f"source row {row_id} intersects protected interval {protected_id!r} with buffer")
        intervals.append((start, end, split, row_id))
        canonical.append(
            {
                "id": row_id,
                "split": split,
                "audio_path": str(audio_path),
                "source_audio": str(source_audio),
                "audio_sha256": audio_sha,
                "source_audio_sha256": source_sha,
                "start_s": start,
                "end_s": end,
                "duration_s": duration,
                "voice": str(row.get("voice", "")),
            }
        )
    if isinstance(proposal_source, Mapping):
        proposal_path_value = proposal_source.get("absolute_path", proposal_source.get("path"))
        proposal_hash_value = proposal_source.get("actual_sha256", proposal_source.get("sha256"))
        if proposal_path_value is not None:
            proposal_source_path = _resolve(str(proposal_path_value), base=proposal.parent)
            if {row["source_audio"] for row in canonical} != {str(proposal_source_path)}:
                raise ValueError("source row source_audio path differs from the source proposal")
        if proposal_hash_value is not None and {row["source_audio_sha256"] for row in canonical} != {str(proposal_hash_value)}:
            raise ValueError("source row source_audio hash differs from the source proposal")
    # No interval may appear twice or straddle the train/valid boundary.
    for index, left in enumerate(intervals):
        for right in intervals[index + 1 :]:
            if left[2] != right[2] and left[0] < right[1] and right[0] < left[1]:
                raise ValueError(f"train/valid intervals overlap: {left[3]} and {right[3]}")
    valid_ids = sorted(row["id"] for row in canonical if row["split"] == "valid")
    declared_valid = sorted(str(value) for value in payload.get("valid_ids", []))
    if valid_ids != declared_valid:
        raise ValueError("source manifest valid_ids does not match row splits")
    counts = {split: sum(row["split"] == split for row in canonical) for split in ("train", "valid")}
    declared_counts = payload.get("counts")
    if isinstance(declared_counts, Mapping) and {key: int(declared_counts.get(key, -1)) for key in counts} != counts:
        raise ValueError("source manifest counts do not match row splits")
    normalized = dict(payload)
    normalized["source_manifest_path"] = str(path)
    normalized["source_manifest_sha256"] = _sha256(path)
    normalized["source_proposal"] = str(proposal)
    normalized["source_proposal_sha256"] = proposal_sha
    normalized["protected_interval_buffer_s"] = buffer_s
    normalized["protected_intervals_sha256"] = _json_sha256(protected_value)
    normalized["rows"] = canonical
    return normalized, canonical, {
        "source_manifest_sha256": normalized["source_manifest_sha256"],
        "source_proposal_sha256": proposal_sha,
        "protected_intervals_sha256": normalized["protected_intervals_sha256"],
        "protected_interval_buffer_s": buffer_s,
    }


def _artifact_path(row: Mapping[str, Any], cache_path: Path) -> Path:
    value = row.get("npz_path", row.get("artifact_path", row.get("feature_path", row.get("mel_path"))))
    if not isinstance(value, str) or not value:
        raise ValueError(f"row {row.get('id')} has no NPZ artifact path")
    return _resolve(value, base=cache_path.parent)


def _validate_aligned_overlap(source_rows: Sequence[Mapping[str, Any]], aligned_path: Path) -> dict[str, Any]:
    aligned_path = aligned_path.resolve()
    if not aligned_path.exists():
        raise FileNotFoundError(aligned_path)
    payload = json.loads(aligned_path.read_text())
    if not isinstance(payload, Mapping) or payload.get("format") != "nano_t3_adaptation_cache_v1":
        raise ValueError("aligned overlap cache must use nano_t3_adaptation_cache_v1")
    aligned_rows = payload.get("rows")
    if not isinstance(aligned_rows, list):
        raise ValueError("aligned overlap cache has no rows")
    by_path: dict[str, Mapping[str, Any]] = {}
    for row in aligned_rows:
        if not isinstance(row, Mapping):
            continue
        audio_path = str(row.get("audio_path", ""))
        if audio_path:
            if audio_path in by_path:
                raise ValueError(f"duplicate aligned row audio path: {audio_path}")
            by_path[audio_path] = row
    overlaps: list[str] = []
    for source in source_rows:
        aligned = by_path.get(str(source["audio_path"]))
        if aligned is None:
            continue
        aligned_hash = aligned.get("audio_sha256")
        if aligned_hash is None and isinstance(aligned.get("transcript_audit"), Mapping):
            aligned_hash = aligned["transcript_audit"].get("audio_sha256")
        if aligned_hash is not None and str(aligned_hash) != str(source["audio_sha256"]):
            raise ValueError(f"aligned overlap hash mismatch for {source['id']}")
        expected = _normalize_tokens(aligned.get("speech_tokens"), row_id=str(source["id"]))
        artifact = source.get("speech_tokens")
        if artifact is None:
            raise ValueError(f"source row {source['id']} has no extracted speech tokens")
        actual = _normalize_tokens(artifact, row_id=str(source["id"]))
        if not np.array_equal(expected, actual):
            raise ValueError(f"aligned cache speech tokens differ for overlap row {source['id']}")
        overlaps.append(str(source["id"]))
    if not overlaps:
        raise ValueError("aligned overlap check found no matching source rows")
    return {
        "status": "passed",
        "path": str(aligned_path),
        "sha256": _sha256(aligned_path),
        "row_count": len(aligned_rows),
        "overlap_ids": overlaps,
        "overlap_count": len(overlaps),
        "token_equality": "exact",
    }


def validate_acoustic_cache(path: Path | str) -> dict[str, Any]:
    """Validate a ready acoustic cache and expand NPZ tokens for the fitter.

    This is the public cache-reader API used by ``fit_decoder_embedding``.
    It returns rows with ``audio_path``, ``speech_tokens``, ``source_mel_path``,
    ``source_sha256``, ``split``, and ``id``.  It never adds a transcript.
    """

    manifest_path = Path(path).expanduser().resolve()
    payload = json.loads(manifest_path.read_text())
    if not isinstance(payload, Mapping) or payload.get("format") != "nano_acoustic_cache_v1":
        raise ValueError("acoustic cache must use nano_acoustic_cache_v1")
    if payload.get("status") != "ready":
        raise ValueError(f"acoustic cache status is not ready: {payload.get('status')!r}")
    if payload.get("transcript_labels_used") is not False:
        raise ValueError("acoustic cache must declare transcript_labels_used=false")
    if payload.get("synthetic_audio") is not False:
        raise ValueError("acoustic cache must declare synthetic_audio=false")
    source_manifest_path = _resolve(str(payload.get("source_manifest")), base=manifest_path.parent)
    source_manifest, source_rows, contract = _validate_source_manifest(source_manifest_path)
    expected_source_sha = payload.get("source_manifest_sha256")
    if expected_source_sha != contract["source_manifest_sha256"]:
        raise ValueError("acoustic cache source manifest SHA-256 does not match current source manifest")
    if _resolve(str(payload.get("source_proposal")), base=manifest_path.parent) != _resolve(str(source_manifest["source_proposal"])):
        raise ValueError("acoustic cache source proposal path differs from the source manifest")
    if payload.get("source_proposal_sha256") != contract["source_proposal_sha256"]:
        raise ValueError("acoustic cache source proposal SHA-256 differs from the source manifest")
    if payload.get("protected_intervals_sha256") != contract["protected_intervals_sha256"]:
        raise ValueError("acoustic cache protected interval hash differs from the source manifest")
    try:
        recorded_buffer = float(payload.get("protected_interval_buffer_s"))
    except (TypeError, ValueError) as exc:
        raise ValueError("acoustic cache protected interval buffer is missing") from exc
    if not math.isfinite(recorded_buffer) or abs(recorded_buffer - float(contract["protected_interval_buffer_s"])) > 1e-9:
        raise ValueError("acoustic cache protected interval buffer differs from the source manifest")
    expected_contract_sha = _json_sha256(
        {
            "sample_rate_16k": SAMPLE_RATE_16K,
            "sample_rate_24k": SAMPLE_RATE_24K,
            "token_rate_hz": TOKEN_RATE_HZ,
            "mel_bands": MEL_BANDS,
            "source_contract": contract,
        }
    )
    if payload.get("source_contract_sha256") != expected_contract_sha:
        raise ValueError("acoustic cache source contract hash is stale or missing")
    for field, expected in (
        ("target_sample_rate", SAMPLE_RATE_16K),
        ("source_sample_rate", SAMPLE_RATE_24K),
        ("target_token_rate_hz", TOKEN_RATE_HZ),
        ("mel_bands", MEL_BANDS),
    ):
        try:
            observed = int(payload.get(field, -1))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"acoustic cache {field} is inconsistent") from exc
        if observed != expected:
            raise ValueError(f"acoustic cache {field} is inconsistent")
    model_dir_value = payload.get("model_dir")
    checkpoint_value = payload.get("model_checkpoint")
    if not isinstance(model_dir_value, str) or not isinstance(checkpoint_value, str):
        raise ValueError("acoustic cache model checkpoint provenance is missing")
    model_dir = _resolve(model_dir_value, base=manifest_path.parent)
    checkpoint = _resolve(checkpoint_value, base=manifest_path.parent)
    expected_checkpoint = (model_dir / S3GEN_CHECKPOINT_NAME).resolve()
    if checkpoint != expected_checkpoint:
        raise ValueError("acoustic cache model checkpoint path does not match model_dir")
    if not checkpoint.exists() or payload.get("model_checkpoint_sha256") != _sha256(checkpoint):
        raise ValueError("acoustic cache model checkpoint SHA-256 does not match current checkpoint")
    aligned_overlap = payload.get("aligned_overlap")
    if not isinstance(aligned_overlap, Mapping) or aligned_overlap.get("status") != "passed" or aligned_overlap.get("token_equality") != "exact":
        raise ValueError("ready acoustic cache needs a passed exact aligned-overlap check")
    overlap_ids = aligned_overlap.get("overlap_ids")
    if not isinstance(overlap_ids, list) or int(aligned_overlap.get("overlap_count", -1)) != len(overlap_ids) or not overlap_ids:
        raise ValueError("ready acoustic cache aligned-overlap check has no verified rows")
    by_id = {row["id"]: row for row in source_rows}
    rows_value = payload.get("rows")
    if not isinstance(rows_value, list) or len(rows_value) != len(source_rows):
        raise ValueError("acoustic cache rows do not match source manifest")
    expanded: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in rows_value:
        if not isinstance(item, Mapping):
            raise ValueError("acoustic cache row is not an object")
        forbidden = _forbidden_label_keys(item)
        if forbidden:
            raise ValueError("acoustic cache contains transcript label fields: " + ", ".join(forbidden[:4]))
        row_id = str(item.get("id", ""))
        if row_id in seen or row_id not in by_id:
            raise ValueError(f"acoustic cache has unknown or duplicate row {row_id!r}")
        seen.add(row_id)
        source = by_id[row_id]
        if str(item.get("split")) != source["split"] or str(item.get("audio_path")) != source["audio_path"]:
            raise ValueError(f"acoustic cache source identity mismatch for {row_id}")
        if str(item.get("audio_sha256")) != source["audio_sha256"] or str(item.get("source_audio_sha256")) != source["source_audio_sha256"]:
            raise ValueError(f"acoustic cache source hash mismatch for {row_id}")
        artifact = _artifact_path(item, manifest_path)
        artifact_sha = item.get("npz_sha256", item.get("artifact_sha256"))
        if not isinstance(artifact_sha, str) or artifact_sha != _sha256(artifact):
            raise ValueError(f"acoustic cache artifact SHA-256 mismatch for {row_id}")
        arrays = _load_npz(artifact)
        if "speech_tokens" not in arrays or "source_mel" not in arrays:
            raise ValueError(f"acoustic cache artifact for {row_id} needs speech_tokens and source_mel")
        tokens = _validate_persisted_tokens(
            arrays["speech_tokens"],
            row_id=row_id,
            expected_count=int(item.get("speech_token_count", -1)),
        )
        mel = _validate_mel(arrays["source_mel"], row_id=row_id)
        expected_frames = {2 * int(tokens.size), 2 * int(tokens.size) - 1}
        if mel.shape[1] not in expected_frames:
            raise ValueError(f"acoustic cache source mel frame count is invalid for {row_id}")
        if item.get("source_mel_shape") != list(mel.shape):
            raise ValueError(f"acoustic cache source mel shape mismatch for {row_id}")
        expanded.append(
            {
                "id": row_id,
                "split": source["split"],
                "audio_path": source["audio_path"],
                "source_sha256": source["audio_sha256"],
                "source_audio": source["source_audio"],
                "source_audio_sha256": source["source_audio_sha256"],
                "start_s": source["start_s"],
                "end_s": source["end_s"],
                "duration_s": source["duration_s"],
                "speech_tokens": [int(value) for value in tokens.tolist()],
                "speech_token_count": int(tokens.size),
                "source_mel_path": str(artifact),
                "mel_path": str(artifact),
                "source_mel_shape": list(mel.shape),
                "artifact_sha256": artifact_sha,
            }
        )
    if seen != set(by_id):
        raise ValueError("acoustic cache is missing source rows")
    result = dict(payload)
    result["source_manifest"] = str(source_manifest_path)
    result["source_manifest_sha256"] = contract["source_manifest_sha256"]
    result["rows"] = expanded
    result["source_contract"] = contract
    result["source_status"] = source_manifest.get("status")
    return result


def _ensure_vendor_path() -> None:
    if str(VENDOR_SRC) not in sys.path:
        sys.path.insert(0, str(VENDOR_SRC))


def _load_audio(path: Path, sample_rate: int) -> np.ndarray:
    import librosa

    wav, _ = librosa.load(str(path), sr=int(sample_rate), mono=True)
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim != 1 or wav.size == 0 or not np.isfinite(wav).all():
        raise ValueError(f"audio is empty or non-finite after resampling: {path}")
    return wav


def _normalize_loudness(wav: np.ndarray, sample_rate: int) -> np.ndarray:
    """Match ChatterboxTurboTTS.norm_loudness at -27 LUFS."""

    import pyloudnorm as pyln

    meter = pyln.Meter(int(sample_rate))
    loudness = float(meter.integrated_loudness(np.asarray(wav, dtype=np.float32)))
    gain_db = -27.0 - loudness
    gain_linear = float(10.0 ** (gain_db / 20.0))
    if math.isfinite(gain_linear) and gain_linear > 0:
        wav = np.asarray(wav, dtype=np.float32) * gain_linear
    if not np.isfinite(wav).all():
        raise ValueError("loudness normalization returned non-finite audio")
    return np.asarray(wav, dtype=np.float32)


def _extract_source_mel(wav24: np.ndarray) -> np.ndarray:
    _ensure_vendor_path()
    import torch
    from chatterbox.models.s3gen.utils.mel import mel_spectrogram

    with torch.no_grad():
        mel = mel_spectrogram(torch.from_numpy(np.asarray(wav24, dtype=np.float32))[None, :])
    return _validate_mel(mel.detach().cpu().numpy(), row_id="audio")


def _stream_tokenizer(torch: Any, model_dir: Path, device: Any) -> tuple[Any, dict[str, Any]]:
    _ensure_vendor_path()
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    from chatterbox.models.s3tokenizer import S3Tokenizer
    from onnx_staged import _stream_load

    checkpoint = model_dir / S3GEN_CHECKPOINT_NAME
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    with torch.device("meta"):
        tokenizer = S3Tokenizer("speech_tokenizer_v2_25hz")
    tokenizer.to_empty(device=device)
    load_report = _stream_load(tokenizer, checkpoint, prefixes=(("tokenizer", ""),), strict=False)
    missing = set(load_report["missing_tensors"])
    if missing - {"_mel_filters", "window"} or load_report["unexpected_tensors"]:
        raise RuntimeError(f"tokenizer checkpoint mismatch: {load_report}")
    # The native checkpoint can omit these deterministic frontend buffers.
    # to_empty() leaves missing buffers uninitialized, so rebuild them before
    # using the tokenizer. All learned tensors still require exact loading.
    if "_mel_filters" in missing:
        import librosa
        filters = librosa.filters.mel(sr=SAMPLE_RATE_16K, n_fft=tokenizer.n_fft,
                                     n_mels=int(tokenizer._mel_filters.shape[0]))
        tokenizer._mel_filters = torch.from_numpy(filters).to(device=device, dtype=torch.float32)
    if "window" in missing:
        tokenizer.window = torch.hann_window(tokenizer.n_fft, device=device, dtype=torch.float32)
    # Rotary frequencies are a plain tensor, not a registered buffer. They
    # remain on meta after to_empty and are absent from the checkpoint.
    from s3tokenizer.model_v2 import precompute_freqs_cis
    for child in tokenizer.modules():
        freqs = getattr(child, "freqs_cis", None)
        if torch.is_tensor(freqs) and freqs.is_meta:
            child.freqs_cis = precompute_freqs_cis(
                int(freqs.shape[-1]), int(freqs.shape[0])
            ).to(device=device)
    tokenizer.eval()
    for parameter in tokenizer.parameters():
        parameter.requires_grad_(False)
    return tokenizer, {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "component": "S3Tokenizer(speech_tokenizer_v2_25hz)",
        "load": load_report,
        "frozen": True,
    }


def _extract_tokens(tokenizer: Any, torch: Any, wav16: np.ndarray, *, row_id: str) -> np.ndarray:
    with torch.no_grad():
        values, lengths = tokenizer.forward([wav16], max_len=None)
    length = int(lengths[0].item())
    return _normalize_tokens(values[0, :length].detach().cpu().numpy(), row_id=row_id)


def _source_rows_for_resume(
    output_manifest: Path,
    source_contract: Mapping[str, Any],
    source_rows: Sequence[Mapping[str, Any]],
    model_sha: str,
    config_sha: str,
) -> dict[str, Any] | None:
    if not output_manifest.exists():
        return None
    payload = json.loads(output_manifest.read_text())
    if not isinstance(payload, Mapping):
        return None
    if payload.get("format") != "nano_acoustic_cache_v1":
        raise ValueError("existing token output has the wrong format")
    if payload.get("source_manifest_sha256") != source_contract.get("source_manifest_sha256"):
        raise ValueError("resume source manifest provenance differs")
    if payload.get("model_checkpoint_sha256") != model_sha or payload.get("source_contract_sha256") != config_sha:
        raise ValueError("resume model or source contract provenance differs")
    if payload.get("status") not in {"running", "partial", "ready"}:
        raise ValueError("resume output has an invalid status")
    current_by_id = {str(row["id"]): row for row in source_rows}
    records = payload.get("rows")
    if not isinstance(records, list):
        raise ValueError("resume output has no row records")
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("resume output contains a non-object row")
        row_id = str(record.get("id", ""))
        if row_id in seen or row_id not in current_by_id:
            raise ValueError(f"resume output has stale or duplicate row {row_id!r}")
        seen.add(row_id)
        source = current_by_id[row_id]
        for field in ("split", "audio_path", "audio_sha256", "source_audio", "source_audio_sha256", "start_s", "end_s", "duration_s"):
            if record.get(field) != source.get(field):
                raise ValueError(f"resume row {row_id} metadata differs for {field}")
        artifact = _artifact_path(record, output_manifest)
        if not artifact.exists() or record.get("npz_sha256") != _sha256(artifact):
            raise ValueError(f"resume row {row_id} artifact is missing or changed")
        arrays = _load_npz(artifact)
        tokens = _validate_persisted_tokens(
            arrays.get("speech_tokens"),
            row_id=row_id,
            expected_count=int(record.get("speech_token_count", -1)),
        )
        mel = _validate_mel(arrays.get("source_mel"), row_id=row_id)
        if record.get("source_mel_shape") != list(mel.shape) or mel.shape[1] not in {2 * int(tokens.size), 2 * int(tokens.size) - 1}:
            raise ValueError(f"resume row {row_id} source mel metadata is stale")
    return dict(payload)


def _tokens_stage(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    started = time.perf_counter()
    if int(args.max_target_tokens) < 2 or int(args.max_target_tokens) > 400:
        raise ValueError("--max-target-tokens must be in 2..400 for bounded acoustic preparation")
    source_manifest_path = _resolve(args.source_manifest)
    source_payload, source_rows, contract = _validate_source_manifest(source_manifest_path)
    model_dir = _resolve(args.model_dir)
    checkpoint = model_dir / S3GEN_CHECKPOINT_NAME
    model_sha = _sha256(checkpoint)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    config_sha = _json_sha256({"sample_rate_16k": SAMPLE_RATE_16K, "sample_rate_24k": SAMPLE_RATE_24K, "token_rate_hz": TOKEN_RATE_HZ, "mel_bands": MEL_BANDS, "source_contract": contract})
    existing = _source_rows_for_resume(manifest_path, contract, source_rows, model_sha, config_sha) if args.resume else None
    records_by_id = {str(item["id"]): dict(item) for item in (existing or {}).get("rows", [])}
    _configure_torch_threads(torch)
    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    tokenizer, tokenizer_report = _stream_tokenizer(torch, model_dir, device)
    manifest: dict[str, Any] = {
        "format": "nano_acoustic_cache_v1",
        "status": "running",
        "transcript_labels_used": False,
        "synthetic_audio": False,
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": contract["source_manifest_sha256"],
        "source_proposal": source_payload["source_proposal"],
        "source_proposal_sha256": contract["source_proposal_sha256"],
        "protected_interval_buffer_s": contract["protected_interval_buffer_s"],
        "protected_intervals_sha256": contract["protected_intervals_sha256"],
        "source_contract_sha256": config_sha,
        "model_dir": str(model_dir),
        "model_checkpoint": str(checkpoint.resolve()),
        "model_checkpoint_sha256": model_sha,
        "tokenizer": tokenizer_report,
        "target_sample_rate": SAMPLE_RATE_16K,
        "source_sample_rate": SAMPLE_RATE_24K,
        "target_token_rate_hz": TOKEN_RATE_HZ,
        "mel_bands": MEL_BANDS,
        "source_preprocessing": {
            "target_loader": "librosa.load(sr=16000, mono=True), unnormalized waveform",
            "source_loader": "librosa.load(sr=24000, mono=True)",
            "normalizer": "ChatterboxTurboTTS.norm_loudness(target_lufs=-27)",
            "mel_extractor": "vendor/chatterbox/src/chatterbox/models/s3gen/utils/mel.py::mel_spectrogram",
            "mel_layout": "bands,time; raw log-mel values",
        },
        "aligned_overlap": None,
        "rows": list(records_by_id.values()),
        "command": list(sys.argv),
        "started_unix": time.time(),
    }
    _write_json(manifest_path, manifest)
    try:
        # Use source manifest order.  It is part of the reproducible contract.
        for source in source_rows:
            row_id = source["id"]
            prior = records_by_id.get(row_id)
            artifact = output_dir / f"{row_id}.npz"
            if prior and artifact.exists() and prior.get("npz_sha256") == _sha256(artifact):
                continue
            wav16 = _load_audio(Path(source["audio_path"]), SAMPLE_RATE_16K)
            tokens = _extract_tokens(tokenizer, torch, wav16, row_id=row_id)
            wav24 = _load_audio(Path(source["audio_path"]), SAMPLE_RATE_24K)
            normalized = _normalize_loudness(wav24, SAMPLE_RATE_24K)
            source_mel = _extract_source_mel(normalized)
            expected_frames = {2 * int(tokens.size), 2 * int(tokens.size) - 1}
            if source_mel.shape[1] not in expected_frames:
                raise ValueError(f"source mel frame count {source_mel.shape[1]} does not match 2N/2N-1 for {row_id}")
            _atomic_npz(artifact, {"speech_tokens": tokens, "source_mel": source_mel})
            record = {
                "id": row_id,
                "split": source["split"],
                "audio_path": source["audio_path"],
                "audio_sha256": source["audio_sha256"],
                "source_audio": source["source_audio"],
                "source_audio_sha256": source["source_audio_sha256"],
                "start_s": source["start_s"],
                "end_s": source["end_s"],
                "duration_s": source["duration_s"],
                "speech_token_count": int(tokens.size),
                "source_mel_shape": list(source_mel.shape),
                "npz_path": str(artifact.resolve()),
                "npz_sha256": _sha256(artifact),
                "source_mel_preprocessing": "normalized source waveform at -27 LUFS",
            }
            records_by_id[row_id] = record
            manifest["rows"] = [records_by_id[item["id"]] for item in source_rows if item["id"] in records_by_id]
            manifest["status"] = "partial"
            manifest["completed_rows"] = len(manifest["rows"])
            manifest["elapsed_seconds"] = time.perf_counter() - started
            manifest["peak_rss_bytes"] = _peak_rss_bytes()
            _write_json(manifest_path, manifest)
        expanded_rows = []
        for source in source_rows:
            item = records_by_id[source["id"]]
            arrays = _load_npz(_artifact_path(item, manifest_path))
            tokens = _normalize_tokens(arrays["speech_tokens"], row_id=source["id"])
            _validate_mel(arrays["source_mel"], row_id=source["id"])
            expanded_rows.append(item)
        manifest["rows"] = expanded_rows
        if args.aligned_cache:
            # Compare after all rows are extracted.  The comparison helper
            # accepts a temporary expanded view without adding it to JSON.
            overlap_rows = []
            for source in source_rows:
                item = records_by_id[source["id"]]
                arrays = _load_npz(_artifact_path(item, manifest_path))
                overlap_rows.append({**source, "speech_tokens": _normalize_tokens(arrays["speech_tokens"], row_id=source["id"]).tolist()})
            manifest["aligned_overlap"] = _validate_aligned_overlap(overlap_rows, _resolve(args.aligned_cache))
        manifest["status"] = "ready"
        manifest["completed_rows"] = len(expanded_rows)
        manifest["elapsed_seconds"] = time.perf_counter() - started
        manifest["peak_rss_bytes"] = _peak_rss_bytes()
        _write_json(manifest_path, manifest)
        # Re-read the completed cache through the public strict validator.  A
        # ready status is never emitted for stale resume metadata or a missing
        # artifact, even when the tokenizer loop skipped every row.
        validate_acoustic_cache(manifest_path)
        return manifest
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error_type"] = type(exc).__name__
        manifest["error"] = str(exc)
        manifest["elapsed_seconds"] = time.perf_counter() - started
        manifest["peak_rss_bytes"] = _peak_rss_bytes()
        _write_json(manifest_path, manifest)
        raise


def _load_acoustic_arrays(row: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    artifact = _resolve(str(row["source_mel_path"]))
    arrays = _load_npz(artifact)
    tokens = _normalize_tokens(arrays["speech_tokens"], row_id=str(row["id"]))
    mel = _validate_mel(arrays["source_mel"], row_id=str(row["id"]))
    return tokens, mel


def _compare_existing_overlap(acoustic: Mapping[str, Any], reference_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_path = {str(row["audio_path"]): row for row in acoustic["rows"]}
    checks: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for ref in reference_rows:
        candidate = by_path.get(str(ref["audio_path"]))
        if candidate is None:
            continue
        arrays = _load_npz(_resolve(str(candidate["source_mel_path"])))
        actual = _validate_mel(arrays["source_mel"], row_id=str(candidate["id"]))
        reference_path = _resolve(str(ref["prepared_mel_path"]))
        reference_arrays = _load_npz(reference_path)
        expected = _validate_mel(reference_arrays["source_mel"], row_id=str(ref["id"]))
        if actual.shape != expected.shape:
            failures.append({"id": ref["id"], "reason": "shape", "actual": list(actual.shape), "expected": list(expected.shape)})
            continue
        delta = actual.astype(np.float64) - expected.astype(np.float64)
        max_abs = float(np.max(np.abs(delta)))
        rmse = float(np.sqrt(np.mean(np.square(delta))))
        passed = bool(max_abs <= 2e-5 and rmse <= 5e-6)
        checks.append({"id": ref["id"], "max_abs": max_abs, "rmse": rmse, "passed": passed})
        if not passed:
            failures.append({"id": ref["id"], "reason": "numeric", "max_abs": max_abs, "rmse": rmse})
    return {"status": "passed" if not failures else "failed", "rows": checks, "failures": failures, "overlap_count": len(checks), "tolerance": {"max_abs": 2e-5, "rmse": 5e-6}}


def _flow_stage(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    started = time.perf_counter()
    _configure_torch_threads(torch)
    if int(args.steps) != FLOW_STEPS:
        raise ValueError("--steps must equal 2 for native meanflow alignment")
    token_manifest = _resolve(args.token_manifest)
    acoustic = validate_acoustic_cache(token_manifest)
    conditionals_path = _resolve(args.conditionals_path)
    conditionals_sha = _sha256(conditionals_path)
    expected_sha = str(args.expected_conditionals_sha256)
    if conditionals_sha != expected_sha:
        raise ValueError("conditionals SHA-256 does not match the required fixed cache")
    model_dir = _resolve(args.model_dir)
    checkpoint = model_dir / S3GEN_CHECKPOINT_NAME
    model_sha = _sha256(checkpoint)
    if acoustic.get("model_checkpoint_sha256") != model_sha:
        raise ValueError("tokenizer cache and flow checkpoint provenance differ")
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_manifest_path = output_dir / "manifest.json"
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    from fit_decoder_embedding import (
        _basic_euler,
        _fixed_noise,
        _load_conditionals_payload,
        _parity,
        _prepare_flow_row,
        _stream_s3gen_modules,
        _validate_prepare_inputs,
    )

    payload = _load_conditionals_payload(torch, conditionals_path)
    reference_prepare_dir = _resolve(args.reference_prepare_dir)
    reference_manifest, reference_rows, reference_conditionals, _, _ = _validate_prepare_inputs(reference_prepare_dir)
    if reference_manifest.get("model_checkpoint_sha256") != model_sha:
        # Legacy prepare records hash the whole model directory rather than
        # just S3Gen. Native parity below remains the authoritative check.
        # Use the producing module's exact names-and-file-bytes algorithm.
        # The fitting helper has a different names-and-file-digests hash.
        from mel_calibration import _sha256_tree
        if reference_manifest.get("model_checkpoint_sha256") != _sha256_tree(model_dir):
            raise ValueError("reference preparation uses a different model checkpoint")
    if reference_conditionals.resolve() != conditionals_path.resolve() or reference_manifest.get("conditioning_sha256") != conditionals_sha:
        raise ValueError("reference prepare directory does not use the requested fixed conditionals cache")
    source_overlap = _compare_existing_overlap(acoustic, reference_rows)
    # An overlap is required.  It proves that this stage uses the same source
    # waveform/mel implementation as the previously audited preparation.
    if source_overlap["overlap_count"] == 0 or source_overlap["status"] != "passed":
        raise ValueError("acoustic source mel did not pass existing preparation overlap checks")
    device = torch.device(args.device)
    rss_before_load = _rss_bytes()
    flow_encoder, estimator, loader_report = _stream_s3gen_modules(torch, model_dir, device)
    reference_states = [_prepare_flow_row(torch, flow_encoder, payload, row, device) for row in reference_rows]
    reference_noises = [_fixed_noise(torch, state, seed=int(row["seed"])) for state, row in zip(reference_states, reference_rows)]
    prepared_recon = {row["id"]: np.asarray(row["prepared_reconstructed_mel"], dtype=np.float32)[None, ...] for row in reference_rows}
    reference_parity = _parity(torch, estimator, reference_states, payload["gen"]["embedding"].to(device=device, dtype=torch.float32), reference_noises, prepared_recon, steps=FLOW_STEPS, atol=1e-4)
    report: dict[str, Any] = {
        "format": "nano_mel_calibration_prepare_v1",
        "status": "running",
        "diagnostic_only": True,
        "acoustic_only": True,
        "transcript_labels_used": False,
        "synthetic_audio": False,
        "cache": str(token_manifest),
        "cache_sha256": _sha256(token_manifest),
        "conditionals_path": str(conditionals_path),
        "conditioning_sha256": conditionals_sha,
        "model_dir": str(model_dir),
        "model_checkpoint_sha256": model_sha,
        "source_sample_rate": SAMPLE_RATE_24K,
        "mel_bands": MEL_BANDS,
        "mel_layout": "bands,time; raw log-mel values",
        "flow_path": "streamed model.s3gen flow encoder and meanflow estimator; HiFT vocoder not called",
        "cfm_steps": FLOW_STEPS,
        "reference_prepare_dir": str(reference_prepare_dir),
        "reference_source_overlap": source_overlap,
        "reference_native_parity": reference_parity,
        "loader": loader_report,
        "rss_before_flow_load_bytes": rss_before_load,
        "rows": [],
        "command": list(sys.argv),
        "started_unix": time.time(),
    }
    _write_json(output_manifest_path, report)
    try:
        if reference_parity.get("status") != "passed":
            report["status"] = "failed_reference_parity"
            _write_json(output_manifest_path, report)
            raise RuntimeError("reused native flow helper failed reference parity")
        row_records: list[dict[str, Any]] = []
        for ordinal, row in enumerate(acoustic["rows"]):
            tokens, source_mel = _load_acoustic_arrays(row)
            flow_row = {
                "id": row["id"],
                "tokens": tokens,
                "source_mel": source_mel,
                "source_frames": int(source_mel.shape[1]),
                "seed": int(args.seed) + ordinal * 1009,
            }
            state = _prepare_flow_row(torch, flow_encoder, payload, flow_row, device)
            noise = _fixed_noise(torch, state, seed=int(flow_row["seed"]))
            with torch.no_grad():
                generated = _basic_euler(torch, estimator, state, payload["gen"]["embedding"].to(device=device, dtype=torch.float32), noise, steps=FLOW_STEPS)
                reconstructed = generated[:, :, state["prompt_len"] :].detach().float().cpu().numpy()[0]
            expected_frames = 2 * (int(tokens.size) + 3) - int(state["prompt_frame_offset"])
            if reconstructed.shape != (MEL_BANDS, expected_frames) or not np.isfinite(reconstructed).all():
                raise ValueError(f"native reconstruction geometry/value failure for {row['id']}: {reconstructed.shape}")
            arrays_path = output_dir / f"{row['id']}.mel.npz"
            _atomic_npz(arrays_path, {"source_mel": source_mel, "reconstructed_mel": reconstructed, "speech_tokens": tokens})
            record = {
                "id": row["id"],
                "split": row["split"],
                "audio_path": row["audio_path"],
                "source_sha256": row["source_sha256"],
                "source_audio": row["source_audio"],
                "source_audio_sha256": row["source_audio_sha256"],
                "start_s": row["start_s"],
                "end_s": row["end_s"],
                "duration_s": row["duration_s"],
                "speech_token_count": int(tokens.size),
                "output_token_count": int(tokens.size + 3),
                "trailing_silence": {"value": S3GEN_SIL, "count": 3},
                "source_mel_shape": list(source_mel.shape),
                "reconstructed_mel_shape": list(reconstructed.shape),
                "seed": int(flow_row["seed"]),
                "cfm_steps": FLOW_STEPS,
                "conditioning_sha256": conditionals_sha,
                "mel_path": str(arrays_path.resolve()),
                "mel_sha256": _sha256(arrays_path),
                "source_preprocessing": {
                    "loader": "librosa.load(sr=24000, mono=True)",
                    "normalizer": "ChatterboxTurboTTS.norm_loudness(target_lufs=-27)",
                    "mel_extractor": "vendor/chatterbox/src/chatterbox/models/s3gen/utils/mel.py::mel_spectrogram",
                },
            }
            row_records.append(record)
            report["rows"] = row_records
            report["status"] = "partial"
            report["completed_rows"] = len(row_records)
            report["elapsed_seconds"] = time.perf_counter() - started
            report["peak_rss_bytes"] = _peak_rss_bytes()
            _write_json(output_manifest_path, report)
        report["rows"] = row_records
        report["status"] = "ready"
        report["completed_rows"] = len(row_records)
        report["elapsed_seconds"] = time.perf_counter() - started
        report["peak_rss_bytes"] = _peak_rss_bytes()
        report["rss_after_flow_bytes"] = _rss_bytes()
        _write_json(output_manifest_path, report)
        return report
    except Exception as exc:
        report["status"] = "failed"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        report["elapsed_seconds"] = time.perf_counter() - started
        report["peak_rss_bytes"] = _peak_rss_bytes()
        _write_json(output_manifest_path, report)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    tokens = subparsers.add_parser("tokens", help="extract source tokens and normalized source mels")
    tokens.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    tokens.add_argument("--output-dir", type=Path, default=DEFAULT_TOKEN_OUTPUT)
    tokens.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    tokens.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    tokens.add_argument("--max-target-tokens", type=int, default=400)
    tokens.add_argument("--aligned-cache", type=Path, default=DEFAULT_ALIGNED_CACHE)
    tokens.add_argument("--resume", action="store_true")
    flow = subparsers.add_parser("flow", help="reconstruct source mels through native S3Gen flow")
    flow.add_argument("--token-manifest", type=Path, default=DEFAULT_TOKEN_OUTPUT / "manifest.json")
    flow.add_argument("--output-dir", type=Path, default=DEFAULT_FLOW_OUTPUT)
    flow.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    flow.add_argument("--conditionals-path", type=Path, default=DEFAULT_CONDITIONALS)
    flow.add_argument("--expected-conditionals-sha256", default=EXPECTED_CONDITIONALS_SHA256)
    flow.add_argument("--reference-prepare-dir", type=Path, default=DEFAULT_REFERENCE_PREPARE)
    flow.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    flow.add_argument("--seed", type=int, default=31)
    flow.add_argument("--steps", type=int, default=FLOW_STEPS)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        report = _tokens_stage(args) if args.command == "tokens" else _flow_stage(args)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}, indent=2), file=sys.stderr)
        return 1
    print(json.dumps({"status": report.get("status"), "format": report.get("format"), "output_dir": str(_resolve(args.output_dir))}, indent=2))
    return 0 if report.get("status") == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
