"""Build an audited, transcript-aligned adaptation manifest.

The old adaptation cache was built from manually selected windows whose labels
do not always match the source audio.  This utility keeps that cache untouched
and proposes a replacement set from the complete original source.

``propose`` performs one full faster-whisper ``tiny.en`` transcription on CPU
with word timestamps.  It creates 3--12 second, punctuation-aligned windows
with a 150 ms boundary pad, rejects all protected reference and held-out
intervals, and writes a raw mono 24 kHz WAV plus an audit manifest.  The WAVs
are decoded source clips, not generated speech.  ``--family`` selects which
source and protected rows in a multi-family references manifest are relevant.
The default remains ``asmr7``.

``stage`` consumes the independent Whisper-small audit JSON.  A row is
accepted only when normalized Tiny/Small word error is below the strict gate,
confidence and no-speech values are sane, and the text is not a known
hallucination.  The final manifest uses a fixed distinct conditioning excerpt
from ``voices/nano/asmr_conversational.json`` and assigns the last two accepted
rows to validation.  It is emitted only when at least eight training and two
validation rows pass.

``rebase`` verifies a clip-level Tiny audit (with word timestamps), rebases
candidate text on that actual clip transcript, and records the unchanged
original labels and source provenance.  It compares the rebased Tiny text with
the supplied Small audit under the same exact normalized-WER gate used by
``stage``; non-exact rows remain marked for stage rejection.

Run both commands through ``bounded_job.py``.  This module performs no model
work when imported and this implementation deliberately does not run a model
itself during development.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REFERENCES = ROOT / "artifacts" / "nano_lab" / "references" / "manifest.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "nano_lab" / "dataset_repair"
DEFAULT_WHISPER_MODEL = ROOT / "Whisper_fast_package" / "models" / "tiny.en"
DEFAULT_PROFILE = ROOT / "voices" / "nano" / "asmr_conversational.json"
ASMR_FAMILY = "asmr7"
HARVEY_FAMILY = "harvey"
FAMILY_CHOICES = (ASMR_FAMILY, HARVEY_FAMILY)
FAMILY_PROFILES = {
    ASMR_FAMILY: DEFAULT_PROFILE,
    HARVEY_FAMILY: ROOT / "voices" / "nano" / "harvey.json",
}
TARGET_SR = 24_000
MIN_WINDOW_S = 3.0
MAX_WINDOW_S = 12.0
PADDING_S = 0.15
MAX_CANDIDATES = 40
MIN_WORDS = 3
MIN_AVG_LOGPROB = -1.5
MAX_NO_SPEECH_PROB = 0.55
MIN_WORD_PROB = 0.35
MAX_CONSENSUS_WER = 0.0
DECODE_ALIGNMENT_TOLERANCE_S = 0.05
HALLUCINATION_MARKERS = (
    "thank you for watching",
    "thanks for watching",
    "please subscribe",
    "subscribe to my channel",
    "subtitles by",
    "amara.org",
    "like and subscribe",
    "music playing",
    "[music]",
    "[applause]",
)
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?")
_PUNCT_END_RE = re.compile(r"[.!?](?:[\"'’”)]*)$")


def _family(args: argparse.Namespace | Mapping[str, Any] | None, default: str = ASMR_FAMILY) -> str:
    """Read and validate a family while keeping old callers ASMR-defaulted."""

    value = args.get("family", default) if isinstance(args, Mapping) else getattr(args, "family", default)
    value = value or default
    if value not in FAMILY_CHOICES:
        raise ValueError(f"unsupported family {value!r}; choose one of {FAMILY_CHOICES}")
    return str(value)


def _default_profile(family: str) -> Path:
    try:
        return FAMILY_PROFILES[family]
    except KeyError as exc:
        raise ValueError(f"unsupported family {family!r}; choose one of {FAMILY_CHOICES}") from exc


def _source_id(family: str, source_hash: str | None) -> str:
    """Use the raw source hash so isolated derivatives cannot mask leakage."""

    return str(source_hash or f"{family}_main")


def _resolve(value: str | os.PathLike[str], *, base: Path = ROOT) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _ffprobe_duration(path: Path) -> float | None:
    command = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        value = float(result.stdout.strip())
        return value if math.isfinite(value) else None
    except (OSError, subprocess.CalledProcessError, ValueError):
        return None


def _provenance_source_records(payload: Any) -> list[dict[str, Any]]:
    """Collect raw-source path/hash records from a separation JSON payload.

    Separation tools in this workspace emit both a list of clip rows and a
    report containing ``reference_rows``.  They use slightly different field
    names, so this parser accepts the source-specific keys without treating a
    derivative clip's ordinary ``path`` as the raw source.
    """

    path_keys = (
        "source",
        "source_path",
        "source_audio",
        "source_file",
        "raw_source",
        "raw_source_path",
        "input_source",
        "input_audio",
    )
    hash_keys = (
        "source_sha256",
        "source_audio_sha256",
        "source_hash",
        "raw_source_sha256",
        "input_source_sha256",
    )
    records: list[dict[str, Any]] = []

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            paths: list[str] = []
            hashes: list[str] = []
            for key in path_keys:
                candidate = value.get(key)
                if isinstance(candidate, Mapping):
                    for nested_key in ("path", "audio_path", "file", "filename"):
                        nested_value = candidate.get(nested_key)
                        if nested_value:
                            paths.append(str(nested_value))
                    for nested_key in ("sha256", "source_sha256", "source_audio_sha256", "hash"):
                        nested_value = candidate.get(nested_key)
                        if nested_value:
                            hashes.append(str(nested_value))
                elif isinstance(candidate, (str, os.PathLike)):
                    paths.append(str(candidate))
            for key in hash_keys:
                candidate = value.get(key)
                if candidate:
                    hashes.append(str(candidate))
            if paths or hashes:
                records.append({"paths": paths, "hashes": hashes})
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    visit(payload)
    return records


def _resolve_from_bases(value: str | os.PathLike[str], bases: Sequence[Path]) -> list[Path]:
    path = Path(value).expanduser()
    if path.is_absolute():
        return [path.resolve()]
    candidates: list[Path] = []
    for base in bases:
        candidate = (base / path).resolve()
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _validate_decode_source(
    source: Path,
    source_hash: str,
    decode_audio: str | os.PathLike[str],
    decode_provenance: str | os.PathLike[str],
    *,
    expected: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate an aligned derivative used for transcription and extraction.

    The raw source remains the identity used for split protection and source
    hashes.  The derivative is accepted only with an explicit provenance JSON
    that identifies that raw source, and only when ffprobe durations agree to
    the fixed one-frame tolerance.
    """

    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"raw source does not exist: {source}")
    actual_source_hash = _sha256(source)
    if source_hash and actual_source_hash != source_hash:
        raise ValueError("raw source hash does not match selected references manifest")
    source_hash = actual_source_hash

    decode_path = _resolve(str(decode_audio))
    provenance_path = _resolve(str(decode_provenance))
    if not decode_path.is_file():
        raise FileNotFoundError(f"decoded source does not exist: {decode_path}")
    if not provenance_path.is_file():
        raise FileNotFoundError(f"decode provenance does not exist: {provenance_path}")

    decode_hash = _sha256(decode_path)
    provenance_hash = _sha256(provenance_path)
    try:
        provenance_payload = _load_json(provenance_path)
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"decode provenance is not valid JSON: {provenance_path}") from exc

    records = _provenance_source_records(provenance_payload)
    source_path_match = False
    source_hash_match = False
    matched_provenance_source: str | None = None
    declared_hash: str | None = None
    for record in records:
        record_hashes = {str(item).lower() for item in record.get("hashes", []) if item}
        hash_matches = source_hash.lower() in record_hashes
        for raw_value in record.get("paths", []):
            for candidate in _resolve_from_bases(raw_value, (provenance_path.parent, ROOT, source.parent)):
                path_matches = candidate == source
                if candidate.is_file():
                    candidate_hash = _sha256(candidate)
                    if record_hashes and candidate_hash.lower() not in record_hashes:
                        continue
                    hash_matches = hash_matches or candidate_hash == source_hash
                if path_matches or hash_matches:
                    source_path_match = source_path_match or path_matches
                    source_hash_match = source_hash_match or hash_matches or path_matches
                    matched_provenance_source = str(candidate)
                    if record_hashes:
                        declared_hash = sorted(record_hashes)[0]
                    break
            if matched_provenance_source:
                break
        if matched_provenance_source:
            break
        if hash_matches:
            source_hash_match = True
            declared_hash = sorted(record_hashes)[0] if record_hashes else None

    if not records:
        raise ValueError("decode provenance contains no raw source path or hash")
    if not source_path_match and not source_hash_match:
        raise ValueError("decode provenance does not identify the selected raw source")
    if declared_hash and declared_hash != source_hash.lower():
        raise ValueError("decode provenance source hash does not match selected raw source")

    source_duration = _ffprobe_duration(source)
    decode_duration = _ffprobe_duration(decode_path)
    if source_duration is None or decode_duration is None:
        raise ValueError("ffprobe could not read raw and decoded source durations")
    duration_delta = abs(float(decode_duration) - float(source_duration))
    if duration_delta > DECODE_ALIGNMENT_TOLERANCE_S:
        raise ValueError(
            f"decoded source duration differs from raw by {duration_delta:.6f}s "
            f"(limit {DECODE_ALIGNMENT_TOLERANCE_S:.3f}s)"
        )

    metadata: dict[str, Any] = {
        "path": str(decode_path),
        "sha256": decode_hash,
        "provenance": str(provenance_path),
        "provenance_sha256": provenance_hash,
        "source_path": str(source),
        "source_sha256": source_hash,
        "source_path_match": bool(source_path_match),
        "source_hash_match": bool(source_hash_match),
        "matched_provenance_source": matched_provenance_source,
        "alignment": {
            "raw_duration_s": round(float(source_duration), 6),
            "decoded_duration_s": round(float(decode_duration), 6),
            "delta_s": round(duration_delta, 6),
            "tolerance_s": DECODE_ALIGNMENT_TOLERANCE_S,
            "method": "ffprobe_format_duration",
            "asserted": True,
        },
        "alignment_asserted": True,
    }
    if expected is not None:
        expected_path = expected.get("path") or expected.get("decode_audio")
        expected_hash = expected.get("sha256") or expected.get("decode_audio_sha256")
        expected_provenance = expected.get("provenance") or expected.get("decode_provenance")
        expected_provenance_hash = expected.get("provenance_sha256") or expected.get("decode_provenance_sha256")
        if expected_path and _resolve(str(expected_path)) != decode_path:
            raise ValueError("decoded source path changed since proposal")
        if expected_hash and str(expected_hash) != decode_hash:
            raise ValueError("decoded source hash changed since proposal")
        if expected_provenance and _resolve(str(expected_provenance)) != provenance_path:
            raise ValueError("decode provenance path changed since proposal")
        if expected_provenance_hash and str(expected_provenance_hash) != provenance_hash:
            raise ValueError("decode provenance hash changed since proposal")
        expected_source_hash = expected.get("source_sha256")
        if expected_source_hash and str(expected_source_hash) != source_hash:
            raise ValueError("decoded source raw hash changed since proposal")
        expected_alignment = expected.get("alignment")
        if expected.get("alignment_asserted") is False or (isinstance(expected_alignment, Mapping) and expected_alignment.get("asserted") is False):
            raise ValueError("decoded source alignment was not asserted in proposal")
    return metadata


def _decoded_source_metadata(payload: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    """Read nested or legacy flat decoded-source fields from a manifest."""

    if not isinstance(payload, Mapping):
        return None
    nested = payload.get("decoded_source")
    if isinstance(nested, Mapping):
        return nested
    if payload.get("decode_audio") or payload.get("decode_provenance"):
        return {
            "path": payload.get("decode_audio"),
            "sha256": payload.get("decode_audio_sha256"),
            "provenance": payload.get("decode_provenance"),
            "provenance_sha256": payload.get("decode_provenance_sha256"),
            "alignment": payload.get("decode_alignment"),
            "alignment_asserted": payload.get("decoded_source_is_aligned", True),
        }
    return None


def _decode_clip(source: Path, start_s: float, end_s: float, destination: Path) -> None:
    if end_s <= start_s:
        raise ValueError(f"invalid clip interval {start_s}..{end_s}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp.wav")
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-ss",
        f"{max(0.0, start_s):.4f}",
        "-i",
        str(source),
        "-t",
        f"{end_s - start_s:.4f}",
        "-ac",
        "1",
        "-ar",
        str(TARGET_SR),
        "-c:a",
        "pcm_s16le",
        "-y",
        str(temporary),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg could not decode {source} [{start_s:.3f},{end_s:.3f}]: {exc}") from exc
    os.replace(temporary, destination)


def _source_from_references(
    references_path: Path,
    family: str = ASMR_FAMILY,
) -> tuple[Path, dict[str, Any]]:
    """Select the raw source associated with ``family``.

    A references manifest can contain several raw sources.  Selection is made
    from family-tagged reference/held-out/training rows and their source hash,
    never from a filename substring.  A one-source manifest remains valid for
    both families, and the old ASMR default still selects ``sources[0]`` when
    older rows carry no family field.
    """

    family = _family({"family": family})
    manifest = _load_json(references_path)
    sources = manifest.get("sources") if isinstance(manifest, Mapping) else None
    if not isinstance(sources, list) or not sources or not all(isinstance(item, Mapping) for item in sources):
        raise ValueError(f"reference manifest has no usable sources: {references_path}")

    family_rows: list[Mapping[str, Any]] = []
    for section in ("references", "heldout", "training"):
        rows = manifest.get(section, []) if isinstance(manifest, Mapping) else []
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, Mapping) and (row.get("family") in (None, family)):
                family_rows.append(row)
    row_hashes: set[str] = set()
    row_basenames: set[str] = set()
    for row in family_rows:
        source_meta = row.get("source")
        if isinstance(source_meta, Mapping):
            if source_meta.get("sha256"):
                row_hashes.add(str(source_meta["sha256"]))
            if source_meta.get("path"):
                row_basenames.add(Path(str(source_meta["path"])).name)
        for key in ("source_audio_sha256", "source_hash"):
            if row.get(key):
                row_hashes.add(str(row[key]))
        for key in ("source_audio", "source_path"):
            if row.get(key):
                row_basenames.add(Path(str(row[key])).name)

    scored: list[tuple[int, int, Mapping[str, Any]]] = []
    for index, item in enumerate(sources):
        score = 0
        if item.get("sha256") in row_hashes:
            score += 100
        if item.get("path") and Path(str(item["path"])).name in row_basenames:
            score += 10
        # Preserve source ordering as a deterministic tie breaker.
        scored.append((score, -index, item))
    best_score, _, chosen = max(scored, key=lambda value: (value[0], value[1]))
    if best_score == 0 and len(sources) > 1:
        if family == ASMR_FAMILY:
            # Existing ASMR manifests predate family tags and intentionally
            # use sources[0].  Keep that default behavior unchanged.
            chosen = sources[0]
        else:
            raise ValueError(
                f"reference manifest has no source rows for family {family!r}; "
                "tag the manifest rows or provide a one-source manifest"
            )
    source_meta = dict(chosen)
    raw_path = source_meta.get("path")
    if not raw_path:
        raise ValueError(f"references manifest source for family {family!r} has no path")
    candidates = [references_path.parent / str(raw_path), ROOT / str(raw_path)]
    source = next((candidate.resolve() for candidate in candidates if candidate.exists()), None)
    if source is None:
        raise FileNotFoundError(f"{family} source from references manifest does not exist: {raw_path}")
    source_meta["absolute_path"] = str(source)
    source_meta["actual_sha256"] = _sha256(source)
    expected_hash = source_meta.get("sha256")
    source_meta["source_hash_matches_manifest"] = not expected_hash or expected_hash == source_meta["actual_sha256"]
    if expected_hash and expected_hash != source_meta["actual_sha256"]:
        raise ValueError(f"{family} source hash does not match references manifest")
    source_meta["family"] = family
    source_meta["source_id"] = _source_id(family, str(source_meta.get("actual_sha256") or expected_hash or ""))
    return source, source_meta


def _protected_intervals(
    references_manifest: Mapping[str, Any],
    source: Path,
    source_hash: str,
    family: str = ASMR_FAMILY,
) -> list[dict[str, Any]]:
    """Return selected-family reference/held-out intervals, never training rows.

    Matching uses the raw source hash and path.  ``source_id`` values from
    isolated derivatives are not trusted as a separate source identity.
    """

    family = _family({"family": family})
    result: list[dict[str, Any]] = []
    for section in ("references", "heldout"):
        rows = references_manifest.get(section, [])
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            row_family = row.get("family")
            if row_family is not None and row_family != family:
                continue
            source_meta = row.get("source") or {}
            row_hash = source_meta.get("sha256") if isinstance(source_meta, Mapping) else None
            row_path = str(source_meta.get("path", "")) if isinstance(source_meta, Mapping) else ""
            row_source_id = source_meta.get("source_id") if isinstance(source_meta, Mapping) else None
            same_source = (
                bool(row_hash and row_hash == source_hash)
                or bool(row_source_id and row_source_id == source_hash)
                or Path(row_path).name == source.name
            )
            if not same_source:
                continue
            try:
                start_s = float(source_meta["start_s"])
                end_s = float(source_meta["end_s"])
            except (KeyError, TypeError, ValueError):
                continue
            if not (math.isfinite(start_s) and math.isfinite(end_s) and end_s > start_s):
                continue
            result.append(
                {
                    "section": section,
                    "id": row.get("id"),
                    "family": family,
                    "source_id": _source_id(family, source_hash),
                    "start_s": start_s,
                    "end_s": end_s,
                }
            )
    return sorted(result, key=lambda item: (item["start_s"], item["end_s"], str(item.get("id"))))


def _path_matches(left: str | os.PathLike[str] | None, right: Path) -> bool:
    if not left:
        return False
    value = Path(str(left))
    try:
        if value.resolve() == right.resolve():
            return True
    except OSError:
        pass
    return value.name == right.name


def _profile_reference_metadata(
    profile: Mapping[str, Any],
    profile_path: Path,
    reference_audio: Path,
    family: str,
    source: Path,
    source_hash: str,
) -> dict[str, Any]:
    """Describe profile audio and recover raw-source provenance when available.

    Harvey's profile points to a Demucs-isolated WAV.  The adjacent separation
    manifest records that it came from raw source seconds 9.0--16.1.  This
    metadata is carried into the staged manifest so overlap checks use the raw
    source hash and interval even though the profile audio has a different
    filename and byte hash.
    """

    origin: dict[str, Any] = {}
    nested = profile.get("source_interval") or profile.get("origin")
    if isinstance(nested, Mapping):
        origin.update(dict(nested))
    for key in ("source_audio", "source_path", "source"):
        if profile.get(key) and key not in origin:
            origin["source_audio"] = profile[key]
            break
    for key in ("source_audio_sha256", "source_sha256", "source_hash"):
        if profile.get(key) and "source_sha256" not in origin:
            origin["source_sha256"] = profile[key]
            break
    if profile.get("source_start_s") is not None and "start_s" not in origin:
        origin["start_s"] = profile.get("source_start_s")
    if profile.get("source_end_s") is not None and "end_s" not in origin:
        origin["end_s"] = profile.get("source_end_s")

    # Search only small JSON sidecars/manifests next to the profile reference.
    # This is read-only provenance discovery; it never changes the profile.
    sidecars = [
        reference_audio.with_suffix(".json"),
        reference_audio.parent / "manifest.json",
        reference_audio.parent / "evaluation_manifest.json",
        reference_audio.parent / "protected_manifest.json",
    ]
    for sidecar in sidecars:
        if not sidecar.exists() or sidecar == profile_path:
            continue
        try:
            payload = _load_json(sidecar)
        except (OSError, ValueError, TypeError):
            continue
        rows: list[Mapping[str, Any]] = []
        if isinstance(payload, list):
            rows = [row for row in payload if isinstance(row, Mapping)]
        elif isinstance(payload, Mapping):
            for section in ("references", "heldout", "training", "rows", "reference_rows"):
                values = payload.get(section)
                if isinstance(values, list):
                    rows.extend(row for row in values if isinstance(row, Mapping))
        for row in rows:
            if not _path_matches(row.get("path") or row.get("reference"), reference_audio):
                continue
            for key in ("source", "source_audio", "source_path"):
                if row.get(key) and "source_audio" not in origin:
                    origin["source_audio"] = row[key]
            for key in ("source_sha256", "source_audio_sha256"):
                if row.get(key) and "source_sha256" not in origin:
                    origin["source_sha256"] = row[key]
            if row.get("start_s") is not None and "start_s" not in origin:
                origin["start_s"] = row["start_s"]
            if row.get("end_s") is not None and "end_s" not in origin:
                origin["end_s"] = row["end_s"]
            origin.setdefault("metadata_path", str(sidecar.resolve()))
            break
        if origin.get("start_s") is not None and origin.get("end_s") is not None:
            break

    source_audio = origin.get("source_audio")
    origin_source_path = _resolve(source_audio) if source_audio else None
    origin_hash = str(origin.get("source_sha256")) if origin.get("source_sha256") else None
    if origin_source_path and origin_source_path.exists() and not origin_hash:
        origin_hash = _sha256(origin_source_path)
    if origin_hash:
        origin["source_sha256"] = origin_hash
    if origin_source_path:
        origin["source_audio"] = str(origin_source_path)
    if origin_hash == source_hash or (origin_source_path and origin_source_path == source):
        origin["source_id"] = _source_id(family, source_hash)
        origin["same_raw_source"] = True
    elif origin_hash or origin_source_path:
        origin["same_raw_source"] = False

    try:
        start_s = float(origin["start_s"])
        end_s = float(origin["end_s"])
        if not (math.isfinite(start_s) and math.isfinite(end_s) and end_s > start_s):
            raise ValueError
        origin["start_s"] = start_s
        origin["end_s"] = end_s
    except (KeyError, TypeError, ValueError):
        origin.pop("start_s", None)
        origin.pop("end_s", None)

    if origin.get("same_raw_source") and "start_s" in origin and "end_s" in origin:
        kind = "isolated_derivative" if reference_audio.resolve() != source.resolve() else "raw_source_window"
    elif reference_audio.resolve() == source.resolve():
        kind = "raw_source"
    else:
        kind = "profile_audio_without_raw_interval"
    return {
        "path": str(reference_audio.resolve()),
        "sha256": _sha256(reference_audio),
        "family": family,
        "kind": kind,
        "origin": origin,
        "overlap_identity": "raw_source_sha256_and_interval",
    }


def _reference_overlap(
    target: Mapping[str, Any],
    profile_reference: Mapping[str, Any],
    family: str,
    source_hash: str,
) -> dict[str, Any]:
    """Return explicit profile/target overlap metadata.

    Unknown provenance is reported as unknown.  It is never treated as a
    disjoint source merely because the profile WAV has another path or hash.
    """

    origin = profile_reference.get("origin") or {}
    target_start = float(target.get("start_s", 0.0))
    target_end = float(target.get("end_s", 0.0))
    profile_start = origin.get("start_s")
    profile_end = origin.get("end_s")
    same_source = origin.get("source_sha256") == source_hash or origin.get("source_id") == _source_id(family, source_hash)
    result: dict[str, Any] = {
        "identity": "raw_source_sha256_and_interval",
        "source_id": _source_id(family, source_hash),
        "same_raw_source": bool(same_source),
        "target_interval": {"start_s": target_start, "end_s": target_end},
        "profile_interval": None,
        "overlap_s": None,
        "overlaps": None,
    }
    if profile_start is None or profile_end is None:
        result["status"] = "unknown_profile_origin_interval"
        return result
    profile_start = float(profile_start)
    profile_end = float(profile_end)
    overlap_s = max(0.0, min(target_end, profile_end) - max(target_start, profile_start)) if same_source else 0.0
    result["profile_interval"] = {"start_s": profile_start, "end_s": profile_end}
    result["overlap_s"] = overlap_s
    result["overlaps"] = bool(same_source and overlap_s > 0.0)
    result["status"] = "checked"
    return result


def _word_text(value: Any) -> str:
    return str(value or "").strip()


def _is_punctuation_boundary(text: str) -> bool:
    return bool(_PUNCT_END_RE.search(text.strip()))


def _normalise_text(text: str) -> str:
    return " ".join(str(text).replace("\u2019", "'").split()).strip()


def _words_from_segment(segment: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = segment.get("words")
    if not isinstance(rows, list):
        return []
    result = []
    for item in rows:
        if not isinstance(item, Mapping):
            continue
        try:
            start_s = float(item.get("start_s", item.get("start")))
            end_s = float(item.get("end_s", item.get("end")))
        except (TypeError, ValueError):
            continue
        text = _word_text(item.get("word", item.get("text")))
        if not text or not (math.isfinite(start_s) and math.isfinite(end_s) and end_s > start_s):
            continue
        probability = item.get("probability")
        try:
            probability = float(probability) if probability is not None else None
        except (TypeError, ValueError):
            probability = None
        result.append({"start_s": start_s, "end_s": end_s, "text": text, "probability": probability})
    return result


def _segment_confidence(segment: Mapping[str, Any], words: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    def finite_float(value: Any) -> float | None:
        try:
            value = float(value)
            return value if math.isfinite(value) else None
        except (TypeError, ValueError):
            return None

    probabilities = [float(word["probability"]) for word in words if word.get("probability") is not None and math.isfinite(float(word["probability"]))]
    return {
        "avg_logprob": finite_float(segment.get("avg_logprob")),
        "no_speech_prob": finite_float(segment.get("no_speech_prob")),
        "compression_ratio": finite_float(segment.get("compression_ratio")),
        "mean_word_probability": float(np.mean(probabilities)) if probabilities else None,
        "min_word_probability": float(np.min(probabilities)) if probabilities else None,
    }


def _sentence_groups(segments: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Carry words across ASR segments and emit complete punctuation boundaries.

    ASR segment boundaries can occur halfway through a sentence. They must not
    become training clip boundaries. Short complete sentences stay with the next
    sentence; long sentences and the final unfinished fragment are discarded.
    """
    groups = []
    current = []
    confidences = []
    for segment in segments:
        words = _words_from_segment(segment)
        if not words:
            # Unknown speech between timestamped segments cannot be skipped
            # while joining labels on either side of it.
            current = []
            confidences = []
            continue
        for word in words:
            current.append(word)
            confidences.append(_segment_confidence(segment, [word]))
            if not _is_punctuation_boundary(word["text"]):
                continue
            duration = current[-1]["end_s"] - current[0]["start_s"]
            if duration < MIN_WINDOW_S:
                continue
            if duration + 2 * PADDING_S <= MAX_WINDOW_S:
                probabilities = [x["probability"] for x in current if x.get("probability") is not None]
                logprobs = [x["avg_logprob"] for x in confidences if x.get("avg_logprob") is not None]
                nonspeech = [x["no_speech_prob"] for x in confidences if x.get("no_speech_prob") is not None]
                groups.append({
                    "words": list(current), "start_s": current[0]["start_s"],
                    "end_s": current[-1]["end_s"],
                    "text": _normalise_text(" ".join(x["text"] for x in current)),
                    "avg_logprob": min(logprobs) if logprobs else None,
                    "no_speech_prob": max(nonspeech) if nonspeech else None,
                    "mean_word_probability": float(np.mean(probabilities)) if probabilities else None,
                    "min_word_probability": min(probabilities) if probabilities else None,
                })
            current = []
            confidences = []
    return groups






def _candidate_windows(
    transcript: Mapping[str, Any],
    protected: Sequence[Mapping[str, Any]],
    source_duration: float | None,
    *,
    max_candidates: int = MAX_CANDIDATES,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    segments = transcript.get("segments")
    if not isinstance(segments, list):
        raise ValueError("transcript has no segments list")
    # ``_sentence_groups`` emits only complete punctuation groups in the
    # accepted duration range.  Do not merge separate groups here: a merge can
    # include timestamped words that were omitted by an overlong sentence.
    groups = _sentence_groups(segments)
    candidates: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for index, group in enumerate(groups, 1):
        text = _normalise_text(group.get("text", ""))
        words = _WORD_RE.findall(text)
        if len(words) < MIN_WORDS:
            rejected.append({"reason": "too_few_words", "text": text, "start_s": group["start_s"], "end_s": group["end_s"]})
            continue
        lower = text.lower()
        markers = [marker for marker in HALLUCINATION_MARKERS if marker in lower]
        if markers:
            rejected.append({"reason": "hallucination_marker", "markers": markers, "text": text, "start_s": group["start_s"], "end_s": group["end_s"]})
            continue
        padded_start = max(0.0, float(group["start_s"]) - PADDING_S)
        padded_end = float(group["end_s"]) + PADDING_S
        if source_duration is not None:
            padded_end = min(source_duration, padded_end)
        padded_duration = padded_end - padded_start
        if padded_duration < MIN_WINDOW_S or padded_duration > MAX_WINDOW_S:
            rejected.append({"reason": "duration_out_of_range", "text": text, "start_s": padded_start, "end_s": padded_end})
            continue
        overlap = next((item for item in protected if padded_start < float(item["end_s"]) and padded_end > float(item["start_s"])), None)
        if overlap is not None:
            rejected.append({"reason": "protected_interval_overlap", "protected": dict(overlap), "text": text, "start_s": padded_start, "end_s": padded_end})
            continue
        avg_logprob = group.get("avg_logprob")
        no_speech = group.get("no_speech_prob")
        mean_prob = group.get("mean_word_probability")
        if avg_logprob is not None and float(avg_logprob) < MIN_AVG_LOGPROB:
            rejected.append({"reason": "low_avg_logprob", "avg_logprob": avg_logprob, "text": text, "start_s": padded_start, "end_s": padded_end})
            continue
        if no_speech is not None and float(no_speech) > MAX_NO_SPEECH_PROB:
            rejected.append({"reason": "high_no_speech_prob", "no_speech_prob": no_speech, "text": text, "start_s": padded_start, "end_s": padded_end})
            continue
        if mean_prob is not None and float(mean_prob) < MIN_WORD_PROB:
            rejected.append({"reason": "low_word_probability", "mean_word_probability": mean_prob, "text": text, "start_s": padded_start, "end_s": padded_end})
            continue
        candidate = {
            "proposal_index": index,
            "start_s": round(padded_start, 4),
            "end_s": round(padded_end, 4),
            "duration_s": round(padded_duration, 4),
            "speech_start_s": round(float(group["start_s"]), 4),
            "speech_end_s": round(float(group["end_s"]), 4),
            "text": text,
            "word_count": len(words),
            "avg_logprob": avg_logprob,
            "no_speech_prob": no_speech,
            "mean_word_probability": mean_prob,
            "min_word_probability": group.get("min_word_probability"),
        }
        if any(candidate["start_s"] < float(previous["end_s"]) and candidate["end_s"] > float(previous["start_s"]) for previous in candidates):
            rejected.append({"reason": "candidate_overlap", **candidate})
            continue
        candidates.append(candidate)
        if len(candidates) >= int(max_candidates):
            break
    return candidates, rejected


def _transcribe_full(source: Path, model_path: Path) -> dict[str, Any]:
    import torch
    from faster_whisper import WhisperModel

    torch.set_num_threads(2)
    model = WhisperModel(str(model_path), device="cpu", compute_type="int8", cpu_threads=2, num_workers=1)
    started = time.perf_counter()
    segments, info = model.transcribe(
        str(source),
        language="en",
        beam_size=3,
        condition_on_previous_text=False,
        vad_filter=False,
        word_timestamps=True,
    )
    saved_segments: list[dict[str, Any]] = []
    texts: list[str] = []
    for segment in segments:
        text = _normalise_text(getattr(segment, "text", ""))
        texts.append(text)
        words = []
        for word in getattr(segment, "words", None) or []:
            words.append(
                {
                    "word": _word_text(getattr(word, "word", "")),
                    "start_s": round(float(getattr(word, "start", 0.0)), 4),
                    "end_s": round(float(getattr(word, "end", 0.0)), 4),
                    "probability": round(float(getattr(word, "probability", 0.0)), 6),
                }
            )
        saved_segments.append(
            {
                "start_s": round(float(getattr(segment, "start", 0.0)), 4),
                "end_s": round(float(getattr(segment, "end", 0.0)), 4),
                "text": text,
                "avg_logprob": round(float(getattr(segment, "avg_logprob", float("nan"))), 6),
                "no_speech_prob": round(float(getattr(segment, "no_speech_prob", float("nan"))), 6),
                "compression_ratio": round(float(getattr(segment, "compression_ratio", float("nan"))), 6),
                "words": words,
            }
        )
    return {
        "schema_version": 1,
        "model": str(model_path.resolve()),
        "compute_type": "int8",
        "device": "cpu",
        "cpu_threads": 2,
        "word_timestamps": True,
        "condition_on_previous_text": False,
        "language": getattr(info, "language", "en"),
        "language_probability": float(getattr(info, "language_probability", float("nan"))),
        "duration_s": float(getattr(info, "duration", float("nan"))),
        "text": _normalise_text(" ".join(texts)),
        "segments": saved_segments,
        "elapsed_s": round(time.perf_counter() - started, 6),
    }


def stage_propose(args: argparse.Namespace) -> dict[str, Any]:
    family = _family(args)
    references_path = _resolve(args.references_manifest)
    references_manifest = _load_json(references_path)
    source, source_meta = _source_from_references(references_path, family)
    source_hash = source_meta["actual_sha256"]
    protected = _protected_intervals(references_manifest, source, source_hash, family)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    decode_metadata: dict[str, Any] | None = None
    decode_audio_value = getattr(args, "decode_audio", None)
    decode_provenance_value = getattr(args, "decode_provenance", None)
    if decode_audio_value is not None:
        if decode_provenance_value is None:
            raise ValueError("--decode-audio requires --decode-provenance JSON")
        decode_metadata = _validate_decode_source(
            source,
            source_hash,
            decode_audio_value,
            decode_provenance_value,
        )
    elif decode_provenance_value is not None:
        raise ValueError("--decode-provenance requires --decode-audio")
    transcription_source = Path(decode_metadata["path"]) if decode_metadata else source
    transcript_path = _resolve(args.reuse_transcript) if args.reuse_transcript else output_dir / f"{family}_tiny_transcript.json"
    if args.reuse_transcript:
        transcript = _load_json(transcript_path)
        transcript_source = transcript.get("source", {}) if isinstance(transcript, Mapping) else {}
        if transcript_source and transcript_source.get("sha256") not in {None, source_hash}:
            raise ValueError("--reuse-transcript source hash does not match references manifest source")
        transcript_decode = _decoded_source_metadata(transcript if isinstance(transcript, Mapping) else None)
        if decode_metadata is not None:
            if not isinstance(transcript_decode, Mapping):
                raise ValueError("--reuse-transcript has no decoded-source identity; provide a transcript made with --decode-audio")
            transcript_decode_checked = _validate_decode_source(
                source,
                source_hash,
                decode_metadata["path"],
                decode_metadata["provenance"],
                expected=transcript_decode,
            )
            input_path = transcript.get("transcription_input") if isinstance(transcript, Mapping) else None
            input_hash = transcript.get("transcription_input_sha256") if isinstance(transcript, Mapping) else None
            if not input_path or not input_hash:
                raise ValueError("--reuse-transcript is missing decoded transcription-input path/hash")
            if _resolve(str(input_path)) != transcription_source:
                raise ValueError("--reuse-transcript transcription input path does not match decoded source")
            if str(input_hash) != transcript_decode_checked["sha256"]:
                raise ValueError("--reuse-transcript transcription input hash does not match decoded source")
        elif transcript_decode:
            raise ValueError("--reuse-transcript was made from a decoded source; pass --decode-audio and --decode-provenance")
    else:
        transcript_body = _transcribe_full(transcription_source, _resolve(args.whisper_model))
        transcript = {
            "source": {**source_meta, "path": str(source), "sha256": source_hash},
            "decoded_source": decode_metadata,
            "transcription_input": str(transcription_source),
            "transcription_input_sha256": _sha256(transcription_source),
            "transcription": transcript_body,
        }
        _write_json(transcript_path, transcript)
    if "transcription" in transcript:
        transcript_body = transcript["transcription"]
    else:
        transcript_body = transcript
    duration = transcript_body.get("duration_s")
    if not isinstance(duration, (int, float)) or not math.isfinite(float(duration)):
        duration = _ffprobe_duration(source)
    candidates, rejected = _candidate_windows(
        transcript_body,
        protected,
        float(duration) if duration else None,
        max_candidates=int(args.max_candidates),
    )
    clips_dir = output_dir / "clips"
    audit_rows = []
    proposal_rows = []
    for ordinal, candidate in enumerate(candidates, 1):
        candidate_id = f"{family}_repair_{ordinal:03d}"
        clip_path = clips_dir / f"{candidate_id}.wav"
        _decode_clip(transcription_source, float(candidate["start_s"]), float(candidate["end_s"]), clip_path)
        candidate = {
            **candidate,
            "id": candidate_id,
            "voice": family,
            "family": family,
            "path": str(clip_path.resolve()),
            "audio_sha256": _sha256(clip_path),
            "synthetic_tts": False,
            "source_audio": str(source),
            "source_audio_sha256": source_hash,
            "source_id": _source_id(family, source_hash),
            "transcript_source": str(transcript_path.resolve()),
        }
        if decode_metadata is not None:
            candidate.update(
                {
                    "decode_audio": decode_metadata["path"],
                    "decode_audio_sha256": decode_metadata["sha256"],
                    "decode_provenance": decode_metadata["provenance"],
                    "decode_provenance_sha256": decode_metadata["provenance_sha256"],
                    "decode_alignment": decode_metadata["alignment"],
                    "decoded_source": dict(decode_metadata),
                    "decoded_source_is_aligned": True,
                }
            )
        proposal_rows.append(candidate)
        audit_row = {
            "id": candidate_id,
            "path": str(clip_path.resolve()),
            "text": candidate["text"],
            "expected_text": candidate["text"],
            "voice": family,
            "family": family,
            "source_id": _source_id(family, source_hash),
        }
        if decode_metadata is not None:
            audit_row.update(
                {
                    "decode_audio": decode_metadata["path"],
                    "decode_audio_sha256": decode_metadata["sha256"],
                    "decode_provenance": decode_metadata["provenance"],
                    "decode_provenance_sha256": decode_metadata["provenance_sha256"],
                }
            )
        audit_rows.append(audit_row)
    audit_path = output_dir / "audit_manifest.json"
    _write_json(audit_path, audit_rows)
    proposal = {
        "schema_version": 1,
        "status": "proposed",
        "kind": "source_window_proposal",
        "voice": family,
        "family": family,
        "synthetic_tts": False,
        "source": {**source_meta, "path": str(source), "sha256": source_hash},
        "references_manifest": str(references_path),
        "references_manifest_sha256": _sha256(references_path),
        "protected_intervals": protected,
        "transcript": str(transcript_path.resolve()),
        "candidates": proposal_rows,
        "rejected": rejected,
        "candidate_count": len(proposal_rows),
        "max_candidates": int(args.max_candidates),
        "window_policy": {"min_s": MIN_WINDOW_S, "max_s": MAX_WINDOW_S, "padding_s": PADDING_S, "non_overlapping": True, "split_only_at_punctuation": True},
        "audit_manifest": str(audit_path.resolve()),
        "command": list(sys.argv),
    }
    if decode_metadata is not None:
        proposal["decoded_source"] = decode_metadata
        proposal["decode_audio"] = decode_metadata["path"]
        proposal["decode_audio_sha256"] = decode_metadata["sha256"]
        proposal["decode_provenance"] = decode_metadata["provenance"]
        proposal["decode_provenance_sha256"] = decode_metadata["provenance_sha256"]
        proposal["decode_alignment"] = decode_metadata["alignment"]
        proposal["decoded_source_is_aligned"] = True
    proposal_path = output_dir / "proposal.json"
    _write_json(proposal_path, proposal)
    return {"proposal": str(proposal_path), "audit_manifest": str(audit_path), "candidate_count": len(proposal_rows), "rejected_count": len(rejected), "protected_count": len(protected)}


def _audit_rows(path: Path) -> list[dict[str, Any]]:
    data = _load_json(path)
    if isinstance(data, list):
        rows = data
    elif isinstance(data, Mapping):
        rows = data.get("inputs") or data.get("rows") or data.get("runs") or []
    else:
        rows = []
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def _audit_text(row: Mapping[str, Any]) -> str:
    value = row.get("transcript")
    if isinstance(value, str):
        return _normalise_text(value)
    asr = row.get("asr")
    if isinstance(asr, Mapping):
        return _normalise_text(asr.get("text", ""))
    return ""


def _audit_payload(path: Path) -> tuple[Mapping[str, Any], list[dict[str, Any]]]:
    """Load an ASR audit report and expose its rows without model imports."""

    payload = _load_json(path)
    if isinstance(payload, Mapping):
        rows_value = payload.get("inputs") or payload.get("rows") or payload.get("runs") or []
        return payload, [dict(row) for row in rows_value if isinstance(row, Mapping)]
    if isinstance(payload, list):
        return {"inputs": payload}, [dict(row) for row in payload if isinstance(row, Mapping)]
    raise ValueError(f"ASR audit must be a JSON object or row list: {path}")


def _verify_audit_model(payload: Mapping[str, Any], expected_kind: str, path: Path) -> dict[str, Any]:
    """Verify an audit report identifies the expected Tiny or Small model."""

    identity = payload.get("model_identity")
    if not isinstance(identity, Mapping):
        identity = {}
    model_value = payload.get("model") or identity.get("path")
    identity_path = identity.get("path")
    if not model_value:
        raise ValueError(f"{expected_kind} audit has no model identity: {path}")
    model_path = str(model_value)
    if identity_path and str(identity_path) != model_path:
        raise ValueError(f"{expected_kind} audit model identity path mismatch: {path}")
    model_name = Path(model_path).name.lower()
    if expected_kind == "tiny":
        if "tiny" not in model_name or "small" in model_name:
            raise ValueError(f"clip Tiny audit does not identify a tiny model: {model_path}")
    elif expected_kind == "small":
        if "small" not in model_name:
            raise ValueError(f"Small audit does not identify a small model: {model_path}")
    else:
        raise ValueError(f"unsupported audit model kind {expected_kind!r}")
    manifest_value = identity.get("download_manifest") or payload.get("model_download_manifest")
    manifest_hash = identity.get("download_manifest_sha256")
    if manifest_value and manifest_hash:
        manifest_path = _resolve(str(manifest_value))
        if not manifest_path.is_file() or _sha256(manifest_path) != str(manifest_hash):
            raise ValueError(f"{expected_kind} audit model download manifest changed: {path}")
    return {
        "kind": expected_kind,
        "path": model_path,
        "identity": dict(identity),
        "report": str(path.resolve()),
        "report_sha256": _sha256(path),
    }


def _verify_audit_audio_row(row: Mapping[str, Any], candidate: Mapping[str, Any], role: str) -> tuple[Path, str]:
    """Verify an audit row still refers to the proposal candidate bytes."""

    candidate_path = _resolve(str(candidate.get("path", "")))
    row_path_value = row.get("path") or row.get("audio_path")
    if not row_path_value:
        raise ValueError(f"{role} audit row {candidate.get('id')} has no audio path")
    row_path = _resolve(str(row_path_value))
    if row_path != candidate_path:
        raise ValueError(f"{role} audit path mismatch for candidate {candidate.get('id')}")
    if not candidate_path.is_file():
        raise FileNotFoundError(candidate_path)
    actual_hash = _sha256(candidate_path)
    if candidate.get("audio_sha256") != actual_hash:
        raise ValueError(f"proposal audio hash changed for candidate {candidate.get('id')}")
    if row.get("audio_sha256") != actual_hash:
        raise ValueError(f"{role} audit audio hash mismatch for candidate {candidate.get('id')}")
    return candidate_path, actual_hash


def _verify_tiny_clip_row(row: Mapping[str, Any], candidate: Mapping[str, Any]) -> tuple[str, dict[str, Any], list[str]]:
    """Inspect Tiny clip confidence and return per-row quality rejections."""

    quality_reasons: list[str] = []
    text = _normalise_text(_audit_text(row))
    if len(_words(text)) < MIN_WORDS:
        quality_reasons.append("too_few_words")
    markers = [marker for marker in HALLUCINATION_MARKERS if marker in text.lower()]
    if markers:
        quality_reasons.append("hallucination_marker")
    segments = row.get("segments")
    if not isinstance(segments, list) or not segments:
        quality_reasons.append("missing_segments")
        segments = []
    has_words = isinstance(row.get("words"), list) or any(
        isinstance(segment, Mapping) and isinstance(segment.get("words"), list) for segment in segments
    )
    if not has_words:
        quality_reasons.append("missing_word_timestamps")
    logprobs: list[float] = []
    no_speech_probs: list[float] = []
    compression_ratios: list[float] = []
    word_probabilities: list[float] = []
    for segment in segments:
        if not isinstance(segment, Mapping):
            quality_reasons.append("invalid_segment")
            continue
        avg_logprob = segment.get("avg_logprob")
        no_speech_prob = segment.get("no_speech_prob")
        compression_ratio = segment.get("compression_ratio")
        try:
            avg_value = float(avg_logprob)
            if not math.isfinite(avg_value):
                raise ValueError
            logprobs.append(avg_value)
            if avg_value < MIN_AVG_LOGPROB:
                quality_reasons.append("low_avg_logprob")
        except (TypeError, ValueError):
            quality_reasons.append("invalid_avg_logprob")
        try:
            no_speech_value = float(no_speech_prob)
            if not math.isfinite(no_speech_value):
                raise ValueError
            no_speech_probs.append(no_speech_value)
            if no_speech_value > MAX_NO_SPEECH_PROB:
                quality_reasons.append("high_no_speech_prob")
        except (TypeError, ValueError):
            quality_reasons.append("invalid_no_speech_prob")
        try:
            compression_value = float(compression_ratio)
            if not math.isfinite(compression_value):
                raise ValueError
            compression_ratios.append(compression_value)
            if compression_value > 2.4:
                quality_reasons.append("high_compression_ratio")
        except (TypeError, ValueError):
            quality_reasons.append("invalid_compression_ratio")
        for word in segment.get("words") or []:
            if not isinstance(word, Mapping):
                quality_reasons.append("invalid_word_timestamp")
                continue
            try:
                word_start = float(word["start_s"])
                word_end = float(word["end_s"])
                if not (math.isfinite(word_start) and math.isfinite(word_end) and word_end >= word_start):
                    raise ValueError
            except (KeyError, TypeError, ValueError):
                quality_reasons.append("invalid_word_timestamp")
            try:
                probability = float(word["probability"])
                if not math.isfinite(probability):
                    raise ValueError
                word_probabilities.append(probability)
            except (KeyError, TypeError, ValueError):
                quality_reasons.append("invalid_word_probability")
    if not word_probabilities:
        quality_reasons.append("missing_word_probabilities")
    elif float(np.mean(word_probabilities)) < MIN_WORD_PROB:
        # Preserve the admission gate's mean-word-confidence semantics.
        # A low-confidence individual word is retained as diagnostic data;
        # independent exact transcript agreement remains mandatory.
        quality_reasons.append("low_word_probability")
    quality_reasons = sorted(set(quality_reasons))
    return text, {
        "text": text,
        "segments": segments,
        "word_timestamps_verified": True,
        "audio_sha256": row.get("audio_sha256"),
        "avg_logprob": min(logprobs) if logprobs else None,
        "no_speech_prob": max(no_speech_probs) if no_speech_probs else None,
        "compression_ratio": max(compression_ratios) if compression_ratios else None,
        "mean_word_probability": float(np.mean(word_probabilities)) if word_probabilities else None,
        "min_word_probability": min(word_probabilities) if word_probabilities else None,
    }, quality_reasons


def rebase_proposal(args: argparse.Namespace) -> dict[str, Any]:
    """Rebase candidate labels on clip-level Tiny, then require Tiny/Small agreement.

    The operation preserves source intervals, protected rows, source hashes,
    and derivative provenance from the original proposal.  The original label
    remains in each candidate's ``original_text`` field for audit.
    """

    proposal_path = _resolve(args.proposal)
    proposal = _load_json(proposal_path)
    if not isinstance(proposal, Mapping) or not isinstance(proposal.get("candidates"), list):
        raise ValueError("proposal must contain a candidates list")
    tiny_path = _resolve(args.tiny_audit)
    small_path = _resolve(args.small_audit)
    tiny_payload, tiny_rows = _audit_payload(tiny_path)
    small_payload, small_rows = _audit_payload(small_path)
    tiny_model = _verify_audit_model(tiny_payload, "tiny", tiny_path)
    small_model = _verify_audit_model(small_payload, "small", small_path)
    if tiny_model["path"] == small_model["path"]:
        raise ValueError("clip Tiny and Small audits must use different model identities")
    if tiny_payload.get("word_timestamps") is not True:
        raise ValueError("clip Tiny audit must be generated with --word-timestamps")
    tiny_by_id = {str(row.get("id")): row for row in tiny_rows if row.get("id") is not None}
    small_by_id = {str(row.get("id")): row for row in small_rows if row.get("id") is not None}
    if len(tiny_by_id) != len(tiny_rows) or len(small_by_id) != len(small_rows):
        raise ValueError("Tiny or Small audit contains duplicate/missing row IDs")
    proposal_ids = {str(candidate.get("id")) for candidate in proposal["candidates"] if isinstance(candidate, Mapping)}
    if set(tiny_by_id) != proposal_ids or set(small_by_id) != proposal_ids:
        raise ValueError("Tiny and Small audit row IDs must exactly match proposal candidates")

    from reference_evaluator import _expand_contractions as evaluator_expand_contractions
    from reference_evaluator import word_error_rate as evaluator_word_error_rate

    rebased = json.loads(json.dumps(proposal))
    candidates: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    exact_count = 0
    for original_candidate in proposal["candidates"]:
        if not isinstance(original_candidate, Mapping):
            raise ValueError("proposal candidate is not an object")
        candidate = dict(original_candidate)
        candidate_id = str(candidate.get("id"))
        tiny_row = tiny_by_id.get(candidate_id)
        small_row = small_by_id.get(candidate_id)
        if tiny_row is None or small_row is None:
            raise ValueError(f"Tiny and Small audits must each contain candidate {candidate_id}")
        _, audio_hash = _verify_audit_audio_row(tiny_row, candidate, "clip Tiny")
        _verify_audit_audio_row(small_row, candidate, "Small")
        tiny_text, tiny_meta, tiny_quality_reasons = _verify_tiny_clip_row(tiny_row, candidate)
        small_text = _normalise_text(_audit_text(small_row))
        consensus = evaluator_word_error_rate(
            evaluator_expand_contractions(tiny_text), evaluator_expand_contractions(small_text)
        )
        exact = float(consensus["wer"]) == 0.0 and not tiny_quality_reasons
        if exact:
            exact_count += 1
        else:
            reasons = []
            if tiny_quality_reasons:
                reasons.append({"reason": "clip_tiny_quality", "details": tiny_quality_reasons})
            if float(consensus["wer"]) != 0.0:
                reasons.append({"reason": "tiny_small_disagreement", "consensus": consensus})
            rejected.append({"id": candidate_id, "reasons": reasons, "consensus": consensus, "tiny_text": tiny_text, "small_text": small_text})
        original_text = _normalise_text(candidate.get("original_text", candidate.get("text", "")))
        original_confidence = {
            key: candidate.get(key)
            for key in ("avg_logprob", "no_speech_prob", "mean_word_probability", "min_word_probability")
            if key in candidate
        }
        candidate["original_text"] = original_text
        candidate["original_confidence"] = original_confidence
        candidate["text"] = tiny_text
        candidate["clip_tiny_text"] = tiny_text
        for key in ("avg_logprob", "no_speech_prob", "mean_word_probability", "min_word_probability"):
            candidate[key] = tiny_meta.get(key)
        candidate["clip_tiny_audit"] = {
            **tiny_meta,
            "quality_reasons": tiny_quality_reasons,
            "report": tiny_model["report"],
            "report_sha256": tiny_model["report_sha256"],
            "model": tiny_model,
        }
        candidate["clip_tiny_small_consensus"] = {
            "tiny_text": tiny_text,
            "small_text": small_text,
            "metrics": consensus,
            "exact": exact,
            "strict_gate": "wer == 0.0 after contraction expansion",
            "small_audit": small_model,
        }
        candidate["rebase"] = {
            "method": "clip_level_tiny_reaudit",
            "original_text": original_text,
            "audio_sha256": audio_hash,
            "tiny_audit": tiny_model,
            "small_audit": small_model,
            "strict_consensus_exact": exact,
            "quality_reasons": tiny_quality_reasons,
        }
        candidates.append(candidate)

    rebased["status"] = "rebased"
    rebased["candidates"] = candidates
    rebased["rebase"] = {
        "method": "clip_level_tiny_reaudit",
        "original_proposal": str(proposal_path.resolve()),
        "original_proposal_sha256": _sha256(proposal_path),
        "tiny_audit": tiny_model,
        "small_audit": small_model,
        "word_timestamps_required": True,
        "strict_consensus": "wer == 0.0 after contraction expansion",
        "candidate_count": len(candidates),
        "exact_count": exact_count,
        "rejected_count": len(rejected),
        "rejected": rejected,
        "protected_intervals_preserved": True,
        "protected_intervals_sha256": _json_sha256(proposal.get("protected_intervals", [])),
        "source_sha256": (proposal.get("source") or {}).get("actual_sha256") if isinstance(proposal.get("source"), Mapping) else None,
        "source_identity_preserved": True,
    }
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = _resolve(args.output_proposal) if args.output_proposal else output_dir / "proposal_rebased.json"
    _write_json(output_path, rebased)
    return {
        "status": "rebased",
        "proposal": str(output_path),
        "candidate_count": len(candidates),
        "exact_count": exact_count,
        "rejected_count": len(rejected),
    }


def _words(text: str) -> list[str]:
    return [word.lower().replace("’", "'") for word in _WORD_RE.findall(text)]






def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def stage_stage(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_consensus_wer != 0.0:
        raise ValueError("Training admission requires exact normalized transcript agreement")
    family = _family(args)
    proposal_path = _resolve(args.proposal)
    proposal = _load_json(proposal_path)
    proposal_family = proposal.get("family") or proposal.get("voice") or ASMR_FAMILY
    if proposal_family != family:
        raise ValueError(f"proposal family {proposal_family!r} does not match requested family {family!r}")
    audit_path = _resolve(args.audit)
    audit_rows = _audit_rows(audit_path)

    # A caller may supply a family-specific protected manifest at stage time.
    # When omitted, retain the proposal's manifest and protected intervals for
    # backwards compatibility.  If supplied, reselect the raw source by hash
    # and reject any candidate overlapping its selected reference/heldout rows.
    proposal_source = proposal.get("source") if isinstance(proposal.get("source"), Mapping) else {}
    source_hash = str(proposal_source.get("actual_sha256") or proposal_source.get("sha256") or "")
    explicit_references = getattr(args, "references_manifest", None)
    references_value = explicit_references
    if references_value is None:
        proposal_references = proposal.get("references_manifest")
        references_value = proposal_references if proposal_references else None
    references_path: Path | None = None
    selected_source_path: Path | None = None
    protected: list[dict[str, Any]] = []
    if references_value:
        candidate_references_path = _resolve(references_value)
        if candidate_references_path.exists():
            references_path = candidate_references_path
            references_manifest = _load_json(references_path)
            source, selected_source_meta = _source_from_references(references_path, family)
            selected_source_path = source
            selected_hash = str(selected_source_meta["actual_sha256"])
            if source_hash and selected_hash != source_hash:
                raise ValueError("stage references manifest source hash does not match proposal source")
            source_hash = selected_hash
            protected = _protected_intervals(references_manifest, source, source_hash, family)
        elif explicit_references is None:
            # A stale proposal can still be staged from its embedded protected
            # intervals.  An explicitly supplied missing manifest is an error.
            protected = [dict(row) for row in proposal.get("protected_intervals", []) if isinstance(row, Mapping)]
        else:
            raise FileNotFoundError(candidate_references_path)
    if not protected:
        protected = [dict(row) for row in proposal.get("protected_intervals", []) if isinstance(row, Mapping)]

    # A decoded proposal must still be bound to the selected raw source at
    # stage time.  Recompute both file hashes and ffprobe alignment before any
    # candidate admission.  This catches replacement of the derivative or its
    # provenance JSON after proposal generation.
    decoded_source = _decoded_source_metadata(proposal)
    if decoded_source is not None:
        if selected_source_path is None:
            proposal_source_path = proposal_source.get("path") or proposal_source.get("absolute_path")
            if not proposal_source_path:
                raise ValueError("decoded proposal has no raw source path for alignment verification")
            selected_source_path = _resolve(str(proposal_source_path))
        decode_path = decoded_source.get("path") or decoded_source.get("decode_audio")
        provenance_path = decoded_source.get("provenance") or decoded_source.get("decode_provenance")
        if not decode_path or not provenance_path:
            raise ValueError("decoded proposal is missing decode path or provenance path")
        decoded_source = _validate_decode_source(
            selected_source_path,
            source_hash,
            str(decode_path),
            str(provenance_path),
            expected=decoded_source,
        )

    proposal_rebase = proposal.get("rebase")
    if isinstance(proposal_rebase, Mapping):
        # Rebase reports are immutable inputs to admission.  Recheck their
        # report hashes and model identities before using rebased labels.
        tiny_audit_meta = proposal_rebase.get("tiny_audit")
        small_audit_meta = proposal_rebase.get("small_audit")
        if not isinstance(tiny_audit_meta, Mapping) or not isinstance(small_audit_meta, Mapping):
            raise ValueError("rebased proposal is missing Tiny/Small audit identities")
        tiny_report = _resolve(str(tiny_audit_meta.get("report", "")))
        small_report = _resolve(str(small_audit_meta.get("report", "")))
        if not tiny_report.is_file() or _sha256(tiny_report) != str(tiny_audit_meta.get("report_sha256")):
            raise ValueError("clip Tiny audit report changed since proposal rebase")
        if not small_report.is_file() or _sha256(small_report) != str(small_audit_meta.get("report_sha256")):
            raise ValueError("Small audit report changed since proposal rebase")
        _verify_audit_model(_load_json(tiny_report), "tiny", tiny_report)
        _verify_audit_model(_load_json(small_report), "small", small_report)
        if small_report != audit_path:
            raise ValueError("stage audit is different from Small audit used for proposal rebase")

    # Reuse the evaluator's exact contraction normalization and Levenshtein
    # tie-break rules.  The import stays inside this admission child, so
    # proposal/transcription remains independent of the evaluator stack.
    from reference_evaluator import _expand_contractions as evaluator_expand_contractions
    from reference_evaluator import word_error_rate as evaluator_word_error_rate

    by_id = {str(row.get("id")): row for row in audit_rows if row.get("id") is not None}
    by_path = {str(Path(row.get("path")).resolve()): row for row in audit_rows if row.get("path")}
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for candidate in proposal.get("candidates", []):
        candidate_id = str(candidate.get("id"))
        candidate_family = candidate.get("family") or candidate.get("voice") or proposal_family
        if candidate_family != family:
            rejected.append({"id": candidate_id, "reasons": ["family_mismatch"], "candidate_family": candidate_family, "family": family})
            continue
        clip_tiny_audit = candidate.get("clip_tiny_audit")
        if isinstance(clip_tiny_audit, Mapping) and clip_tiny_audit.get("quality_reasons"):
            rejected.append({"id": candidate_id, "reasons": ["clip_tiny_quality_rejection"], "quality_reasons": list(clip_tiny_audit.get("quality_reasons"))})
            continue
        if decoded_source is not None:
            candidate_decode = _decoded_source_metadata(candidate)
            decode_mismatch = not isinstance(candidate_decode, Mapping)
            if isinstance(candidate_decode, Mapping):
                for key in ("path", "sha256", "provenance", "provenance_sha256"):
                    if candidate_decode.get(key) and str(candidate_decode.get(key)) != str(decoded_source.get(key)):
                        decode_mismatch = True
            if decode_mismatch:
                rejected.append({"id": candidate_id, "reasons": ["decoded_source_metadata_mismatch"]})
                continue
        if source_hash and candidate.get("source_audio_sha256") and candidate.get("source_audio_sha256") != source_hash:
            rejected.append({"id": candidate_id, "reasons": ["source_hash_mismatch"], "proposal_source_sha256": source_hash, "candidate_source_sha256": candidate.get("source_audio_sha256")})
            continue
        try:
            candidate_start = float(candidate["start_s"])
            candidate_end = float(candidate["end_s"])
        except (KeyError, TypeError, ValueError):
            rejected.append({"id": candidate_id, "reasons": ["invalid_source_interval"]})
            continue
        overlap = next((item for item in protected if candidate_start < float(item["end_s"]) and candidate_end > float(item["start_s"])), None)
        if overlap is not None:
            rejected.append({"id": candidate_id, "reasons": ["protected_interval_overlap"], "protected": dict(overlap)})
            continue
        audio_path = _resolve(candidate.get("path", ""))
        if not audio_path.is_file():
            rejected.append({"id": candidate_id, "reasons": ["missing_audio"]})
            continue
        actual_audio_hash = _sha256(audio_path)
        if candidate.get("audio_sha256") != actual_audio_hash:
            rejected.append(
                {
                    "id": candidate_id,
                    "reasons": ["stale_audio_hash"],
                    "proposal_audio_sha256": candidate.get("audio_sha256"),
                    "actual_audio_sha256": actual_audio_hash,
                }
            )
            continue
        audit = by_id.get(candidate_id) or by_path.get(str(Path(candidate.get("path", "")).resolve()))
        if audit is None:
            rejected.append({"id": candidate_id, "reason": "missing_small_audit"})
            continue
        if Path(audit.get("path", "")).resolve() != audio_path:
            rejected.append({"id": candidate_id, "reason": "audit_audio_path_mismatch"})
            continue
        if audit.get("audio_sha256") != actual_audio_hash:
            rejected.append({"id": candidate_id, "reason": "audit_audio_hash_missing_or_changed"})
            continue
        tiny_text = _normalise_text(candidate.get("text", ""))
        small_text = _audit_text(audit)
        normalized_wer = evaluator_word_error_rate(
            evaluator_expand_contractions(tiny_text), evaluator_expand_contractions(small_text)
        )
        reasons = []
        if len(_words(tiny_text)) < MIN_WORDS or len(_words(small_text)) < MIN_WORDS:
            reasons.append("too_few_words")
        if not small_text:
            reasons.append("empty_small_transcript")
        if float(normalized_wer["wer"]) > float(args.max_consensus_wer):
            reasons.append("tiny_small_disagreement")
        lower = f"{tiny_text} {small_text}".lower()
        markers = [marker for marker in HALLUCINATION_MARKERS if marker in lower]
        if markers:
            reasons.append("hallucination_marker")
        avg_logprob = candidate.get("avg_logprob")
        no_speech_prob = candidate.get("no_speech_prob")
        mean_word_probability = candidate.get("mean_word_probability")
        if avg_logprob is not None and float(avg_logprob) < float(args.min_avg_logprob):
            reasons.append("low_avg_logprob")
        if no_speech_prob is not None and float(no_speech_prob) > float(args.max_no_speech_prob):
            reasons.append("high_no_speech_prob")
        if mean_word_probability is not None and float(mean_word_probability) < float(args.min_word_probability):
            reasons.append("low_word_probability")
        for segment in audit.get("segments", []):
            if (segment.get("avg_logprob", -99) < args.min_avg_logprob
                or segment.get("no_speech_prob", 1) > args.max_no_speech_prob
                or segment.get("compression_ratio", 99) > 2.4):
                reasons.append("low_small_confidence")
                break
        if reasons:
            rejected.append({"id": candidate_id, "reasons": reasons, "tiny_text": tiny_text, "small_text": small_text, "consensus": normalized_wer})
            continue
        text = tiny_text
        accepted.append(
            {
                "id": candidate_id,
                "audio_path": str(audio_path),
                "text": text,
                "source_text": text,
                "speaker_id": family,
                "family": family,
                "voice": family,
                "source_id": _source_id(family, source_hash or candidate.get("source_audio_sha256")),
                "start_s": float(candidate["start_s"]),
                "end_s": float(candidate["end_s"]),
                "speech_start_s": float(candidate.get("speech_start_s", candidate["start_s"])),
                "speech_end_s": float(candidate.get("speech_end_s", candidate["end_s"])),
                "duration_s": float(candidate["duration_s"]),
                "transcript_audit": {
                    "status": "accepted",
                    "method": "independent_tiny_small_consensus",
                    "human_verified": False,
                    "tiny_text": tiny_text,
                    "small_text": small_text,
                    "audit_path": str(audit_path.resolve()),
                    "consensus": normalized_wer,
                    "text_sha256": _text_sha256(text),
                    "audio_sha256": _sha256(audio_path),
                },
                "proposal_path": str(proposal_path.resolve()),
                "source_audio": candidate.get("source_audio"),
                "source_audio_sha256": candidate.get("source_audio_sha256"),
            }
        )
        if "original_text" in candidate:
            accepted[-1]["original_text"] = candidate["original_text"]
        if "original_confidence" in candidate:
            accepted[-1]["original_confidence"] = candidate["original_confidence"]
        if "clip_tiny_text" in candidate:
            accepted[-1]["clip_tiny_text"] = candidate["clip_tiny_text"]
        if "clip_tiny_audit" in candidate:
            accepted[-1]["clip_tiny_audit"] = candidate["clip_tiny_audit"]
        if "clip_tiny_small_consensus" in candidate:
            accepted[-1]["clip_tiny_small_consensus"] = candidate["clip_tiny_small_consensus"]
        if "rebase" in candidate:
            accepted[-1]["rebase"] = candidate["rebase"]
        if decoded_source is not None:
            accepted[-1].update(
                {
                    "decode_audio": decoded_source["path"],
                    "decode_audio_sha256": decoded_source["sha256"],
                    "decode_provenance": decoded_source["provenance"],
                    "decode_provenance_sha256": decoded_source["provenance_sha256"],
                    "decode_alignment": decoded_source["alignment"],
                    "decoded_source": dict(decoded_source),
                    "decoded_source_is_aligned": True,
                }
            )
    accepted.sort(key=lambda row: (float(row["start_s"]), str(row["id"])))
    required_total = int(args.min_train) + int(args.min_valid)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_report: dict[str, Any] = {
        "schema_version": 1,
        "family": family,
        "status": "accepted" if len(accepted) >= required_total else "insufficient",
        "proposal": str(proposal_path),
        "audit": str(audit_path),
        "references_manifest": str(references_path) if references_path else None,
        "protected_intervals": protected,
        "decoded_source": decoded_source,
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "min_train": int(args.min_train),
        "min_valid": int(args.min_valid),
        "accepted": accepted,
        "rejected": rejected,
        "production_manifest": None,
        "command": list(sys.argv),
    }
    if len(accepted) < required_total:
        _write_json(output_dir / "stage_report.json", stage_report)
        return {"status": "insufficient", "accepted_count": len(accepted), "required": required_total, "stage_report": str(output_dir / "stage_report.json")}

    reference_profile_value = getattr(args, "reference_profile", None) or _default_profile(family)
    reference_profile_path = _resolve(reference_profile_value)
    profile = _load_json(reference_profile_path)
    reference_audio = _resolve(str(profile["reference"]))
    if not reference_audio.exists():
        raise FileNotFoundError(f"fixed Nano reference does not exist: {reference_audio}")
    profile_reference = _profile_reference_metadata(
        profile,
        reference_profile_path,
        reference_audio,
        family,
        selected_source_path
        or _resolve(str(proposal_source.get("path") or proposal_source.get("absolute_path") or reference_audio)),
        source_hash,
    )
    if profile_reference["origin"].get("same_raw_source") is False:
        raise ValueError("reference profile origin source does not match selected family source")
    # Spread validation across source time instead of reserving only the final
    # delivery style. This rule is fixed before any adaptation is trained.
    valid_count = int(args.min_valid)
    valid_indices = {int((j + 1) * len(accepted) / (valid_count + 1)) for j in range(valid_count)}
    for index, row in enumerate(accepted):
        row["split"] = "valid" if index in valid_indices else "train"
        row["reference_audio_path"] = str(reference_audio)
        row["reference_source"] = str(reference_profile_path)
        row["reference_audio_sha256"] = _sha256(reference_audio)
        row["reference_profile_kind"] = profile_reference["kind"]
        row["reference_overlap"] = _reference_overlap(row, profile_reference, family, source_hash)
    manifest = {
        "format": "nano_repaired_adaptation_manifest_v1",
        "generated_by": "scripts/nano_lab/repair_dataset.py stage",
        "created_at_unix": time.time(),
        "source_proposal": str(proposal_path),
        "source_audit": str(audit_path),
        "family": family,
        "source_id": _source_id(family, source_hash),
        "references_manifest": str(references_path) if references_path else proposal.get("references_manifest"),
        "references_manifest_sha256": _sha256(references_path) if references_path else proposal.get("references_manifest_sha256"),
        "protected_intervals": protected,
        "reference_profile": str(reference_profile_path),
        "reference_audio": str(reference_audio),
        "reference_audio_sha256": _sha256(reference_audio),
        "reference_profile_metadata": profile_reference,
        "voice": family,
        "rows": accepted,
        "counts": {"all": len(accepted), "train": sum(row["split"] == "train" for row in accepted), "valid": sum(row["split"] == "valid" for row in accepted)},
        "human_verified": False,
        "production_ready": False,
        "training_admission": "automatic_transcript_consensus_passed",
        "provenance": {
            "candidate_audio_is_decoded_source": decoded_source is not None,
            "old_cache_modified": False,
            "split_rule": f"{valid_count} validation rows at evenly spaced source-time ranks",
            "valid_indices": sorted(valid_indices),
            "reference_audio_origin": profile_reference["kind"],
            "overlap_identity": "raw_source_sha256_and_interval",
            "profile_reference_isolated_from_raw_source": profile_reference["kind"] == "isolated_derivative",
        },
    }
    manifest_path = _resolve(args.output_manifest) if args.output_manifest else output_dir / "adaptation_repaired.json"
    _write_json(manifest_path, manifest)
    stage_report["production_manifest"] = str(manifest_path)
    stage_report["manifest_counts"] = manifest["counts"]
    _write_json(output_dir / "stage_report.json", stage_report)
    return {"status": "accepted", "accepted_count": len(accepted), "manifest": str(manifest_path), "counts": manifest["counts"], "stage_report": str(output_dir / "stage_report.json")}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    propose = sub.add_parser("propose", help="transcribe source and propose protected, non-overlapping windows")
    propose.add_argument("--family", choices=FAMILY_CHOICES, default=ASMR_FAMILY)
    propose.add_argument("--references-manifest", "--manifest", dest="references_manifest", type=Path, default=DEFAULT_REFERENCES)
    propose.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    propose.add_argument("--whisper-model", type=Path, default=DEFAULT_WHISPER_MODEL)
    propose.add_argument("--reuse-transcript", type=Path, help="reuse a previously saved tiny.en transcript JSON")
    propose.add_argument("--decode-audio", type=Path, help="optional time-aligned derivative used for transcription and clip extraction")
    propose.add_argument("--decode-provenance", type=Path, help="JSON provenance for --decode-audio; must identify the selected raw source")
    propose.add_argument("--max-candidates", type=int, default=MAX_CANDIDATES)

    stage = sub.add_parser("stage", help="admit rows after independent Whisper-small audit")
    stage.add_argument("--family", choices=FAMILY_CHOICES, default=ASMR_FAMILY)
    stage.add_argument("--references-manifest", "--manifest", dest="references_manifest", type=Path, help="family-specific protected references manifest (defaults to proposal manifest)")
    stage.add_argument("--proposal", type=Path, required=True)
    stage.add_argument("--audit", type=Path, required=True)
    stage.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    stage.add_argument("--output-manifest", type=Path)
    stage.add_argument("--reference-profile", type=Path, help="conditioning profile; defaults by family")
    stage.add_argument("--max-consensus-wer", type=float, default=MAX_CONSENSUS_WER)
    stage.add_argument("--min-avg-logprob", type=float, default=MIN_AVG_LOGPROB)
    stage.add_argument("--max-no-speech-prob", type=float, default=MAX_NO_SPEECH_PROB)
    stage.add_argument("--min-word-probability", type=float, default=MIN_WORD_PROB)
    stage.add_argument("--min-train", type=int, default=8)
    stage.add_argument("--min-valid", type=int, default=2)

    rebase = sub.add_parser("rebase", help="rebase proposal labels on verified clip-level Tiny ASR")
    rebase.add_argument("--proposal", type=Path, required=True, help="original source-window proposal")
    rebase.add_argument("--tiny-audit", type=Path, required=True, help="clip-level Tiny audit made with --word-timestamps")
    rebase.add_argument("--small-audit", type=Path, required=True, help="independent Whisper-small audit for the same clips")
    rebase.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    rebase.add_argument("--output-proposal", type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "propose":
            result = stage_propose(args)
        elif args.command == "stage":
            result = stage_stage(args)
        else:
            result = rebase_proposal(args)
    except Exception as exc:
        output_dir = _resolve(getattr(args, "output_dir", DEFAULT_OUTPUT))
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(output_dir / f"{args.command}_error.json", {"status": "error", "error": f"{type(exc).__name__}: {exc}", "command": list(sys.argv)})
        raise
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 1 if result.get("status") == "insufficient" else 0


if __name__ == "__main__":
    raise SystemExit(main())
