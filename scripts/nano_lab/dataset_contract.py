"""Reject unaudited or changed text/audio pairs before feature loading/training."""
import hashlib
from pathlib import Path


def require_audited_rows(rows):
    for row in rows:
        if row.get("split") == "reference":
            continue
        label = row.get("id", "unknown")
        audit = row.get("transcript_audit", {})
        if audit.get("status") != "accepted" or not audit.get("method"):
            raise ValueError(f"Transcript audit required for {label}; do not train on unverified source labels")
        text = row.get("source_text", row.get("text", ""))
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != audit.get("text_sha256"):
            raise ValueError(f"Audited transcript changed for {label}")
        path = Path(row["audio_path"])
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest != audit.get("audio_sha256"):
            raise ValueError(f"Audited source audio changed for {label}")


def require_inference_conditioning(rows):
    """Reject caches made with the former independent reference encoders."""
    verified = set()
    for row in rows:
        if row.get("split") == "reference":
            continue
        label = row.get("id", "unknown")
        provenance = row.get("conditioning_provenance", {})
        preprocessing = provenance.get("preprocessing", {})
        if (preprocessing.get("entrypoint") != "ChatterboxTurboTTS.prepare_conditionals"
                or preprocessing.get("norm_loudness") is not True
                or preprocessing.get("prompt_tokens_unmodified") is not True
                or preprocessing.get("speaker_embedding_unmodified") is not True):
            raise ValueError(f"Inference-compatible conditioning required for {label}; rebuild the feature cache")
        path = Path(row["reference_audio_path"]).resolve()
        if path != Path(provenance.get("path", "")).resolve():
            raise ValueError(f"Conditioning reference path changed for {label}")
        expected = provenance.get("sha256")
        if expected != row.get("reference_sha256"):
            raise ValueError(f"Conditioning reference hash mismatch for {label}")
        key = (path, expected)
        if key not in verified:
            with path.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            if digest != expected:
                raise ValueError(f"Conditioning reference audio changed for {label}")
            verified.add(key)
