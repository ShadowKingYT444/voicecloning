"""Resolve and integrity-check profile conditioning caches.

This module is deliberately model-free.  Native and ONNX launchers use it
before importing Torch or constructing an inference runtime.  Profiles that
do not opt in to cache integrity retain the native auto-prepare fallback.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def _sha256_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    """Return the streamed SHA-256 digest for one regular file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _expected_sha256(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("conditioning_cache_sha256 must be a 64-character hexadecimal SHA-256 string")
    normalised = value.lower()
    if any(character not in "0123456789abcdef" for character in normalised):
        raise ValueError("conditioning_cache_sha256 must be a 64-character hexadecimal SHA-256 string")
    return normalised


def resolve_conditioning_cache(
    profile: object,
    *,
    root: str | Path,
    require_exists: bool = False,
) -> dict[str, object]:
    """Resolve a profile cache and enforce its optional integrity contract.

    ``conditioning_cache_required`` and ``conditioning_cache_sha256`` opt a
    profile into strict cache use.  A missing cache is allowed only when both
    fields are absent and ``require_exists`` is false.  This is the native
    launcher compatibility path: it can create a cache from the reference
    audio after the model is loaded.  ONNX preparation passes
    ``require_exists=True`` because it cannot create conditionals.

    The return value contains the resolved ``path``, whether it existed, and
    the expected/verified digest.  No model package is imported here.
    """

    if not hasattr(profile, "get"):
        raise TypeError("voice profile must provide mapping-style get()")
    cache_value = profile.get("conditioning_cache")  # type: ignore[union-attr]
    if cache_value in (None, ""):
        raise ValueError("voice profile has no conditioning_cache")

    path = Path(str(cache_value)).expanduser()
    root_path = Path(root).expanduser().resolve()
    if not path.is_absolute():
        path = root_path / path
    path = path.resolve()

    required_value = profile.get("conditioning_cache_required", False)  # type: ignore[union-attr]
    if not isinstance(required_value, bool):
        raise ValueError("conditioning_cache_required must be a boolean when present")
    expected = _expected_sha256(profile.get("conditioning_cache_sha256"))  # type: ignore[union-attr]
    strict = bool(require_exists or required_value or expected is not None)
    exists = path.exists()
    if not exists:
        if strict:
            raise FileNotFoundError(f"required conditioning cache not found: {path}")
        return {
            "path": path,
            "exists": False,
            "required": False,
            "legacy_fallback": True,
            "expected_sha256": expected,
            "sha256": None,
        }
    if not path.is_file():
        raise FileNotFoundError(f"conditioning cache is not a regular file: {path}")

    actual = _sha256_file(path) if expected is not None else None
    if expected is not None and actual != expected:
        raise ValueError(
            "conditioning cache SHA-256 mismatch: "
            f"{path} has {actual}, profile requires {expected}"
        )
    return {
        "path": path,
        "exists": True,
        "required": strict,
        "legacy_fallback": False,
        "expected_sha256": expected,
        "sha256": actual,
    }


__all__ = ["resolve_conditioning_cache"]
