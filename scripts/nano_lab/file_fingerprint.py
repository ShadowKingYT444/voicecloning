"""Bounded, race-aware SHA-256 fingerprints for local files.

The staged ONNX pipeline verifies a graph before it creates an ORT session.
Several checks in one process can refer to the same large graph.  This module
keeps a small process-local cache for those checks.  The cache stores digest
strings only.  It never stores file contents or model tensors.

Every cache key contains the resolved path and the file identity and size
metadata used by the pipeline.  A cache miss hashes the file as a stream and
checks the metadata again before publishing the digest.  A file that changes
while it is being read is rejected instead of producing a digest for an
unknown version.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path


MAX_CACHE_ENTRIES = 64


class FileChangedDuringHashError(RuntimeError):
    """Raised when a file's identity or metadata changes during a hash read."""


@dataclass(frozen=True)
class FileSignature:
    """The metadata used to identify one immutable file version."""

    path: str
    dev: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int


# OrderedDict provides a small LRU.  The value is only a hexadecimal digest.
_CACHE: "OrderedDict[FileSignature, str]" = OrderedDict()


def _stat_signature(path: Path) -> FileSignature:
    stat_result = path.stat()
    return FileSignature(
        path=str(path),
        dev=int(getattr(stat_result, "st_dev", 0)),
        inode=int(getattr(stat_result, "st_ino", 0)),
        size=int(stat_result.st_size),
        mtime_ns=int(getattr(stat_result, "st_mtime_ns", int(stat_result.st_mtime * 1_000_000_000))),
        ctime_ns=int(getattr(stat_result, "st_ctime_ns", int(stat_result.st_ctime * 1_000_000_000))),
    )


def _stream_sha256(path: Path, *, chunk_size: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def clear_cache() -> None:
    """Drop all process-local fingerprints.

    This is intended for tests and long-running callers that want an explicit
    cache boundary.  Normal file changes invalidate entries automatically via
    the metadata in :class:`FileSignature`.
    """

    _CACHE.clear()


def cache_info() -> dict[str, int]:
    """Return bounded cache counters without exposing file contents."""

    return {"size": len(_CACHE), "max_entries": MAX_CACHE_ENTRIES}


def fingerprint_file(path: str | os.PathLike[str], *, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 digest for one stable local file.

    The first metadata snapshot is used for the cache key.  On a miss, the
    file is read in chunks and a second snapshot is taken.  A mismatch raises
    :class:`FileChangedDuringHashError` and does not populate the cache.
    """

    if int(chunk_size) <= 0:
        raise ValueError("chunk_size must be positive")
    resolved = Path(path).expanduser().resolve()
    before = _stat_signature(resolved)
    cached = _CACHE.get(before)
    if cached is not None:
        _CACHE.move_to_end(before)
        return cached

    digest = _stream_sha256(resolved, chunk_size=int(chunk_size))
    after = _stat_signature(resolved)
    if before != after:
        raise FileChangedDuringHashError(
            f"file changed while hashing {resolved}: metadata changed during read"
        )

    _CACHE[before] = digest
    _CACHE.move_to_end(before)
    while len(_CACHE) > MAX_CACHE_ENTRIES:
        _CACHE.popitem(last=False)
    return digest


# Explicit aliases keep the helper convenient for callers that use the common
# ``sha256_file`` name while sharing the same race and cache behavior.
sha256_file = fingerprint_file
file_sha256 = fingerprint_file


__all__ = [
    "MAX_CACHE_ENTRIES",
    "FileChangedDuringHashError",
    "FileSignature",
    "cache_info",
    "clear_cache",
    "file_sha256",
    "fingerprint_file",
    "sha256_file",
]
