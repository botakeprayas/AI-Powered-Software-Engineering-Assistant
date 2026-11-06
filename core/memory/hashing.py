"""Content hashes. A hash is a fingerprint: same bytes, same hash; any change, a new hash."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

from core.memory.models import FileState


def file_digest(data: bytes) -> str:
    """SHA-256 of some bytes, as 64 hex characters."""
    return hashlib.sha256(data).hexdigest()


def tree_digest(files: Mapping[str, FileState]) -> str:
    """One hash for the whole indexed content. Independent of the order of `files`."""
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(files[path].content_hash.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()