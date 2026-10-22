"""Getting a repository onto disk safely: a local folder path or an uploaded .zip."""

from __future__ import annotations

import dataclasses
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO

from core.repo.models import SkippedItem
from core.repo.scanner import dir_skip_reason, name_skip_reason

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 100 * 1024 * 1024
MAX_ZIP_ENTRIES = 10_000
CHUNK_BYTES = 64 * 1024


class RepoLoadError(ValueError):
    """A problem with the chosen folder or zip, phrased for the user."""


@dataclass
class LoadedRepo:
    root: Path
    temp_dir: Path
    skipped: list[SkippedItem]

    def cleanup(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)


def validate_folder(raw: str) -> Path:
    """Turn user text into a safe, existing folder path."""
    cleaned = raw.strip().strip('"').strip("'")  # Windows "Copy as path" adds quotes
    if not cleaned:
        raise RepoLoadError("Enter a folder path.")
    try:
        resolved = Path(cleaned).expanduser().resolve()
    except (OSError, RuntimeError) as exc:
        raise RepoLoadError(f"Invalid path: {exc}") from exc
    if not resolved.exists():
        raise RepoLoadError(f"Folder not found: {resolved}")
    if not resolved.is_dir():
        raise RepoLoadError(f"That is a file, not a folder: {resolved}")
    if resolved.parent == resolved:
        raise RepoLoadError("Refusing to scan a drive root. Choose a project folder.")
    return resolved


def _safe_parts(filename: str) -> tuple[str, ...] | None:
    """Path parts of a zip entry, or None if the entry could escape the target folder."""
    name = filename.replace("\\", "/")
    if name.startswith("/"):
        return None
    parts = PurePosixPath(name).parts
    if not parts or any(part == ".." or ":" in part for part in parts):
        return None
    return parts


def _blocked_dir(parts: tuple[str, ...]) -> tuple[str, str] | None:
    for index, part in enumerate(parts[:-1]):
        reason = dir_skip_reason(part)
        if reason:
            return "/".join(parts[: index + 1]), reason
    return None


def _copy_limited(zf: zipfile.ZipFile, info: zipfile.ZipInfo, target: Path, total: int) -> int:
    """Copy one entry, counting REAL bytes (headers can lie) against the size limit."""
    with zf.open(info) as source, open(target, "wb") as out:
        while chunk := source.read(CHUNK_BYTES):
            total += len(chunk)
            if total > MAX_UNCOMPRESSED_BYTES:
                limit_mb = MAX_UNCOMPRESSED_BYTES // (1024 * 1024)
                raise RepoLoadError(
                    f"The zip expands to more than {limit_mb} MB. Refusing (possible zip bomb)."
                )
            out.write(chunk)
    return total


def extract_zip(source: str | Path | BinaryIO, dest: Path) -> tuple[Path, list[SkippedItem]]:
    """Safely extract `source` into `dest`. Returns (repo_root, skipped_items).

    Skipped folders and sensitive files are never written to disk.
    """
    dest = Path(dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    skipped: list[SkippedItem] = []
    try:
        zf = zipfile.ZipFile(source)
    except (zipfile.BadZipFile, OSError) as exc:
        raise RepoLoadError(f"That is not a valid zip file: {exc}") from exc
    with zf:
        infos = zf.infolist()
        if len(infos) > MAX_ZIP_ENTRIES:
            raise RepoLoadError(f"The zip has {len(infos):,} entries (limit {MAX_ZIP_ENTRIES:,}).")
        if any(info.flag_bits & 0x1 for info in infos):
            raise RepoLoadError("Encrypted zip files are not supported.")
        dir_seen: set[str] = set()
        total = 0
        try:
            for info in infos:
                if info.is_dir():
                    continue
                parts = _safe_parts(info.filename)
                if parts is None:
                    skipped.append(SkippedItem(info.filename, "file", "unsafe path in zip"))
                    continue
                rel = "/".join(parts)
                if (info.external_attr >> 16) & 0o170000 == 0o120000:
                    skipped.append(SkippedItem(rel, "file", "symbolic link (not extracted)"))
                    continue
                blocked = _blocked_dir(parts)
                if blocked:
                    if blocked[0] not in dir_seen:
                        dir_seen.add(blocked[0])
                        skipped.append(SkippedItem(blocked[0], "dir", blocked[1]))
                    continue
                reason = name_skip_reason(parts[-1])
                if reason:
                    skipped.append(SkippedItem(rel, "file", reason))
                    continue
                target = dest.joinpath(*parts).resolve()
                if dest not in target.parents:
                    skipped.append(SkippedItem(rel, "file", "unsafe path in zip"))
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                total = _copy_limited(zf, info, target, total)
        except (zipfile.BadZipFile, RuntimeError, OSError) as exc:
            raise RepoLoadError(f"Could not extract the zip: {exc}") from exc

    top_entries = list(dest.iterdir())
    root = dest
    if len(top_entries) == 1 and top_entries[0].is_dir():
        root = top_entries[0]  # a zip of one project folder: use that folder as the root
        prefix = root.name + "/"
        skipped = [
            dataclasses.replace(item, path=item.path[len(prefix):])
            if item.path.startswith(prefix)
            else item
            for item in skipped
        ]
    return root, skipped


def load_zip_to_temp(source: str | Path | BinaryIO) -> LoadedRepo:
    """Extract into a fresh temp folder. The caller must call .cleanup() when done."""
    temp_dir = Path(tempfile.mkdtemp(prefix="ai_se_repo_"))
    try:
        root, skipped = extract_zip(source, temp_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return LoadedRepo(root=root, temp_dir=temp_dir, skipped=skipped)