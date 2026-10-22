import io
import zipfile
from pathlib import Path

import pytest

from core.repo import loader
from core.repo.loader import RepoLoadError, extract_zip, load_zip_to_temp, validate_folder


def make_zip(entries):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for name, content in entries.items():
            zf.writestr(name, content)
    buffer.seek(0)
    return buffer


def test_single_top_folder_is_unwrapped_and_junk_is_not_written(tmp_path):
    source = make_zip(
        {
            "proj/app.py": "def f():\n    pass\n",
            "proj/.env": "KEY=1",
            "proj/id_rsa": "FAKE",
            "proj/.venv/lib/x.py": "x = 1\n",
        }
    )
    root, skipped = extract_zip(source, tmp_path / "out")
    assert root.name == "proj"
    assert (root / "app.py").exists()
    assert not (root / ".env").exists()
    assert not (root / "id_rsa").exists()
    assert not (root / ".venv").exists()
    reasons = {item.path: item.reason for item in skipped}
    assert "sensitive" in reasons[".env"]
    assert "sensitive" in reasons["id_rsa"]
    assert reasons[".venv"] == "virtual environment"


def test_path_traversal_and_absolute_paths_are_blocked(tmp_path):
    source = make_zip(
        {
            "proj/app.py": "x = 1\n",
            "proj/../../evil.py": "boom",
            "/abs.py": "boom",
            "C:/win.py": "boom",
        }
    )
    _, skipped = extract_zip(source, tmp_path / "out")
    assert not (tmp_path / "evil.py").exists()
    assert not (tmp_path / "out" / "evil.py").exists()
    assert sum("unsafe path" in item.reason for item in skipped) == 3


def test_not_a_zip_gives_a_friendly_error(tmp_path):
    with pytest.raises(RepoLoadError, match="not a valid zip"):
        extract_zip(io.BytesIO(b"this is not a zip"), tmp_path / "out")


def test_zip_bomb_limit_is_enforced(tmp_path, monkeypatch):
    monkeypatch.setattr(loader, "MAX_UNCOMPRESSED_BYTES", 10)
    source = make_zip({"proj/big.py": "x" * 1000})
    with pytest.raises(RepoLoadError, match="zip bomb"):
        extract_zip(source, tmp_path / "out")


def test_load_zip_to_temp_can_be_cleaned_up():
    loaded = load_zip_to_temp(make_zip({"proj/app.py": "x = 1\n"}))
    assert (loaded.root / "app.py").exists()
    loaded.cleanup()
    assert not loaded.temp_dir.exists()


@pytest.mark.parametrize("raw", ["", "   "])
def test_empty_folder_text_is_rejected(raw):
    with pytest.raises(RepoLoadError):
        validate_folder(raw)


def test_missing_folder_and_file_paths_are_rejected(tmp_path):
    (tmp_path / "file.txt").write_text("x", encoding="utf-8")
    with pytest.raises(RepoLoadError, match="not found"):
        validate_folder(str(tmp_path / "nope"))
    with pytest.raises(RepoLoadError, match="file, not a folder"):
        validate_folder(str(tmp_path / "file.txt"))


def test_quoted_path_is_accepted(tmp_path):
    assert validate_folder(f'"{tmp_path}"') == tmp_path.resolve()


def test_drive_root_is_refused(tmp_path):
    with pytest.raises(RepoLoadError, match="drive root"):
        validate_folder(str(Path(tmp_path.anchor)))