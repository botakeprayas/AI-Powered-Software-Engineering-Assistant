import pytest
from impact_helpers import IMPACT_SAMPLE
from memory_helpers import commit_all, edit, git, needs_git, sample_repo, write

from core.memory.changes import detect_changes, git_renames
from core.memory.gitrepo import GitRepo
from core.memory.hashing import file_digest, tree_digest
from core.memory.manifest import build_manifest
from core.memory.models import SOURCE_CONTENT, SOURCE_GIT, FileState, RenameInfo


def state(path: str, text: str) -> FileState:
    data = text.encode("utf-8")
    return FileState(path, file_digest(data), len(data), True)


def files(texts: dict[str, str]) -> dict[str, FileState]:
    return {path: state(path, text) for path, text in texts.items()}


def test_file_digest_is_sha256_and_any_change_alters_it():
    expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert file_digest(b"abc") == expected
    assert file_digest(b"x = 1\n") == file_digest(b"x = 1\n")
    assert file_digest(b"x = 1\n") != file_digest(b"x = 2\n")


def test_tree_digest_depends_on_paths_and_bytes_but_not_on_order():
    first = files({"a.py": "x = 1\n", "b.py": "y = 1\n"})
    reordered = {"b.py": first["b.py"], "a.py": first["a.py"]}
    assert tree_digest(first) == tree_digest(reordered)
    assert tree_digest(first) != tree_digest(files({"a.py": "x = 2\n", "b.py": "y = 1\n"}))
    assert tree_digest(first) != tree_digest(files({"c.py": "x = 1\n", "b.py": "y = 1\n"}))


def test_identical_manifests_have_no_changes():
    stored = files({"a.py": "x = 1\n", "b.py": "y = 1\n"})
    changes = detect_changes(stored, dict(stored))
    assert not changes.has_changes
    assert changes.unchanged == ["a.py", "b.py"]
    assert changes.to_process == [] and changes.to_remove == []


def test_added_modified_and_deleted_files():
    stored = files({"keep.py": "k = 1\n", "edit.py": "e = 1\n", "gone.py": "g = 1\n"})
    current = files({"keep.py": "k = 1\n", "edit.py": "e = 2\n", "new.py": "n = 1\n"})
    changes = detect_changes(stored, current)
    assert (changes.added, changes.modified, changes.deleted) == (
        ["new.py"], ["edit.py"], ["gone.py"],
    )
    assert changes.renamed == [] and changes.unchanged == ["keep.py"]
    assert changes.to_process == ["edit.py", "new.py"] and changes.to_remove == ["gone.py"]
    assert changes.summary() == "1 added, 1 modified, 1 deleted, 0 renamed, 1 unchanged"


def test_a_moved_file_with_identical_content_is_a_rename_found_by_content():
    changes = detect_changes(files({"old/name.py": "x = 1\n"}), files({"new/name.py": "x = 1\n"}))
    assert changes.renamed == [RenameInfo("old/name.py", "new/name.py", 100, SOURCE_CONTENT)]
    assert changes.added == [] and changes.deleted == []
    assert changes.to_process == ["new/name.py"] and changes.to_remove == ["old/name.py"]


def test_gits_rename_report_covers_a_rename_that_also_edits_the_file():
    stored = files({"old.py": "x = 1\ny = 2\n"})
    current = files({"new.py": "x = 1\ny = 3\n"})
    hint = RenameInfo("old.py", "new.py", 80, SOURCE_GIT)
    without = detect_changes(stored, current)
    assert without.renamed == [] and without.added == ["new.py"]
    assert without.deleted == ["old.py"]
    with_hint = detect_changes(stored, current, [hint])
    assert with_hint.renamed == [hint]
    assert with_hint.added == [] and with_hint.deleted == []


def test_a_git_hint_that_disagrees_with_the_files_is_ignored():
    stored = files({"a.py": "x = 1\n", "c.py": "z = 1\n"})
    current = files({"a.py": "x = 1\n", "b.py": "y = 1\n"})
    hint = RenameInfo("a.py", "b.py", 90, SOURCE_GIT)  # a.py still exists: not renamed
    changes = detect_changes(stored, current, [hint])
    assert changes.renamed == []
    assert changes.added == ["b.py"] and changes.deleted == ["c.py"]


def test_a_copy_is_an_added_file_not_a_rename():
    stored = files({"a.py": "x = 1\n"})
    current = files({"a.py": "x = 1\n", "copy.py": "x = 1\n"})
    changes = detect_changes(stored, current)
    assert changes.added == ["copy.py"] and changes.renamed == []
    assert changes.unchanged == ["a.py"]


def test_several_identical_files_are_paired_in_a_fixed_order():
    stored = files({"a/__init__.py": "", "b/__init__.py": ""})
    current = files({"c/__init__.py": "", "d/__init__.py": ""})
    changes = detect_changes(stored, current)
    assert [(r.old_path, r.new_path) for r in changes.renamed] == [
        ("a/__init__.py", "c/__init__.py"), ("b/__init__.py", "d/__init__.py"),
    ]


def test_the_manifest_holds_only_the_python_files_the_scanner_keeps(tmp_path):
    for rel, text in {
        "app.py": "x = 1\n",
        "notes.txt": "plain\n",
        ".env": "KEY=secret\n",
        "secrets.py": "TOKEN = 1\n",
        "node_modules/m.py": "y = 1\n",
        ".venv/pyvenv.cfg": "home = x\n",
        ".venv/lib/v.py": "z = 1\n",
    }.items():
        write(tmp_path, rel, text)
    result = build_manifest(tmp_path)
    assert sorted(result.files) == ["app.py"]
    assert result.files["app.py"].content_hash == file_digest(b"x = 1\n")
    assert result.warnings == []
    assert {item.path for item in result.skipped} >= {".env", "secrets.py", ".venv"}


def test_the_impact_sample_manifest_has_its_twelve_python_files():
    result = build_manifest(IMPACT_SAMPLE)
    assert len(result.files) == 12 and "pricing/rates.py" in result.files
    data = (IMPACT_SAMPLE / "pricing" / "rates.py").read_bytes()
    assert result.files["pricing/rates.py"].content_hash == file_digest(data)
    assert all(item.is_python for item in result.files.values())


@needs_git
def test_git_reports_an_exact_rename_between_commits(tmp_path):
    root, old = sample_repo(tmp_path)
    git(root, "mv", "shop/receipt.py", "shop/receipts.py")
    new = commit_all(root)
    renames, note = git_renames(GitRepo(root), old, new)
    assert note == ""
    assert renames == [RenameInfo("shop/receipt.py", "shop/receipts.py", 100, SOURCE_GIT)]


@needs_git
def test_git_renames_explains_why_it_has_no_answer(tmp_path):
    root, old = sample_repo(tmp_path)
    assert "no indexed commit" in git_renames(GitRepo(root), None)[1]
    renames, note = git_renames(GitRepo(root), "0" * 40)
    assert renames == [] and "not in this repository" in note
    renames, note = git_renames(GitRepo(tmp_path), old)  # tmp_path itself is not a repository
    assert renames == [] and "not available" in note


@needs_git
def test_a_real_commit_with_every_kind_of_change(tmp_path):
    root, old = sample_repo(tmp_path)
    before = build_manifest(root).files
    edit(root, "pricing/rates.py", "TAX_RATE = 0.08", "TAX_RATE = 0.09")
    stock = (
        '"""Stock levels."""\n\n\ndef in_stock(item, stock):\n'
        '    """True when the item has units left."""\n    return stock.get(item, 0) > 0\n'
    )
    write(root, "shop/inventory.py", stock)
    (root / "tools" / "formatting.py").unlink()
    git(root, "mv", "shop/receipt.py", "shop/receipts.py")
    edit(root, "shop/receipts.py", '"""Receipts."""', '"""Receipts and slips."""')
    new = commit_all(root, "many changes")
    after = build_manifest(root).files
    hints, note = git_renames(GitRepo(root), old, new)
    assert note == ""
    changes = detect_changes(before, after, hints)
    assert changes.added == ["shop/inventory.py"]
    assert changes.modified == ["pricing/rates.py"]
    assert changes.deleted == ["tools/formatting.py"]
    (rename,) = changes.renamed
    assert (rename.old_path, rename.new_path) == ("shop/receipt.py", "shop/receipts.py")
    assert rename.source == SOURCE_GIT
    assert rename.similarity is not None and 50 <= rename.similarity < 100
    assert len(changes.unchanged) == 9


@needs_git
def test_an_uncommitted_move_is_still_found_by_identical_content(tmp_path):
    root, old = sample_repo(tmp_path)
    before = build_manifest(root).files
    (root / "shop" / "receipt.py").rename(root / "shop" / "receipts.py")
    after = build_manifest(root).files
    hints, _ = git_renames(GitRepo(root), old)
    assert hints == []  # git cannot pair an untracked new file with the deleted one
    changes = detect_changes(before, after, hints)
    assert [(r.old_path, r.new_path, r.source) for r in changes.renamed] == [
        ("shop/receipt.py", "shop/receipts.py", SOURCE_CONTENT),
    ]


def test_changed_bytes_are_the_only_thing_that_counts_as_a_modification(tmp_path):
    write(tmp_path, "m.py", "x = 1\n")
    before = build_manifest(tmp_path).files
    write(tmp_path, "m.py", "x = 1\n")  # rewritten with identical bytes
    assert not detect_changes(before, build_manifest(tmp_path).files).has_changes
    write(tmp_path, "m.py", "x = 2\n")
    assert detect_changes(before, build_manifest(tmp_path).files).modified == ["m.py"]


@pytest.mark.parametrize("text", ["", "\n", "x = 1"])
def test_empty_and_unusual_files_are_hashed_like_any_other(tmp_path, text):
    write(tmp_path, "m.py", text)
    manifest = build_manifest(tmp_path).files
    assert manifest["m.py"].content_hash == file_digest(text.encode("utf-8"))