from datetime import datetime

import pytest
from memory_helpers import commit_all, edit, git, init_repo, needs_git, sample_repo, write

from core.memory.gitrepo import GitRepo

pytestmark = needs_git


def test_head_branch_commit_info_and_a_clean_tree(tmp_path):
    root, head = sample_repo(tmp_path)
    repo = GitRepo(root)
    assert repo.available()
    assert repo.head() == head and len(head) == 40
    assert repo.branch() is not None
    assert repo.is_dirty() is False
    info = repo.commit_info(head)
    assert info is not None and info.commit == head and info.subject == "initial"
    assert datetime.fromisoformat(info.committed_at).tzinfo is not None
    assert repo.commit_exists(head) and repo.commit_exists(head[:10])


def test_edits_and_new_files_make_the_tree_dirty(tmp_path):
    root, _ = sample_repo(tmp_path)
    repo = GitRepo(root)
    edit(root, "tools/formatting.py", "upper()", "lower()")
    assert repo.is_dirty() is True
    commit_all(root)
    assert repo.is_dirty() is False
    write(root, "tools/untracked.py", "x = 1\n")
    assert repo.is_dirty() is True


def test_a_folder_that_is_not_a_repository_gives_none_everywhere(tmp_path):
    repo = GitRepo(tmp_path)
    assert not repo.available()
    assert repo.head() is None and repo.branch() is None and repo.is_dirty() is None
    assert repo.changed_files("a" * 40) is None and repo.commit_info("a" * 40) is None


def test_a_repository_without_commits_has_no_head(tmp_path):
    init_repo(tmp_path / "empty")
    repo = GitRepo(tmp_path / "empty")
    assert repo.available() and repo.head() is None


def test_changed_files_between_two_commits_cover_every_kind_of_change(tmp_path):
    root, old = sample_repo(tmp_path)
    edit(root, "pricing/rates.py", "TAX_RATE = 0.08", "TAX_RATE = 0.09")
    stock = (
        '"""Stock levels."""\n\n\ndef in_stock(item, stock):\n'
        '    """True when the item has units left."""\n    return stock.get(item, 0) > 0\n'
    )
    write(root, "shop/inventory.py", stock)
    (root / "tools" / "formatting.py").unlink()
    git(root, "mv", "shop/receipt.py", "shop/receipts.py")
    new = commit_all(root)
    changes = GitRepo(root).changed_files(old, new)
    assert changes is not None and len(changes) == 4
    by_path = {c.path: c for c in changes}
    assert by_path["pricing/rates.py"].status == "M"
    assert by_path["shop/inventory.py"].status == "A"
    assert by_path["tools/formatting.py"].status == "D"
    moved = by_path["shop/receipts.py"]
    assert (moved.status, moved.old_path, moved.similarity) == ("R", "shop/receipt.py", 100)


def test_changed_files_between_a_commit_and_the_working_tree(tmp_path):
    root, old = sample_repo(tmp_path)
    edit(root, "pricing/rates.py", "TAX_RATE = 0.08", "TAX_RATE = 0.09")
    changes = GitRepo(root).changed_files(old)
    assert changes is not None
    assert [(c.status, c.path) for c in changes] == [("M", "pricing/rates.py")]


def test_paths_are_relative_to_the_analyzed_folder_and_other_folders_are_ignored(tmp_path):
    base = tmp_path / "mono"
    init_repo(base)
    write(base, "proj/a.py", "x = 1\n")
    write(base, "other/b.py", "y = 1\n")
    old = commit_all(base)
    write(base, "proj/a.py", "x = 2\n")
    write(base, "other/b.py", "y = 2\n")
    new = commit_all(base)
    repo = GitRepo(base / "proj")
    assert repo.head() == new
    changes = repo.changed_files(old, new)
    assert changes is not None and [(c.status, c.path) for c in changes] == [("M", "a.py")]


@pytest.mark.parametrize("revision", ["", "--output=x", "HEAD; rm", "zzzzzzz", "abc"])
def test_malformed_revisions_are_refused_without_running_git(tmp_path, revision):
    root, _ = sample_repo(tmp_path)
    repo = GitRepo(root)
    assert repo.commit_exists(revision) is False
    assert repo.changed_files(revision) is None
    assert repo.file_at(revision, "app.py") is None
    assert repo.commit_info(revision) is None


def test_a_well_formed_but_unknown_hash_is_reported_as_unknown(tmp_path):
    root, _ = sample_repo(tmp_path)
    repo = GitRepo(root)
    assert repo.commit_exists("0" * 40) is False
    assert repo.changed_files("0" * 40) is None


def test_file_at_reads_an_old_version_of_a_file(tmp_path):
    root, old = sample_repo(tmp_path)
    edit(root, "pricing/rates.py", "TAX_RATE = 0.08", "TAX_RATE = 0.09")
    commit_all(root)
    repo = GitRepo(root)
    assert "TAX_RATE = 0.08" in (repo.file_at(old, "pricing/rates.py") or "")
    assert repo.file_at(old, "no/such/file.py") is None


@pytest.mark.parametrize("path", ["../outside.py", "/etc/passwd", "a/../../b.py", "-x", ""])
def test_file_at_refuses_paths_that_could_escape_the_folder(tmp_path, path):
    root, old = sample_repo(tmp_path)
    assert GitRepo(root).file_at(old, path) is None