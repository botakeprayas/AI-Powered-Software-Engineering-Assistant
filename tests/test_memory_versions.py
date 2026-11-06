import pytest
from memory_e2e_helpers import RATES, assert_matches_scratch
from memory_helpers import commit_all, copy_sample, edit, git, needs_git, sample_repo, write

from core.memory.indexer import IncrementalIndexer
from core.memory.models import SOURCE_GIT
from core.memory.store import MemoryStore

TAX_EDIT = ("(1 + TAX_RATE), 2)", "(1 + TAX_RATE), 3)")


@needs_git
def test_versions_follow_commits_and_the_working_tree(tmp_path):
    root, head = sample_repo(tmp_path)
    store = MemoryStore()
    indexer = IncrementalIndexer(root, store)
    one = indexer.update()
    v1 = one.version
    assert one.first_index and one.created_version
    assert (v1.commit_hash, v1.dirty, v1.parent_id) == (head, False, None) and v1.branch

    edit(root, RATES, *TAX_EDIT)
    two = indexer.update()
    v2 = two.version
    assert two.changes.modified == [RATES]
    assert (v2.commit_hash, v2.dirty, v2.parent_id) == (head, True, v1.id)
    assert v2.label.endswith("+dirty") and v2.tree_digest != v1.tree_digest

    new_head = commit_all(root, "tune the tax")
    three = indexer.update()
    v3 = three.version
    assert (v3.commit_hash, v3.dirty, v3.parent_id) == (new_head, False, v2.id)
    assert three.created_version and three.files_extracted == [] and three.graph_mode == "none"
    assert v3.tree_digest == v2.tree_digest  # the edit was already indexed
    assert any("No indexed Python content changed" in note for note in three.notes)
    assert [v.id for v in store.list_versions()] == [3, 2, 1]

    four = indexer.update()
    assert not four.created_version and four.version.id == v3.id


@needs_git
def test_a_commit_that_only_touches_other_files_makes_a_version_but_reprocesses_nothing(tmp_path):
    root, head = sample_repo(tmp_path)
    indexer = IncrementalIndexer(root, MemoryStore())
    first = indexer.update().version
    write(root, "README.md", "# notes\n")
    new_head = commit_all(root, "docs")
    update = indexer.update()
    assert update.created_version and update.version.commit_hash == new_head != head
    assert update.version.parent_id == first.id
    assert update.files_extracted == [] and update.graph_mode == "none"
    assert update.version.tree_digest == first.tree_digest


@needs_git
def test_a_new_untracked_python_file_makes_the_version_dirty(tmp_path):
    root, head = sample_repo(tmp_path)
    indexer = IncrementalIndexer(root, MemoryStore())
    indexer.update()
    write(root, "tools/extra.py", "def extra():\n    return 1\n")
    update = indexer.update()
    assert update.changes.added == ["tools/extra.py"]
    assert update.version.dirty and update.version.commit_hash == head


@needs_git
def test_a_rename_with_edits_is_followed_through_gits_report(tmp_path):
    root, _ = sample_repo(tmp_path)
    store = MemoryStore()
    indexer = IncrementalIndexer(root, store)
    indexer.update()
    git(root, "mv", "shop/receipt.py", "shop/receipts.py")
    edit(root, "shop/receipts.py", '"""Receipts."""', '"""Receipts and slips."""')
    commit_all(root, "rename and edit")
    update = indexer.update()
    (rename,) = update.changes.renamed
    assert (rename.old_path, rename.new_path, rename.source) == (
        "shop/receipt.py", "shop/receipts.py", SOURCE_GIT,
    )
    assert rename.similarity is not None and rename.similarity < 100
    assert update.files_extracted == ["shop/receipts.py"]
    assert "shop/receipt.py" not in store.manifest()
    assert_matches_scratch(store, root)


def test_a_folder_without_git_is_versioned_by_its_content(tmp_path):
    root = copy_sample(tmp_path / "repo")
    store = MemoryStore()
    indexer = IncrementalIndexer(root, store)
    one = indexer.update().version
    edit(root, RATES, *TAX_EDIT)
    two = indexer.update().version
    assert one.commit_hash is None and two.commit_hash is None and not two.dirty
    assert two.parent_id == one.id and two.tree_digest != one.tree_digest
    assert one.label.startswith("files:") and one.label != two.label


@pytest.mark.parametrize("note", ["", "after a pull"])
def test_a_note_is_kept_with_the_version(tmp_path, note):
    root = copy_sample(tmp_path / "repo")
    version = IncrementalIndexer(root, MemoryStore()).update(note=note).version
    assert ("after a pull" in version.note) == bool(note)
    assert "graph: full" in version.note