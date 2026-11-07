from memory_e2e_helpers import RATES
from memory_helpers import commit_all, copy_sample, edit, needs_git, sample_repo, write

from core.memory.freshness import BEHIND, FRESH, NOT_INDEXED, STALE, check_freshness
from core.memory.indexer import IncrementalIndexer
from core.memory.store import MemoryStore

TAX_EDIT = ("(1 + TAX_RATE), 2)", "(1 + TAX_RATE), 3)")


def indexed(root) -> MemoryStore:
    store = MemoryStore()
    IncrementalIndexer(root, store).update()
    return store


def reasons_text(result) -> str:
    return " ".join(result.reasons)


@needs_git
def test_a_clean_indexed_repository_is_fresh(tmp_path):
    root, head = sample_repo(tmp_path)
    result = check_freshness(root, indexed(root))
    assert result.status == FRESH and result.is_fresh and result.reasons == ()
    assert result.head == head and not result.dirty and result.indexed.commit_hash == head
    assert "matches the repository" in result.headline


@needs_git
def test_an_uncommitted_edit_makes_the_index_stale(tmp_path):
    root, _ = sample_repo(tmp_path)
    store = indexed(root)
    edit(root, RATES, *TAX_EDIT)
    result = check_freshness(root, store)
    assert result.status == STALE and result.changes.modified == [RATES]
    text = reasons_text(result)
    assert "modified: pricing/rates.py" in text and "uncommitted changes" in text
    assert "HEAD is" not in text
    assert "may use outdated code" in result.headline


@needs_git
def test_a_new_commit_names_both_hashes_and_makes_the_index_stale(tmp_path):
    root, head = sample_repo(tmp_path)
    store = indexed(root)
    edit(root, RATES, *TAX_EDIT)
    new_head = commit_all(root, "tune the tax")
    result = check_freshness(root, store)
    assert result.status == STALE and result.head == new_head
    text = reasons_text(result)
    assert f"HEAD is {new_head[:12]} but the index was built at {head[:12]}" in text
    assert "Latest commit: tune the tax" in text


@needs_git
def test_committing_what_was_already_indexed_is_behind_not_stale(tmp_path):
    root, _ = sample_repo(tmp_path)
    edit(root, RATES, *TAX_EDIT)
    store = indexed(root)  # indexed while dirty
    commit_all(root, "tune the tax")
    result = check_freshness(root, store)
    assert result.status == BEHIND and not result.changes.has_changes
    text = reasons_text(result)
    assert "HEAD is" in text and "no longer has the uncommitted changes" in text
    assert "evidence is still valid" in result.headline


@needs_git
def test_a_commit_that_touches_only_other_files_is_behind(tmp_path):
    root, head = sample_repo(tmp_path)
    store = indexed(root)
    write(root, "README.md", "# notes\n")
    new_head = commit_all(root, "docs")
    result = check_freshness(root, store)
    assert result.status == BEHIND and result.head == new_head != head
    assert not result.changes.has_changes


@needs_git
def test_a_renamed_file_is_reported_as_a_rename(tmp_path):
    root, _ = sample_repo(tmp_path)
    store = indexed(root)
    (root / "shop" / "receipt.py").rename(root / "shop" / "receipts.py")
    result = check_freshness(root, store)
    assert result.status == STALE
    assert "renamed: shop/receipt.py -> shop/receipts.py" in reasons_text(result)


def test_an_unindexed_repository_is_reported_as_such(tmp_path):
    root = copy_sample(tmp_path / "repo")
    result = check_freshness(root, MemoryStore())
    assert result.status == NOT_INDEXED and result.indexed is None
    assert "has not been indexed" in result.headline


def test_without_git_freshness_follows_the_file_contents(tmp_path):
    root = copy_sample(tmp_path / "repo")
    store = indexed(root)
    assert check_freshness(root, store).status == FRESH
    edit(root, RATES, *TAX_EDIT)
    result = check_freshness(root, store)
    assert result.status == STALE and result.head is None
    assert not any("HEAD" in reason for reason in result.reasons)


def test_checking_freshness_never_writes_to_the_store(tmp_path):
    root = copy_sample(tmp_path / "repo")
    store = indexed(root)
    edit(root, RATES, *TAX_EDIT)
    before = (len(store.list_versions()), store.manifest())
    check_freshness(root, store)
    assert (len(store.list_versions()), store.manifest()) == before


def test_after_an_update_the_index_is_fresh_again(tmp_path):
    root = copy_sample(tmp_path / "repo")
    store = indexed(root)
    edit(root, RATES, *TAX_EDIT)
    assert check_freshness(root, store).status == STALE
    IncrementalIndexer(root, store).update()
    assert check_freshness(root, store).status == FRESH