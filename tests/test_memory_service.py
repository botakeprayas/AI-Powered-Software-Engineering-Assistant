import pytest
from memory_e2e_helpers import RATES
from memory_helpers import commit_all, copy_sample, edit, needs_git, sample_repo, write
from memory_service_helpers import HashEmbedder, ask, make_service

from core.chat.models import QAConfig
from core.memory.freshness import BEHIND, FRESH, STALE
from core.validation.evallog import read_log
from services.llm_service import MockLLM

TAX_EDIT = ("(1 + TAX_RATE), 2)", "(1 + TAX_RATE), 3)")


@pytest.fixture
def open_service(tmp_path):
    opened = []

    def make(root, embedder=None, policy="update"):
        service = make_service(root, tmp_path, embedder, policy)
        opened.append(service)
        return service

    yield make
    for service in opened:
        service.close()


@needs_git
def test_the_first_answer_indexes_the_repository_and_is_tagged_with_the_commit(
    tmp_path, open_service
):
    root, head = sample_repo(tmp_path)
    service = open_service(root)
    answer = ask(service, MockLLM(), tmp_path)
    assert not answer.blocked and answer.validated.answer.status == "answered"
    assert answer.refreshed is not None and answer.refreshed.first_index
    assert answer.freshness.status == FRESH
    stamp = answer.stamp
    assert (stamp.commit_hash, stamp.dirty) == (head, False)
    assert answer.citations
    for citation in answer.citations:
        assert citation["commit"] == head and citation["version_id"] == stamp.version_id
        assert len(citation["code_sha256"]) == 64
    (record,) = service.history()
    assert record.id == answer.record_id and record.version == stamp and not record.outdated
    assert record.citations == answer.citations
    records, bad = read_log(tmp_path / "log.jsonl")
    assert bad == 0 and len(records) == 1
    version = records[0]["version"]
    assert version["commit"] == head and version["freshness"] == "fresh"
    assert version["history_id"] == record.id
    assert records[0]["citations_versioned"] == answer.citations


@needs_git
def test_an_edit_is_picked_up_before_answering_and_the_old_answer_is_marked_outdated(
    tmp_path, open_service
):
    root, head = sample_repo(tmp_path)
    service = open_service(root)
    first = ask(service, MockLLM(), tmp_path)
    edit(root, RATES, *TAX_EDIT)
    second = ask(service, MockLLM(), tmp_path)
    update = second.refreshed
    assert update.changes.modified == [RATES] and update.files_extracted == [RATES]
    assert (update.semantic.embedded, update.semantic.reused) == (1, 14)
    assert second.freshness_before.status == STALE and second.freshness.status == FRESH
    assert second.stamp.version_id == first.stamp.version_id + 1
    assert second.stamp.dirty and second.stamp.commit_hash == head
    assert second.stamp.tree_digest != first.stamp.tree_digest
    newest, oldest = service.history()
    assert not newest.outdated and newest.version == second.stamp
    assert oldest.outdated and oldest.version == first.stamp
    assert first.stamp.label in oldest.outdated_reason
    assert second.stamp.label in oldest.outdated_reason


@needs_git
def test_with_the_warn_policy_a_stale_answer_is_labeled_and_uses_the_old_version(
    tmp_path, open_service
):
    root, _ = sample_repo(tmp_path)
    service = open_service(root)
    first = ask(service, MockLLM(), tmp_path)
    edit(root, RATES, *TAX_EDIT)
    second = ask(service, MockLLM(), tmp_path, policy="warn")
    assert second.refreshed is None and second.freshness.status == STALE
    assert second.stamp == first.stamp
    assert any(w.startswith("STALE INDEX") for w in second.warnings)
    assert all(not record.outdated for record in service.history())
    records, _ = read_log(tmp_path / "log.jsonl")
    assert [r["version"]["freshness"] for r in records] == ["fresh", "stale"]
    assert records[1]["version"]["version_id"] == records[0]["version"]["version_id"]


@needs_git
def test_with_the_block_policy_a_stale_index_gives_no_answer(tmp_path, open_service):
    root, _ = sample_repo(tmp_path)
    service = open_service(root)
    llm = MockLLM()
    ask(service, llm, tmp_path)
    edit(root, RATES, *TAX_EDIT)
    second = ask(service, llm, tmp_path, policy="block")
    assert second.blocked and "stale" in second.reason.lower() and second.validated is None
    assert llm.calls == 1 and len(service.history()) == 1


@needs_git
def test_a_commit_that_changes_no_python_content_warns_but_does_not_invalidate(
    tmp_path, open_service
):
    root, head = sample_repo(tmp_path)
    service = open_service(root)
    first = ask(service, MockLLM(), tmp_path)
    write(root, "README.md", "# notes\n")
    new_head = commit_all(root, "docs")
    blocked_policy = ask(service, MockLLM(), tmp_path, policy="block")
    assert not blocked_policy.blocked and blocked_policy.freshness.status == BEHIND
    assert blocked_policy.stamp == first.stamp
    assert any("HEAD" in warning for warning in blocked_policy.warnings)
    updated = ask(service, MockLLM(), tmp_path, policy="update")
    assert updated.refreshed.created_version and updated.refreshed.files_extracted == []
    assert updated.stamp.commit_hash == new_head != head
    assert all(not record.outdated for record in service.history())


def test_a_failed_embedding_update_blocks_answers_instead_of_using_old_vectors(
    tmp_path, open_service
):
    root = copy_sample(tmp_path / "repo")
    embedder = HashEmbedder()
    service = open_service(root, embedder)
    llm = MockLLM()
    first = ask(service, llm, tmp_path)
    edit(root, RATES, *TAX_EDIT)
    embedder.fail = True
    blocked = ask(service, llm, tmp_path)
    assert blocked.blocked and "vector index" in blocked.reason
    assert "simulated embedding failure" in service.engine_error
    assert any("Embeddings were not updated" in w for w in blocked.refreshed.warnings)
    assert len(service.history()) == 1 and service.history()[0].outdated
    assert service.store.current_version().id == first.stamp.version_id + 1
    embedder.fail = False
    recovered = ask(service, llm, tmp_path)
    assert not recovered.blocked and recovered.refreshed is None
    assert recovered.stamp.version_id == first.stamp.version_id + 1


def test_a_stored_answer_is_reused_only_for_identical_content_and_settings(
    tmp_path, open_service
):
    root = copy_sample(tmp_path / "repo")
    service = open_service(root)
    llm = MockLLM()
    first = ask(service, llm, tmp_path)
    again = ask(service, llm, tmp_path, reuse=True)
    assert again.cached is not None and again.cached.id == first.record_id
    assert again.validated is None and llm.calls == 1 and again.stamp == first.stamp
    other = ask(service, llm, tmp_path, reuse=True, qa_config=QAConfig(min_cosine=-0.9))
    assert other.cached is None and llm.calls == 2
    edit(root, RATES, *TAX_EDIT)
    changed = ask(service, llm, tmp_path, reuse=True)
    assert changed.cached is None and changed.refreshed is not None and llm.calls == 3


def test_a_new_version_gets_a_new_empty_retrieval_cache(tmp_path, open_service):
    root = copy_sample(tmp_path / "repo")
    service = open_service(root)
    first = ask(service, MockLLM(), tmp_path)
    cache = service.retrieval_cache
    assert cache.version_id == first.stamp.version_id
    assert (len(cache), cache.misses, cache.hits) == (1, 1, 0)
    ask(service, MockLLM(), tmp_path)
    assert (cache.hits, cache.misses) == (1, 1)
    edit(root, RATES, *TAX_EDIT)
    second = ask(service, MockLLM(), tmp_path)
    fresh = service.retrieval_cache
    assert fresh is not cache and fresh.version_id == second.stamp.version_id
    assert (len(fresh), fresh.misses, fresh.hits) == (1, 1, 0)


def test_a_folder_without_git_is_tagged_by_its_content(tmp_path, open_service):
    root = copy_sample(tmp_path / "repo")
    service = open_service(root)
    first = ask(service, MockLLM(), tmp_path)
    assert first.stamp.commit_hash is None and first.stamp.label.startswith("files:")
    edit(root, RATES, *TAX_EDIT)
    second = ask(service, MockLLM(), tmp_path)
    assert second.stamp.label != first.stamp.label
    records, _ = read_log(tmp_path / "log.jsonl")
    assert [r["version"]["commit"] for r in records] == [None, None]
    assert records[0]["version"]["label"] == first.stamp.label


def test_a_bad_policy_is_rejected(tmp_path, open_service):
    service = open_service(copy_sample(tmp_path / "repo"))
    with pytest.raises(ValueError, match="policy must be one of"):
        ask(service, MockLLM(), tmp_path, policy="sometimes")