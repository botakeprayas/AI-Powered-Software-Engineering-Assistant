import pytest
from validation_helpers import evidence_for, load_symbols, make_answer

from core.memory.answers import AnswerStore, VersionStamp, question_key
from core.memory.retrieval import VersionedRetriever
from core.memory.store import MemoryStore


def stamp(version_id=1, digest="d" * 64, commit="a" * 40, dirty=False) -> VersionStamp:
    label = commit[:12] + ("+dirty" if dirty else "")
    stamped = "2026-11-01T10:00:00+00:00"
    return VersionStamp(version_id, commit, dirty, digest, label, stamped, "main")


@pytest.fixture(scope="module")
def symbols():
    return load_symbols()


def answer_for(symbols):
    evidence = [evidence_for(symbols["ShoppingCart.total"])]
    return make_answer(
        facts=[("total sums the items", ["E1"])],
        inferences=[("it is probably used at checkout", ["E1"])],
        evidence=evidence,
    )


def test_a_recorded_answer_round_trips_with_its_version_and_citations(symbols):
    answers = AnswerStore(MemoryStore())
    version = stamp()
    record_id = answers.record(
        "How does total work?", version, answer_for(symbols), None, "mock/mock-model", "fresh", "k1"
    )
    record = answers.get(record_id)
    assert record.version == version and record.status == "answered" and not record.outdated
    assert record.question == "How does total work?" and record.llm == "mock/mock-model"
    assert record.freshness == "fresh" and record.level is None
    assert record.answer["facts"] == [{"text": "total sums the items", "evidence_ids": ["E1"]}]
    (citation,) = record.citations
    assert (citation["id"], citation["location"]) == ("E1", "shop/cart.py:18-20")
    assert (citation["commit"], citation["version"]) == ("a" * 40, version.label)
    assert len(citation["code_sha256"]) == 64 and citation["valid"] is None
    assert answers.get(999) is None and answers.count() == 1


def test_cached_answers_need_the_same_content_model_and_settings(symbols):
    answers = AnswerStore(MemoryStore())
    version = stamp()
    record_id = answers.record(
        "How does TOTAL work?", version, answer_for(symbols), None, "mock/m", "fresh", "k1"
    )
    hit = answers.cached("  how does total   work? ", version, "mock/m", "k1")
    assert hit is not None and hit.id == record_id
    same_content = stamp(2, digest="d" * 64, commit="b" * 40)  # a new commit, identical content
    moved = answers.cached("how does total work?", same_content, "mock/m", "k1")
    assert moved is not None and moved.version.commit_hash == "a" * 40
    other_content = stamp(3, digest="e" * 64, commit="c" * 40)
    assert answers.cached("how does total work?", other_content, "mock/m", "k1") is None
    assert answers.cached("how does total work?", version, "mock/other", "k1") is None
    assert answers.cached("how does total work?", version, "mock/m", "k2") is None
    assert answers.cached("something else", version, "mock/m", "k1") is None


def test_changed_content_marks_older_answers_outdated_but_keeps_them(symbols):
    answers = AnswerStore(MemoryStore())
    old, middle = stamp(1), stamp(2, digest="e" * 64, commit="b" * 40)
    first = answers.record("q one", old, answer_for(symbols), None, "m", "fresh", "k")
    second = answers.record("q two", middle, answer_for(symbols), None, "m", "fresh", "k")
    current = stamp(3, digest="f" * 64, commit="c" * 40)
    assert answers.invalidate_content_changes(current) == 2
    marked = answers.get(first)
    assert marked.outdated and marked.outdated_at
    assert old.label in marked.outdated_reason and current.label in marked.outdated_reason
    assert answers.cached("q one", old, "m", "k") is None
    assert answers.invalidate_content_changes(current) == 0
    assert [r.id for r in answers.recent(10)] == [second, first]  # still visible for audit
    identical = stamp(4, digest="f" * 64, commit="d" * 40)
    answers.record("q three", current, answer_for(symbols), None, "m", "fresh", "k")
    assert answers.invalidate_content_changes(identical) == 0


def test_question_keys_ignore_case_and_spacing():
    assert question_key("  How  Does\nX? ") == "how does x?"


class FakeEngine:
    def __init__(self):
        self.calls = []
        self.closed = False

    def search(self, query, mode="hybrid", top_k=5):
        self.calls.append((query, mode, top_k))
        return [f"{query}|{mode}|{top_k}"]

    def stored_locations(self):
        return {"k": ("a.py", 1, 2)}

    def close(self):
        self.closed = True


def test_the_retrieval_cache_serves_repeats_and_normalizes_whitespace():
    engine = FakeEngine()
    cache = VersionedRetriever(engine, 7)
    first = cache.search("how  is tax", "hybrid", 8)
    again = cache.search("how is tax", "hybrid", 8)
    assert first == again == ["how  is tax|hybrid|8"] and len(engine.calls) == 1
    assert (cache.hits, cache.misses, cache.version_id, len(cache)) == (1, 1, 7, 1)
    cache.search("how is tax", "keyword", 8)
    cache.search("how is tax", "hybrid", 3)
    assert len(engine.calls) == 3 and cache.last == ["how is tax|hybrid|3"]
    first.append("changed by the caller")
    assert cache.search("how is tax", "hybrid", 8) == ["how  is tax|hybrid|8"]


def test_the_retrieval_cache_evicts_the_oldest_entries_and_delegates():
    engine = FakeEngine()
    cache = VersionedRetriever(engine, 1, max_entries=2)
    for query in ("one", "two", "three"):
        cache.search(query)
    assert len(cache) == 2
    cache.search("one")  # was evicted, so the engine is asked again
    assert len(engine.calls) == 4
    assert cache.stored_locations() == {"k": ("a.py", 1, 2)}
    cache.close()
    assert engine.closed