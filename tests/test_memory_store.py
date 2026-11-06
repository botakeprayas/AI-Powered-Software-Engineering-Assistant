from datetime import datetime

import pytest

from core.memory.hashing import file_digest
from core.memory.models import FileState
from core.memory.store import MemoryStore, memory_db_path


def state(path: str, text: str) -> FileState:
    data = text.encode("utf-8")
    return FileState(path, file_digest(data), len(data), True)


def test_the_schema_has_the_expected_tables():
    store = MemoryStore()
    rows = store.conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    assert {row["name"] for row in rows} >= {"meta", "versions", "files", "symbols"}


def test_versions_are_recorded_and_the_current_one_is_tracked():
    store = MemoryStore()
    assert store.current_version() is None
    first = store.add_version("a" * 40, False, "1" * 64, "main", file_count=3)
    second = store.add_version(
        "b" * 40, True, "2" * 64, None, file_count=4, parent_id=first, note="edited"
    )
    assert (first, second) == (1, 2)
    store.set_current_version(second)
    current = store.current_version()
    assert current is not None and current.id == 2 and current.parent_id == 1
    assert current.dirty and current.note == "edited" and current.branch is None
    assert current.label == "bbbbbbbbbbbb+dirty"
    assert store.get_version(1).label == "aaaaaaaaaaaa"
    assert [v.id for v in store.list_versions()] == [2, 1]
    assert datetime.fromisoformat(current.indexed_at).tzinfo is not None


def test_a_version_without_a_commit_is_named_after_its_file_digest():
    store = MemoryStore()
    version_id = store.add_version(None, False, "f" * 64)
    assert store.get_version(version_id).label == "files:ffffffffffff"
    assert store.get_version(999) is None


def test_version_counts_can_be_updated_later():
    store = MemoryStore()
    version_id = store.add_version("a" * 40, False, "1" * 64)
    store.update_version(version_id, file_count=7, symbol_count=40, note="done")
    version = store.get_version(version_id)
    assert (version.file_count, version.symbol_count, version.note) == (7, 40, "done")


def test_the_manifest_round_trips_and_files_can_be_deleted_with_their_symbols():
    store = MemoryStore()
    states = [state("a.py", "x = 1\n"), state("b.py", "y = 1\n")]
    for item in states:
        store.put_file_state(item, 1)
    assert store.manifest() == {item.path: item for item in states}
    store.conn.execute(
        "INSERT INTO symbols VALUES ('a.py::f', 'a.py', 'function', 'f', 'f', 1, 2, '', 'code')"
    )
    store.conn.commit()
    store.delete_files(["a.py"])
    store.delete_files([])
    assert list(store.manifest()) == ["b.py"]
    assert store.conn.execute("SELECT COUNT(*) FROM symbols").fetchone()[0] == 0


def test_a_changed_hash_discards_derived_data_but_the_same_hash_keeps_it():
    store = MemoryStore()
    original = state("a.py", "x = 1\n")
    store.put_file_state(original, 1)
    store.conn.execute("UPDATE files SET status = 'parsed', edges_json = '[1]'")
    store.conn.commit()
    query = "SELECT status, edges_json, version_changed FROM files"
    store.put_file_state(original, 2)
    row = store.conn.execute(query).fetchone()
    assert (row["status"], row["edges_json"], row["version_changed"]) == ("parsed", "[1]", 1)
    store.put_file_state(state("a.py", "x = 2\n"), 3)
    row = store.conn.execute(query).fetchone()
    assert (row["status"], row["edges_json"], row["version_changed"]) == (None, None, 3)


def test_a_failed_transaction_rolls_everything_back():
    store = MemoryStore()
    with pytest.raises(RuntimeError):
        with store.transaction():
            store.put_file_state(state("a.py", "x = 1\n"), 1)
            store.add_version("a" * 40, False, "1" * 64)
            raise RuntimeError("boom")
    assert store.manifest() == {} and store.list_versions() == []


def test_nested_transactions_commit_only_once_at_the_end():
    store = MemoryStore()
    with pytest.raises(RuntimeError):
        with store.transaction():
            with store.transaction():
                store.put_file_state(state("a.py", "x = 1\n"), 1)
            raise RuntimeError("after the inner block")
    assert store.manifest() == {}  # the inner block did not commit early


def test_everything_survives_closing_and_reopening(tmp_path):
    path = tmp_path / "indexes" / "mem.db"
    store = MemoryStore(path)
    version_id = store.add_version(None, False, "e" * 64)
    store.set_current_version(version_id)
    store.put_file_state(state("a.py", "x = 1\n"), version_id)
    store.close()
    reopened = MemoryStore(path)
    assert reopened.current_version().id == version_id
    assert list(reopened.manifest()) == ["a.py"]


def test_each_repository_gets_its_own_database_file(tmp_path, monkeypatch):
    monkeypatch.setenv("INDEX_DIR", str(tmp_path / "indexes"))
    one = memory_db_path(tmp_path / "one")
    two = memory_db_path(tmp_path / "two")
    assert one != two and one == memory_db_path(tmp_path / "one")
    assert one.parent == tmp_path / "indexes"
    assert one.name.startswith("memory-") and one.suffix == ".db"