import pytest
from memory_e2e_helpers import (
    ALL_FILES,
    RATES,
    assert_matches_scratch,
    canonical,
    indexed,
    parseable,
    scratch,
)
from memory_helpers import copy_sample, edit, write

from core.memory import indexer as indexer_module
from core.memory.indexer import (
    IncrementalIndexer,
    load_graph,
    load_issues,
    load_statuses,
    load_symbols,
)
from core.memory.models import SOURCE_CONTENT
from core.memory.store import MemoryStore

NEW_FILE = "shop/inventory.py"
STOCK = (
    '"""Stock levels."""\n\n\ndef in_stock(item, stock):\n'
    '    """True when the item has units left."""\n    return stock.get(item, 0) > 0\n'
)
SCREEN = (
    '"""Screen."""\n\n\nclass Screen:\n    """A screen."""\n\n'
    '    def show(self, value):\n        """Show a value."""\n        return value\n'
)
TAX_EDIT = ("(1 + TAX_RATE), 2)", "(1 + TAX_RATE), 3)")


@pytest.fixture
def spy(monkeypatch):
    """Records which files the indexer extracts and parses."""
    calls = {"extract": [], "parse": []}
    real_extract = indexer_module.extract_file
    real_parse = indexer_module.parse_file

    def extract(root, repo_file):
        calls["extract"].append(repo_file.path)
        return real_extract(root, repo_file)

    def parse(root, rel):
        calls["parse"].append(rel)
        return real_parse(root, rel)

    monkeypatch.setattr(indexer_module, "extract_file", extract)
    monkeypatch.setattr(indexer_module, "parse_file", parse)
    return calls


def forget(calls):
    calls["extract"].clear()
    calls["parse"].clear()


def test_the_first_index_equals_a_from_scratch_build(tmp_path, spy):
    box = indexed(tmp_path)
    update = box.first
    assert update.first_index and update.created_version and update.graph_mode == "full"
    assert sorted(spy["extract"]) == ALL_FILES
    # an empty file (such as an empty __init__.py) is recorded but never parsed, as in Phase 4
    assert sorted(spy["parse"]) == parseable(box.root, ALL_FILES)
    assert update.symbols_total == 15 and len(update.graph_files_recomputed) == 12
    assert update.version.commit_hash is None and update.version.label.startswith("files:")
    assert (update.version.file_count, update.version.symbol_count) == (12, 15)
    assert_matches_scratch(box.store, box.root)


def test_an_unchanged_repository_is_not_reprocessed(tmp_path, spy):
    box = indexed(tmp_path)
    forget(spy)
    update = box.indexer.update()
    assert not update.created_version and update.version.id == box.first.version.id
    assert spy["extract"] == [] and spy["parse"] == []
    assert update.graph_mode == "none" and update.files_reused == 12
    assert not update.changes.has_changes


def test_a_body_edit_reprocesses_only_that_file(tmp_path, fake_embedder, spy):
    box = indexed(tmp_path, fake_embedder)
    assert box.first.semantic.embedded == 15
    forget(spy)
    before = fake_embedder.texts_embedded
    edit(box.root, RATES, *TAX_EDIT)
    update = box.indexer.update(fake_embedder, semantic_db=box.db)
    assert update.changes.modified == [RATES] and not update.changes.added
    assert spy["extract"] == [RATES] and spy["parse"] == [RATES]
    assert update.symbols_extracted == 2 and update.files_reused == 11
    assert update.semantic.embedded == 1 and update.semantic.reused == 14
    assert fake_embedder.texts_embedded - before == 1
    assert update.graph_mode == "partial"
    assert update.graph_files_recomputed == [RATES]
    assert (update.graph_files_parsed, update.graph_files_reparsed) == (1, 0)
    assert_matches_scratch(box.store, box.root)


def test_inserting_a_line_keeps_line_ranges_correct_without_reembedding_neighbours(
    tmp_path, fake_embedder
):
    box = indexed(tmp_path, fake_embedder)
    doc = '    """Add sales tax to an amount."""\n'
    edit(box.root, RATES, doc, doc + "    rate = TAX_RATE\n")
    update = box.indexer.update(fake_embedder, semantic_db=box.db, keep_engine=True)
    try:
        locations = update.engine.stored_locations()
    finally:
        update.engine.close()
    assert update.graph_mode == "partial"
    assert update.semantic.embedded == 1  # only apply_tax; apply_discount just moved
    assert locations["pricing/rates.py::apply_discount"] == (RATES, 12, 15)
    _, symbols = scratch(box.root)
    for symbol in symbols:
        assert locations[symbol.key] == (symbol.file, symbol.start_line, symbol.end_line)
    assert_matches_scratch(box.store, box.root)


def test_a_body_edit_that_changes_a_call_target_updates_only_that_files_edges(tmp_path):
    box = indexed(tmp_path)
    edit(
        box.root, "pricing/invoice.py",
        "return apply_tax(amount) - amount", "return apply_discount(amount, 0) - amount",
    )
    update = box.indexer.update()
    assert update.graph_mode == "partial"
    assert update.graph_files_recomputed == ["pricing/invoice.py"]
    callers = load_graph(box.store).callers_of("pricing/rates.py::apply_tax")
    labels = {rel.source_label for rel in callers}
    assert "tax_only" not in labels and "invoice_total" in labels
    assert_matches_scratch(box.store, box.root)


def test_adding_a_file_extracts_only_it_but_the_graph_is_resolved_again(
    tmp_path, fake_embedder, spy
):
    box = indexed(tmp_path, fake_embedder)
    old_parsed = len(parseable(box.root, ALL_FILES))
    forget(spy)
    write(box.root, NEW_FILE, STOCK)
    update = box.indexer.update(fake_embedder, semantic_db=box.db)
    assert update.changes.added == [NEW_FILE]
    assert spy["extract"] == [NEW_FILE]
    assert update.symbols_extracted == 1 and update.symbols_total == 16
    assert update.semantic.embedded == 1 and update.semantic.reused == 15
    assert update.graph_mode == "full" and "added, deleted or renamed" in update.graph_reason
    assert sorted(spy["parse"]) == parseable(box.root, [*ALL_FILES, NEW_FILE])
    graph_counts = (update.graph_files_parsed, update.graph_files_reparsed)
    assert graph_counts == (old_parsed + 1, old_parsed)
    assert_matches_scratch(box.store, box.root)


def test_a_new_method_in_a_new_file_changes_an_unchanged_files_uncertain_call(tmp_path, spy):
    box = indexed(tmp_path)

    def receipt_call(graph):
        (found,) = [
            rel for rel in graph.relations("calls", "uncertain") if rel.file == "shop/receipt.py"
        ]
        return found

    before = receipt_call(load_graph(box.store))
    assert len(before.candidates) == 1
    forget(spy)
    write(box.root, "shop/screen.py", SCREEN)
    update = box.indexer.update()
    graph = load_graph(box.store)
    after = receipt_call(graph)
    assert {graph.label(key) for key in after.candidates} == {"ConsoleDisplay.show", "Screen.show"}
    assert after.reason != before.reason
    assert "shop/receipt.py" in update.graph_files_recomputed
    assert "shop/receipt.py" not in spy["extract"] and "shop/receipt.py" in spy["parse"]
    assert_matches_scratch(box.store, box.root)


@pytest.mark.parametrize("path", ["tools/formatting.py", "shop/checkout.py"])
def test_deleting_a_file_removes_its_data_and_embeds_nothing(tmp_path, fake_embedder, spy, path):
    box = indexed(tmp_path, fake_embedder)
    forget(spy)
    before = fake_embedder.texts_embedded
    (box.root / path).unlink()
    update = box.indexer.update(fake_embedder, semantic_db=box.db)
    assert update.changes.deleted == [path] and spy["extract"] == []
    assert update.semantic.embedded == 0 and update.semantic.removed == 1
    assert fake_embedder.texts_embedded == before
    assert update.symbols_total == 14 and update.graph_mode == "full"
    assert path not in box.store.manifest()
    remaining = box.store.conn.execute("SELECT 1 FROM symbols WHERE file = ?", (path,))
    assert remaining.fetchone() is None
    assert not load_graph(box.store).has_node(path)
    assert_matches_scratch(box.store, box.root)


def test_renaming_a_file_reembeds_only_its_symbols(tmp_path, fake_embedder, spy):
    box = indexed(tmp_path, fake_embedder)
    forget(spy)
    (box.root / "shop" / "receipt.py").rename(box.root / "shop" / "receipts.py")
    update = box.indexer.update(fake_embedder, semantic_db=box.db)
    (rename,) = update.changes.renamed
    assert (rename.old_path, rename.new_path) == ("shop/receipt.py", "shop/receipts.py")
    assert rename.source == SOURCE_CONTENT
    assert update.files_extracted == ["shop/receipts.py"] == spy["extract"]
    # the embedding text contains the file path, so the one symbol of this file is redone
    assert update.semantic.embedded == 1 and update.semantic.removed == 1
    assert update.semantic.reused == 14
    assert "shop/receipt.py" not in box.store.manifest()
    assert update.graph_mode == "full"
    assert_matches_scratch(box.store, box.root)


def test_a_syntax_error_is_recorded_and_the_rest_keeps_working(tmp_path):
    box = indexed(tmp_path)
    path = box.root / "tools" / "formatting.py"
    original = path.read_text(encoding="utf-8")
    path.write_text("def shout(:\n    pass\n", encoding="utf-8")
    update = box.indexer.update()
    assert load_statuses(box.store)["tools/formatting.py"] == "syntax_error"
    issues = load_issues(box.store)
    assert any(i.path == "tools/formatting.py" and "syntax error" in i.message for i in issues)
    assert update.symbols_total == 14
    assert_matches_scratch(box.store, box.root)
    path.write_text(original, encoding="utf-8")
    fixed = box.indexer.update()
    assert load_statuses(box.store)["tools/formatting.py"] == "parsed"
    assert fixed.symbols_total == 15
    assert_matches_scratch(box.store, box.root)


def test_a_failed_update_writes_nothing_and_the_change_is_found_again(tmp_path, monkeypatch):
    box = indexed(tmp_path)
    version = box.store.current_version().id
    edit(box.root, RATES, *TAX_EDIT)

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(indexer_module, "compute_file_graphs", boom)
    with pytest.raises(RuntimeError, match="boom"):
        box.indexer.update()
    assert box.store.current_version().id == version
    assert len(box.store.list_versions()) == 1
    monkeypatch.undo()
    update = box.indexer.update()
    assert update.changes.modified == [RATES]
    assert_matches_scratch(box.store, box.root)


def test_stored_data_loads_without_reading_or_parsing_anything(tmp_path, monkeypatch):
    root = copy_sample(tmp_path / "repo")
    path = tmp_path / "memory.db"
    store = MemoryStore(path)
    IncrementalIndexer(root, store).update()
    store.close()

    def refuse(*args, **kwargs):
        raise AssertionError("nothing should be read or parsed")

    monkeypatch.setattr(indexer_module, "extract_file", refuse)
    monkeypatch.setattr(indexer_module, "parse_file", refuse)
    reopened = MemoryStore(path)
    update = IncrementalIndexer(root, reopened).update()
    assert not update.created_version and update.graph_mode == "none"
    assert_matches_scratch(reopened, root)


def test_a_partial_update_repairs_inconsistent_stored_graph_data(tmp_path):
    box = indexed(tmp_path)
    box.store.conn.execute("UPDATE files SET nodes_json = NULL WHERE path = 'shop/display.py'")
    box.store.conn.commit()
    edit(box.root, RATES, *TAX_EDIT)
    update = box.indexer.update()
    assert update.graph_mode == "full"
    assert any("inconsistent" in note or "missing" in note for note in update.notes)
    assert_matches_scratch(box.store, box.root)


def test_symbols_keep_the_scanner_order_when_folder_and_file_order_differ(tmp_path):
    root = tmp_path / "repo"
    write(root, "z.py", "def zed():\n    return 1\n")
    write(root, "a/x.py", "def ex():\n    return 1\n")
    write(root, "a/b/y.py", "def why():\n    return 1\n")
    store = MemoryStore()
    IncrementalIndexer(root, store).update()
    keys = [symbol.key for symbol in load_symbols(store)]
    assert keys == ["z.py::zed", "a/x.py::ex", "a/b/y.py::why"]
    assert_matches_scratch(store, root)


def test_the_incremental_graph_equals_a_scratch_build_after_a_whole_series_of_edits(tmp_path):
    box = indexed(tmp_path)
    edit(box.root, RATES, *TAX_EDIT)
    box.indexer.update()
    write(box.root, NEW_FILE, STOCK)
    box.indexer.update()
    (box.root / "tools" / "formatting.py").unlink()
    box.indexer.update()
    (box.root / "shop" / "receipt.py").rename(box.root / "shop" / "receipts.py")
    box.indexer.update()
    edit(box.root, "pricing/invoice.py", "def tax_only(amount):", "def tax_part(amount):")
    box.indexer.update()
    assert_matches_scratch(box.store, box.root)
    graph, _ = scratch(box.root)
    assert canonical(load_graph(box.store)) == canonical(graph)
    assert len(box.store.list_versions()) == 6


def test_the_update_report_says_what_was_reprocessed_and_why(tmp_path):
    box = indexed(tmp_path)
    edit(box.root, RATES, *TAX_EDIT)
    text = box.indexer.update().describe()
    assert "Files: 0 added, 1 modified, 0 deleted, 0 renamed, 11 unchanged" in text
    assert "re-extracted 2 from 1 file(s)" in text
    assert "Graph: partial - only function bodies changed" in text
    assert "recomputed 1 file(s); parsed 1 for the graph (0 unchanged file(s) re-parsed)" in text