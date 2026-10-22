import pytest

from core.semantic.engine import CodeSearchEngine
from core.semantic.symbol_source import AstSymbolSource
from services.embedding_service import EmbeddingError

TOTAL = 10  # symbols in the sample repo (see tests/conftest.py)


def test_first_sync_embeds_everything(engine, fake_embedder):
    stats = engine.sync()
    assert stats.total == TOTAL
    assert stats.embedded == TOTAL
    assert stats.reused == 0
    assert fake_embedder.texts_embedded == TOTAL


def test_second_sync_reuses_every_vector(engine, fake_embedder):
    engine.sync()
    before = fake_embedder.texts_embedded
    stats = engine.sync()
    assert stats.embedded == 0
    assert stats.reused == TOTAL
    assert fake_embedder.texts_embedded == before


def test_changing_one_function_reembeds_only_it(engine, repo_dir):
    engine.sync()
    path = repo_dir / "billing.py"
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace("(1 + tax_rate), 2)", "(1 + tax_rate) + 1, 2)"), encoding="utf-8")
    stats = engine.sync()
    assert stats.embedded == 1
    assert stats.embedded_files == ["billing.py"]
    assert stats.reused == TOTAL - 1


def test_line_shift_updates_locations_without_reembedding(engine, repo_dir):
    engine.sync()
    key = "utils.py::read_json_config"
    old_file, old_start, old_end = engine.stored_locations()[key]
    path = repo_dir / "utils.py"
    path.write_text("# a new comment\n\n" + path.read_text(encoding="utf-8"), encoding="utf-8")
    stats = engine.sync()
    assert stats.embedded == 0
    assert engine.stored_locations()[key] == (old_file, old_start + 2, old_end + 2)


def test_deleted_file_symbols_are_removed(engine, repo_dir):
    engine.sync()
    (repo_dir / "auth.py").unlink()
    stats = engine.sync()
    assert stats.removed == 2
    assert stats.total == TOTAL - 2
    assert len(engine.stored_locations()) == TOTAL - 2


def test_changing_model_reembeds_everything(engine, repo_dir, tmp_path, make_fake_embedder):
    engine.sync()
    other = CodeSearchEngine(
        AstSymbolSource(repo_dir), make_fake_embedder(model_name="fake-model-2"),
        tmp_path / "index.db",
    )
    try:
        assert other.sync().embedded == TOTAL
    finally:
        other.close()


def test_failed_embedding_leaves_cache_intact(
    engine, repo_dir, tmp_path, make_failing_embedder
):
    engine.sync()
    before = engine.stored_locations()
    path = repo_dir / "billing.py"
    path.write_text(path.read_text(encoding="utf-8").replace("amount", "sum"), encoding="utf-8")
    broken = CodeSearchEngine(
        AstSymbolSource(repo_dir), make_failing_embedder(), tmp_path / "index.db"
    )
    try:
        with pytest.raises(EmbeddingError):
            broken.sync()
        assert broken.search("compute_invoice_total", mode="keyword")
    finally:
        broken.close()
    assert engine.stored_locations() == before


def test_locations_match_source(engine, repo_dir):
    engine.sync()
    stored = engine.stored_locations()
    symbols = AstSymbolSource(repo_dir).load()
    assert len(stored) == len(symbols) == TOTAL
    for symbol in symbols:
        lines = (repo_dir / symbol.file).read_text(encoding="utf-8").split("\n")
        assert "\n".join(lines[symbol.start_line - 1 : symbol.end_line]) == symbol.code
        assert stored[symbol.key] == (symbol.file, symbol.start_line, symbol.end_line)


def test_decorated_function_location_includes_decorator(repo_dir):
    symbols = {s.name: s for s in AstSymbolSource(repo_dir).load()}
    sym = symbols["read_json_config"]
    lines = (repo_dir / sym.file).read_text(encoding="utf-8").split("\n")
    assert lines[sym.start_line - 1].startswith("@log_call")