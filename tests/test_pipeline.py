from repo_helpers import SAMPLE_REPO, build_hazard_repo

from core.repo.pipeline import analyze_repository
from core.repo.source import ScannedSymbolSource


def test_sample_repo_end_to_end():
    analysis = analyze_repository(SAMPLE_REPO)
    try:
        assert len(analysis.symbols) == 13
        assert {i.path for i in analysis.issues} == {"broken_syntax.py", "empty_module.py"}
        assert analysis.file_status["broken_syntax.py"] == "syntax_error"
        assert analysis.file_status["empty_module.py"] == "empty"
        assert analysis.file_status["notes.txt"] == "not_parsed"
        tree = analysis.tree_text()
        assert "cart.py  [6 symbols]" in tree
        assert "broken_syntax.py  [syntax error]" in tree
    finally:
        analysis.close()


def test_hazard_repo_never_crashes_and_hides_junk(tmp_path):
    analysis = analyze_repository(build_hazard_repo(tmp_path / "repo"))
    try:
        names = {s.name for s in analysis.symbols}
        assert names.isdisjoint({"hidden", "hidden2", "built", "packaged", "gen", "auto"})
        paths = {f.path for f in analysis.scan.files}
        assert ".env" not in paths
        assert "id_rsa" not in analysis.tree_text()
        assert {i.path for i in analysis.issues} == {
            "broken_syntax.py", "empty_module.py", "bad_encoding.py", "binary_like.py",
        }
        assert analysis.file_status["bad_encoding.py"] == "encoding_error"
        assert len(analysis.symbols) == 13  # the good files still work
    finally:
        analysis.close()


def test_phase2_bridge_returns_the_same_symbols():
    source = ScannedSymbolSource(SAMPLE_REPO)
    symbols = source.load()
    assert len(symbols) == 13
    assert len(source.errors) == 2
    assert all(hasattr(s, "key") and hasattr(s, "embedding_text") for s in symbols)