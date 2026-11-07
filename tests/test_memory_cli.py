import importlib.util
from pathlib import Path

import pytest
from memory_e2e_helpers import RATES
from memory_helpers import copy_sample, edit
from memory_service_helpers import QUESTION, HashEmbedder

ROOT = Path(__file__).resolve().parents[1]
TAX_EDIT = ("(1 + TAX_RATE), 2)", "(1 + TAX_RATE), 3)")


def load_cli():
    spec = importlib.util.spec_from_file_location("memory_cli", ROOT / "scripts" / "memory.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CLI = load_cli()


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.setenv("INDEX_DIR", str(tmp_path / "idx"))
    monkeypatch.setenv("EVAL_LOG_PATH", str(tmp_path / "log.jsonl"))
    monkeypatch.setenv("QA_MIN_COSINE", "-1")
    monkeypatch.setattr(CLI, "make_embedder", lambda: HashEmbedder())
    return copy_sample(tmp_path / "repo")


def test_status_of_a_repository_that_was_never_indexed(repo, capsys):
    assert CLI.main(["status", "--repo", str(repo)]) == 3
    assert "not been indexed" in capsys.readouterr().out


def test_update_then_status_then_an_edit(repo, capsys):
    assert CLI.main(["update", "--repo", str(repo), "--no-embeddings"]) == 0
    out = capsys.readouterr().out
    assert "Indexed version files:" in out and "Graph: full" in out
    assert CLI.main(["status", "--repo", str(repo)]) == 0
    assert "matches the repository" in capsys.readouterr().out
    edit(repo, RATES, *TAX_EDIT)
    assert CLI.main(["status", "--repo", str(repo)]) == 3
    out = capsys.readouterr().out
    assert "STALE" in out and "modified: pricing/rates.py" in out


def test_answers_are_tagged_listed_and_marked_outdated_after_a_change(repo, capsys):
    args = ["ask", "--repo", str(repo), "--question", QUESTION, "--mock", "--no-judge"]
    assert CLI.main(args) == 0
    out = capsys.readouterr().out
    assert "Based on version files:" in out and "History id: #1" in out
    assert "Citations:" in out
    edit(repo, RATES, *TAX_EDIT)
    assert CLI.main(args) == 0
    capsys.readouterr()
    assert CLI.main(["history", "--repo", str(repo)]) == 0
    table = capsys.readouterr().out
    assert "OUTDATED" in table and "[valid]" in table
    assert CLI.main(["history", "--repo", str(repo), "--show", "1"]) == 0
    shown = capsys.readouterr().out
    assert "OUTDATED" in shown and "was based on" in shown and "Based on version files:" in shown


def test_the_block_policy_refuses_to_answer_from_a_stale_index(repo, capsys):
    args = ["ask", "--repo", str(repo), "--question", QUESTION, "--mock", "--no-judge"]
    assert CLI.main(args) == 0
    edit(repo, RATES, *TAX_EDIT)
    capsys.readouterr()
    assert CLI.main([*args, "--policy", "block"]) == 3
    out = capsys.readouterr().out
    assert "BLOCKED:" in out and "stale" in out.lower()


def test_an_empty_history_and_an_unknown_answer_are_reported(repo, capsys):
    assert CLI.main(["history", "--repo", str(repo)]) == 0
    assert "No stored answers" in capsys.readouterr().out
    assert CLI.main(["history", "--repo", str(repo), "--show", "9"]) == 1
    assert "no stored answer #9" in capsys.readouterr().err


def test_a_missing_folder_gives_a_clear_error(capsys):
    assert CLI.main(["status", "--repo", "no_such_folder_anywhere"]) == 1
    assert "ERROR:" in capsys.readouterr().err