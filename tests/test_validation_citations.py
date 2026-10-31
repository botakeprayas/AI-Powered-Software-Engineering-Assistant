import dataclasses

import pytest
from repo_helpers import SAMPLE_REPO
from validation_helpers import copy_sample, evidence_for, load_symbols, make_answer, snapshot_for

from core.validation.citations import (
    check_citations,
    check_evidence,
    find_locations,
    line_count,
)
from core.validation.models import (
    CITE_MISSING,
    CITE_OUT_OF_RANGE,
    CITE_STALE,
    CITE_VALID,
    LOC_INVALID,
    LOC_OK,
    LOC_UNGROUNDED,
)
from core.validation.snapshot import IndexSnapshot


@pytest.fixture(scope="module")
def symbols():
    return load_symbols()


@pytest.fixture
def snapshot(symbols):
    return snapshot_for(SAMPLE_REPO, symbols)


def test_valid_citation_passes(symbols, snapshot):
    check = check_evidence(evidence_for(symbols["ShoppingCart.total"]), snapshot)
    assert check.status == CITE_VALID
    assert check.ok


def test_fabricated_evidence_id_is_rejected(symbols, snapshot):
    answer = make_answer(
        facts=[("total sums the items", ["E1", "E9"])],
        evidence=[evidence_for(symbols["ShoppingCart.total"])],
    )
    citations, _, _ = check_citations(answer, snapshot)
    assert {c.evidence_id: c.status for c in citations} == {
        "E1": CITE_VALID,
        "E9": CITE_MISSING,
    }


@pytest.mark.parametrize(
    "written",
    [
        "see shop/cart.py:90-120",  # beyond the end of the file
        "see shop/refunds.py:3-9",  # a file that does not exist
        "see ../outside.py:1-2",  # outside the repository
        "see shop/cart.py:30-20",  # reversed range
        "see shop/cart.py:0-3",  # line zero
    ],
)
def test_fabricated_locations_are_rejected(symbols, snapshot, written):
    answer = make_answer(
        facts=[(f"total is computed {written}", ["E1"])],
        evidence=[evidence_for(symbols["ShoppingCart.total"])],
    )
    _, locations, _ = check_citations(answer, snapshot)
    assert [x.status for x in locations] == [LOC_INVALID]


def test_location_inside_evidence_is_ok_and_outside_is_ungrounded(symbols, snapshot):
    answer = make_answer(
        facts=[("total at shop/cart.py:19-20 and email at utils/validation.py:7-9", ["E1"])],
        evidence=[evidence_for(symbols["ShoppingCart.total"])],
    )
    _, locations, _ = check_citations(answer, snapshot)
    assert [x.status for x in locations] == [LOC_OK, LOC_UNGROUNDED]


def test_unindexed_files_are_never_opened(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "credentials.json").write_text('{"k": 1}\n', encoding="utf-8")
    (tmp_path / "main.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    snap = IndexSnapshot(tmp_path, {"main.py::f": ("main.py", 1, 2)})
    assert snap.lines("config/credentials.json") is None
    assert snap.lines("main.py") is not None
    answer = make_answer(facts=[("secret in config/credentials.json:1-1", ["E1"])])
    _, locations, _ = check_citations(answer, snap)
    assert [x.status for x in locations] == [LOC_INVALID]


def test_edited_file_makes_the_citation_stale(tmp_path):
    root = copy_sample(tmp_path / "repo")
    symbols = load_symbols(root)
    snap = snapshot_for(root, symbols)
    evidence = evidence_for(symbols["ShoppingCart.total"])
    assert check_evidence(evidence, snap).status == CITE_VALID
    cart = root / "shop" / "cart.py"
    cart.write_text("# added\n\n" + cart.read_text(encoding="utf-8"), encoding="utf-8")
    assert check_evidence(evidence, snap).status == CITE_STALE


def test_file_truncated_after_indexing_is_out_of_range(tmp_path):
    root = copy_sample(tmp_path / "repo")
    symbols = load_symbols(root)
    snap = snapshot_for(root, symbols)
    (root / "shop" / "cart.py").write_text("x = 1\n", encoding="utf-8")
    check = check_evidence(evidence_for(symbols["ShoppingCart.total"]), snap)
    assert check.status == CITE_OUT_OF_RANGE


def test_deleted_file_is_missing(tmp_path):
    root = copy_sample(tmp_path / "repo")
    symbols = load_symbols(root)
    snap = snapshot_for(root, symbols)
    (root / "shop" / "cart.py").unlink()
    check = check_evidence(evidence_for(symbols["ShoppingCart.total"]), snap)
    assert check.status == CITE_MISSING


def test_moved_symbol_is_stale(symbols, snapshot):
    moved = dataclasses.replace(symbols["ShoppingCart.total"], start_line=19)
    assert check_evidence(evidence_for(moved), snapshot).status == CITE_STALE


def test_symbol_missing_from_the_index(symbols):
    others = {q: s for q, s in symbols.items() if q != "ShoppingCart.total"}
    snap = snapshot_for(SAMPLE_REPO, others)
    check = check_evidence(evidence_for(symbols["ShoppingCart.total"]), snap)
    assert check.status == CITE_MISSING


def test_find_locations_reads_ranges_single_lines_and_backslashes():
    text = "see shop/cart.py:22-27, utils/validation.py:7 and shop\\cart.py:3"
    assert find_locations(text) == [
        ("shop/cart.py:22-27", "shop/cart.py", 22, 27),
        ("utils/validation.py:7", "utils/validation.py", 7, 7),
        ("shop\\cart.py:3", "shop/cart.py", 3, 3),
    ]


def test_line_count_ignores_the_empty_piece_after_a_trailing_newline():
    assert line_count("a\nb\n".split("\n")) == 2
    assert line_count("a\nb".split("\n")) == 2