import pytest
from repo_helpers import SAMPLE_REPO

from core.repo.pipeline import analyze_repository


@pytest.fixture(scope="module")
def analysis():
    result = analyze_repository(SAMPLE_REPO)
    yield result
    result.close()


def test_function_name_returns_correct_file_and_lines(analysis):
    hits = analysis.index.search("validate_email")
    assert len(hits) == 1
    sym = hits[0].symbol
    assert (sym.file, sym.start_line, sym.end_line) == ("utils/validation.py", 7, 9)
    assert sym.location == "utils/validation.py:7-9"


def test_exact_name_match_ranks_first(analysis):
    first = analysis.index.search("ShoppingCart")[0].symbol
    assert (first.qualname, first.start_line, first.end_line) == ("ShoppingCart", 4, 27)


def test_async_method_is_found_with_correct_range(analysis):
    first = analysis.index.search("checkout")[0].symbol
    assert (first.kind, first.file, first.start_line, first.end_line) == (
        "async_method", "shop/cart.py", 22, 27,
    )


def test_search_is_case_insensitive(analysis):
    assert analysis.index.search("VALIDATE_EMAIL")[0].symbol.name == "validate_email"


def test_docstring_search(analysis):
    hits = analysis.index.search("percentage", fields=["docstring"])
    assert [h.symbol.name for h in hits] == ["apply_discount"]
    assert hits[0].matched == ("docstring",)


def test_restricting_fields_changes_results(analysis):
    assert analysis.index.search("percentage", fields=["name"]) == []


def test_all_words_must_match(analysis):
    hits = analysis.index.search("discount percent")
    assert [h.symbol.name for h in hits] == ["apply_discount"]


def test_code_search_finds_usage(analysis):
    names = {h.symbol.qualname for h in analysis.index.search("isinstance", fields=["code"])}
    assert names == {"validate_age"}


def test_percent_sign_is_a_literal_not_a_wildcard(analysis):
    assert analysis.index.search("%") == []


def test_no_match_returns_empty_list(analysis):
    assert analysis.index.search("zzz_not_in_this_repo") == []


def test_limit_is_respected(analysis):
    assert len(analysis.index.search("def", limit=2)) == 2


def test_invalid_arguments_raise(analysis):
    with pytest.raises(ValueError):
        analysis.index.search("   ")
    with pytest.raises(ValueError):
        analysis.index.search("x", fields=["bogus"])
    with pytest.raises(ValueError):
        analysis.index.search("x", limit=0)


def test_every_hit_location_matches_the_file_text(analysis):
    for hit in analysis.index.search("return"):
        sym = hit.symbol
        lines = (SAMPLE_REPO / sym.file).read_text(encoding="utf-8").split("\n")
        assert "\n".join(lines[sym.start_line - 1 : sym.end_line]) == sym.code


def test_counts_by_kind(analysis):
    assert analysis.index.count() == 13
    assert analysis.index.counts_by_kind() == {
        "class": 1, "method": 4, "async_method": 1, "function": 6, "async_function": 1,
    }