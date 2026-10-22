import pytest

DESCRIPTION_QUERIES = [
    ("function that validates user input", "sanitize_submission"),
    ("protect a login credential before saving it to the database", "salted_digest"),
    ("calculate how much a customer has to pay", "compute_invoice_total"),
    ("try a flaky network call again when it fails", "retry_request"),
    ("load settings from disk", "read_json_config"),
]


@pytest.mark.parametrize("mode", ["semantic", "hybrid"])
@pytest.mark.parametrize(("query", "expected"), DESCRIPTION_QUERIES)
def test_description_query_finds_function_without_its_name(real_engine, query, expected, mode):
    # Guard: the query must not simply contain the function's name words.
    assert not any(part in query.lower() for part in expected.split("_"))
    results = real_engine.search(query, mode=mode, top_k=3)
    names = [r.symbol.name for r in results]
    assert expected in names, f"{mode}: expected {expected} in top 3, got {names}"


@pytest.mark.parametrize("identifier", ["compute_invoice_total", "retry_request"])
def test_exact_identifier_still_works_in_hybrid(real_engine, identifier):
    names = [r.symbol.name for r in real_engine.search(identifier, mode="hybrid", top_k=3)]
    assert identifier in names


def test_results_point_at_correct_lines(real_engine, real_repo):
    for query, _ in DESCRIPTION_QUERIES:
        for result in real_engine.search(query, mode="hybrid", top_k=3):
            sym = result.symbol
            lines = (real_repo / sym.file).read_text(encoding="utf-8").split("\n")
            assert "\n".join(lines[sym.start_line - 1 : sym.end_line]) == sym.code