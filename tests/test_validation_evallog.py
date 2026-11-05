from core.validation.evallog import append_record, format_summary, read_log, summarize

FIRST = {
    "status": "answered",
    "policy": {"level": "sufficient"},
    "counts": {
        "evidence_cited": 2, "evidence_cited_valid": 2, "claims_total": 4,
        "claims_supported": 3, "claims_weak": 1, "claims_unsupported": 0,
        "locations_invalid": 0,
    },
}
SECOND = {
    "status": "answered",
    "policy": {"level": "limited"},
    "counts": {
        "evidence_cited": 2, "evidence_cited_valid": 1, "claims_total": 4,
        "claims_supported": 2, "claims_weak": 0, "claims_unsupported": 2,
        "locations_invalid": 1,
    },
}


def test_records_round_trip_and_malformed_lines_are_skipped(tmp_path):
    path = tmp_path / "logs" / "eval.jsonl"
    assert append_record(path, FIRST) and append_record(path, SECOND)
    with path.open("a", encoding="utf-8") as handle:
        handle.write("this is not json\n[1, 2]\n\n")
    records, bad = read_log(path)
    assert len(records) == 2
    assert bad == 2


def test_summary_counts_citation_correctness_and_unsupported_claims():
    summary = summarize([FIRST, SECOND])
    assert summary["records"] == 2
    assert summary["citation_valid_rate"] == 0.75
    assert summary["claim_supported_rate"] == 0.625
    assert summary["claim_unsupported_rate"] == 0.25
    assert summary["levels"] == {"sufficient": 1, "limited": 1}
    assert summary["totals"]["locations_invalid"] == 1


def test_empty_summary_has_no_rates():
    summary = summarize([])
    assert summary["records"] == 0
    assert summary["citation_valid_rate"] is None
    assert "n/a" in format_summary(summary)


def test_format_summary_says_these_are_automatic_checks():
    text = format_summary(summarize([FIRST, SECOND]), bad_lines=1)
    assert "Citations valid: 3 of 4 (75.0%)" in text
    assert "Claims unsupported: 2 of 8 (25.0%)" in text
    assert "not human-verified" in text


def test_a_missing_log_reads_as_empty(tmp_path):
    assert read_log(tmp_path / "nope.jsonl") == ([], 0)


def test_an_unwritable_path_returns_false(tmp_path):
    assert append_record(tmp_path, {"a": 1}) is False