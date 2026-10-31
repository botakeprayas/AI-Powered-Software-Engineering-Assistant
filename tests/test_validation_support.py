import pytest
from validation_helpers import evidence_for, load_symbols

from core.validation.config import ValidationConfig
from core.validation.models import (
    JUDGE_NOT_SUPPORTED,
    JUDGE_PARTIAL,
    JUDGE_SKIPPED,
    JUDGE_SUPPORTED,
    JUDGE_UNPARSED,
    KIND_FACT,
    KIND_INFERENCE,
    RULE_FAIL_ANCHOR,
    RULE_FAIL_WORDS,
    RULE_NOTHING,
    RULE_PASS,
    SUPPORTED,
    UNCHECKED,
    UNSUPPORTED,
    WEAK,
)
from core.validation.support import decide, extract_anchors, rule_check, seen_text

CONFIG = ValidationConfig()


@pytest.fixture(scope="module")
def checkout_seen():
    return seen_text(evidence_for(load_symbols()["ShoppingCart.checkout"]))


def test_anchors_are_code_names_numbers_and_strings_only():
    text = (
        "The `checkout` method awaits payment_gateway.charge and returns ShoppingCart "
        'total 0.08 e.g. in cart.py and "paid".'
    )
    assert set(extract_anchors(text)) == {
        "checkout", "payment_gateway.charge", "ShoppingCart", "0.08", "paid",
    }


def test_written_locations_are_not_anchors():
    assert extract_anchors("see shop/cart.py:22-27") == []


def test_claim_naming_real_code_passes(checkout_seen):
    text = "`checkout` awaits `payment_gateway.charge`"
    result = rule_check(text, KIND_FACT, checkout_seen, CONFIG)
    assert result.verdict == RULE_PASS
    assert result.score == 1.0


def test_fact_naming_missing_code_fails_but_an_inference_may_pass(checkout_seen):
    text = "`checkout` calls `payment_gateway.refund`"
    fact = rule_check(text, KIND_FACT, checkout_seen, CONFIG)
    inference = rule_check(text, KIND_INFERENCE, checkout_seen, CONFIG)
    assert fact.verdict == RULE_FAIL_ANCHOR
    assert fact.missing == ("payment_gateway.refund",)
    assert inference.verdict == RULE_PASS  # 1 of 2 names found meets the 50% inference rule


def test_plain_words_found_in_the_code_pass(checkout_seen):
    text = "Charges the customer and empties the cart."
    assert rule_check(text, KIND_FACT, checkout_seen, CONFIG).verdict == RULE_PASS


def test_plain_words_not_in_the_code_fail(checkout_seen):
    text = "Sends a confirmation email to the warehouse"
    assert rule_check(text, KIND_FACT, checkout_seen, CONFIG).verdict == RULE_FAIL_WORDS


def test_claim_with_nothing_checkable(checkout_seen):
    assert rule_check("It does this.", KIND_FACT, checkout_seen, CONFIG).verdict == RULE_NOTHING


@pytest.mark.parametrize(
    ("citation_ok", "rule", "judge", "final"),
    [
        (False, RULE_PASS, JUDGE_SUPPORTED, UNSUPPORTED),
        (True, RULE_PASS, JUDGE_NOT_SUPPORTED, UNSUPPORTED),
        (True, RULE_FAIL_ANCHOR, JUDGE_SUPPORTED, WEAK),
        (True, RULE_FAIL_ANCHOR, JUDGE_SKIPPED, UNSUPPORTED),
        (True, RULE_FAIL_ANCHOR, JUDGE_PARTIAL, UNSUPPORTED),
        (True, RULE_PASS, JUDGE_PARTIAL, WEAK),
        (True, RULE_PASS, JUDGE_SUPPORTED, SUPPORTED),
        (True, RULE_FAIL_WORDS, JUDGE_SUPPORTED, SUPPORTED),
        (True, RULE_PASS, JUDGE_SKIPPED, SUPPORTED),
        (True, RULE_FAIL_WORDS, JUDGE_SKIPPED, WEAK),
        (True, RULE_NOTHING, JUDGE_SKIPPED, UNCHECKED),
        (True, RULE_NOTHING, JUDGE_UNPARSED, UNCHECKED),
    ],
)
def test_final_verdict_table(citation_ok, rule, judge, final):
    assert decide(citation_ok, rule, judge)[0] == final


def test_basis_says_whether_a_judge_took_part():
    assert decide(True, RULE_PASS, JUDGE_SUPPORTED)[1] == "judge + rules"
    assert decide(True, RULE_PASS, JUDGE_SKIPPED)[1] == "rules only"
    assert decide(False, RULE_PASS, JUDGE_SUPPORTED)[1] == "citation check"