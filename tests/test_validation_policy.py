import dataclasses

import pytest
from validation_helpers import evidence_for, load_symbols, make_answer

from core.chat.models import STATUS_INSUFFICIENT
from core.settings import PROJECT_ROOT
from core.validation.config import (
    NUMERIC_SETTINGS,
    ValidationConfig,
    load_validation_config,
)
from core.validation.models import (
    CITE_STALE,
    CITE_VALID,
    KIND_FACT,
    LEVEL_INSUFFICIENT,
    LEVEL_LIMITED,
    LEVEL_NOT_APPLICABLE,
    LEVEL_SUFFICIENT,
    LOC_INVALID,
    CitationCheck,
    ClaimCheck,
    LocationCheck,
)
from core.validation.policy import evaluate_policy

ENV_NAMES = [row[1] for row in NUMERIC_SETTINGS] + [
    "VALIDATION_JUDGE", "EVAL_LOG_ENABLED", "EVAL_LOG_PATH", "EVAL_LOG_CODE",
]


@pytest.fixture(scope="module")
def symbols():
    return load_symbols()


def policy_for(symbols, finals=("supported",), cite_status=CITE_VALID, locations=(),
               ev_kwargs=None, config=None):
    evidence = [evidence_for(symbols["ShoppingCart.total"], **(ev_kwargs or {}))]
    answer = make_answer(facts=[("t", ["E1"])], evidence=evidence)
    citations = [CitationCheck("E1", "shop/cart.py:18-20", cite_status, "detail")]
    claims = [
        ClaimCheck(
            label=f"F{i}", kind=KIND_FACT, text="t", evidence_ids=("E1",),
            citation_ok=cite_status == CITE_VALID, final=final,
        )
        for i, final in enumerate(finals, start=1)
    ]
    return evaluate_policy(answer, citations, list(locations), claims, config or ValidationConfig())


def test_all_rules_passing_is_sufficient(symbols):
    result = policy_for(symbols)
    assert result.level == LEVEL_SUFFICIENT
    assert result.limitations == ()


def test_no_valid_citation_is_insufficient_with_an_explicit_limitation(symbols):
    result = policy_for(symbols, cite_status=CITE_STALE, finals=("unsupported",))
    assert result.level == LEVEL_INSUFFICIENT
    assert any("valid citation" in text for text in result.limitations)


def test_weak_retrieval_scores_add_a_limitation(symbols):
    result = policy_for(symbols, ev_kwargs={"semantic": 0.12, "rank": 5, "score": 2.0})
    assert result.level == LEVEL_LIMITED
    assert any("Retrieval scores" in text for text in result.limitations)


def test_unsupported_fact_is_a_limitation(symbols):
    result = policy_for(symbols, finals=("supported", "unsupported"))
    text = " ".join(result.limitations)
    assert result.level == LEVEL_LIMITED
    assert "not supported by their cited code" in text
    assert "F2" in text


def test_fabricated_location_is_a_limitation(symbols):
    bad = LocationCheck("F1", "shop/cart.py:90-120", "shop/cart.py", 90, 120, LOC_INVALID, "x")
    result = policy_for(symbols, locations=[bad])
    assert result.level == LEVEL_LIMITED
    assert any("shop/cart.py:90-120" in text for text in result.limitations)


def test_an_abstention_has_no_level(symbols):
    answer = make_answer(status=STATUS_INSUFFICIENT)
    result = evaluate_policy(answer, [], [], [], ValidationConfig())
    assert result.level == LEVEL_NOT_APPLICABLE


def test_thresholds_are_configurable(symbols):
    strict = dataclasses.replace(ValidationConfig(), min_valid_citations=2)
    result = policy_for(symbols, config=strict)
    assert result.level == LEVEL_LIMITED
    assert any("policy minimum: 2" in text for text in result.limitations)


@pytest.fixture
def clean_env(monkeypatch):
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def test_config_defaults(clean_env):
    assert load_validation_config(env_file=None) == ValidationConfig()


def test_config_reads_the_environment(clean_env, monkeypatch):
    monkeypatch.setenv("VALIDATION_MIN_VALID_CITATIONS", "3")
    monkeypatch.setenv("VALIDATION_JUDGE", "off")
    monkeypatch.setenv("EVAL_LOG_PATH", "logs/x.jsonl")
    config = load_validation_config(env_file=None)
    assert config.min_valid_citations == 3
    assert config.judge_enabled is False
    assert config.log_path == PROJECT_ROOT / "logs" / "x.jsonl"


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("VALIDATION_MIN_COSINE", "2"),
        ("VALIDATION_MIN_SUPPORTED_FRACTION", "abc"),
        ("VALIDATION_JUDGE", "maybe"),
        ("VALIDATION_MIN_VALID_CITATIONS", "-1"),
    ],
)
def test_config_rejects_bad_values(clean_env, monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        load_validation_config(env_file=None)