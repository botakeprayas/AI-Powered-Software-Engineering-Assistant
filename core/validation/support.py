"""Rule-based claim-support check and the final verdict table.

The rules are cheap and transparent. They can catch a claim that names code which is not
in the cited excerpt. They cannot tell whether a claim's logic is right.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from core.chat.models import Evidence
from core.validation.citations import FILE_EXTENSIONS, LOCATION_PATTERN
from core.validation.config import ValidationConfig
from core.validation.models import (
    JUDGE_NOT_SUPPORTED,
    JUDGE_PARTIAL,
    JUDGE_SUPPORTED,
    KIND_FACT,
    RULE_FAIL_ANCHOR,
    RULE_FAIL_WORDS,
    RULE_NOTHING,
    RULE_PASS,
    SUPPORTED,
    UNCHECKED,
    UNSUPPORTED,
    WEAK,
)

BACKTICK = re.compile(r"`([^`\n]{1,80})`")
QUOTED = re.compile(r'"([^"\n]{2,40})"')
IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")
NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?!\w)")
WORD = re.compile(r"[A-Za-z]+")
CAMEL_SPLIT = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
HAS_VERDICT = (JUDGE_SUPPORTED, JUDGE_PARTIAL, JUDGE_NOT_SUPPORTED)
STOPWORDS = frozenset(
    "this that with from which when then than into their there these those where while will "
    "would could should also each every some such only other does done have been being code "
    "method function class returns return value values used uses using calls called calling "
    "takes given after before about over under between through because itself means like "
    "just very more most many much both same what your they them here make makes made".split()
)


@dataclass(frozen=True)
class RuleResult:
    verdict: str
    score: float | None
    anchors: tuple[str, ...]
    missing: tuple[str, ...]


def seen_text(evidence: Evidence) -> str:
    """What the model was shown for this excerpt: the header facts plus the (maybe cut) code."""
    sym = evidence.symbol
    return f"{sym.kind} {sym.qualname} {sym.file}\n{evidence.code}"


def stem(word: str) -> str:
    """A crude stemmer: 'charges' and 'charge' both become 'charg'."""
    if word.endswith("ies") and len(word) > 4:
        word = word[:-3] + "y"
    else:
        for suffix in ("ing", "ed", "es", "s"):
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                word = word[: -len(suffix)]
                break
    return word.rstrip("e")


def _code_like(token: str) -> bool:
    """Does this token look like a code name (snake_case, camelCase, dotted.name)?"""
    if len(token) < 2 or token.rsplit(".", 1)[-1].lower() in FILE_EXTENSIONS:
        return False
    if "." in token:
        return all(len(part) >= 2 for part in token.split("."))
    return "_" in token or re.search(r"[a-z][A-Z]", token) is not None


def extract_anchors(text: str) -> list[str]:
    """Things a claim names that must appear in the cited code: identifiers, numbers, strings."""
    cleaned = LOCATION_PATTERN.sub(" ", text)
    found: list[str] = []
    for match in BACKTICK.finditer(cleaned):
        item = match.group(1).strip().replace("()", "")
        if item:
            found.append(item)
    outside = BACKTICK.sub(" ", cleaned)
    found += [m.group(1) for m in QUOTED.finditer(outside)]
    found += [m.group(0) for m in IDENT.finditer(outside) if _code_like(m.group(0))]
    found += [m.group(0) for m in NUMBER.finditer(outside) if len(m.group(0)) >= 2]
    return list(dict.fromkeys(found))


def content_words(text: str) -> list[str]:
    plain = BACKTICK.sub(" ", LOCATION_PATTERN.sub(" ", text))
    words = []
    for raw in WORD.findall(plain):
        word = raw.lower()
        if len(word) >= 4 and word not in STOPWORDS:
            words.append(stem(word))
    return list(dict.fromkeys(words))


def word_set(text: str) -> set[str]:
    spaced = CAMEL_SPLIT.sub(" ", text.replace("_", " "))
    return {stem(w.lower()) for w in WORD.findall(spaced) if len(w) >= 3}


def rule_check(text: str, kind: str, seen: str, config: ValidationConfig) -> RuleResult:
    """Do the code items a claim names (or, failing that, its words) appear in the cited code?"""
    anchors = extract_anchors(text)
    if anchors:
        haystack = " ".join(seen.split())
        missing = [a for a in anchors if " ".join(a.split()) not in haystack]
        score = 1 - len(missing) / len(anchors)
        if kind == KIND_FACT:
            need = config.fact_anchor_coverage
        else:
            need = config.inference_anchor_coverage
        verdict = RULE_PASS if round(score, 6) >= need else RULE_FAIL_ANCHOR
        return RuleResult(verdict, score, tuple(anchors), tuple(missing))
    words = content_words(text)
    if not words:
        return RuleResult(RULE_NOTHING, None, (), ())
    known = word_set(seen)
    score = sum(w in known for w in words) / len(words)
    verdict = RULE_PASS if round(score, 6) >= config.word_coverage else RULE_FAIL_WORDS
    return RuleResult(verdict, score, (), ())


def decide(citation_ok: bool, rule: str, judge: str) -> tuple[str, str]:
    """Final verdict and its basis. The full table is in the README."""
    if not citation_ok:
        return UNSUPPORTED, "citation check"
    basis = "judge + rules" if judge in HAS_VERDICT else "rules only"
    if judge == JUDGE_NOT_SUPPORTED:
        return UNSUPPORTED, basis
    if rule == RULE_FAIL_ANCHOR:
        return (WEAK if judge == JUDGE_SUPPORTED else UNSUPPORTED), basis
    if judge == JUDGE_PARTIAL:
        return WEAK, basis
    if judge == JUDGE_SUPPORTED:
        return SUPPORTED, basis
    if rule == RULE_PASS:
        return SUPPORTED, basis
    if rule == RULE_FAIL_WORDS:
        return WEAK, basis
    return UNCHECKED, basis