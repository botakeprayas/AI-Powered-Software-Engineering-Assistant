"""Constants and plain data types for answer validation (Phase 5)."""

from __future__ import annotations

from dataclasses import dataclass, field

KIND_FACT = "fact"
KIND_INFERENCE = "inference"

# Result of checking one cited excerpt against the CURRENT index and files
CITE_VALID = "valid"
CITE_STALE = "stale"
CITE_MISSING = "missing"
CITE_OUT_OF_RANGE = "out_of_range"

# Result of checking a file:line location the model wrote inside a claim
LOC_OK = "ok"
LOC_UNGROUNDED = "ungrounded"
LOC_INVALID = "invalid"

# Rule-based check of a claim against the code it cites
RULE_PASS = "pass"
RULE_FAIL_ANCHOR = "fail_anchor"
RULE_FAIL_WORDS = "fail_words"
RULE_NOTHING = "nothing_to_check"

# LLM judge verdicts (plus two "no verdict" states)
JUDGE_SUPPORTED = "supported"
JUDGE_PARTIAL = "partial"
JUDGE_NOT_SUPPORTED = "not_supported"
JUDGE_UNPARSED = "unparsed"
JUDGE_SKIPPED = "skipped"

# Final verdict on a claim
SUPPORTED = "supported"
WEAK = "weak"
UNSUPPORTED = "unsupported"
UNCHECKED = "unchecked"

# Evidence levels. They are rule outcomes, NOT probabilities.
LEVEL_SUFFICIENT = "sufficient"
LEVEL_LIMITED = "limited"
LEVEL_INSUFFICIENT = "insufficient"
LEVEL_NOT_APPLICABLE = "not_applicable"

LEVEL_DISCLAIMER = (
    "This level comes from fixed rules in the configuration. "
    "It is not a probability and does not guarantee the answer is correct."
)


@dataclass(frozen=True)
class CitationCheck:
    evidence_id: str
    location: str
    status: str
    detail: str

    @property
    def ok(self) -> bool:
        return self.status == CITE_VALID


@dataclass(frozen=True)
class LocationCheck:
    """A file:line location the model wrote inside a claim (it was told not to)."""

    claim_label: str
    text: str
    file: str
    start: int
    end: int
    status: str
    detail: str


@dataclass
class ClaimCheck:
    label: str  # F1, F2, ... for facts and I1, I2, ... for inferences
    kind: str
    text: str
    evidence_ids: tuple[str, ...]
    citation_ok: bool
    rule_verdict: str = RULE_NOTHING
    rule_score: float | None = None
    anchors: tuple[str, ...] = ()
    missing_anchors: tuple[str, ...] = ()
    judge_verdict: str = JUDGE_SKIPPED
    judge_reason: str = ""
    final: str = UNCHECKED
    basis: str = ""
    reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PolicyResult:
    level: str
    limitations: tuple[str, ...]
    notes: tuple[str, ...]
    # (rule name, passed, limitation text that applies when it failed)
    checks: tuple[tuple[str, bool, str], ...]


@dataclass
class ValidationReport:
    citations: list[CitationCheck] = field(default_factory=list)
    locations: list[LocationCheck] = field(default_factory=list)
    claims: list[ClaimCheck] = field(default_factory=list)
    policy: PolicyResult | None = None
    judge_model: str = ""
    judge_error: str = ""
    cited_by: dict[str, list[str]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        """Numbers used by the evaluation log and the summary script."""
        return {
            "evidence_cited": len(self.citations),
            "evidence_cited_valid": sum(c.ok for c in self.citations),
            "locations_written": len(self.locations),
            "locations_invalid": sum(x.status == LOC_INVALID for x in self.locations),
            "locations_ungrounded": sum(x.status == LOC_UNGROUNDED for x in self.locations),
            "claims_total": len(self.claims),
            "claims_supported": sum(c.final == SUPPORTED for c in self.claims),
            "claims_weak": sum(c.final == WEAK for c in self.claims),
            "claims_unsupported": sum(c.final == UNSUPPORTED for c in self.claims),
            "claims_unchecked": sum(c.final == UNCHECKED for c in self.claims),
        }