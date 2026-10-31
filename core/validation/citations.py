"""Citation checks against the currently indexed repository version."""

from __future__ import annotations

import re
from collections.abc import Iterable

from core.chat.models import Answer, Claim, Evidence
from core.validation.models import (
    CITE_MISSING,
    CITE_OUT_OF_RANGE,
    CITE_STALE,
    CITE_VALID,
    KIND_FACT,
    KIND_INFERENCE,
    LOC_INVALID,
    LOC_OK,
    LOC_UNGROUNDED,
    CitationCheck,
    LocationCheck,
)
from core.validation.snapshot import IndexSnapshot

FILE_EXTENSIONS = ("py", "pyi", "md", "txt", "toml", "cfg", "ini", "json", "yaml", "yml")
LOCATION_PATTERN = re.compile(
    r"(?P<file>[\w./\\-]+\.(?:py|pyi|md|txt|toml|cfg|ini|json|ya?ml))"
    r":(?P<start>\d{1,9})(?:\s*[-\u2013]\s*(?P<end>\d{1,9}))?"
)


def normalize_path(raw: str) -> str:
    path = raw.replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    return path


def line_count(lines: list[str]) -> int:
    """Real line count (the final empty piece after a trailing newline is not a line)."""
    return len(lines) - 1 if lines and lines[-1] == "" else len(lines)


def labeled_claims(answer: Answer) -> list[tuple[str, str, Claim]]:
    """(label, kind, claim) with labels F1, F2, ... for facts and I1, I2, ... for inferences."""
    items = [(f"F{n}", KIND_FACT, c) for n, c in enumerate(answer.facts, start=1)]
    items += [(f"I{n}", KIND_INFERENCE, c) for n, c in enumerate(answer.inferences, start=1)]
    return items


def find_locations(text: str) -> list[tuple[str, str, int, int]]:
    """(matched text, file, start, end) for every file:line written in `text`."""
    found = []
    for match in LOCATION_PATTERN.finditer(text):
        start = int(match.group("start"))
        end = int(match.group("end") or start)
        found.append((match.group(0), normalize_path(match.group("file")), start, end))
    return found


def check_evidence(evidence: Evidence, snapshot: IndexSnapshot) -> CitationCheck:
    """Is this cited excerpt still real? Checks the index AND the file text on disk."""
    sym = evidence.symbol

    def result(status: str, detail: str) -> CitationCheck:
        return CitationCheck(evidence.id, evidence.location, status, detail)

    lines = snapshot.lines(sym.file)
    if lines is None:
        return result(CITE_MISSING, f"{sym.file} is not indexed or cannot be read")
    total = line_count(lines)
    if sym.start_line < 1 or sym.end_line < sym.start_line:
        return result(CITE_OUT_OF_RANGE, "the line range is not valid")
    if sym.end_line > total:
        detail = f"the range ends at line {sym.end_line} but the file has {total}"
        return result(CITE_OUT_OF_RANGE, detail)
    indexed = snapshot.indexed_location(sym.key)
    if indexed is None:
        return result(CITE_MISSING, "the symbol is no longer in the index")
    if indexed != (sym.file, sym.start_line, sym.end_line):
        where = f"{indexed[0]}:{indexed[1]}-{indexed[2]}"
        return result(CITE_STALE, f"the index now lists this symbol at {where}")
    current = "\n".join(lines[sym.start_line - 1 : sym.end_line])
    if current != sym.code:
        return result(CITE_STALE, "the file text at these lines changed after indexing")
    return result(CITE_VALID, "the location exists and its text matches the index")


def check_location(
    label: str,
    text: str,
    file: str,
    start: int,
    end: int,
    evidence: Iterable[Evidence],
    snapshot: IndexSnapshot,
) -> LocationCheck:
    """Check a file:line the model wrote itself. It must exist and sit inside the evidence."""

    def result(status: str, detail: str) -> LocationCheck:
        return LocationCheck(label, text, file, start, end, status, detail)

    lines = snapshot.lines(file)
    if lines is None:
        return result(LOC_INVALID, "the file is not in the index")
    total = line_count(lines)
    if start < 1 or end < start:
        return result(LOC_INVALID, "not a valid line range")
    if end > total:
        return result(LOC_INVALID, f"the file has only {total} lines")
    for item in evidence:
        sym = item.symbol
        if sym.file == file and sym.start_line <= start and end <= sym.end_line:
            return result(LOC_OK, f"inside retrieved evidence {item.id}")
    return result(LOC_UNGROUNDED, "exists, but was not part of the retrieved evidence")


def check_citations(
    answer: Answer, snapshot: IndexSnapshot
) -> tuple[list[CitationCheck], list[LocationCheck], dict[str, list[str]]]:
    """Returns (evidence citation checks, written-location checks, evidence id -> claim labels)."""
    by_id = answer.evidence_by_id()
    cited_by: dict[str, list[str]] = {}
    for label, _kind, claim in labeled_claims(answer):
        for eid in claim.evidence_ids:
            cited_by.setdefault(eid, []).append(label)
    citations = []
    for eid in cited_by:
        evidence = by_id.get(eid)
        if evidence is None:
            detail = "no such excerpt was retrieved for this answer"
            citations.append(CitationCheck(eid, "?", CITE_MISSING, detail))
        else:
            citations.append(check_evidence(evidence, snapshot))
    locations: list[LocationCheck] = []
    seen: set[tuple[str, str]] = set()
    for label, _kind, claim in labeled_claims(answer):
        for text, file, start, end in find_locations(claim.text):
            if (label, text) in seen:
                continue
            seen.add((label, text))
            locations.append(
                check_location(label, text, file, start, end, answer.evidence, snapshot)
            )
    return citations, locations, cited_by