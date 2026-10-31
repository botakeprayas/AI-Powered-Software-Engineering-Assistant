"""Builders shared by the Phase 5 tests. The 'judge' here is a MOCK, not a real model."""

from __future__ import annotations

import shutil
from pathlib import Path

from repo_helpers import SAMPLE_REPO

from core.chat.models import STATUS_ANSWERED, Answer, Claim, Evidence
from core.repo.pipeline import analyze_repository
from core.semantic.models import Symbol
from core.validation.snapshot import IndexSnapshot
from services.llm_service import LLMError, LLMResponse, LLMService, LLMStatus


def load_symbols(root: Path = SAMPLE_REPO) -> dict[str, Symbol]:
    analysis = analyze_repository(root)
    analysis.close()
    return {s.qualname: s for s in analysis.symbols}


def snapshot_for(root: Path, symbols: dict[str, Symbol]) -> IndexSnapshot:
    locations = {s.key: (s.file, s.start_line, s.end_line) for s in symbols.values()}
    return IndexSnapshot(root, locations)


def evidence_for(
    symbol: Symbol, eid: str = "E1", semantic: float = 0.55, rank: int = 1, score: float = 12.0
) -> Evidence:
    return Evidence(
        id=eid, symbol=symbol, code=symbol.code, code_truncated=False,
        semantic_score=semantic, keyword_rank=rank, keyword_score=score,
    )


def make_answer(facts=(), inferences=(), evidence=(), status: str = STATUS_ANSWERED) -> Answer:
    return Answer(
        question="q",
        status=status,
        facts=[Claim(text, tuple(ids)) for text, ids in facts],
        inferences=[Claim(text, tuple(ids)) for text, ids in inferences],
        evidence=list(evidence),
    )


def copy_sample(destination: Path) -> Path:
    shutil.copytree(SAMPLE_REPO, destination)
    return destination


class ScriptedJudge(LLMService):
    """MOCK judge: returns a scripted reply or raises a scripted error. NOT a real model."""

    def __init__(self, reply: str = "", error: LLMError | None = None) -> None:
        self._reply = reply
        self._error = error
        self.prompts: list[str] = []

    @property
    def provider(self) -> str:
        return "scripted-mock"

    @property
    def model_name(self) -> str:
        return "scripted-judge"

    @property
    def is_mock(self) -> bool:
        return True

    def check_ready(self) -> LLMStatus:
        return LLMStatus(True, "MOCK scripted judge")

    def generate(self, prompt, *, system=None, temperature=0.1, max_tokens=800):
        self.prompts.append(prompt)
        if self._error is not None:
            raise self._error
        return LLMResponse(
            text=self._reply, model=self.model_name, provider=self.provider, is_mock=True
        )