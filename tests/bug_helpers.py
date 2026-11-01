"""Helpers for the Phase 6 tests. SequenceLLM is a MOCK that replays scripted replies."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from core.graph.expand import GraphContext, load_graph_context
from core.validation.config import ValidationConfig
from services.llm_service import LLMError, LLMResponse, LLMService, LLMStatus

BUG_SAMPLE = Path(__file__).resolve().parents[1] / "data" / "bug_sample"

HYPOTHESIS_REPLY = (
    "STATUS: ANSWERED\n"
    "FACTS:\n"
    "- The exception is ZeroDivisionError, raised at `total / len(items)` [E1]\n"
    "INFERENCES:\n"
    "- Hypothesis: `average_price` divides by `len(items)`, which is 0 for an empty cart "
    "[E1][E2]\n"
    "UNKNOWNS:\n"
    "- Whether an empty cart is allowed is not shown.\n"
)
UNSUPPORTED_HYPOTHESIS_REPLY = (
    "STATUS: ANSWERED\n"
    "FACTS:\n"
    "- The exception is ZeroDivisionError, raised at `total / len(items)` [E1]\n"
    "INFERENCES:\n"
    "- Hypothesis: `average_price` calls `refund_all_orders` and `send_email_report` [E1]\n"
    "UNKNOWNS:\n"
    "- none\n"
)
FABRICATED_REPLY = (
    "STATUS: ANSWERED\nFACTS:\n- The division fails [E9]\n"
    "INFERENCES:\n- Hypothesis: bad input [E9]\nUNKNOWNS:\n- none\n"
)
FIX_REPLY = (
    "RATIONALE: average_price divides by len(items), which is 0 for an empty cart.\n"
    "FILE: cart/pricing.py\n"
    "<<<<<<< SEARCH\n"
    "    total = sum(price for _, price in items)\n"
    "    return total / len(items)\n"
    "=======\n"
    "    if not items:\n"
    "        return 0.0\n"
    "    total = sum(price for _, price in items)\n"
    "    return total / len(items)\n"
    ">>>>>>> REPLACE\n"
)
TEST_REPLY = (
    "FILE: tests/test_regression_average_price.py\n"
    "```python\n"
    "from cart.pricing import average_price\n"
    "\n"
    "\n"
    "def test_average_of_empty_cart_is_zero():\n"
    "    assert average_price([]) == 0.0\n"
    "```\n"
)


class SequenceLLM(LLMService):
    """MOCK: replays scripted replies in order (the last one repeats). NOT a real model.

    `errors` maps a 1-based call number to an LLMError that call raises.
    """

    def __init__(self, replies: list[str], errors: dict[int, LLMError] | None = None) -> None:
        self._replies = list(replies)
        self._errors = dict(errors or {})
        self.prompts: list[str] = []
        self.systems: list[str | None] = []

    @property
    def provider(self) -> str:
        return "scripted-test"

    @property
    def model_name(self) -> str:
        return "scripted-model"

    def check_ready(self) -> LLMStatus:
        return LLMStatus(True, "scripted test double")

    def generate(self, prompt, *, system=None, temperature=0.1, max_tokens=800):
        self.prompts.append(prompt)
        self.systems.append(system)
        number = len(self.prompts)
        if number in self._errors:
            raise self._errors[number]
        text = self._replies[min(number - 1, len(self._replies) - 1)]
        return LLMResponse(text=text, model=self.model_name, provider=self.provider)


def validation_config() -> ValidationConfig:
    """Rule-based claim checks only (no judge call) and no log file."""
    return dataclasses.replace(ValidationConfig(), judge_enabled=False, log_enabled=False)


def sample_context() -> GraphContext:
    return load_graph_context(BUG_SAMPLE)


def sample_traceback() -> str:
    return (BUG_SAMPLE / "traceback.txt").read_text(encoding="utf-8")