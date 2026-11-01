import pytest
from bug_helpers import BUG_SAMPLE

from core.bugs.proposal import (
    DRAFT_NONE,
    DRAFT_READY,
    DRAFT_REJECTED,
    FIX_NONE,
    FIX_PROPOSED,
    FIX_REJECTED,
    PROPOSED_LABEL,
    build_fix_proposal,
    build_regression_draft,
    find_risks,
)

PRICING = (BUG_SAMPLE / "cart" / "pricing.py").read_text(encoding="utf-8")
CHECKOUT = (BUG_SAMPLE / "cart" / "checkout.py").read_text(encoding="utf-8")
ALLOWED = {"cart/pricing.py", "cart/checkout.py"}
SEARCH = "    total = sum(price for _, price in items)\n    return total / len(items)"
REPLACE = (
    "    if not items:\n        return 0.0\n"
    "    total = sum(price for _, price in items)\n    return total / len(items)"
)


def reader(rel):
    return {"cart/pricing.py": PRICING, "cart/checkout.py": CHECKOUT}.get(rel)


def block(file="cart/pricing.py", search=SEARCH, replace=REPLACE):
    return f"FILE: {file}\n<<<<<<< SEARCH\n{search}\n=======\n{replace}\n>>>>>>> REPLACE\n"


def propose(raw, **limits):
    return build_fix_proposal(raw, reader, ALLOWED, **limits)


def test_a_good_edit_becomes_a_real_labeled_diff():
    fix = propose("RATIONALE: guard the empty cart.\n" + block())
    assert fix.status == FIX_PROPOSED and fix.usable
    assert fix.label == PROPOSED_LABEL == "PROPOSED, NOT VERIFIED"
    assert fix.rationale == "guard the empty cart."
    assert fix.changed_lines == 2
    assert "--- a/cart/pricing.py" in fix.diff and "+++ b/cart/pricing.py" in fix.diff
    assert "+    if not items:" in fix.diff
    assert "if not items:" in fix.patched["cart/pricing.py"]
    assert PRICING == (BUG_SAMPLE / "cart" / "pricing.py").read_text(encoding="utf-8")


def test_fenced_replies_are_accepted():
    assert propose("```diff\n" + block() + "```\n").usable


def test_declining_and_unparseable_answers():
    declined = propose("NO_FIX: not enough evidence")
    assert declined.status == FIX_NONE
    assert declined.rationale == "not enough evidence"
    assert propose("Here is my idea: check for zero.").status == FIX_REJECTED


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (block(search="    return nothing_like_this"), "was not found"),
        (block(search="items", replace="things"), "matches"),
        (block(file="tests/test_pricing.py"), "not an editable file"),
        (block(file="cart/other.py"), "not an editable file"),
        (block(replace=SEARCH), "changes nothing"),
        (block(search="   "), "SEARCH text is empty"),
        (block(replace="    return total / (len(items)"), "syntax error"),
        (
            block(
                search="    return total / len(items)",
                replace="    import subprocess\n    subprocess.run(['echo'])\n"
                "    return total / len(items)",
            ),
            "introduces risky code",
        ),
    ],
)
def test_bad_edits_are_rejected_with_a_reason(raw, message):
    fix = propose(raw)
    assert fix.status == FIX_REJECTED and not fix.usable
    assert any(message in problem for problem in fix.problems), fix.problems
    assert fix.patched == {} and fix.diff == ""


def test_edit_and_size_limits():
    first = block()
    second = block(
                "cart/checkout.py",
        "    average = average_price(items)",
        "    average = average_price(items) or 0",
    )
    assert any("limit is 1" in p for p in propose(first + second, max_edits=1).problems)
    assert any("limit is 1" in p for p in propose(first, max_changed_lines=1).problems)


def test_risk_scan_flags_network_processes_and_eval():
    source = "import socket\nimport os\n\n\ndef f():\n    os.system('x')\n    return eval('1')\n"
    assert find_risks(source) == [
        "imports socket (line 1)", "uses .system (line 6)", "calls eval() (line 7)",
    ]
    assert find_risks("def broken(:\n") == []


GOOD_TEST = (
    "from cart.pricing import average_price\n\n\n"
    "def test_empty():\n    assert average_price([]) == 0\n"
)

def reply(code=GOOD_TEST, file="tests/test_regression_x.py"):
    return f"FILE: {file}\n```python\n{code}```\n"


def draft(raw, exists=lambda rel: False):
    return build_regression_draft(raw, "fallback", exists)


def test_a_good_draft_is_a_new_file_under_tests():
    result = draft(reply())
    assert result.status == DRAFT_READY and result.usable
    assert result.path == "tests/test_regression_x.py"
    assert result.label == "DRAFT, NOT RUN"


def test_the_path_is_forced_and_never_overwrites_an_existing_file():
    assert draft(reply(file="../../evil.py")).path == "tests/test_regression_evil.py"
    taken = {"tests/test_regression_x.py", "tests/test_regression_x_2.py"}
    assert draft(reply(), exists=lambda rel: rel in taken).path == "tests/test_regression_x_3.py"
    assert draft("```python\n" + GOOD_TEST + "```\n").path == "tests/test_regression_fallback.py"


@pytest.mark.parametrize(
    ("code", "message"),
    [
        ("def test_x(:\n    pass\n", "syntax error"),
        ("def helper():\n    return 1\n", "no function whose name starts with 'test'"),
        ("import socket\n\n\ndef test_x():\n    socket.socket()\n", "imports socket"),
        ("def test_x():\n    eval('1')\n", "calls eval()"),
        ("def test_x():\n    pass\n# " + "x" * 5000 + "\n", "longer than"),
    ],
)
def test_unsafe_or_broken_drafts_are_rejected(code, message):
    result = draft(reply(code))
    assert result.status == DRAFT_REJECTED and not result.usable
    assert any(message in problem for problem in result.problems), result.problems


def test_declined_or_missing_code_blocks():
    assert draft("NO_TEST: needs a database").status == DRAFT_NONE
    assert draft("just words").status == DRAFT_REJECTED