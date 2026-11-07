from pathlib import Path

import pytest
from impact_e2e_helpers import edit_diff
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app"
PAGES = {
    "main.py": "AI-Powered Software Engineering Assistant",
    "pages/1_Chat.py": "Repository chat",
    "pages/2_Dependencies.py": "Dependency graph",
    "pages/3_Validated_chat.py": "Validated chat",
    "pages/4_Graph_chat.py": "Graph chat",
    "pages/5_Bug_investigation.py": "Bug investigation",
    "pages/6_Change_impact.py": "Change impact",
    "pages/7_Repository_memory.py": "Repository memory",
}


def load(page):
    at = AppTest.from_file(str(APP / page), default_timeout=90)
    at.run()
    return at


@pytest.mark.parametrize(("page", "title"), PAGES.items())
def test_every_page_renders_its_title_without_errors(page, title):
    at = load(page)
    assert not at.exception, [e.value for e in at.exception]
    assert [t.value for t in at.title] == [title]


@pytest.mark.parametrize(
    "page",
    [
        "pages/5_Bug_investigation.py",
        "pages/6_Change_impact.py",
        "pages/7_Repository_memory.py",
    ],
)
def test_the_newer_pages_show_their_controls_before_a_repository_is_loaded(page):
    at = load(page)
    assert [b.label for b in at.sidebar.button] == ["Load repository"]
    assert any("Load repository" in info.value for info in at.info)


def test_the_change_impact_page_renders_a_real_report():
    diff = edit_diff("pricing/rates.py", 8, "    return round(amount * (1 + TAX_RATE), 3)")
    at = AppTest.from_file(str(APP / "pages" / "6_Change_impact.py"), default_timeout=90)
    at.session_state["impact_diff"] = diff
    at.run()
    at.sidebar.button[0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    analyze = next(b for b in at.button if b.label == "Analyze impact")
    analyze.click().run()
    assert not at.exception, [e.value for e in at.exception]
    headings = [s.value for s in at.subheader]
    assert "Known dependencies (confirmed edges)" in headings
    assert "Inferred impacts (not confirmed)" in headings
    assert "Recommended tests" in headings
    assert any("not probabilities" in w.value for w in at.warning)