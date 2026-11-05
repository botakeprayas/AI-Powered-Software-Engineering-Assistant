from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app"
PAGES = {
    "main.py": "AI-Powered Software Engineering Assistant",
    "pages/1_Chat.py": "Repository chat",
    "pages/2_Dependencies.py": "Dependency graph",
    "pages/3_Validated_chat.py": "Validated chat",
    "pages/4_Graph_chat.py": "Graph chat",
    "pages/5_Bug_investigation.py": "Bug investigation",
}


@pytest.mark.parametrize(("page", "title"), PAGES.items())
def test_every_page_renders_its_title_without_errors(page, title):
    at = AppTest.from_file(str(APP / page), default_timeout=90)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert [t.value for t in at.title] == [title]


def test_the_bug_page_shows_its_controls_before_a_repository_is_loaded():
    at = AppTest.from_file(str(APP / "pages" / "5_Bug_investigation.py"), default_timeout=90)
    at.run()
    assert [b.label for b in at.sidebar.button] == ["Load repository"]
    assert any("Load repository" in info.value for info in at.info)