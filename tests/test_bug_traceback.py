import pytest
from repo_helpers import SAMPLE_REPO

from core.bugs.mapping import (
    AMBIGUOUS,
    EXTERNAL,
    MAPPED,
    MISMATCH,
    MODULE_LEVEL,
    UNINDEXED,
    FrameMapper,
    is_external_path,
    is_test_path,
    pick_primary,
    select_frames,
)
from core.bugs.traceback_parser import MAX_INPUT_CHARS, Frame, parse_traceback
from core.repo.pipeline import analyze_repository
from core.semantic.models import Symbol

CLASSIC = r'''Traceback (most recent call last):
  File "C:\work\sample_repo\app.py", line 8, in main
    cart.add_item("book", 12.5, 2)
  File "C:\work\sample_repo\shop\cart.py", line 12, in add_item
    self.items.append((name, price, quantity))
AttributeError: 'ShoppingCart' object has no attribute 'items'
'''
CHAINED = '''Traceback (most recent call last):
  File "a.py", line 3, in first
    d["x"]
KeyError: 'x'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "b.py", line 9, in second
    raise ValueError("bad")
ValueError: bad
'''
PYTEST_SHORT = '''_____ test_total _____
tests/test_cart.py:7: in test_total
    assert cart.total() == 5
shop/cart.py:19: in total
    return sum(p * q for _, p, q in self.items) / len(self.items)
E   ZeroDivisionError: division by zero
'''
PYTEST_LONG = '''shop/cart.py:19: ZeroDivisionError
E   assert 1 == 2
'''
CARETS = '''Traceback (most recent call last):
  File "m.py", line 4, in run
    return helper()(1)
           ~~~~~~~~^^^
TypeError: 'int' object is not callable
'''


def test_classic_traceback():
    parsed = parse_traceback(CLASSIC)
    assert [(f.file, f.line, f.func) for f in parsed.frames] == [
        (r"C:\work\sample_repo\app.py", 8, "main"),
        (r"C:\work\sample_repo\shop\cart.py", 12, "add_item"),
    ]
    assert parsed.frames[1].code == "self.items.append((name, price, quantity))"
    assert parsed.exception_type == "AttributeError"
    assert "no attribute 'items'" in parsed.exception_message
    assert parsed.warnings == []


def test_chained_exceptions_use_the_final_error():
    parsed = parse_traceback(CHAINED)
    assert [f.file for f in parsed.frames] == ["b.py"]
    assert [f.file for f in parsed.earlier_frames] == ["a.py"]
    assert (parsed.exception_type, parsed.exception_message) == ("ValueError", "bad")


def test_pytest_short_style():
    parsed = parse_traceback(PYTEST_SHORT)
    assert [(f.file, f.line, f.func) for f in parsed.frames] == [
        ("tests/test_cart.py", 7, "test_total"),
        ("shop/cart.py", 19, "total"),
    ]
    assert parsed.exception_type == "ZeroDivisionError"


def test_pytest_long_style_and_bare_assert():
    parsed = parse_traceback(PYTEST_LONG)
    assert [(f.file, f.line, f.func) for f in parsed.frames] == [("shop/cart.py", 19, None)]
    assert parsed.exception_type == "AssertionError"


def test_caret_marker_lines_are_not_code():
    parsed = parse_traceback(CARETS)
    assert parsed.frames[0].code == "return helper()(1)"
    assert parsed.exception_type == "TypeError"


@pytest.mark.parametrize(
    "text", ["", "   \n", "just some words", "Traceback (most recent call last):"]
)
def test_text_without_frames_gives_a_warning_not_an_exception(text):
    parsed = parse_traceback(text)
    assert parsed.frames == []
    assert parsed.warnings


@pytest.mark.parametrize("text", [None, 5, b"bytes"])
def test_non_text_input_never_raises(text):
    assert parse_traceback(text).warnings  # type: ignore[arg-type]


def test_huge_input_is_cut():
    text = CLASSIC + "x" * (MAX_INPUT_CHARS + 10)
    parsed = parse_traceback(text)
    assert any("first" in w for w in parsed.warnings)
    assert parsed.frames


def test_absurd_line_numbers_are_ignored():
    text = 'File "a.py", line 99999999999, in f\n'
    assert parse_traceback(text).frames == []


def sample():
    analysis = analyze_repository(SAMPLE_REPO)
    analysis.close()
    return analysis.symbols, set(analysis.file_status)


def make_symbol(file, qualname, start, end, code="x"):
    return Symbol(
        key=f"{file}::{qualname}", file=file, kind="function", name=qualname,
        qualname=qualname, start_line=start, end_line=end, docstring="", code=code,
    )


def test_frames_map_to_the_innermost_symbol_with_exact_paths():
    symbols, files = sample()
    mapper = FrameMapper(symbols, files, root_hint=r"C:\work\sample_repo")
    first, second = mapper.map(parse_traceback(CLASSIC).frames)
    assert (first.status, first.symbol.qualname, first.file) == (MAPPED, "main", "app.py")
    assert second.symbol.qualname == "ShoppingCart.add_item"  # not the class around it
    assert second.reason == "path matches the indexed file"


def test_frames_map_by_path_ending_when_the_root_is_unknown():
    symbols, files = sample()
    second = FrameMapper(symbols, files).map(parse_traceback(CLASSIC).frames)[1]
    assert second.status == MAPPED
    assert second.file == "shop/cart.py"
    assert "end of the path" in second.reason


def test_a_changed_line_is_refused_not_mapped():
    symbols, files = sample()
    frame = Frame("shop/cart.py", 12, "add_item", "self.items.clear()")
    mapped = FrameMapper(symbols, files).map_frame(frame)
    assert mapped.status == MISMATCH
    assert mapped.symbol is None


def test_module_level_unindexed_external_and_ambiguous_frames():
    symbols, files = sample()
    mapper = FrameMapper(symbols, files)
    assert mapper.map_frame(Frame("app.py", 2)).status == MODULE_LEVEL
    assert mapper.map_frame(Frame("not_here.py", 3)).status == UNINDEXED
    for path in (
        r"C:\Python313\Lib\json\decoder.py",
        "/venv/lib/site-packages/requests/utils.py",
        "<frozen importlib._bootstrap>",
    ):
        assert mapper.map_frame(Frame(path, 10)).status == EXTERNAL
    twins = [make_symbol("a/utils.py", "f", 1, 5), make_symbol("b/utils.py", "g", 1, 5)]
    assert FrameMapper(twins).map_frame(Frame("utils.py", 3)).status == AMBIGUOUS


def test_a_library_file_never_maps_to_a_repo_file_with_the_same_name():
    symbols = [make_symbol("utils.py", "f", 1, 5)]
    frame = Frame(r"C:\venv\Lib\site-packages\requests\utils.py", 3)
    assert FrameMapper(symbols).map_frame(frame).status == EXTERNAL


def test_path_helpers():
    assert is_test_path("tests/test_x.py") and is_test_path("pkg/conftest.py")
    assert not is_test_path("shop/cart.py")
    assert is_external_path("C:/Users/p/AppData/Local/Programs/Python/Python313/Lib/os.py")
    assert not is_external_path("shop/cart.py")


def test_primary_and_selected_frames_prefer_repo_code_over_tests():
    symbols = [
        make_symbol("tests/test_cart.py", "test_total", 5, 8),
        make_symbol("shop/cart.py", "total", 18, 20),
        make_symbol("shop/pricing.py", "apply", 3, 9),
    ]
    mapper = FrameMapper(symbols)
    mapped = mapper.map(
        [Frame("tests/test_cart.py", 7), Frame("shop/pricing.py", 4), Frame("shop/cart.py", 19)]
    )
    assert pick_primary(mapped).symbol.qualname == "total"
    chosen = [m.symbol.qualname for m in select_frames(mapped)]
    assert chosen == ["total", "apply", "test_total"]
    assert [m.symbol.qualname for m in select_frames(mapped, 1, include_test=False)] == ["total"]
    only_test = mapper.map([Frame("tests/test_cart.py", 7)])
    assert pick_primary(only_test).symbol.qualname == "test_total"
    assert pick_primary([]) is None