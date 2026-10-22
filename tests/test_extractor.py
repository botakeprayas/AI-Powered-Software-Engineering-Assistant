from repo_helpers import SAMPLE_REPO, build_hazard_repo

from core.repo.extractor import extract_file, extract_symbols
from core.repo.models import RepoFile
from core.repo.pipeline import analyze_repository

# qualname -> (file, kind, start_line, end_line), checked against data/sample_repo
EXPECTED = {
    "main": ("app.py", "function", 5, 10),
    "fetch_data": ("app.py", "async_function", 13, 15),
    "ShoppingCart": ("shop/cart.py", "class", 4, 27),
    "ShoppingCart.__init__": ("shop/cart.py", "method", 7, 8),
    "ShoppingCart.add_item": ("shop/cart.py", "method", 10, 12),
    "ShoppingCart.remove_item": ("shop/cart.py", "method", 14, 16),
    "ShoppingCart.total": ("shop/cart.py", "method", 18, 20),
    "ShoppingCart.checkout": ("shop/cart.py", "async_method", 22, 27),
    "log_call": ("shop/pricing.py", "function", 5, 13),
    "apply_discount": ("shop/pricing.py", "function", 16, 18),
    "compute_tax": ("shop/pricing.py", "function", 21, 24),  # starts at its decorator
    "validate_email": ("utils/validation.py", "function", 7, 9),
    "validate_age": ("utils/validation.py", "function", 12, 16),
}


def sample_symbols():
    analysis = analyze_repository(SAMPLE_REPO)
    analysis.close()
    return {s.qualname: s for s in analysis.symbols}


def test_sample_symbols_have_expected_kind_and_lines():
    symbols = sample_symbols()
    assert set(symbols) == set(EXPECTED)
    for qualname, (file, kind, start, end) in EXPECTED.items():
        sym = symbols[qualname]
        assert (sym.file, sym.kind, sym.start_line, sym.end_line) == (file, kind, start, end)


def test_stored_code_is_exactly_the_line_range():
    for sym in sample_symbols().values():
        lines = (SAMPLE_REPO / sym.file).read_text(encoding="utf-8").split("\n")
        assert "\n".join(lines[sym.start_line - 1 : sym.end_line]) == sym.code
        assert lines[sym.start_line - 1].lstrip().startswith(("def ", "async def ", "class ", "@"))


def test_docstring_is_extracted():
    sym = sample_symbols()["ShoppingCart.total"]
    assert sym.docstring == "Return the sum of price times quantity for all items."


def test_function_nested_inside_a_function_is_not_indexed():
    assert "wrapper" not in {s.name for s in sample_symbols().values()}


def test_definition_inside_try_except_is_found():
    text = "try:\n    import x\nexcept ImportError:\n    def fallback():\n        return 1\n"
    (sym,) = extract_symbols("m.py", text)
    assert (sym.name, sym.start_line, sym.end_line) == ("fallback", 4, 5)


def test_duplicate_names_get_unique_keys():
    text = "if True:\n    def f():\n        return 1\nelse:\n    def f():\n        return 2\n"
    keys = [s.key for s in extract_symbols("m.py", text)]
    assert keys == ["m.py::f", "m.py::f#2"]


def test_windows_line_endings_keep_line_numbers_correct():
    text = "def a():\r\n    return 1\r\n\r\n\r\ndef b():\r\n    return 2\r\n"
    b = extract_symbols("m.py", text)[1]
    assert (b.start_line, b.end_line) == (5, 6)
    assert b.code == "def b():\n    return 2"


def test_syntax_error_becomes_a_warning_not_an_exception():
    result = extract_file(SAMPLE_REPO, RepoFile("broken_syntax.py", 0, True))
    assert result.status == "syntax_error"
    assert result.symbols == []
    assert "syntax error" in result.issues[0].message


def test_empty_file_becomes_a_warning():
    result = extract_file(SAMPLE_REPO, RepoFile("empty_module.py", 0, True))
    assert result.status == "empty"
    assert result.issues


def test_bad_encoding_and_nul_bytes_become_warnings(tmp_path):
    root = build_hazard_repo(tmp_path / "repo")
    bad = extract_file(root, RepoFile("bad_encoding.py", 0, True))
    nul = extract_file(root, RepoFile("binary_like.py", 0, True))
    assert bad.status == "encoding_error"
    assert "UTF-8" in bad.issues[0].message
    assert nul.status == "error"


def test_missing_file_becomes_a_warning(tmp_path):
    result = extract_file(tmp_path, RepoFile("ghost.py", 0, True))
    assert result.status == "error"


def test_utf8_bom_is_accepted(tmp_path):
    (tmp_path / "bom.py").write_bytes(b"\xef\xbb\xbfdef f():\n    pass\n")
    result = extract_file(tmp_path, RepoFile("bom.py", 0, True))
    assert result.status == "parsed"
    assert [s.name for s in result.symbols] == ["f"]