import pytest
from impact_helpers import (
    CART,
    IMPACT_SAMPLE,
    insert_after,
    make_delete_diff,
    make_diff,
    make_new_diff,
    make_view,
    replace_line,
)

from core.graph.expand import load_graph_context
from core.impact.changes import (
    MODULE_LEVEL,
    OUT_BINARY,
    OUT_EMPTY,
    OUT_MAPPED,
    OUT_MISMATCH,
    OUT_NEW,
    OUT_NOT_INDEXED,
    OUT_NOT_PYTHON,
    OUT_UNPARSED,
    SIDE_NEW,
    SIDE_OLD,
    ChangedUnit,
    IndexView,
    map_diff,
)
from core.impact.diff_parser import parse_diff

PATH = "shop/cart.py"
BINARY = (
    "diff --git a/img/logo.png b/img/logo.png\n"
    "index 1111111..2222222 100644\n"
    "Binary files a/img/logo.png and b/img/logo.png differ\n"
)
MODE_ONLY = "diff --git a/shop/cart.py b/shop/cart.py\nold mode 100644\nnew mode 100755\n"
RENAME = (
    "diff --git a/shop/cart.py b/shop/basket.py\n"
    "similarity index 100%\nrename from shop/cart.py\nrename to shop/basket.py\n"
)


def run(diff, texts=None, base="auto", unparsed=()):
    view = make_view(texts or {PATH: CART}, unparsed)
    return map_diff(parse_diff(diff), view, base)


def only(result):
    (change,) = result.files
    return change


def names(change):
    return [unit.qualname for unit in change.units]


def test_an_edited_statement_maps_to_its_method():
    new = replace_line(CART, 10, "        self.items.insert(0, item)")
    change = only(run(make_diff(PATH, CART, new)))
    assert (change.outcome, change.side) == (OUT_MAPPED, SIDE_OLD)
    assert names(change) == ["Cart.add"]
    unit = change.units[0]
    assert (unit.start_line, unit.end_line) == (8, 10)
    assert (unit.lines_added, unit.lines_removed) == (1, 1)
    assert unit.touched == {10}


def test_a_statement_appended_to_a_method_belongs_to_that_method():
    new = insert_after(CART, 10, "        self.count += 1")
    change = only(run(make_diff(PATH, CART, new)))
    assert names(change) == ["Cart.add"]
    unit = change.units[0]
    assert unit.inserted_after == {10} and unit.touched == set()
    assert (unit.lines_added, unit.lines_removed) == (1, 0)


def test_a_new_method_at_the_end_of_a_class_belongs_to_the_class():
    new = insert_after(CART, 14, "", "    def clear(self):", "        self.items.clear()")
    change = only(run(make_diff(PATH, CART, new)))
    assert names(change) == ["Cart"]
    unit = change.units[0]
    assert unit.kind == "class" and (unit.start_line, unit.end_line) == (5, 14)
    assert unit.lines_added == 3


def test_a_new_function_after_the_last_function_is_module_level():
    new = insert_after(CART, 19, "", "", "def other():", "    return 2")
    change = only(run(make_diff(PATH, CART, new)))
    (unit,) = change.units
    assert unit.is_module_level and unit.key == PATH
    assert unit.qualname == f"module {PATH}"
    assert unit.lines_added == 4


def test_a_changed_constant_is_module_level():
    new = replace_line(CART, 22, "LIMIT = 20")
    (unit,) = only(run(make_diff(PATH, CART, new))).units
    assert unit.kind == MODULE_LEVEL and unit.touched == {22}


def test_a_docstring_edit_touches_the_class_not_its_methods():
    new = replace_line(CART, 6, '    """A shopping cart."""')
    assert names(only(run(make_diff(PATH, CART, new)))) == ["Cart"]


def test_a_signature_edit_touches_the_method():
    new = replace_line(CART, 8, "    def add(self, item, front=False):")
    assert names(only(run(make_diff(PATH, CART, new)))) == ["Cart.add"]


def test_two_edits_give_two_units_in_order():
    edited = replace_line(CART, 10, "        self.items.insert(0, item)")
    new = replace_line(edited, 19, "    return x")
    change = only(run(make_diff(PATH, CART, new)))
    assert names(change) == ["Cart.add", "helper"]
    assert [u.touched for u in change.units] == [{10}, {19}]


def test_a_comment_between_definitions_is_module_level():
    new = insert_after(CART, 16, "# helper below")
    (unit,) = only(run(make_diff(PATH, CART, new))).units
    assert unit.is_module_level


def test_a_new_decorator_belongs_to_the_decorated_function():
    new = insert_after(CART, 16, "@cache")
    assert names(only(run(make_diff(PATH, CART, new)))) == ["helper"]


def test_a_diff_that_is_already_applied_is_mapped_on_the_new_version():
    new = replace_line(CART, 10, "        self.items.insert(0, item)")
    change = only(run(make_diff(PATH, CART, new), {PATH: new}))
    assert (change.outcome, change.side) == (OUT_MAPPED, SIDE_NEW)
    assert names(change) == ["Cart.add"] and change.units[0].touched == {10}
    appended = insert_after(CART, 10, "        self.count += 1")
    change = only(run(make_diff(PATH, CART, appended), {PATH: appended}))
    assert change.side == SIDE_NEW and names(change) == ["Cart.add"]
    assert change.units[0].touched == {11}


def test_forcing_the_wrong_side_is_a_mismatch():
    new = replace_line(CART, 10, "        self.items.insert(0, item)")
    diff = make_diff(PATH, CART, new)
    assert only(run(diff, {PATH: new}, base="old")).outcome == OUT_MISMATCH
    assert only(run(diff, {PATH: CART}, base="new")).outcome == OUT_MISMATCH
    assert only(run(diff, {PATH: CART}, base="old")).outcome == OUT_MAPPED


def test_a_diff_for_a_different_version_is_refused_not_guessed():
    other = replace_line(CART, 9, '        """Add one item."""')
    changed = replace_line(other, 10, "        self.items.insert(0, item)")
    change = only(run(make_diff(PATH, other, changed)))
    assert change.outcome == OUT_MISMATCH
    assert "differs from the diff" in change.detail
    assert change.units == []


def test_diffs_without_context_lines_are_checked_when_possible():
    new = insert_after(CART, 10, "        self.count += 1")
    diff = make_diff(PATH, CART, new, context=0)
    on_old = only(run(diff, {PATH: CART}))
    assert (on_old.outcome, on_old.side) == (OUT_MAPPED, SIDE_OLD)
    assert "no context lines" in on_old.detail and names(on_old) == ["Cart.add"]
    on_new = only(run(diff, {PATH: new}))
    assert on_new.side == SIDE_NEW and "no context lines" not in on_new.detail


@pytest.mark.parametrize(
    ("diff", "outcome"),
    [
        (make_diff("README.md", "a\n", "b\n"), OUT_NOT_PYTHON),
        (make_diff("other/mod.py", "a = 1\n", "a = 2\n"), OUT_NOT_INDEXED),
        (make_new_diff("shop/fresh.py", "x = 1\n"), OUT_NEW),
        (BINARY, OUT_BINARY),
        (MODE_ONLY, OUT_EMPTY),
    ],
)
def test_changes_outside_the_analysis_are_listed_not_mapped(diff, outcome):
    change = only(run(diff))
    assert change.outcome == outcome
    assert change.units == []


def test_a_deleted_file_changes_all_its_symbols():
    change = only(run(make_delete_diff(PATH, CART)))
    assert (change.status, change.outcome, change.side) == ("deleted", OUT_MAPPED, SIDE_OLD)
    assert names(change) == ["Cart", "Cart.add", "Cart.total", "helper", f"module {PATH}"]
    assert {unit.whole for unit in change.units} == {"deleted"}
    assert change.units[0].lines_removed == 10


def test_a_renamed_file_changes_all_its_symbols():
    change = only(run(RENAME))
    assert (change.status, change.outcome, change.path) == ("renamed", OUT_MAPPED, PATH)
    assert names(change)[:4] == ["Cart", "Cart.add", "Cart.total", "helper"]
    assert {unit.whole for unit in change.units} == {"renamed"}


def test_a_file_that_could_not_be_parsed_gets_a_module_level_unit():
    old = "def f(:\n    pass\n"
    diff = make_diff("bad.py", old, "def f(:\n    return 1\n")
    change = only(run(diff, {"bad.py": old}, unparsed=("bad.py",)))
    assert change.outcome == OUT_UNPARSED
    (unit,) = change.units
    assert unit.is_module_level and unit.touched == {2}


def test_the_sample_repository_maps_through_the_real_graph_context():
    context = load_graph_context(IMPACT_SAMPLE)
    view = IndexView.from_context(context)
    assert len(context.symbols) == 15
    assert sorted(view.files) == [
        "app.py", "pricing/__init__.py", "pricing/invoice.py", "pricing/rates.py",
        "shop/__init__.py", "shop/checkout.py", "shop/display.py", "shop/receipt.py",
        "tests/test_checkout.py", "tests/test_invoice.py", "tests/test_rates.py",
        "tools/formatting.py",
    ]
    old = (IMPACT_SAMPLE / "pricing" / "rates.py").read_text(encoding="utf-8")
    new = replace_line(old, 14, "    return amount")
    change = only(map_diff(parse_diff(make_diff("pricing/rates.py", old, new)), view))
    assert change.outcome == OUT_MAPPED and names(change) == ["apply_discount"]
    unit = change.units[0]
    assert (unit.start_line, unit.end_line, unit.touched) == (11, 14, {14})
    assert view.read_lines(".env") is None  # files outside the index are never opened


def test_a_unit_describes_its_changed_lines_in_words():
    unit = ChangedUnit("k", PATH, "method", "Cart.add", 8, 10)
    unit.touched.update({18, 19, 22})
    unit.inserted_after.add(14)
    assert unit.describe() == "lines 18-19, 22 changed; new lines inserted after line 14"
    whole = ChangedUnit(PATH, PATH, MODULE_LEVEL, "module", whole="deleted")
    assert whole.describe() == "the whole file is deleted" and whole.is_module_level


def test_an_invalid_base_is_rejected():
    diff = make_diff(PATH, CART, CART + "x = 1\n")
    with pytest.raises(ValueError, match="base must be"):
        map_diff(parse_diff(diff), make_view({PATH: CART}), "sideways")