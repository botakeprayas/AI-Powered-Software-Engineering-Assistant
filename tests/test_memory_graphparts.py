import ast
import json

import pytest

from core.graph.analyzer import collect_imports, module_assigned_names, package_parts_for
from core.memory.graphparts import MemoryIndexError, imports_from_json, imports_to_json
from core.memory.graphparts import interface_hash
from core.memory.indexer import load_graph, scan_order_key
from core.memory.models import FileState
from core.memory.store import MemoryStore
from core.repo.extractor import extract_symbols


def interface(text, path="m.py"):
    tree = ast.parse(text)
    imports = collect_imports(tree, package_parts_for(path))
    symbols = extract_symbols(path, text)
    return interface_hash("parsed", symbols, imports, module_assigned_names(tree.body))


BASE = (
    "import os\nLIMIT = 1\n\n\ndef first(a):\n    \"\"\"One.\"\"\"\n    return a\n\n\n"
    "class Box:\n    def open(self):\n        return 1\n"
)


def test_the_interface_ignores_everything_other_files_cannot_see():
    reference = interface(BASE)
    assert interface(BASE.replace("return a", "return a + 1")) == reference  # a body edit
    assert interface(BASE.replace("One.", "Another.")) == reference  # a docstring edit
    assert interface(BASE.replace("def first(a)", "def first(a, b)")) == reference  # parameters
    assert interface("\n\n" + BASE) == reference  # lines shift


@pytest.mark.parametrize(
    "changed",
    [
        BASE + "\n\ndef extra():\n    return 1\n",  # a new function
        BASE.replace("def first(a)", "def second(a)"),  # a renamed function
        BASE.replace("import os", "import os\nimport sys"),  # a new import
        BASE.replace("LIMIT = 1", "LIMIT = 1\nOTHER = 2"),  # a new module variable
        BASE.replace("def open(self)", "async def open(self)"),  # a different kind
        BASE.replace("class Box:", "class Crate:"),  # a renamed class
    ],
)
def test_the_interface_changes_when_other_files_could_notice(changed):
    assert interface(changed) != interface(BASE)


def test_reordering_definitions_changes_the_interface():
    one = "def a():\n    return 1\n\n\ndef b():\n    return 2\n"
    two = "def b():\n    return 2\n\n\ndef a():\n    return 1\n"
    assert interface(one) != interface(two)


def test_the_parse_status_is_part_of_the_interface():
    assert interface_hash("parsed", [], [], set()) != interface_hash("syntax_error", [], None, None)


def test_imports_survive_a_json_round_trip():
    text = (
        "import os\nimport a.b as c\nfrom pkg import x as y, z\nfrom . import sibling\n"
        "from .mod import *\n"
    )
    imports = collect_imports(ast.parse(text), package_parts_for("pkg/m.py"))
    assert len(imports) == 5 and any(item.star for item in imports)
    assert imports_from_json(imports_to_json(imports)) == imports


def test_the_scanner_order_puts_files_before_sub_folders():
    paths = ["a/b/y.py", "a/x.py", "z.py", "a/b/c/deep.py", "a.py"]
    assert sorted(paths, key=scan_order_key) == [
        "a.py", "z.py", "a/x.py", "a/b/y.py", "a/b/c/deep.py",
    ]


def make_store_with_one_file():
    store = MemoryStore()
    store.put_file_state(FileState("a.py", "hash", 1), 1)
    return store


def test_a_file_without_graph_data_is_reported_not_guessed():
    store = make_store_with_one_file()
    with pytest.raises(MemoryIndexError, match="graph data is missing for a.py"):
        load_graph(store)


def test_an_edge_pointing_at_a_missing_node_is_reported_as_inconsistent():
    store = make_store_with_one_file()
    edge = {
        "source": "a.py", "target": "nowhere", "kind": "calls", "certainty": "confirmed",
        "file": "a.py", "line": 1, "reason": "x", "candidates": [],
    }
    store.conn.execute(
        "UPDATE files SET nodes_json = '[]', edges_json = ?, ignored_json = '{}'",
        (json.dumps([edge]),),
    )
    with pytest.raises(MemoryIndexError, match="inconsistent"):
        load_graph(store)