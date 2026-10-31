from graph_helpers import calls, graph_of

from core.graph.models import CALLS, UNCERTAIN


def test_same_file_call_is_confirmed(tmp_path):
    graph = graph_of(tmp_path, {
        "m.py": "def helper():\n    return 1\n\n\ndef go():\n    return helper()\n",
    })
    assert ("go", "helper", "confirmed") in calls(graph)


def test_imported_alias_is_confirmed(tmp_path):
    graph = graph_of(tmp_path, {
        "a.py": "def real():\n    return 1\n",
        "b.py": "from a import real as r\n\n\ndef go():\n    return r()\n",
    })
    assert ("go", "real", "confirmed") in calls(graph)


def test_variable_assigned_once_from_a_class_gets_its_method(tmp_path):
    graph = graph_of(tmp_path, {
        "m.py": (
            "class Box:\n    def open(self):\n        return 1\n\n\n"
            "def go():\n    b = Box()\n    return b.open()\n"
        ),
    })
    found = calls(graph)
    assert ("go", "Box", "confirmed") in found
    assert ("go", "Box.open", "confirmed") in found


def test_reassigned_variable_is_uncertain_with_candidates(tmp_path):
    graph = graph_of(tmp_path, {
        "m.py": (
            "class Box:\n    def open(self):\n        return 1\n\n\n"
            "class Other:\n    def open(self):\n        return 2\n\n\n"
            "def go(flag):\n    b = Box()\n    if flag:\n        b = Other()\n"
            "    return b.open()\n"
        ),
    })
    found = calls(graph)
    assert ("go", "b.open", "uncertain") in found
    assert ("go", "Box.open", "confirmed") not in found
    rel = next(r for r in graph.relations(CALLS, UNCERTAIN) if r.target_label == "b.open")
    assert {graph.label(c) for c in rel.candidates} == {"Box.open", "Other.open"}


def test_getattr_result_is_uncertain(tmp_path):
    graph = graph_of(tmp_path, {
        "m.py": "def go(obj, name):\n    handler = getattr(obj, name)\n    return handler()\n",
    })
    found = calls(graph)
    assert ("go", "handler", "uncertain") in found
    assert not [item for item in found if item[0] == "go" and item[2] == "confirmed"]


def test_call_of_a_parameter_is_uncertain(tmp_path):
    graph = graph_of(tmp_path, {"m.py": "def go(callback):\n    return callback(1)\n"})
    assert ("go", "callback", "uncertain") in calls(graph)


def test_unknown_receiver_lists_candidates_but_confirms_nothing(tmp_path):
    graph = graph_of(tmp_path, {
        "m.py": (
            "class A:\n    def run(self):\n        return 1\n\n\n"
            "def go(thing):\n    return thing.run()\n"
        ),
    })
    assert ("go", "thing.run", "uncertain") in calls(graph)
    (key,) = graph.find_symbols("A.run")
    assert graph.callers_of(key) == []
    assert [r.source_label for r in graph.possible_callers_of(key)] == ["go"]


def test_attribute_call_with_no_matching_repo_method_is_not_recorded(tmp_path):
    graph = graph_of(tmp_path, {
        "m.py": "def go(thing):\n    return thing.nothing_like_this()\n",
    })
    assert not [item for item in calls(graph) if item[0] == "go"]
    assert graph.stats.ignored.get(
        "attribute call on an unknown object (no repo method has that name)"
    ) == 1


def test_star_import_makes_unknown_names_uncertain(tmp_path):
    graph = graph_of(tmp_path, {
        "a.py": "def exported():\n    return 1\n",
        "b.py": "from a import *\n\n\ndef go():\n    return mystery()\n",
    })
    assert ("go", "mystery", "uncertain") in calls(graph)


def test_standard_library_calls_are_not_recorded(tmp_path):
    graph = graph_of(tmp_path, {
        "m.py": "import os\n\n\ndef go():\n    return os.path.join('a', 'b')\n",
    })
    assert not [item for item in calls(graph) if item[0] == "go"]
    assert graph.stats.ignored["external package or standard library"] == 1


def test_importlib_import_module_is_uncertain(tmp_path):
    graph = graph_of(tmp_path, {
        "m.py": "import importlib\n\n\ndef load(name):\n    return importlib.import_module(name)\n",
    })
    assert ("load", "importlib.import_module", "uncertain") in calls(graph)


def test_reexport_through_init_is_followed(tmp_path):
    graph = graph_of(tmp_path, {
        "pkg/__init__.py": "from .impl import helper\n",
        "pkg/impl.py": "def helper():\n    return 1\n",
        "main.py": "from pkg import helper\n\n\ndef go():\n    return helper()\n",
    })
    rel = next(r for r in graph.relations(CALLS) if r.source_label == "go")
    assert (rel.target, rel.certainty) == ("pkg/impl.py::helper", "confirmed")


def test_self_method_inside_a_class_is_confirmed(tmp_path):
    graph = graph_of(tmp_path, {
        "m.py": (
            "class A:\n    def a(self):\n        return self.b()\n\n"
            "    def b(self):\n        return 1\n"
        ),
    })
    assert ("A.a", "A.b", "confirmed") in calls(graph)


def test_calls_in_a_nested_function_count_for_the_enclosing_function(tmp_path):
    graph = graph_of(tmp_path, {
        "m.py": (
            "def helper():\n    return 1\n\n\n"
            "def outer():\n    def inner():\n        return helper()\n    return inner()\n"
        ),
    })
    found = calls(graph)
    assert ("outer", "helper", "confirmed") in found
    assert not [item for item in found if item[0] == "outer" and item[2] == "uncertain"]


def test_a_file_with_a_syntax_error_does_not_stop_the_graph(tmp_path):
    graph = graph_of(tmp_path, {
        "good.py": "def a():\n    return b()\n\n\ndef b():\n    return 1\n",
        "broken.py": "def oops(:\n    pass\n",
    })
    assert graph.stats.files_skipped == 1
    assert ("a", "b", "confirmed") in calls(graph)