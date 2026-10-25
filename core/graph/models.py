"""Constants and plain data types for the dependency graph."""

from __future__ import annotations

from dataclasses import dataclass, field

IMPORTS = "imports"
DEFINES = "defines"
CALLS = "calls"
EDGE_KINDS = (IMPORTS, DEFINES, CALLS)

CONFIRMED = "confirmed"
UNCERTAIN = "uncertain"

NODE_FILE = "file"
NODE_MODULE = "module"
NODE_FUNCTION = "function"
NODE_CLASS = "class"
NODE_UNRESOLVED = "unresolved"

MAX_CANDIDATES = 5

# What static analysis cannot resolve. Shown in the UI and the CLI.
LIMITATIONS = (
    "Python only. Code is parsed with ast and never executed.",
    "Calls on objects of unknown type (parameters, attributes, results) are uncertain.",
    "Dynamic code is uncertain: getattr, eval, exec, importlib, __import__, callbacks.",
    "Inheritance is not resolved: a method missing from its class is uncertain.",
    "Only x = ClassName(...) assigned once in a function gives x a known type.",
    "Attributes set in __init__ (self.gw = Gateway()) are not tracked.",
    "A star import (from x import *) makes unknown names uncertain.",
    "Imports inside functions count file-wide; a name bound two ways is uncertain.",
    "Calls into the standard library, packages and builtins are not recorded.",
    "Nested functions are not nodes; their calls count for the enclosing function.",
    "Decorators are recorded as calls from the scope that defines the function.",
    "Calls made by frameworks, threads or callbacks cannot be seen.",
    "Only the repo root and a src/ folder are import roots; others look external.",
    "A missing edge is not proof that no call exists.",
)


@dataclass(frozen=True)
class Relation:
    """One edge, flattened for display and lookups."""

    kind: str
    certainty: str
    source: str
    target: str
    source_label: str
    target_label: str
    file: str  # file where the relationship appears
    line: int  # 1-based line in that file
    reason: str  # how it was resolved (or why it could not be)
    candidates: tuple[str, ...] = ()  # possible targets of an uncertain call (not confirmed)

    @property
    def location(self) -> str:
        return f"{self.file}:{self.line}"


@dataclass
class GraphStats:
    nodes: dict[str, int] = field(default_factory=dict)
    edges: dict[str, int] = field(default_factory=dict)  # "calls/confirmed" -> count
    ignored: dict[str, int] = field(default_factory=dict)  # calls not recorded, by reason
    files_analyzed: int = 0
    files_skipped: int = 0