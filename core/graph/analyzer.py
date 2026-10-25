"""Syntax-level helpers: imports, local names and call sites. Code is parsed, never run."""

from __future__ import annotations

import ast
import builtins
import warnings
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from core.repo.extractor import normalize_newlines

DEF_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
BUILTIN_NAMES = frozenset(dir(builtins))
DYNAMIC_CALLS = frozenset({"eval", "exec", "__import__"})


@dataclass(frozen=True)
class ImportStmt:
    """One import statement. `module` is absolute ('' if it cannot be worked out)."""

    line: int
    module: str
    names: tuple[tuple[str, str | None], ...] = ()  # from-imports: (name, alias)
    alias: str | None = None  # plain import: import a.b as alias
    is_from: bool = False
    star: bool = False
    unresolved_relative: bool = False


@dataclass(frozen=True)
class CallSite:
    line: int
    func: ast.expr  # the expression being called
    call: ast.Call | None = None  # None for a bare decorator such as @log
    is_decorator: bool = False


@dataclass
class LocalInfo:
    params: set[str] = field(default_factory=set)
    names: set[str] = field(default_factory=set)  # every name bound inside the function
    nested_defs: set[str] = field(default_factory=set)
    constructor_assigns: dict[str, ast.Call] = field(default_factory=dict)  # x = f(...) once
    getattr_names: set[str] = field(default_factory=set)  # x = getattr(...) once


def parse_file(root: Path, rel_path: str) -> ast.Module:
    """Parse one file the way Phase 1 did (UTF-8, normalized line endings)."""
    data = (Path(root) / rel_path).read_bytes()
    text = normalize_newlines(data.decode("utf-8-sig"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ast.parse(text, filename=rel_path)


def package_parts_for(path: str) -> tuple[str, ...]:
    """Folders above a file. They form its package, which relative imports start from."""
    return tuple(path.split("/")[:-1])


def _absolute_module(node: ast.ImportFrom, package: tuple[str, ...]) -> tuple[str, bool]:
    if node.level == 0:
        return node.module or "", True
    up = node.level - 1
    if up > len(package):
        return "", False
    base = list(package[: len(package) - up])
    if node.module:
        base += node.module.split(".")
    return ".".join(base), True


def collect_imports(tree: ast.AST, package: tuple[str, ...]) -> list[ImportStmt]:
    found: list[ImportStmt] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                found.append(ImportStmt(node.lineno, item.name, alias=item.asname))
        elif isinstance(node, ast.ImportFrom):
            module, ok = _absolute_module(node, package)
            names = tuple((a.name, a.asname) for a in node.names if a.name != "*")
            star = any(a.name == "*" for a in node.names)
            found.append(
                ImportStmt(
                    node.lineno, module, names, is_from=True, star=star,
                    unresolved_relative=not ok,
                )
            )
    found.sort(key=lambda stmt: stmt.line)
    return found


def child_blocks(node: ast.AST) -> Iterator[list[ast.stmt]]:
    """Statement lists nested in if/try/with/for/while/match (same rule as Phase 1)."""
    for attr in ("body", "orelse", "finalbody"):
        block = getattr(node, attr, None)
        if isinstance(block, list):
            yield block
    for handler in getattr(node, "handlers", []):
        yield handler.body
    for case in getattr(node, "cases", []):
        yield case.body


def iter_definitions(
    body: list[ast.stmt], prefix: str = "", in_class: bool = False
) -> Iterator[tuple[ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef, str, bool]]:
    """Yield (node, qualname, in_class) exactly like Phase 1's extractor walks definitions."""
    for node in body:
        if isinstance(node, DEF_TYPES):
            qualname = f"{prefix}{node.name}"
            yield node, qualname, in_class
            if isinstance(node, ast.ClassDef):
                yield from iter_definitions(node.body, f"{qualname}.", True)
        else:
            for block in child_blocks(node):
                yield from iter_definitions(block, prefix, in_class)


def start_line_of(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> int:
    """First line of a definition; decorators count (same as Phase 1)."""
    return min([node.lineno, *(d.lineno for d in node.decorator_list)])


def dotted_parts(expr: ast.expr) -> list[str] | None:
    """['a', 'b', 'c'] for a.b.c built only from names and attributes, else None."""
    parts: list[str] = []
    while isinstance(expr, ast.Attribute):
        parts.append(expr.attr)
        expr = expr.value
    if isinstance(expr, ast.Name):
        parts.append(expr.id)
        return parts[::-1]
    return None


def short_expr(expr: ast.AST, limit: int = 60) -> str:
    try:
        text = ast.unparse(expr)
    except Exception:  # noqa: BLE001 - a label is never worth a crash
        text = "<expression>"
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def collect_locals(func: ast.FunctionDef | ast.AsyncFunctionDef) -> LocalInfo:
    """Names bound inside a function, and which are assigned from a call exactly once."""
    info = LocalInfo()
    counts: Counter[str] = Counter()
    blocked: set[str] = set()
    assigned: dict[str, ast.Call] = {}
    arguments = func.args
    for arg in [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs]:
        info.params.add(arg.arg)
    for extra in (arguments.vararg, arguments.kwarg):
        if extra is not None:
            info.params.add(extra.arg)
    counts.update(info.params)
    for stmt in func.body:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store | ast.Del):
                counts[node.id] += 1
            elif isinstance(node, DEF_TYPES):
                counts[node.name] += 1
                info.nested_defs.add(node.name)
            elif isinstance(node, ast.Global | ast.Nonlocal):
                blocked.update(node.names)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                counts[node.name] += 1
            elif isinstance(node, ast.Import | ast.ImportFrom):
                for alias in node.names:
                    counts[alias.asname or alias.name.split(".")[0]] += 1
            elif isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and isinstance(node.value, ast.Call):
                    assigned[target.id] = node.value
            elif isinstance(node, ast.AnnAssign) and isinstance(node.value, ast.Call):
                if isinstance(node.target, ast.Name):
                    assigned[node.target.id] = node.value
    info.names = set(counts)
    for name, call in assigned.items():
        if counts[name] != 1 or name in blocked:
            continue
        if isinstance(call.func, ast.Name) and call.func.id == "getattr":
            info.getattr_names.add(name)
        else:
            info.constructor_assigns[name] = call
    return info


def module_assigned_names(body: list[ast.stmt]) -> set[str]:
    """Names assigned at module level (not inside any function or class)."""
    names: set[str] = set()
    stack: list[ast.AST] = list(body)
    while stack:
        node = stack.pop()
        if isinstance(node, DEF_TYPES):
            continue
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store | ast.Del):
            names.add(node.id)
        stack.extend(ast.iter_child_nodes(node))
    return names


def iter_calls(nodes: list[ast.stmt], enter_defs: bool) -> Iterator[CallSite]:
    """Call sites in a scope.

    enter_defs=True (function scope): nested functions and lambdas count for this scope.
    enter_defs=False (module or class scope): bodies of definitions are separate scopes,
    but their decorators are evaluated here, so those count for this scope.
    """
    stack: list[ast.AST] = list(reversed(nodes))
    while stack:
        node = stack.pop()
        if isinstance(node, DEF_TYPES):
            for deco in node.decorator_list:
                if not isinstance(deco, ast.Call):  # @name or @a.b: applying it is a call
                    yield CallSite(deco.lineno, deco, None, is_decorator=True)
            if not enter_defs:
                stack.extend(reversed(node.decorator_list))
                continue
        elif isinstance(node, ast.Call):
            yield CallSite(node.lineno, node.func, node)
        stack.extend(reversed(list(ast.iter_child_nodes(node))))