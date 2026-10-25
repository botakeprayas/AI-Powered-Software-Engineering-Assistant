"""Name and call resolution. Anything that cannot be proven from the text stays uncertain."""

from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass, field

from core.graph.analyzer import (
    BUILTIN_NAMES,
    DYNAMIC_CALLS,
    CallSite,
    ImportStmt,
    LocalInfo,
    dotted_parts,
    short_expr,
)
from core.graph.models import CONFIRMED, MAX_CANDIDATES, UNCERTAIN
from core.semantic.models import Symbol

IGNORED = "ignored"
METHOD_KINDS = ("method", "async_method")
DYNAMIC_EXTERNALS = frozenset({"importlib.import_module", "importlib.__import__"})
MAX_CHAIN_DEPTH = 3


@dataclass(frozen=True)
class Resolution:
    """Result of resolving one call. For 'ignored', `reason` is a short category."""

    status: str  # confirmed | uncertain | ignored
    target: str = ""  # symbol key (confirmed only)
    reason: str = ""
    candidates: tuple[str, ...] = ()  # possible targets of an uncertain call (not confirmed)
    label: str = ""  # call expression text (uncertain only)


def _ok(target: str, reason: str) -> Resolution:
    return Resolution(CONFIRMED, target=target, reason=reason)


def _unsure(reason: str, label: str, candidates: tuple[str, ...] = ()) -> Resolution:
    return Resolution(UNCERTAIN, reason=reason, candidates=candidates, label=label)


def _ignore(category: str) -> Resolution:
    return Resolution(IGNORED, reason=category)


@dataclass(frozen=True)
class Binding:
    kind: str  # symbol | module | external | local | builtin | unknown | missing
    target: str = ""  # symbol key, dotted module name, or external dotted name
    reason: str = ""
    candidates: tuple[str, ...] = ()


@dataclass(frozen=True)
class ImportResult:
    line: int
    edge_kind: str  # file | symbol | external | unresolved | none
    edge_target: str
    certainty: str
    reason: str
    bind_name: str | None = None
    binding: Binding | None = None
    star_module: str | None = None
    candidates: tuple[str, ...] = ()


@dataclass
class FileScope:
    bindings: dict[str, Binding] = field(default_factory=dict)
    conflicts: set[str] = field(default_factory=set)
    star_modules: list[str] = field(default_factory=list)
    results: list[ImportResult] = field(default_factory=list)


@dataclass(frozen=True)
class CallContext:
    path: str
    loc: LocalInfo | None = None
    class_key: str | None = None  # set inside methods, so self.x() can be resolved


def module_names(path: str) -> list[str]:
    """Dotted module names a file can be imported as ('' roots and src/ handled)."""
    stem = path[:-3] if path.endswith(".py") else path
    parts = stem.split("/")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    if not parts:
        return []
    names = [".".join(parts)]
    if parts[0] == "src" and len(parts) > 1:
        names.append(".".join(parts[1:]))
    return names


def package_names(path: str) -> list[str]:
    """Dotted names of the folders above a file (namespace packages included)."""
    parts = path.split("/")[:-1]
    names = [".".join(parts[: i + 1]) for i in range(len(parts))]
    if parts and parts[0] == "src":
        names += [".".join(parts[1 : i + 1]) for i in range(1, len(parts))]
    return names


def _pick_parent(options: list[Symbol], child: Symbol) -> Symbol | None:
    for option in options:
        if option.start_line <= child.start_line and child.end_line <= option.end_line:
            return option
    return None


class ModuleIndex:
    """Tables built from the Phase 1 symbols, plus import resolution and call resolution."""

    def __init__(self, symbols: list[Symbol], python_files: list[str]) -> None:
        self.symbols: dict[str, Symbol] = {s.key: s for s in symbols}
        self.module_files: dict[str, list[str]] = {}
        self.packages: set[str] = set()
        for path in python_files:
            for name in module_names(path):
                files = self.module_files.setdefault(name, [])
                if path not in files:
                    files.append(path)
            self.packages.update(package_names(path))
        self.top: dict[str, dict[str, list[str]]] = {}
        self.class_methods: dict[str, dict[str, list[str]]] = {}
        self.method_index: dict[str, list[str]] = {}
        self.parent_of: dict[str, str] = {}
        by_file_qual: dict[tuple[str, str], list[Symbol]] = defaultdict(list)
        for s in symbols:
            by_file_qual[(s.file, s.qualname)].append(s)
        for s in symbols:
            if "." not in s.qualname:
                self.top.setdefault(s.file, {}).setdefault(s.name, []).append(s.key)
                continue
            parent_qual = s.qualname.rsplit(".", 1)[0]
            parent = _pick_parent(by_file_qual.get((s.file, parent_qual), []), s)
            if parent is None:
                continue
            self.parent_of[s.key] = parent.key
            self.class_methods.setdefault(parent.key, {}).setdefault(s.name, []).append(s.key)
            if s.kind in METHOD_KINDS:
                self.method_index.setdefault(s.name, []).append(s.key)
        self._imports: dict[str, list[ImportStmt]] = {}
        self._assigned: dict[str, set[str]] = {}
        self._scopes: dict[str, FileScope] = {}
        self._building: set[str] = set()

    def register_file(self, path: str, imports: list[ImportStmt], assigned: set[str]) -> None:
        self._imports[path] = imports
        self._assigned[path] = assigned

    def is_repo_module(self, name: str) -> bool:
        return name in self.module_files or name in self.packages

    # ---- imports -----------------------------------------------------------------
    def scope(self, path: str) -> FileScope:
        """Import bindings of one file (memoized; an import cycle yields an empty scope)."""
        if path in self._scopes:
            return self._scopes[path]
        if path in self._building:
            return FileScope()
        self._building.add(path)
        try:
            scope = FileScope()
            for stmt in self._imports.get(path, []):
                for result in self._resolve_import(path, stmt):
                    scope.results.append(result)
                    if result.star_module:
                        scope.star_modules.append(result.star_module)
                    if result.bind_name and result.binding is not None:
                        old = scope.bindings.get(result.bind_name)
                        if old is not None and old != result.binding:
                            scope.conflicts.add(result.bind_name)
                        scope.bindings[result.bind_name] = result.binding
            self._scopes[path] = scope
            return scope
        finally:
            self._building.discard(path)

    def _ambiguous(
        self, line: int, module: str, files: list[str], bind: str | None = None
    ) -> ImportResult:
        reason = f"several files match module '{module}'"
        return ImportResult(
            line, "unresolved", f"unresolved:{module}", UNCERTAIN, reason, bind,
            Binding("unknown", "", reason), candidates=tuple(files[:MAX_CANDIDATES]),
        )

    def _unresolved_name(
        self, line: int, module: str, name: str, bind: str, reason: str,
        candidates: tuple[str, ...],
    ) -> ImportResult:
        return ImportResult(
            line, "unresolved", f"unresolved:{module}.{name}", UNCERTAIN, reason, bind,
            Binding("unknown", "", reason, candidates), candidates=candidates,
        )

    def _plain_import(self, stmt: ImportStmt, bind: str, bound: str) -> ImportResult:
        name = stmt.module
        reason = f"import {name}"
        files = self.module_files.get(name, [])
        kind = "module" if self.is_repo_module(bound) else "external"
        binding = Binding(kind, bound, reason)
        if len(files) == 1:
            return ImportResult(stmt.line, "file", files[0], CONFIRMED, reason, bind, binding)
        if len(files) > 1:
            return self._ambiguous(stmt.line, name, files, bind)
        if self.is_repo_module(name):  # a package folder without its own file
            return ImportResult(stmt.line, "none", "", CONFIRMED, reason, bind, binding)
        return ImportResult(stmt.line, "external", name, CONFIRMED, reason, bind, binding)

    def _resolve_import(self, path: str, stmt: ImportStmt) -> list[ImportResult]:
        if not stmt.is_from:
            bound = stmt.module if stmt.alias else stmt.module.split(".")[0]
            return [self._plain_import(stmt, stmt.alias or bound, bound)]
        if stmt.unresolved_relative or not stmt.module:
            return [
                ImportResult(
                    stmt.line, "unresolved", f"unresolved:relative import in {path}:{stmt.line}",
                    UNCERTAIN, "relative import that cannot be placed inside the repository",
                )
            ]
        module = stmt.module
        files = self.module_files.get(module, [])
        names = ", ".join(n for n, _ in stmt.names) or "*"
        if not files and module not in self.packages:
            out = [ImportResult(stmt.line, "external", module, CONFIRMED,
                                f"from {module} import {names}")]
            for name, alias in stmt.names:
                binding = Binding("external", f"{module}.{name}", f"from {module} import {name}")
                out.append(
                    ImportResult(stmt.line, "none", "", CONFIRMED, "", alias or name, binding)
                )
            return out
        if len(files) > 1:
            return [self._ambiguous(stmt.line, module, files)]
        file = files[0] if files else None
        out = []
        if file is not None:
            out.append(ImportResult(stmt.line, "file", file, CONFIRMED,
                                    f"from {module} import {names}"))
        if stmt.star:
            out.append(ImportResult(stmt.line, "none", "", CONFIRMED, "", star_module=module))
        for name, alias in stmt.names:
            out.append(self._from_name_result(stmt.line, module, file, name, alias or name))
        return out

    def _from_name_result(
        self, line: int, module: str, file: str | None, name: str, bind: str
    ) -> ImportResult:
        reason = f"from {module} import {name}"
        if file is not None:
            keys = self.top.get(file, {}).get(name, [])
            if len(keys) == 1:
                binding = Binding("symbol", keys[0], reason)
                return ImportResult(line, "symbol", keys[0], CONFIRMED, reason, bind, binding)
            if len(keys) > 1:
                return self._unresolved_name(
                    line, module, name, bind, f"{reason}: defined {len(keys)} times in {file}",
                    tuple(keys[:MAX_CANDIDATES]),
                )
            inner = self.scope(file)
            reexport = inner.bindings.get(name)
            if reexport is not None and name not in inner.conflicts:
                if reexport.kind == "symbol":
                    note = f"{reason} (re-exported by {file})"
                    return ImportResult(
                        line, "symbol", reexport.target, CONFIRMED, note, bind, reexport
                    )
                return ImportResult(line, "none", "", CONFIRMED, reason, bind, reexport)
        sub = f"{module}.{name}"
        sub_files = self.module_files.get(sub, [])
        if len(sub_files) == 1:
            binding = Binding("module", sub, reason)
            note = f"{reason} (submodule)"
            return ImportResult(line, "file", sub_files[0], CONFIRMED, note, bind, binding)
        if len(sub_files) > 1:
            return self._ambiguous(line, sub, sub_files, bind)
        if sub in self.packages:
            return ImportResult(line, "none", "", CONFIRMED, reason, bind,
                                Binding("module", sub, reason))
        return self._unresolved_name(
            line, module, name, bind,
            f"{reason}: name not found in the repository module (may be dynamic)", (),
        )

    # ---- names ---------------------------------------------------------------------
    def resolve_name(self, path: str, name: str, loc: LocalInfo | None) -> Binding:
        if loc is not None and name in loc.names:
            if name in loc.nested_defs:
                return Binding("local", name, "nested function")
            if name in loc.params:
                return Binding("local", name, "parameter")
            if name in loc.getattr_names:
                return Binding("local", name, "getattr result")
            return Binding("local", name, "local variable")
        scope = self.scope(path)
        keys = self.top.get(path, {}).get(name, [])
        imported = scope.bindings.get(name)
        assigned = name in self._assigned.get(path, set())
        if name in scope.conflicts or sum(map(bool, (keys, imported, assigned))) > 1:
            return Binding("unknown", name, "name is bound more than once in this file",
                           tuple(keys[:MAX_CANDIDATES]))
        if keys:
            if len(keys) == 1:
                return Binding("symbol", keys[0], "defined in this file")
            return Binding("unknown", name, f"name is defined {len(keys)} times in this file",
                           tuple(keys[:MAX_CANDIDATES]))
        if imported is not None:
            return imported
        if assigned:
            return Binding("local", name, "module variable")
        if name in BUILTIN_NAMES:
            return Binding("builtin", name)
        if scope.star_modules:
            mods = ", ".join(scope.star_modules)
            return Binding("unknown", name, f"name may come from a star import of {mods}")
        return Binding("missing", name)

    # ---- calls ---------------------------------------------------------------------
    def resolve_call(self, ctx: CallContext, site: CallSite, _depth: int = 0) -> Resolution:
        func = site.func
        label = short_expr(func)
        if _depth > MAX_CHAIN_DEPTH:
            return _unsure("chain of assignments too deep to follow", label)
        if isinstance(func, ast.Name):
            binding = self.resolve_name(ctx.path, func.id, ctx.loc)
            return self._call_of_binding(binding, label, func.id)
        if isinstance(func, ast.Attribute):
            return self._resolve_attribute(ctx, func, label, _depth)
        if (
            isinstance(func, ast.Call)
            and isinstance(func.func, ast.Name)
            and func.func.id == "getattr"
        ):
            return _unsure("getattr(...) returns a function chosen at runtime", label)
        return _unsure("the called expression's target cannot be determined statically", label)

    def _call_of_binding(self, binding: Binding, label: str, shown: str) -> Resolution:
        kind = binding.kind
        if kind == "symbol":
            return _ok(binding.target, binding.reason or "resolved name")
        if kind == "builtin":
            if shown in DYNAMIC_CALLS:
                return _unsure(f"{shown}() runs or imports code chosen at runtime", label)
            return _ignore("builtin")
        if kind == "external":
            if binding.target in DYNAMIC_EXTERNALS:
                return _unsure(f"{binding.target}() imports a module chosen at runtime", label)
            return _ignore("external package or standard library")
        if kind == "module":
            return _ignore("module object used as a call")
        if kind == "local":
            return self._local_call(binding, label)
        if kind == "unknown":
            return _unsure(binding.reason, label, binding.candidates)
        return _ignore("name not defined in the repository")

    def _local_call(self, binding: Binding, label: str) -> Resolution:
        if binding.reason == "nested function":
            return _ignore("nested function (not a graph node)")
        notes = {
            "parameter": "call of a parameter: the caller decides the target at runtime",
            "getattr result": "result of getattr() is called: the target is chosen at runtime",
            "local variable": "call of a local variable whose value is not tracked",
            "module variable": "call of a module-level variable whose value is not tracked",
        }
        return _unsure(notes.get(binding.reason, binding.reason), label)

    def _resolve_attribute(
        self, ctx: CallContext, func: ast.Attribute, label: str, depth: int
    ) -> Resolution:
        attr = func.attr
        parts = dotted_parts(func)
        if parts is None:  # the receiver is an expression: a call result, a subscript, ...
            value = func.value
            if (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id == "super"
            ):
                why = "super() depends on inheritance, which is not resolved"
                return self._by_method_name(attr, label, why)
                why = "the receiver is an expression of unknown type"
                return self._by_method_name(attr, label, why)
        root, rest = parts[0], parts[1:]
        if root in ("self", "cls") and ctx.class_key is not None:
            class_name = self.symbols[ctx.class_key].qualname
            if len(rest) == 1:
                methods = self.class_methods.get(ctx.class_key, {}).get(attr, [])
                if len(methods) == 1:
                    return _ok(methods[0], f"{root}.{attr}() inside class {class_name}")
                if len(methods) > 1:
                    return _unsure(f"class {class_name} defines '{attr}' more than once",
                                   label, tuple(methods[:MAX_CANDIDATES]))
                why = f"'{attr}' is not defined in class {class_name} (inherited or dynamic)"
                return self._by_method_name(attr, label, why)
            why = f"receiver '{'.'.join(parts[:-1])}' has unknown type (attributes are not tracked)"
            return self._by_method_name(attr, label, why)
        binding = self.resolve_name(ctx.path, root, ctx.loc)
        return self._attribute_of_binding(ctx, binding, root, rest, label, depth)

    def _attribute_of_binding(
        self, ctx: CallContext, binding: Binding, root: str, rest: list[str], label: str,
        depth: int,
    ) -> Resolution:
        kind = binding.kind
        attr = rest[-1]
        if kind == "symbol":
            if self.symbols[binding.target].kind == "class":
                return self._symbol_with_attrs(binding.target, rest, label, binding.reason)
            return _ignore("attribute of a function object")
        if kind == "module":
            return self._in_module(binding.target, rest, label)
        if kind == "external":
            full = ".".join([binding.target, *rest])
            if full in DYNAMIC_EXTERNALS:
                return _unsure(f"{full}() imports a module chosen at runtime", label)
            return _ignore("external package or standard library")
        if kind == "builtin":
            return _ignore("builtin")
        if kind == "local":
            ctor = ctx.loc.constructor_assigns.get(root) if ctx.loc else None
            if ctor is not None and binding.reason == "local variable":
                made = self.resolve_call(ctx, CallSite(ctor.lineno, ctor.func, ctor), depth + 1)
                made_symbol = self.symbols.get(made.target) if made.status == CONFIRMED else None
                if made_symbol is not None and made_symbol.kind == "class":
                    why = (
                        f"{root} = {made_symbol.qualname}(...) is assigned once "
                        f"(line {ctor.lineno})"
                    )
                    return self._symbol_with_attrs(made_symbol.key, rest, label, why)
            return self._by_method_name(
                attr, label, f"receiver '{root}' is a {binding.reason} of unknown type"
            )
        if kind == "unknown":
            return self._by_method_name(attr, label, binding.reason)
        return self._by_method_name(attr, label, f"receiver '{root}' is not defined in the repo")

    def _symbol_with_attrs(
        self, key: str, attrs: list[str], label: str, reason: str
    ) -> Resolution:
        if not attrs:
            return _ok(key, reason)
        symbol = self.symbols[key]
        if symbol.kind == "class" and len(attrs) == 1:
            methods = self.class_methods.get(key, {}).get(attrs[0], [])
            if len(methods) == 1:
                return _ok(methods[0], f"{reason}; method of class {symbol.qualname}")
        why = f"{reason}; '{'.'.join(attrs)}' is not resolved"
        return self._by_method_name(attrs[-1], label, why)

    def _in_module(self, module: str, rest: list[str], label: str) -> Resolution:
        current = module
        index = 0
        while index < len(rest) and self.is_repo_module(f"{current}.{rest[index]}"):
            current = f"{current}.{rest[index]}"
            index += 1
        remaining = rest[index:]
        if not remaining:
            return _ignore("module object used as a call")
        files = self.module_files.get(current, [])
        if len(files) > 1:
            return _unsure(f"several files match module '{current}'", label,
                           tuple(files[:MAX_CANDIDATES]))
        if not files:
            return _ignore("name inside a package folder without a module file")
        file = files[0]
        name = remaining[0]
        reason = f"{label} resolved through module {current}"
        keys = self.top.get(file, {}).get(name, [])
        if len(keys) == 1:
            return self._symbol_with_attrs(keys[0], remaining[1:], label, reason)
        if len(keys) > 1:
            return _unsure(f"'{name}' is defined {len(keys)} times in {file}", label,
                           tuple(keys[:MAX_CANDIDATES]))
        scope = self.scope(file)
        reexport = scope.bindings.get(name)
        if reexport is not None and name not in scope.conflicts and reexport.kind == "symbol":
            return self._symbol_with_attrs(
                reexport.target, remaining[1:], label, f"{reason} (re-exported)"
            )
        return _unsure(f"'{name}' was not found in {file} (may be dynamic)", label)

    def _by_method_name(self, attr: str, label: str, why: str) -> Resolution:
        """Receiver type unknown: list repo methods with that name as CANDIDATES only."""
        keys = self.method_index.get(attr, [])
        if not keys:
            return _ignore("attribute call on an unknown object (no repo method has that name)")
        reason = f"{why}; {len(keys)} repo method(s) named '{attr}'"
        return _unsure(reason, label, tuple(keys[:MAX_CANDIDATES]))