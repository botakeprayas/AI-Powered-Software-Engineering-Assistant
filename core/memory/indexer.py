"""Incremental indexer: update the memory store from the working tree and reprocess only
what changed (Phase 8).

Symbols and embeddings are never recomputed for an unchanged file. Graph edges are reused
per file only while no changed file's interface (definitions, imports, module variables)
differs and no file was added, deleted or renamed. Otherwise every file's edges are
re-resolved, because an unchanged file's edges can depend on other files. The report says
which case applied and why. Nothing is written unless the whole update succeeds.
"""

from __future__ import annotations

import ast
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from core.graph.analyzer import (
    ImportStmt,
    collect_imports,
    module_assigned_names,
    package_parts_for,
    parse_file,
)
from core.graph.expand import GraphContext
from core.graph.graph import DependencyGraph
from core.memory.changes import detect_changes, git_renames
from core.memory.gitrepo import GitRepo
from core.memory.graphparts import (
    FileGraph,
    MemoryIndexError,
    assemble_graph,
    compute_file_graphs,
    imports_from_json,
    imports_to_json,
    interface_hash,
)
from core.memory.hashing import tree_digest
from core.memory.manifest import build_manifest
from core.memory.models import FileChanges, FileState, VersionRecord
from core.memory.store import MemoryStore
from core.repo.extractor import extract_file
from core.repo.models import FileIssue, RepoFile
from core.semantic.config import default_db_path
from core.semantic.engine import CodeSearchEngine
from core.semantic.models import Symbol, SyncStats
from services.embedding_service import EmbeddingError, EmbeddingService

logger = logging.getLogger(__name__)

GRAPH_NONE = "none"
GRAPH_PARTIAL = "partial"
GRAPH_FULL = "full"
PARSE_ERRORS = (OSError, ValueError, SyntaxError, RecursionError)
INSERT_SYMBOL = (
    "INSERT INTO symbols (symbol_key, file, kind, name, qualname, start_line, end_line, "
    "docstring, code) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
)


def scan_order_key(path: str) -> tuple[tuple[int, str], ...]:
    """Sort key that reproduces the Phase 1 scanner's order: in each folder the files come
    first (A-Z), then the sub-folders (A-Z), each walked the same way."""
    parts = path.split("/")
    return tuple((1, part) for part in parts[:-1]) + ((0, parts[-1]),)


# ---- reading the stored index (no file is read or parsed) -------------------------------


def load_symbols(store: MemoryStore) -> list[Symbol]:
    """Every symbol, in the same order a from-scratch Phase 1 scan produces."""
    by_file: dict[str, list[Symbol]] = {}
    for row in store.conn.execute("SELECT * FROM symbols ORDER BY rowid"):
        symbol = Symbol(
            key=row["symbol_key"], file=row["file"], kind=row["kind"], name=row["name"],
            qualname=row["qualname"], start_line=row["start_line"], end_line=row["end_line"],
            docstring=row["docstring"], code=row["code"],
        )
        by_file.setdefault(symbol.file, []).append(symbol)
    return [s for path in sorted(by_file, key=scan_order_key) for s in by_file[path]]


def load_statuses(store: MemoryStore) -> dict[str, str | None]:
    rows = store.conn.execute("SELECT path, status FROM files ORDER BY path")
    return {row["path"]: row["status"] for row in rows}


def load_issues(store: MemoryStore) -> list[FileIssue]:
    found: list[FileIssue] = []
    rows = store.conn.execute(
        "SELECT path, issues_json FROM files WHERE issues_json IS NOT NULL ORDER BY path"
    )
    for row in rows:
        found += [FileIssue(row["path"], message) for message in json.loads(row["issues_json"])]
    return found


def load_graph(store: MemoryStore) -> DependencyGraph:
    """Assemble the dependency graph from stored per-file data. No file is parsed."""
    rows = [
        row
        for row in store.conn.execute(
            "SELECT path, imports_json, nodes_json, edges_json, ignored_json "
            "FROM files ORDER BY path"
        )
        if row["path"].endswith(".py")
    ]
    parts: dict[str, FileGraph] = {}
    for row in rows:
        if row["nodes_json"] is None or row["edges_json"] is None:
            raise MemoryIndexError(f"graph data is missing for {row['path']}")
        parts[row["path"]] = FileGraph(
            json.loads(row["nodes_json"]),
            json.loads(row["edges_json"]),
            json.loads(row["ignored_json"] or "{}"),
        )
    analyzed = sum(1 for row in rows if row["imports_json"] is not None)
    try:
        return assemble_graph(parts, analyzed, len(rows) - analyzed)
    except KeyError as exc:
        raise MemoryIndexError(f"the stored graph is inconsistent: {exc}") from exc


def load_graph_context(store: MemoryStore, root: Path) -> GraphContext:
    """A Phase 4 GraphContext built from the stored index."""
    symbols = {symbol.key: symbol for symbol in load_symbols(store)}
    return GraphContext(load_graph(store), symbols, Path(root).resolve())


class StoredSymbolSource:
    """A Phase 2 symbol source backed by the memory store: no file is read or parsed."""

    def __init__(self, store: MemoryStore) -> None:
        self._store = store
        self.errors: list[str] = []

    def load(self) -> list[Symbol]:
        self.errors = [f"{i.path}: {i.message}" for i in load_issues(self._store)]
        return load_symbols(self._store)


def semantic_sync(
    store: MemoryStore, embedder: EmbeddingService, db_path: Path
) -> tuple[CodeSearchEngine, SyncStats]:
    """Sync the Phase 2 engine from stored symbols. Unchanged symbols reuse their vectors
    (content-hash cache); only new or changed ones are embedded. The caller closes the engine."""
    engine = CodeSearchEngine(StoredSymbolSource(store), embedder, Path(db_path))
    try:
        stats = engine.sync()
    except Exception:
        engine.close()
        raise
    return engine, stats


# ---- the update report ----------------------------------------------------------------


@dataclass
class IndexUpdate:
    """What one update did. Everything here is a count of work that really happened."""

    changes: FileChanges
    first_index: bool = False
    created_version: bool = False
    version: VersionRecord | None = None
    files_extracted: list[str] = field(default_factory=list)
    files_reused: int = 0
    symbols_extracted: int = 0
    symbols_total: int = 0
    graph_mode: str = GRAPH_NONE
    graph_reason: str = ""
    graph_files_recomputed: list[str] = field(default_factory=list)
    graph_files_parsed: int = 0  # ASTs parsed for the graph in this update
    graph_files_reparsed: int = 0  # ... of which belong to unchanged files
    semantic: SyncStats | None = None
    engine: CodeSearchEngine | None = None
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    seconds: float = 0.0

    def describe(self) -> str:
        lines: list[str] = []
        if self.version is not None:
            state = "new version" if self.created_version else "same version"
            lines.append(f"Indexed version {self.version.label} ({state})")
        lines.append(f"Files: {self.changes.summary()}")
        lines.append(
            f"Symbols: re-extracted {self.symbols_extracted} from {len(self.files_extracted)} "
            f"file(s); {self.files_reused} unchanged file(s) reused; "
            f"{self.symbols_total} in total"
        )
        if self.semantic is not None:
            sem = self.semantic
            lines.append(
                f"Embeddings: {sem.embedded} computed, {sem.reused} reused, "
                f"{sem.removed} removed"
            )
        lines.append(f"Graph: {self.graph_mode} - {self.graph_reason}")
        if self.graph_mode != GRAPH_NONE:
            lines.append(
                f"  recomputed {len(self.graph_files_recomputed)} file(s); parsed "
                f"{self.graph_files_parsed} for the graph "
                f"({self.graph_files_reparsed} unchanged file(s) re-parsed)"
            )
        lines += [f"Note: {note}" for note in self.notes]
        lines += [f"Warning: {warning}" for warning in self.warnings]
        return "\n".join(lines)


@dataclass
class _FileData:
    state: FileState
    status: str
    issues: list[str]
    symbols: list[Symbol]
    tree: ast.Module | None
    imports: list[ImportStmt] | None
    assigned: set[str] | None
    interface: str


class IncrementalIndexer:
    def __init__(self, root: Path, store: MemoryStore, git: GitRepo | None = None) -> None:
        self.root = Path(root).resolve()
        self.store = store
        self.git = git if git is not None else GitRepo(self.root)

    def update(
        self,
        embedder: EmbeddingService | None = None,
        semantic_db: Path | None = None,
        keep_engine: bool = False,
        force_full: bool = False,
        note: str = "",
    ) -> IndexUpdate:
        """Bring the memory store up to date with the working tree.

        Raises on unexpected errors, and in that case NOTHING is written. An embedding
        failure is reported as a warning: the memory store stays updated, and the
        semantic version marker is not advanced.
        """
        started = time.perf_counter()
        store = self.store
        manifest = build_manifest(self.root)
        current = manifest.files
        stored = store.manifest()
        previous = store.current_version()
        git_ok = self.git.available()
        head = self.git.head() if git_ok else None
        branch = self.git.branch() if git_ok else None
        dirty = bool(self.git.is_dirty()) if git_ok else False
        digest = tree_digest(current)
        hints, hint_note = self._rename_hints(previous)
        changes = detect_changes(stored, current, hints)
        result = IndexUpdate(
            changes=changes, first_index=not stored, warnings=list(manifest.warnings)
        )
        if hint_note and changes.added and changes.deleted:
            result.notes.append(hint_note)
        same = (
            previous is not None
            and previous.commit_hash == head
            and previous.dirty == dirty
            and previous.tree_digest == digest
        )
        if same and not changes.has_changes and not force_full:
            result.version = previous
            result.files_reused = len(changes.unchanged)
            result.symbols_total = previous.symbol_count
            result.notes.append(
                "Nothing changed since the indexed version, so nothing was reprocessed."
            )
        else:
            with store.transaction():
                version_id = store.add_version(
                    head, dirty, digest, branch, parent_id=previous.id if previous else None
                )
                self._apply(changes, current, version_id, result, force_full, note)
                store.set_current_version(version_id)
            result.created_version = True
            result.version = store.get_version(version_id)
        self._semantic(result, embedder, semantic_db, keep_engine)
        result.seconds = time.perf_counter() - started
        return result

    # ---- steps ---------------------------------------------------------------------

    def _rename_hints(self, previous: VersionRecord | None):
        if previous is None or not previous.commit_hash:
            return [], ""
        return git_renames(self.git, previous.commit_hash)

    def _extract(self, state: FileState) -> _FileData:
        """Read and parse ONE changed file (symbols, imports, module variables)."""
        extraction = extract_file(self.root, RepoFile(state.path, state.size, True))
        tree = None
        imports = None
        assigned = None
        if extraction.status == "parsed":
            try:
                tree = parse_file(self.root, state.path)
            except PARSE_ERRORS:
                tree = None
            else:
                imports = collect_imports(tree, package_parts_for(state.path))
                assigned = module_assigned_names(tree.body)
        symbols = list(extraction.symbols)
        return _FileData(
            state, extraction.status, [issue.message for issue in extraction.issues],
            symbols, tree, imports, assigned,
            interface_hash(extraction.status, symbols, imports, assigned),
        )

    def _store_extracted(self, data: dict[str, _FileData]) -> None:
        conn = self.store.conn
        for path, item in data.items():
            conn.execute("DELETE FROM symbols WHERE file = ?", (path,))
            conn.executemany(
                INSERT_SYMBOL,
                [
                    (s.key, s.file, s.kind, s.name, s.qualname, s.start_line, s.end_line,
                     s.docstring, s.code)
                    for s in item.symbols
                ],
            )
            imports = None if item.imports is None else imports_to_json(item.imports)
            assigned = None if item.assigned is None else json.dumps(sorted(item.assigned))
            conn.execute(
                "UPDATE files SET status = ?, issues_json = ?, interface_hash = ?, "
                "imports_json = ?, assigned_json = ? WHERE path = ?",
                (item.status, json.dumps(item.issues), item.interface, imports, assigned, path),
            )

    def _choose_mode(
        self,
        changes: FileChanges,
        data: dict[str, _FileData],
        old_interface: dict[str, str | None],
        first: bool,
        force_full: bool,
    ) -> tuple[str, str]:
        if first:
            return GRAPH_FULL, "this is the first index of the repository"
        if force_full:
            return GRAPH_FULL, "a full graph recompute was requested"
        if changes.added or changes.deleted or changes.renamed:
            return GRAPH_FULL, (
                "files were added, deleted or renamed, which can change how imports and "
                "names resolve in other files"
            )
        moved = sorted(p for p in changes.modified if data[p].interface != old_interface.get(p))
        if moved:
            names = ", ".join(moved[:3]) + (" ..." if len(moved) > 3 else "")
            return GRAPH_FULL, (
                f"the definitions or imports of {len(moved)} file(s) changed ({names}), "
                "which can change how other files resolve calls"
            )
        if changes.modified:
            return GRAPH_PARTIAL, (
                "only function bodies changed, so only the changed files' graph data "
                "was recomputed"
            )
        return GRAPH_NONE, "no Python file changed"

    def _graph_inputs(self):
        rows = self.store.conn.execute(
            "SELECT path, imports_json, assigned_json FROM files"
        ).fetchall()
        python_files = sorted(row["path"] for row in rows if row["path"].endswith(".py"))
        imports: dict[str, list[ImportStmt]] = {}
        assigned: dict[str, set[str]] = {}
        for row in rows:
            if row["imports_json"] is None or not row["path"].endswith(".py"):
                continue
            imports[row["path"]] = imports_from_json(row["imports_json"])
            assigned[row["path"]] = set(json.loads(row["assigned_json"] or "[]"))
        return python_files, imports, assigned

    def _graph_step(self, mode: str, data: dict[str, _FileData], result: IndexUpdate) -> None:
        """Recompute and store graph data for the files the mode requires."""
        symbols = load_symbols(self.store)
        python_files, imports, assigned = self._graph_inputs()
        targets = python_files if mode == GRAPH_FULL else sorted(data)
        trees = {path: item.tree for path, item in data.items() if item.tree is not None}
        reparsed = 0
        for path in targets:
            if path in trees or path not in imports:
                continue
            try:
                trees[path] = parse_file(self.root, path)
            except PARSE_ERRORS:
                result.warnings.append(
                    f"{path}: could not be parsed again for the graph; its edges were left out"
                )
            else:
                reparsed += 1
        parts = compute_file_graphs(symbols, python_files, imports, assigned, trees, targets)
        conn = self.store.conn
        for path, part in parts.items():
            conn.execute(
                "UPDATE files SET nodes_json = ?, edges_json = ?, ignored_json = ? "
                "WHERE path = ?",
                (json.dumps(part.nodes), json.dumps(part.edges), json.dumps(part.ignored), path),
            )
        result.graph_files_recomputed = sorted(parts)
        result.graph_files_parsed = sum(1 for path in parts if path in trees)
        result.graph_files_reparsed = reparsed

    def _validate_graph(
        self, mode: str, data: dict[str, _FileData], result: IndexUpdate
    ) -> None:
        """The stored pieces must assemble into a consistent graph. If a partial update left
        them inconsistent, recompute everything instead of keeping bad data."""
        try:
            load_graph(self.store)
        except MemoryIndexError as exc:
            if mode == GRAPH_FULL:
                raise
            result.notes.append(
                f"The stored graph was inconsistent ({exc}), so it was rebuilt from all files."
            )
            result.graph_mode = GRAPH_FULL
            result.graph_reason += " (the stored graph was inconsistent, so everything was redone)"
            self._graph_step(GRAPH_FULL, data, result)
            load_graph(self.store)

    def _apply(
        self,
        changes: FileChanges,
        current: dict[str, FileState],
        version_id: int,
        result: IndexUpdate,
        force_full: bool,
        note: str,
    ) -> None:
        store = self.store
        old_interface = {
            row["path"]: row["interface_hash"]
            for row in store.conn.execute("SELECT path, interface_hash FROM files")
        }
        store.delete_files(changes.to_remove)
        data: dict[str, _FileData] = {}
        for path in changes.to_process:
            store.put_file_state(current[path], version_id)  # resets this file's derived data
            data[path] = self._extract(current[path])
        self._store_extracted(data)
        result.files_extracted = sorted(data)
        result.symbols_extracted = sum(len(item.symbols) for item in data.values())
        result.files_reused = len(changes.unchanged)
        mode, reason = self._choose_mode(
            changes, data, old_interface, result.first_index, force_full
        )
        result.graph_mode, result.graph_reason = mode, reason
        if mode != GRAPH_NONE:
            self._graph_step(mode, data, result)
            self._validate_graph(mode, data, result)
        elif not changes.has_changes:
            result.notes.append(
                "No indexed Python content changed. The version record was added because "
                "the commit or the working-tree state differs."
            )
        total = int(store.conn.execute("SELECT COUNT(*) FROM symbols").fetchone()[0])
        result.symbols_total = total
        summary = f"{changes.summary()}; graph: {result.graph_mode}"
        store.update_version(
            version_id, len(store.manifest()), total, f"{note}; {summary}" if note else summary
        )

    def _semantic(
        self,
        result: IndexUpdate,
        embedder: EmbeddingService | None,
        semantic_db: Path | None,
        keep_engine: bool,
    ) -> None:
        if embedder is None:
            return
        db_path = Path(semantic_db) if semantic_db is not None else default_db_path(self.root)
        try:
            engine, stats = semantic_sync(self.store, embedder, db_path)
        except EmbeddingError as exc:
            result.warnings.append(f"Embeddings were not updated: {exc}")
            return
        result.semantic = stats
        if result.version is not None:
            self.store.set_meta("semantic_version", str(result.version.id))
            self.store.set_meta("semantic_model", embedder.model_name)
        if keep_engine:
            result.engine = engine
        else:
            engine.close()