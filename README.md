# AI-Powered Software Engineering Assistant

Analyzes a software repository to help developers understand, navigate, debug, test and
maintain it. Built phase by phase with 100% free and open-source tools.

**Current phase:** 0, environment and project setup (no AI features yet).

## Setup (Windows PowerShell)

```powershell
cd C:\PROJECTS\ai-se-assistant
python --version                      # must be 3.10 or newer
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env           # then edit .env if needed
python scripts\verify_env.py
```

## Run

```powershell
streamlit run app/main.py             # opens http://localhost:8501
pytest                                # run tests
ruff check .                          # lint
```

## Structure

| Path | Purpose |
|---|---|
| `app/` | Streamlit UI only |
| `core/` | Logic independent of the UI (settings now; parsing, indexing, search later) |
| `services/` | Wrappers for external systems (LLM, embeddings), later |
| `tests/` | pytest tests |
| `scripts/` | Helper scripts (`verify_env.py`) |
| `data/sample_repo/` | Sample codebase used during development |

## Architecture decisions

1. **Isolated virtual environment** (`.venv`) so dependencies are reproducible.
2. **Config in `.env`, committed template in `.env.example`**. Secrets never enter Git.
3. **Frozen `Settings` dataclass.** Configuration is read once and cannot be changed by accident.
4. **Real environment variables override `.env`** (`override=False`), the common convention.
5. **UI / logic / services split** so the UI, the model and the search code can change independently.
6. **`logging.basicConfig(force=True)`** because Streamlit re-runs the script on every interaction.
7. **`pyproject.toml`** holds Ruff and pytest settings in one place.

## Known limitations (Phase 0)

- The Streamlit page is a placeholder with no features.
- Dependency versions are minimums (`>=`), not pinned. We will pin them after verification.
- `verify_env.py` can only confirm `.env` is ignored with `git check-ignore` once `git init` has been run.
- `app/main.py` adds the project root to `sys.path`. A proper installable package may replace this later.
- Not yet verified on a clean machine. Results will be recorded here.

**Current phase:** 1 (repository scanner and keyword search) on top of Phases 0 and 2.
## Phase 1: Repository scanner, symbol extractor, keyword search

Run: `streamlit run app/main.py` (UI) or `python scripts\scan_repo.py data\sample_repo --search validate_email` (CLI).
No LLM, no model download and no API key are used in this phase.

### Decisions
- AST extraction (`ast`): functions, async functions, classes and methods with exact line ranges. Decorators count as part of the symbol. Line endings are normalized to `\n` before parsing, so line numbers match.
- SQLite symbol index, in-memory by default (`SymbolIndex(db_path=...)` for a file). `check_same_thread=False` because Streamlit reruns scripts on different threads.
- Exact keyword search: case-insensitive substring, every word must match. Score: whole-query name match +100, name +40, docstring +15, code +5.
- The scanner skips by name or marker: virtual environments (including any folder containing `pyvenv.cfg`), `.git`, caches, build output, generated files, binaries, symlinks and sensitive files. Each skip is recorded with its reason and shown in the UI. Sensitive files are never opened.
- Zip upload is extracted safely: path traversal, absolute paths and symlinks are rejected; 50 MB upload, 100 MB expanded and 10,000 entry limits; skipped and sensitive files are never written to disk.
- One shared `Symbol` type (from `core/semantic/models.py`). `core/repo/source.py` is a bridge so Phase 2 can use this scanner later; Phase 2 is unchanged.

### Known limitations (Phase 1)
- Python only. Functions nested inside other functions are not indexed.
- Keyword search has no stemming or fuzzy matching ("validates" will not match "validate").
- Files over 1 MB, more than 5,000 files, and non-UTF-8 files are skipped with a warning.
- The scanner skips files named like secrets, but cannot detect secrets hard-coded inside normal source files, and the UI shows source code as written.
- The folder-path box can read any folder the user can read. Run the app only on your own machine.
- Folders named `env`, `build` or `dist` are skipped even if they hold real code.
- No persistence between runs: the index is rebuilt on each Analyze.

## Phase 2: Semantic code search

Run: `python scripts\semantic_search.py --repo . --query "load configuration"`
Modes: `--mode keyword | semantic | hybrid` (default hybrid; keyword needs no model).

### Decisions
- Model `sentence-transformers/all-MiniLM-L6-v2` (small, CPU-friendly); swap via `EMBEDDING_MODEL`.
- Embedding text = name (also split into words) + docstring + file + code (truncated).
- SQLite is the source of truth (location, content hash, vector); FAISS (`IndexFlatIP`) is
  rebuilt from it after each sync. FAISS vector id = SQLite row id.
- Content hash = sha256(format version + model name + embedding text); line numbers are not
  hashed, so moved code only gets its line range updated, not re-embedded.
- Hybrid = Reciprocal Rank Fusion (k=60): rank-based, so cosine and BM25 scales never clash.
- One index file per repo path under `data/indexes/` (git-ignored).
- `trust_remote_code` is never enabled.

### Known limitations
- `AstSymbolSource` (Python only, no nested functions) and the built-in BM25 are FALLBACKS
  pending an adapter to the Phase 1 symbol/keyword code.
- Exact FAISS search; fine for tens of thousands of symbols, not millions.
- MiniLM is not code-specific and reads only ~256 tokens per symbol.
- No stemming: "validates" will not keyword-match "validate" (semantic search covers this).
- The index is rebuilt in memory on every run; the vector cache avoids re-embedding.
- No minimum-relevance cutoff: semantic search always returns top-k, so weak matches (cos < ~0.25) are noise. Check the cosine score.
- Test functions are indexed and can outrank production code; no filter yet.
- Each CLI run pays ~15 s for importing PyTorch and loading the model; the UI should keep the model loaded.

**Current phase:** 3 (repository-aware chat) on top of Phases 0-2.

## Phase 3: Repository-aware chat (local LLM via Ollama)

Run: `streamlit run app/main.py`, then open **Chat** in the sidebar. CLI: `python scripts\ask.py --repo data\sample_repo --question "How does checkout work?" --mock`.

### Decisions
- One LLM service (`services/llm_service.py`) behind the `LLMService` interface. Providers: `OllamaLLM` (real, local, no API key) and `MockLLM` (clearly labeled fake for tests). URL and model come from `.env`.
- Workflow: hybrid retrieval (Phase 2) -> relevance gate -> prompt -> LLM -> parse and validate.
- The model cites evidence IDs (`[E1]`). The app maps IDs to the indexed `file:start-end`, so locations cannot be invented. Unknown IDs are removed. Uncited statements are dropped.
- Prompt separates FACTS, INFERENCES and UNKNOWNS and treats code excerpts as untrusted data (prompt-injection defense).
- Abstention happens before the model (no relevant evidence), in the model (STATUS: INSUFFICIENT_EVIDENCE) and after it (no valid citation).
- Retries with exponential backoff for connection errors, timeouts and HTTP 5xx/429. Missing model and bad requests are not retried.
- `num_ctx` is sent explicitly so the rules at the start of the prompt are never silently cut off.
- Ollama calls bypass Windows proxy settings when the host is localhost.

### Known limitations (Phase 3)
- The relevance thresholds (`QA_MIN_COSINE`, `QA_MIN_KEYWORD_SCORE`) are heuristics. Tune them with `scripts\ask.py --scores`.
- A question that is topically close to the code can pass the gate; the model then has to abstain, and small models do not always do so.
- Each question is answered on its own; follow-ups that depend on earlier turns are not resolved.
- Small models (3B) may ignore the output format. Such answers are rejected, not shown.
- Class excerpts can duplicate their methods' code, which wastes evidence budget.
- Python files only. If `OLLAMA_BASE_URL` points to another computer, code excerpts leave this machine.
- Answers have not been evaluated beyond the sample repository.

## Phase 4: Dependency graph (Part 1: graph and CLI)

Run: `python scripts\graph.py data\graph_sample --callees OrderService.checkout`
Other options: `--callers NAME`, `--uncertain`, `--edges`, `--limitations`.

### What it does
- Parses Python files with `ast` (code is never executed) and builds a NetworkX `MultiDiGraph`.
- Nodes: files, functions, classes, external modules, and `unresolved` placeholders.
- Edges: `imports`, `defines`, `calls`. Every edge stores `kind`, `certainty`, `file`, `line` and a `reason`.
- Symbol ids reuse the Phase 1 keys (`file::qualname`), so the graph lines up with the search index.

### Decisions
- Each edge is `confirmed` (resolved from the code text) or `uncertain` (cannot be proven).
- Uncertain calls point to an `unresolved` node. Possible targets are stored as candidates and are never drawn as real edges.
- Confirmed cases: names defined or imported in the file (including re-exports), `self.method()` inside its class, `ClassName.method()`, and `x.method()` when `x = ClassName(...)` is assigned once.
- Uncertain cases: parameters and unknown receivers, `getattr`, `importlib`, `eval`/`exec`/`__import__`, star imports, ambiguous names, `super()`.
- Calls to builtins, the standard library and third-party packages are counted but not recorded.
- Caller and callee lookups use confirmed edges only. Possible callers are listed separately and labeled "NOT confirmed".

### Known limitations (Phase 4)
- Python only. Inheritance is not resolved. Attributes set in `__init__` are not tracked (`self.gateway.charge()` stays uncertain).
- Nested functions are not nodes. Their calls count for the enclosing function.
- Calls made by frameworks, threads or callbacks cannot be seen.
- A missing edge does not prove that no call exists. Run `--limitations` for the full list.
- Verified so far on `data/graph_sample` only: 15 symbols, 43 edges (12 confirmed calls, 4 uncertain calls, 15 defines, 12 imports).
- Not built yet: graph-expanded retrieval for chat, the dependency visualization, and the Phase 4 tests (Part 2).