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

### Phase 4, part 2: graph-expanded retrieval and visualization

Run: `streamlit run app/main.py`, then open **Dependencies** and **Graph chat**. CLI: `python scripts\ask_graph.py --repo data\graph_sample --question "What does OrderService.checkout call?" --mock`.

- After hybrid retrieval and the Phase 3 relevance gate, the code's confirmed callers and callees are added (`GraphQAEngine`, a subclass of the Phase 3 `QAEngine`). Defaults: depth 1, 2 per direction, 4 added, 3 seeds, 11,000 characters (`GRAPH_EXPAND_*` in `.env`).
- Each added excerpt records why: `callee of E1 (confirmed: OrderService.checkout calls apply_tax at orders/service.py:18)`.
- Only confirmed edges are followed. Uncertain relationships and module-level callers become notes the model must report as unknowns. Notes cannot be cited.
- The graph never rescues an irrelevant question: with no relevant retrieved code, the answer abstains before the model is called.
- If expansion fails, the answer is built from search results only, with a warning.
- Visualization uses matplotlib (offline, static): files are columns, solid arrows are confirmed calls, dashed arrows are uncertain calls ending in an unresolved box.

### Known limitations (Phase 4, part 2)
- The graph is built when you press the button and is not refreshed after edits. Rebuild it.
- Graph-added excerpts have no retrieval scores, so the Phase 5 "strong evidence" rule does not count them.
- Expansion follows calls only, not imports, inheritance or data flow.
- A missing relationship is not proof that no call exists.
- Tested on the `graph_sample` repository only.

## Phase 5: Evidence-first answer validation

Run: `streamlit run app/main.py`, then open **Validated chat**. CLI: `python scripts\ask_validated.py --repo data\sample_repo --question "How does checkout work?"`. Count results with `python scripts\eval_log_summary.py`.

### What it does
- Re-checks every citation against the currently indexed repository: the file is indexed, the range exists, the index lists that symbol there, and the file text still matches (`valid`, `stale`, `missing`, `out_of_range`).
- Checks `file:line` locations the model wrote itself (it is told not to): `ok`, `ungrounded`, or `invalid` (invented).
- Checks each claim against the code it cites: rules (code names the claim mentions must appear in the cited code) plus one batched LLM-as-judge call (`SUPPORTED`, `PARTIAL`, `NOT_SUPPORTED`).
- Applies a configurable evidence policy (`VALIDATION_*` in `.env`) and turns every failed rule into an explicit limitation. Levels are `sufficient`, `limited`, `insufficient`.
- Writes one JSONL record per question (`EVAL_LOG_PATH`) so citation correctness and unsupported claims can be counted later.

### Final verdict for a claim
| Citations | Judge | Rules | Final |
|---|---|---|---|
| invalid or invented | any | any | unsupported |
| valid | NOT_SUPPORTED | any | unsupported |
| valid | SUPPORTED | a named code item is missing | weak |
| valid | other or none | a named code item is missing | unsupported |
| valid | PARTIAL | other | weak |
| valid | SUPPORTED | other | supported |
| valid | none | pass | supported (rules only) |
| valid | none | word overlap low | weak |
| valid | none | nothing checkable | unchecked |

### Decisions
- Validation wraps the Phase 3 `QAEngine`; Phase 3 code is unchanged.
- Levels are rule outcomes, never probabilities. Model confidence is not shown, because it is not calibrated.
- Only files that appear in the index are opened, so skipped sensitive files are never read.
- Logs store a code hash and length, not code (`EVAL_LOG_CODE=on` changes that). `data/logs/` is git-ignored.

### Known limitations (Phase 5)
- "Supported" means no problem was found by these checks, not that the claim is correct.
- Name and word overlap cannot detect wrong logic: "never calls charge" passes if `charge` appears in the code.
- The judge is the same kind of fallible model as the answerer, sees only the cited excerpts, and is weak at negation and cross-file reasoning.
- Only identifiers, numbers and double-quoted strings are checked. The word stemmer is crude.
- The default thresholds (0.8, 0.5, 0.4, minimum 1 valid citation) are untuned guesses. Tune them with labeled data.
- Phase 3 already drops unknown evidence IDs, so invented locations written inside claim text are the main fabrication this phase catches.
- Counts from the log are automatic checks. The `human_label` field exists for manual labeling later.
- Phase 4 part 2 (graph-expanded retrieval) is not built, so validation covers search-based evidence only.


## Phase 6: Bug investigation and test generation

Run: `streamlit run app/main.py`, then open **Bug investigation**. CLI: `python scripts\investigate.py --repo data\bug_sample --traceback-file data\bug_sample\traceback.txt` (add `--mock` for no model, `--verify` to run the draft test in the sandbox).

### What it does
- Parses a Python or pytest traceback into frames and maps each to an indexed symbol. Library frames, unknown files, ambiguous paths and lines that differ from the index are refused with a reason, never guessed.
- Gathers evidence with the reason each excerpt was chosen: the failing frame, calling frames, one test frame, Phase 4 confirmed callers and callees, and related existing tests.
- Asks the model for a root-cause hypothesis in the Phase 3 format and validates it with Phase 5 (citations, claim support, evidence level). A fix is proposed only if a hypothesis claim passed validation.
- The fix is SEARCH/REPLACE blocks that must match exactly once, are applied in memory, and are shown as a diff computed with `difflib`. Limits: 3 edits, 40 changed lines, only files shown as evidence, never test files, the patched file must parse, and no newly introduced risky constructs. It is labeled PROPOSED, NOT VERIFIED.
- The regression test is a draft written to a NEW file under `tests/` (the path is forced), refused if it imports network, process or file-removal modules or calls `eval`/`exec`. It is labeled DRAFT, NOT RUN.
- On request, the sandbox runs the draft test on the current code and again with the fix, and a verdict is decided from the real pytest output (see the table below).

### Verdicts
| Verdict | Meaning |
|---|---|
| verified | draft test failed before (with the reported exception) and passes after; related tests that passed still pass |
| fix_ineffective | draft test still fails or errors after the fix |
| regression | a related test passed before and fails after |
| test_not_reproducing | draft test passes on the current code |
| different_failure | draft fails before, but its output never mentions the reported exception |
| test_broken | draft errors before the fix |
| inconclusive | timeout, sandbox error, no readable summary, or the fix no longer applies |
| not_run | no fix or test to run, or the MOCK provider |

### Sandbox (what is and is not achievable on Windows)
- A temporary copy of only the files the Phase 1 scanner keeps (no `.env`, keys, venvs, `.git`). The project folder is never the working directory.
- A minimal environment (no host variables), `-E -s -B` Python flags, pinned pytest root, stdin closed, a timeout that kills the whole process tree, and capped output.
- A best-effort Python audit-hook guard inside the child process refuses file access outside the copy, sockets, name lookups, links and starting programs. It is NOT a security boundary: native code (ctypes) and interpreter attacks can bypass it, and tests run in the same process as pytest, so a hostile test could forge a result.
- Not achievable without Docker or Job Objects: CPU and memory limits, a real network namespace, a separate filesystem. Docker or a virtual machine is the recommended next step for untrusted code.

### Decisions
- A hypothesis explains why a failure happens and is validated with evidence; a fix is a code change and is only called verified when a real test run supports it.
- "Verified" means exactly the table row above. The test is model-written and may encode the wrong behavior.
- The MOCK provider writes no fix or test, so nothing is verified.

### Known limitations (Phase 6)
- Python tracebacks only. Frames in files that are not indexed are refused.
- Small local models often fail the exact-match SEARCH rule. A rejected fix is the safe outcome.
- Related-test selection is a scored heuristic (confirmed graph callers, file name, name mention).
- The repository's third-party packages must be installed in the app's virtual environment, because the sandbox uses it.
- No CPU or memory limits; only the timeout applies.
- Verified on the `bug_sample` repository only.

## Phase 7: Change-impact analysis

Run: `streamlit run app/main.py`, then open **Change impact**. CLI: `python scripts\impact.py --repo data\impact_sample --diff-file change.diff` (`--no-git`, `--no-ruff`, `--depth N`, `--base auto|old|new`, `--expect-files a.py,b.py`, `--expect-tests ...`).

### What it does
- Parses a unified diff (git diff or diff -u) into files, hunks and change blocks, then checks it against the indexed file text (the version before the change, or the version that already has it). A diff that matches neither is refused, not mapped by line number.
- Maps changed lines to the innermost function or class, or to module-level code. Deleted and renamed files change every symbol in the file.
- Traverses the Phase 4 graph. **Known** dependents are reached over confirmed edges only. **Inferred** dependents need at least one uncertain edge (for example `printer.show(x)` with several possible targets). They are listed separately and never mixed.
- Ranks changed code with six visible factors (dependents, graph position, test reach, git churn, Ruff findings, change size) and shows the points and a plain explanation for each. Factors that cannot be computed (no git, no Ruff) are left out and the total is scaled.
- Reports affected files, recommended tests (closest first) and a pytest command. `--expect-files` and `--expect-tests` compare the predictions with a hand-checked list (recall and precision of the known items).

### Decisions
- The score is a **heuristic rank** from 0 to 100, labeled as such everywhere. It is not a probability. The weights are judgment calls and have not been calibrated against real defects.
- Test reach is static reachability over confirmed edges, not measured coverage.
- Only files in the index are ever opened. A diff naming `.env` or a key file is listed as not indexed and never read.
- Git and Ruff are optional, run with fixed arguments and timeouts, read only, and never execute repository code.

### Known limitations (Phase 7)
- Python only, and only code the Phase 1 scanner indexed.
- Dynamic calls (getattr, callbacks, frameworks, threads) are invisible. A missing item is not proof of safety.
- The use of module-level variables and constants is not tracked: a changed constant reaches only the files that import its module.
- A signature edit is detected from the definition line, but which callers must change is not worked out.
- Churn is per file, and Ruff findings use default rules on the indexed version of the file.
- Validated on `data/impact_sample` and a few hand-made diffs only. The point values are untuned.

## Phase 8: Commit-aware repository memory

Run: `streamlit run app/main.py`, then open **Repository memory**. CLI: `python scripts\memory.py status|update|ask|history --repo .` (`--policy update|warn|block`, `--reuse`, `--no-embeddings`, `--full`).

### What it does
- Identifies a version by commit hash (read through the git command line), a dirty-working-tree flag and a digest of the indexed file contents. Versions live in SQLite (`data/indexes/memory-<hash>.db`, git-ignored).
- Detects added, modified, deleted and renamed files by comparing per-file content hashes with the stored manifest. Git's rename report is a hint, used only where it agrees with the files.
- Updates incrementally: only changed files are parsed, unchanged symbols reuse their vectors (the Phase 2 content-hash cache), and graph edges are stored per file. They are reused per file only while no interface (definitions, imports, module variables) changed and no file was added, deleted or renamed. Otherwise every file's edges are re-resolved, because an unchanged file's edges can depend on other files.
- Freshness levels: `fresh`, `behind` (HEAD moved but the indexed content is identical), `stale` (the content differs), `not_indexed`.
- Stale policy: `update` (refresh, then answer), `warn` (answer from the old version with a loud warning), `block` (refuse). A `behind` index never blocks.
- Every answer records the version it was based on (commit, dirty flag, content digest, label), and every citation records it too, in the answer history and in the evaluation log (`version` and `citations_versioned`).
- When the indexed content changes, older answers are marked OUTDATED. They stay visible for audit and are never served as current. Stored answers can be reused only for identical content, question, LLM and settings.
- The retrieval cache belongs to one index version and is dropped with it. If the vectors cannot be updated (for example the embedding model fails), answering is blocked.

### Decisions
- The commit hash alone cannot identify the indexed content (the index is built from the working tree), so the content digest is part of the version.
- git by subprocess, not GitPython: no new dependency, the same git calls Phase 7 makes, and fewer Windows handle problems.
- The memory service keeps its own vector cache so other pages that sync the Phase 2 index cannot change the evidence behind a stored version.

### Known limitations (Phase 8)
- Python files only. A change to other files moves HEAD (`behind`) but reprocesses nothing.
- A freshness check reads and hashes every kept Python file (no parsing).
- Reuse ignores changes outside the settings fingerprint (embedding model name and QA settings).
- The full graph mode re-parses unchanged files. The update report says so.
- A history rewrite that drops the indexed commit makes git unable to report renames. Renames then use identical content.
- Verified on `data/impact_sample` copies and throwaway git repositories only.