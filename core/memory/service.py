"""MemoryService: version-aware indexing and answering (Phase 8).

Before answering, it compares the repository with the indexed version. Depending on the stale
policy it updates the index first, answers with a loud warning, or refuses. It never answers
from vectors it could not bring up to date. Every answer is tagged with the version it was
based on, in the answer history and in the evaluation log.
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

from core.chat.models import STATUS_ERROR, QAConfig, RepoInfo
from core.memory.answers import AnswerRecord, AnswerStore, VersionStamp, citation_records
from core.memory.config import POLICY_BLOCK, POLICY_UPDATE, STALE_POLICIES, MemoryConfig
from core.memory.freshness import BEHIND, NOT_INDEXED, STALE, Freshness, check_freshness
from core.memory.gitrepo import GitRepo
from core.memory.indexer import IncrementalIndexer, IndexUpdate, semantic_sync
from core.memory.retrieval import VersionedRetriever
from core.memory.store import MemoryStore, memory_db_path
from core.validation.config import ValidationConfig
from core.validation.engine import ValidatedAnswer, ValidatingQAEngine
from core.validation.evallog import append_record, build_record
from services.embedding_service import EmbeddingError, EmbeddingService
from services.llm_service import LLMService

logger = logging.getLogger(__name__)


def semantic_db_path(root: Path) -> Path:
    """The memory service's own vector cache. Other pages that sync the shared Phase 2
    index cannot change the evidence behind a stored version."""
    path = memory_db_path(root)
    return path.with_name(path.name.replace("memory-", "memory-vectors-", 1))


@dataclass
class VersionedAnswer:
    validated: ValidatedAnswer | None = None  # a fresh answer with its Phase 5 validation
    cached: AnswerRecord | None = None  # or a stored answer that was reused
    stamp: VersionStamp | None = None  # the version the answer is based on
    freshness: Freshness | None = None  # the index state used for answering
    freshness_before: Freshness | None = None  # the state before any update
    refreshed: IndexUpdate | None = None
    citations: list[dict] = field(default_factory=list)
    record_id: int | None = None
    blocked: bool = False
    reason: str = ""
    warnings: list[str] = field(default_factory=list)
    logged: bool = False

    @property
    def status(self) -> str:
        if self.blocked:
            return "blocked"
        if self.cached is not None:
            return self.cached.status
        return self.validated.answer.status if self.validated is not None else "none"


class MemoryService:
    def __init__(
        self,
        root: Path,
        embedder: EmbeddingService | None = None,
        store: MemoryStore | None = None,
        semantic_db: Path | None = None,
        config: MemoryConfig | None = None,
        git: GitRepo | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.store = store if store is not None else MemoryStore(memory_db_path(self.root))
        self._owns_store = store is None
        self.embedder = embedder
        self.semantic_db = Path(semantic_db) if semantic_db else semantic_db_path(self.root)
        self.config = config or MemoryConfig()
        self.git = git if git is not None else GitRepo(self.root)
        self.indexer = IncrementalIndexer(self.root, self.store, self.git)
        self.answers = AnswerStore(self.store)
        self._engine = None
        self._retriever: VersionedRetriever | None = None
        self._engine_version: int | None = None
        self.engine_error = ""

    @property
    def retrieval_cache(self) -> VersionedRetriever | None:
        return self._retriever

    def close(self) -> None:
        self._drop_engine()
        if self._owns_store:
            self.store.close()

    def freshness(self) -> Freshness:
        return check_freshness(self.root, self.store, self.git)

    def history(self, limit: int = 50) -> list[AnswerRecord]:
        return self.answers.recent(limit)

    # ---- the index ------------------------------------------------------------------

    def _drop_engine(self) -> None:
        if self._engine is not None:
            self._engine.close()
        self._engine = None
        self._retriever = None
        self._engine_version = None

    def _attach(self, engine, version_id: int) -> None:
        self._engine = engine
        self._retriever = VersionedRetriever(engine, version_id, self.config.retrieval_cache_size)
        self._engine_version = version_id
        self.engine_error = ""

    def refresh(self, force_full: bool = False, note: str = "") -> IndexUpdate:
        """Update the memory store (and the vectors), then drop everything tied to old
        versions: the retrieval cache goes with the old engine, and stored answers whose
        content differs are marked outdated."""
        self._drop_engine()
        update = self.indexer.update(
            self.embedder, self.semantic_db, keep_engine=True, force_full=force_full, note=note
        )
        version = update.version
        if version is None:
            raise RuntimeError("The update produced no version record.")
        if update.created_version:
            self.answers.invalidate_content_changes(VersionStamp.from_record(version))
        if update.engine is not None:
            self._attach(update.engine, version.id)
        else:
            failed = [w for w in update.warnings if w.startswith("Embeddings were not updated")]
            self.engine_error = failed[0] if failed else "No embedding service is available."
        return update

    def _ensure_engine(self, version) -> bool:
        """Make sure the vector index matches `version`. False when it cannot be done."""
        if self._engine is not None and self._engine_version == version.id:
            return True
        if self.embedder is None:
            self.engine_error = "No embedding service is available."
            return False
        self._drop_engine()
        try:
            engine, _stats = semantic_sync(self.store, self.embedder, self.semantic_db)
        except EmbeddingError as exc:
            self.engine_error = f"Embeddings were not updated: {exc}"
            return False
        self.store.set_meta("semantic_version", str(version.id))
        self._attach(engine, version.id)
        return True

    # ---- answering ------------------------------------------------------------------

    def _settings_key(self, qa_config: QAConfig) -> str:
        model = self.embedder.model_name if self.embedder is not None else "none"
        return hashlib.sha256(f"{model}|{qa_config!r}".encode()).hexdigest()[:16]

    @staticmethod
    def _freshness_warning(now: Freshness, stamp: VersionStamp) -> str:
        first = now.reasons[0] if now.reasons else ""
        if now.status == STALE:
            return (
                f"STALE INDEX: {now.headline} This answer is based on version "
                f"{stamp.label}, not on the current files. {first}"
            )
        return (
            "The repository moved on (HEAD or the working-tree state differs from the "
            "indexed version), but the indexed Python content is identical, so this "
            f"evidence is still valid. {first}"
        )

    @staticmethod
    def _block(result: VersionedAnswer, reason: str) -> VersionedAnswer:
        result.blocked = True
        result.reason = reason
        return result

    def ask(
        self,
        question: str,
        llm: LLMService,
        qa_config: QAConfig | None = None,
        validation_config: ValidationConfig | None = None,
        policy: str | None = None,
        judge: LLMService | None = None,
        reuse: bool = False,
    ) -> VersionedAnswer:
        qa_config = qa_config or QAConfig()
        validation_config = validation_config or ValidationConfig()
        policy = policy or self.config.stale_policy
        if policy not in STALE_POLICIES:
            raise ValueError(f"policy must be one of {STALE_POLICIES}, got {policy!r}.")
        result = VersionedAnswer()
        before = self.freshness()
        now = before
        result.freshness_before = before
        needs_update = before.status == NOT_INDEXED or (
            before.status in (STALE, BEHIND) and policy == POLICY_UPDATE
        )
        if needs_update:
            result.refreshed = self.refresh(note=f"refreshed before answering ({before.status})")
            now = self.freshness()
            summary = result.refreshed.changes.summary()
            result.warnings.append(f"The index was updated before answering ({summary}).")
        result.freshness = now
        version = self.store.current_version()
        if version is None:
            return self._block(result, "The repository could not be indexed.")
        if now.status == STALE and policy == POLICY_BLOCK:
            first = now.reasons[0] if now.reasons else ""
            return self._block(
                result,
                f"The index is stale, so no answer was given. {now.headline} {first} "
                "Update the index, or change the stale policy.",
            )
        if not self._ensure_engine(version):
            return self._block(
                result,
                f"The vector index could not be brought up to date ({self.engine_error}), "
                "so answers would use outdated evidence.",
            )
        stamp = VersionStamp.from_record(version)
        result.stamp = stamp
        if now.status in (STALE, BEHIND):
            result.warnings.append(self._freshness_warning(now, stamp))
        key = self._settings_key(qa_config)
        llm_name = f"{llm.provider}/{llm.model_name}"
        if reuse:
            cached = self.answers.cached(question, stamp, llm_name, key)
            if cached is not None:
                result.cached = cached
                result.record_id = cached.id
                result.citations = cached.citations
                return result
        return self._answer(
            result, question, llm, qa_config, validation_config, policy, judge, key, llm_name
        )

    def _answer(
        self, result, question, llm, qa_config, validation_config, policy, judge, key, llm_name
    ) -> VersionedAnswer:
        stamp = result.stamp
        version = self.store.current_version()
        engine = ValidatingQAEngine(
            self._retriever, llm, self.root, qa_config,
            dataclasses.replace(validation_config, log_enabled=False),  # logged below, tagged
            RepoInfo(self.root.name, version.symbol_count if version else 0), judge,
        )
        validated = engine.ask(question)
        answer = validated.answer
        result.validated = validated
        result.citations = citation_records(answer, stamp, validated.report)
        if answer.status != STATUS_ERROR:
            result.record_id = self.answers.record(
                question, stamp, answer, validated.report, llm_name,
                result.freshness.status, key,
            )
        if validation_config.log_enabled:
            record = build_record(
                answer, validated.report, self._retriever.last, validation_config,
                self.root.name, (judge or llm).model_name,
            )
            record["version"] = {
                **stamp.to_dict(),
                "freshness": result.freshness.status,
                "freshness_before": result.freshness_before.status,
                "stale_policy": policy,
                "history_id": result.record_id,
            }
            record["citations_versioned"] = result.citations
            result.logged = append_record(validation_config.log_path, record)
        return result