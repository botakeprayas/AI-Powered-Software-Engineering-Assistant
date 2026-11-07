"""Answer history and answer cache, tied to repository versions (Phase 8).

Every stored answer records the version (commit hash, dirty flag, content digest) its evidence
came from. An answer may be reused only while the indexed CONTENT is identical to the content
it was based on. When the indexed content changes, older answers are marked outdated: they are
kept for audit (you can still see which version supported them) but never served as current.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

from core.chat.models import Answer
from core.memory.models import VersionRecord
from core.memory.store import MemoryStore
from core.validation.models import ValidationReport

SCHEMA = """
CREATE TABLE IF NOT EXISTS answer_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    asked_at        TEXT NOT NULL,
    question        TEXT NOT NULL,
    question_key    TEXT NOT NULL,
    version_id      INTEGER NOT NULL,
    version_label   TEXT NOT NULL,
    commit_hash     TEXT,
    dirty           INTEGER NOT NULL,
    tree_digest     TEXT NOT NULL,
    branch          TEXT,
    indexed_at      TEXT NOT NULL,
    status          TEXT NOT NULL,
    llm             TEXT NOT NULL,
    freshness       TEXT NOT NULL,
    level           TEXT,
    settings_key    TEXT NOT NULL,
    answer_json     TEXT NOT NULL,
    citations_json  TEXT NOT NULL,
    outdated        INTEGER NOT NULL DEFAULT 0,
    outdated_reason TEXT NOT NULL DEFAULT '',
    outdated_at     TEXT
);
CREATE INDEX IF NOT EXISTS idx_answer_lookup ON answer_history(question_key, tree_digest);
"""
INSERT = (
    "INSERT INTO answer_history (asked_at, question, question_key, version_id, "
    "version_label, commit_hash, dirty, tree_digest, branch, indexed_at, status, llm, "
    "freshness, level, settings_key, answer_json, citations_json) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def question_key(question: str) -> str:
    """The question without case and spacing differences, used to find stored answers."""
    return " ".join(question.lower().split())


@dataclass(frozen=True)
class VersionStamp:
    """The repository version an answer was based on."""

    version_id: int
    commit_hash: str | None
    dirty: bool
    tree_digest: str
    label: str
    indexed_at: str
    branch: str | None = None

    @classmethod
    def from_record(cls, version: VersionRecord) -> VersionStamp:
        return cls(
            version.id, version.commit_hash, version.dirty, version.tree_digest,
            version.label, version.indexed_at, version.branch,
        )

    def to_dict(self) -> dict:
        return {
            "version_id": self.version_id, "commit": self.commit_hash, "dirty": self.dirty,
            "tree_digest": self.tree_digest, "label": self.label,
            "indexed_at": self.indexed_at, "branch": self.branch,
        }


@dataclass(frozen=True)
class AnswerRecord:
    id: int
    asked_at: str
    question: str
    version: VersionStamp
    status: str
    llm: str
    freshness: str  # the index state the answer was given in
    level: str | None  # the Phase 5 evidence level, when validation ran
    answer: dict
    citations: list[dict]
    outdated: bool
    outdated_reason: str
    outdated_at: str | None


def answer_to_dict(answer: Answer) -> dict:
    def claims(items) -> list[dict]:
        return [{"text": c.text, "evidence_ids": list(c.evidence_ids)} for c in items]

    return {
        "status": answer.status,
        "message": answer.message,
        "facts": claims(answer.facts),
        "inferences": claims(answer.inferences),
        "unknowns": list(answer.unknowns),
        "warnings": list(answer.warnings),
        "abstained_before_llm": answer.abstained_before_llm,
        "is_mock": answer.is_mock,
    }


def citation_records(
    answer: Answer, stamp: VersionStamp, report: ValidationReport | None
) -> list[dict]:
    """One record per cited excerpt: where it is, and which version it came from."""
    valid = {c.evidence_id: c.ok for c in report.citations} if report is not None else {}
    cited_by = report.cited_by if report is not None else {}
    rows = []
    for item in answer.citations():
        symbol = item.symbol
        rows.append(
            {
                "id": item.id,
                "location": item.location,
                "symbol": symbol.qualname,
                "kind": symbol.kind,
                "symbol_key": symbol.key,
                "version_id": stamp.version_id,
                "commit": stamp.commit_hash,
                "dirty": stamp.dirty,
                "version": stamp.label,
                "code_sha256": hashlib.sha256(symbol.code.encode("utf-8")).hexdigest(),
                "cited_by": list(cited_by.get(item.id, [])),
                "valid": valid.get(item.id),
            }
        )
    return rows


def _record(row: sqlite3.Row) -> AnswerRecord:
    stamp = VersionStamp(
        row["version_id"], row["commit_hash"], bool(row["dirty"]), row["tree_digest"],
        row["version_label"], row["indexed_at"], row["branch"],
    )
    return AnswerRecord(
        row["id"], row["asked_at"], row["question"], stamp, row["status"], row["llm"],
        row["freshness"], row["level"], json.loads(row["answer_json"]),
        json.loads(row["citations_json"]), bool(row["outdated"]), row["outdated_reason"],
        row["outdated_at"],
    )


class AnswerStore:
    def __init__(self, store: MemoryStore) -> None:
        self._store = store
        store.conn.executescript(SCHEMA)

    def record(
        self,
        question: str,
        stamp: VersionStamp,
        answer: Answer,
        report: ValidationReport | None,
        llm: str,
        freshness: str,
        settings_key: str = "",
    ) -> int:
        level = report.policy.level if report is not None and report.policy else None
        values = (
            _now(), question, question_key(question), stamp.version_id, stamp.label,
            stamp.commit_hash, int(stamp.dirty), stamp.tree_digest, stamp.branch,
            stamp.indexed_at, answer.status, llm, freshness, level, settings_key,
            json.dumps(answer_to_dict(answer), ensure_ascii=False),
            json.dumps(citation_records(answer, stamp, report), ensure_ascii=False),
        )
        with self._store.transaction() as conn:
            cursor = conn.execute(INSERT, values)
        return int(cursor.lastrowid)

    def get(self, record_id: int) -> AnswerRecord | None:
        row = self._store.conn.execute(
            "SELECT * FROM answer_history WHERE id = ?", (record_id,)
        ).fetchone()
        return None if row is None else _record(row)

    def recent(self, limit: int = 50) -> list[AnswerRecord]:
        rows = self._store.conn.execute(
            "SELECT * FROM answer_history ORDER BY id DESC LIMIT ?", (limit,)
        )
        return [_record(row) for row in rows]

    def count(self) -> int:
        return int(self._store.conn.execute("SELECT COUNT(*) FROM answer_history").fetchone()[0])

    def cached(
        self, question: str, stamp: VersionStamp, llm: str, settings_key: str
    ) -> AnswerRecord | None:
        """A stored answer that is still valid: same question, same indexed content, same
        LLM and settings, and not marked outdated."""
        row = self._store.conn.execute(
            "SELECT * FROM answer_history WHERE question_key = ? AND tree_digest = ? "
            "AND llm = ? AND settings_key = ? AND outdated = 0 ORDER BY id DESC LIMIT 1",
            (question_key(question), stamp.tree_digest, llm, settings_key),
        ).fetchone()
        return None if row is None else _record(row)

    def invalidate_content_changes(self, current: VersionStamp) -> int:
        """Mark every answer whose indexed content differs from `current` as outdated.
        Answers with identical content are left alone. Returns how many were marked."""
        rows = self._store.conn.execute(
            "SELECT id, version_label FROM answer_history "
            "WHERE outdated = 0 AND tree_digest != ?",
            (current.tree_digest,),
        ).fetchall()
        when = _now()
        with self._store.transaction() as conn:
            for row in rows:
                reason = (
                    f"The repository content changed: this answer was based on version "
                    f"{row['version_label']}, and the index now holds {current.label}."
                )
                conn.execute(
                    "UPDATE answer_history SET outdated = 1, outdated_reason = ?, "
                    "outdated_at = ? WHERE id = ?",
                    (reason, when, row["id"]),
                )
        return len(rows)