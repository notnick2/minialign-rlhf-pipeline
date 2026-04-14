"""
SQLite-backed annotation storage for MiniAlign.

Stores all annotation types in structured tables with annotator tracking,
timestamps, and efficient retrieval for IAA computation.
"""

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class AnnotationStore:
    """
    Thread-safe SQLite annotation storage.

    Supports five annotation task types:
      1. general_labeling  — classification, NER, sentiment, intent
      2. pairwise          — A vs B preference comparisons
      3. instruction_quality — 1-5 rubric ratings
      4. factuality        — claim-level true/false/uncertain labels
      5. toxicity_bias     — toxicity and bias flags
    """

    def __init__(self, db_path: str = "annotations.db"):
        self.db_path = Path(db_path)
        self._local = threading.local()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Return a thread-local SQLite connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
        return self._local.conn

    def _init_db(self):
        """Create all annotation tables if they don't exist."""
        conn = self._get_conn()
        cur = conn.cursor()

        cur.executescript("""
        CREATE TABLE IF NOT EXISTS general_labels (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            annotator   TEXT    NOT NULL,
            text        TEXT    NOT NULL,
            category    TEXT,
            ner_spans   TEXT,           -- JSON array of {start, end, label, text}
            sentiment   TEXT,
            intent      TEXT,
            notes       TEXT,
            created_at  TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS pairwise_preferences (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            annotator       TEXT    NOT NULL,
            prompt          TEXT    NOT NULL,
            response_a      TEXT    NOT NULL,
            response_b      TEXT    NOT NULL,
            preference      TEXT    NOT NULL,   -- 'A' | 'B' | 'tie'
            reasons         TEXT,               -- JSON array of reason strings
            confidence      INTEGER NOT NULL,   -- 1-3
            notes           TEXT,
            created_at      TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS instruction_quality (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            annotator           TEXT    NOT NULL,
            prompt              TEXT    NOT NULL,
            response            TEXT    NOT NULL,
            relevance           INTEGER NOT NULL,   -- 1-5
            completeness        INTEGER NOT NULL,
            accuracy            INTEGER NOT NULL,
            format_adherence    INTEGER NOT NULL,
            conciseness         INTEGER NOT NULL,
            overall_score       REAL    NOT NULL,   -- weighted avg
            notes               TEXT,
            created_at          TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS factuality_labels (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            annotator   TEXT    NOT NULL,
            response    TEXT    NOT NULL,
            claims      TEXT    NOT NULL,   -- JSON: [{claim, label, citation}, ...]
            notes       TEXT,
            created_at  TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS toxicity_bias (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            annotator           TEXT    NOT NULL,
            response            TEXT    NOT NULL,
            is_toxic            INTEGER NOT NULL,   -- 0 | 1
            toxicity_categories TEXT,               -- JSON array
            is_biased           INTEGER NOT NULL,
            bias_types          TEXT,               -- JSON array
            severity            TEXT    NOT NULL,   -- none|mild|moderate|severe
            notes               TEXT,
            created_at          TEXT    NOT NULL
        );
        """)
        conn.commit()

    # ------------------------------------------------------------------ #
    #  General Labeling                                                    #
    # ------------------------------------------------------------------ #

    def save_general_label(
        self,
        annotator: str,
        text: str,
        category: Optional[str],
        ner_spans: Optional[List[Dict]],
        sentiment: Optional[str],
        intent: Optional[str],
        notes: str = "",
    ) -> int:
        conn = self._get_conn()
        cur = conn.execute(
            """INSERT INTO general_labels
               (annotator, text, category, ner_spans, sentiment, intent, notes, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                annotator,
                text,
                category,
                json.dumps(ner_spans) if ner_spans else None,
                sentiment,
                intent,
                notes,
                datetime.utcnow().isoformat(),
            ),
        )
        conn.commit()
        return cur.lastrowid

    def get_general_labels(self, limit: int = 1000) -> List[Dict]:
        cur = self._get_conn().execute(
            "SELECT * FROM general_labels ORDER BY created_at DESC LIMIT ?", (limit,)
        )
        rows = cur.fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ #
    #  Pairwise Preferences                                               #
    # ------------------------------------------------------------------ #

    def save_pairwise(
        self,
        annotator: str,
        prompt: str,
        response_a: str,
        response_b: str,
        preference: str,
        reasons: List[str],
        confidence: int,
        notes: str = "",
    ) -> int:
        conn = self._get_conn()
        cur = conn.execute(
            """INSERT INTO pairwise_preferences
               (annotator, prompt, response_a, response_b, preference, reasons,
                confidence, notes, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                annotator,
                prompt,
                response_a,
                response_b,
                preference,
                json.dumps(reasons),
                confidence,
                notes,
                datetime.utcnow().isoformat(),
            ),
        )
        conn.commit()
        return cur.lastrowid

    def get_pairwise(self, limit: int = 1000) -> List[Dict]:
        cur = self._get_conn().execute(
            "SELECT * FROM pairwise_preferences ORDER BY created_at DESC LIMIT ?", (limit,)
        )
        return [dict(r) for r in cur.fetchall()]

    # ------------------------------------------------------------------ #
    #  Instruction Quality                                                 #
    # ------------------------------------------------------------------ #

    def save_instruction_quality(
        self,
        annotator: str,
        prompt: str,
        response: str,
        relevance: int,
        completeness: int,
        accuracy: int,
        format_adherence: int,
        conciseness: int,
        notes: str = "",
    ) -> int:
        # Weighted average: accuracy and completeness weighted more
        weights = [0.2, 0.25, 0.3, 0.15, 0.1]
        scores = [relevance, completeness, accuracy, format_adherence, conciseness]
        overall = sum(w * s for w, s in zip(weights, scores))

        conn = self._get_conn()
        cur = conn.execute(
            """INSERT INTO instruction_quality
               (annotator, prompt, response, relevance, completeness, accuracy,
                format_adherence, conciseness, overall_score, notes, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                annotator, prompt, response,
                relevance, completeness, accuracy, format_adherence, conciseness,
                round(overall, 3), notes, datetime.utcnow().isoformat(),
            ),
        )
        conn.commit()
        return cur.lastrowid

    def get_instruction_quality(self, min_score: float = 0.0, limit: int = 1000) -> List[Dict]:
        cur = self._get_conn().execute(
            "SELECT * FROM instruction_quality WHERE overall_score >= ? ORDER BY overall_score DESC LIMIT ?",
            (min_score, limit),
        )
        return [dict(r) for r in cur.fetchall()]

    # ------------------------------------------------------------------ #
    #  Factuality                                                          #
    # ------------------------------------------------------------------ #

    def save_factuality(
        self,
        annotator: str,
        response: str,
        claims: List[Dict],   # [{claim, label, citation}, ...]
        notes: str = "",
    ) -> int:
        conn = self._get_conn()
        cur = conn.execute(
            """INSERT INTO factuality_labels
               (annotator, response, claims, notes, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (annotator, response, json.dumps(claims), notes, datetime.utcnow().isoformat()),
        )
        conn.commit()
        return cur.lastrowid

    def get_factuality(self, limit: int = 1000) -> List[Dict]:
        cur = self._get_conn().execute(
            "SELECT * FROM factuality_labels ORDER BY created_at DESC LIMIT ?",(limit,)
        )
        rows = [dict(r) for r in cur.fetchall()]
        for r in rows:
            r["claims"] = json.loads(r["claims"])
        return rows

    # ------------------------------------------------------------------ #
    #  Toxicity & Bias                                                     #
    # ------------------------------------------------------------------ #

    def save_toxicity_bias(
        self,
        annotator: str,
        response: str,
        is_toxic: bool,
        toxicity_categories: List[str],
        is_biased: bool,
        bias_types: List[str],
        severity: str,
        notes: str = "",
    ) -> int:
        conn = self._get_conn()
        cur = conn.execute(
            """INSERT INTO toxicity_bias
               (annotator, response, is_toxic, toxicity_categories, is_biased,
                bias_types, severity, notes, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                annotator, response,
                int(is_toxic), json.dumps(toxicity_categories),
                int(is_biased), json.dumps(bias_types),
                severity, notes, datetime.utcnow().isoformat(),
            ),
        )
        conn.commit()
        return cur.lastrowid

    def get_toxicity_bias(self, limit: int = 1000) -> List[Dict]:
        cur = self._get_conn().execute(
            "SELECT * FROM toxicity_bias ORDER BY created_at DESC LIMIT ?", (limit,)
        )
        rows = [dict(r) for r in cur.fetchall()]
        for r in rows:
            r["toxicity_categories"] = json.loads(r["toxicity_categories"] or "[]")
            r["bias_types"] = json.loads(r["bias_types"] or "[]")
        return rows

    # ------------------------------------------------------------------ #
    #  Statistics                                                          #
    # ------------------------------------------------------------------ #

    def get_statistics(self) -> Dict[str, Any]:
        """Return aggregate statistics across all annotation tables."""
        conn = self._get_conn()
        stats = {}
        tables = {
            "general_labeling": "general_labels",
            "pairwise": "pairwise_preferences",
            "instruction_quality": "instruction_quality",
            "factuality": "factuality_labels",
            "toxicity_bias": "toxicity_bias",
        }
        total = 0
        for label, table in tables.items():
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            stats[label] = count
            total += count
        stats["total"] = total

        # Annotator distribution across all tables
        annotators: Dict[str, int] = {}
        for table in tables.values():
            rows = conn.execute(f"SELECT annotator, COUNT(*) as cnt FROM {table} GROUP BY annotator").fetchall()
            for row in rows:
                annotators[row[0]] = annotators.get(row[0], 0) + row[1]
        stats["annotators"] = annotators

        return stats

    def export_for_iaa(self, task_type: str) -> List[Dict]:
        """
        Export annotations grouped by item for IAA computation.
        Returns a list of dicts: {item_id, annotator, label}.
        """
        conn = self._get_conn()
        if task_type == "pairwise":
            rows = conn.execute(
                "SELECT id, annotator, preference as label FROM pairwise_preferences"
            ).fetchall()
        elif task_type == "instruction_quality":
            rows = conn.execute(
                "SELECT id, annotator, overall_score as label FROM instruction_quality"
            ).fetchall()
        elif task_type == "toxicity_bias":
            rows = conn.execute(
                "SELECT id, annotator, severity as label FROM toxicity_bias"
            ).fetchall()
        elif task_type == "general_labeling":
            rows = conn.execute(
                "SELECT id, annotator, sentiment as label FROM general_labels"
            ).fetchall()
        elif task_type == "factuality":
            rows = conn.execute(
                "SELECT id, annotator, id as label FROM factuality_labels"
            ).fetchall()
        else:
            return []
        return [dict(r) for r in rows]
