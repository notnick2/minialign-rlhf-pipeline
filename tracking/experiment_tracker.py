"""Experiment tracker with SQLite backend.
Covers: Prompt versioning and experiment tracking (Cat 6)
        Reproducible eval harnesses (Cat 6)

Tracks training runs, metrics, checkpoints, and prompt templates.
Supports run comparison, export, and prompt versioning via SHA-256 hashing.

Design:
- SQLite for zero-dependency local storage
- Prompt hashing enables tracking of prompt version changes across runs
- Config diffs enable ablation study analysis
- All timestamps stored as ISO-8601 UTC strings for portability
"""

import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    timestamp       TEXT NOT NULL,
    algorithm       TEXT,
    config_json     TEXT NOT NULL,
    prompt_hash     TEXT,
    prompt_template TEXT,
    notes           TEXT
);

CREATE TABLE IF NOT EXISTS metrics (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL,
    step        INTEGER NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS checkpoints (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT NOT NULL,
    checkpoint_path TEXT NOT NULL,
    eval_score      REAL,
    saved_at        TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS prompts (
    prompt_hash     TEXT PRIMARY KEY,
    prompt_template TEXT NOT NULL,
    first_seen_at   TEXT NOT NULL,
    times_used      INTEGER DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_metrics_run_id ON metrics(run_id);
CREATE INDEX IF NOT EXISTS idx_metrics_run_step ON metrics(run_id, step);
CREATE INDEX IF NOT EXISTS idx_checkpoints_run ON checkpoints(run_id);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _config_diff(config_a: dict, config_b: dict) -> dict:
    """Compute differences between two config dicts."""
    all_keys = set(config_a) | set(config_b)
    diff: dict = {}
    for key in sorted(all_keys):
        val_a = config_a.get(key, "<missing>")
        val_b = config_b.get(key, "<missing>")
        if val_a != val_b:
            diff[key] = {"run_a": val_a, "run_b": val_b}
    return diff


# ---------------------------------------------------------------------------
# ExperimentTracker
# ---------------------------------------------------------------------------


class ExperimentTracker:
    """SQLite-backed experiment tracker for MiniAlign training runs.

    Stores per-run configs, step-level metrics, checkpoint paths, and
    prompt templates with SHA-256 version hashing.

    Example usage::

        tracker = ExperimentTracker("experiments.db")
        run_id = tracker.start_run(config={"algorithm": "dpo", "lr": 5e-5},
                                   prompt_template="### Instruction:\n{instruction}")
        for step in range(100):
            tracker.log_metrics(run_id, step, {"loss": 0.5 - step * 0.004})
        tracker.log_checkpoint(run_id, "checkpoints/dpo/final", eval_score=0.87)
        print(tracker.get_run_history(run_id))
    """

    def __init__(self, db_path: str = "experiments.db") -> None:
        """Initialize tracker and create tables if they don't exist.

        Args:
            db_path: Path to SQLite database file. Created if absent.
        """
        self.db_path = str(Path(db_path).resolve())
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")  # better concurrency
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        conn = self._connect()
        conn.executescript(_DDL)
        conn.commit()
        conn.close()

    # ------------------------------------------------------------------
    # Run management
    # ------------------------------------------------------------------

    def start_run(
        self,
        config: dict,
        prompt_template: Optional[str] = None,
        algorithm: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> str:
        """Start a new experiment run.

        Args:
            config: Training configuration dict (serialized as JSON).
            prompt_template: Optional prompt template string. Hashed for versioning.
            algorithm: Optional algorithm label (e.g., "dpo", "ppo").
            notes: Optional free-text notes.

        Returns:
            Unique run_id (UUID4 string).
        """
        run_id = str(uuid.uuid4())
        timestamp = _now_iso()

        # Determine algorithm from config if not provided
        if algorithm is None:
            algorithm = config.get("algorithm", config.get("trainer", None))

        # Hash prompt template
        prompt_hash = None
        if prompt_template:
            prompt_hash = _sha256(prompt_template)

        conn = self._connect()
        try:
            conn.execute(
                """INSERT INTO runs (run_id, timestamp, algorithm, config_json,
                   prompt_hash, prompt_template, notes)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    timestamp,
                    algorithm,
                    json.dumps(config, default=str),
                    prompt_hash,
                    prompt_template,
                    notes,
                ),
            )

            # Track prompt in prompts table
            if prompt_template and prompt_hash:
                existing = conn.execute(
                    "SELECT times_used FROM prompts WHERE prompt_hash = ?",
                    (prompt_hash,),
                ).fetchone()

                if existing:
                    conn.execute(
                        "UPDATE prompts SET times_used = times_used + 1 WHERE prompt_hash = ?",
                        (prompt_hash,),
                    )
                else:
                    conn.execute(
                        "INSERT INTO prompts (prompt_hash, prompt_template, first_seen_at) VALUES (?, ?, ?)",
                        (prompt_hash, prompt_template, timestamp),
                    )

            conn.commit()
        finally:
            conn.close()

        print(f"Started run {run_id} (algorithm={algorithm})")
        return run_id

    def log_metrics(
        self, run_id: str, step: int, metrics: dict[str, float]
    ) -> None:
        """Log a dict of metric name → value at a given training step.

        Args:
            run_id: Run identifier from start_run().
            step: Global training step number.
            metrics: Dict of metric_name → metric_value.
        """
        if not metrics:
            return

        conn = self._connect()
        try:
            rows = [
                (run_id, step, name, float(value))
                for name, value in metrics.items()
            ]
            conn.executemany(
                "INSERT INTO metrics (run_id, step, metric_name, metric_value) VALUES (?, ?, ?, ?)",
                rows,
            )
            conn.commit()
        finally:
            conn.close()

    def log_checkpoint(
        self,
        run_id: str,
        checkpoint_path: str,
        eval_score: Optional[float] = None,
    ) -> None:
        """Record a model checkpoint path and optional evaluation score.

        Args:
            run_id: Run identifier.
            checkpoint_path: File system path to checkpoint directory.
            eval_score: Optional scalar evaluation metric (e.g., val_loss, accuracy).
        """
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO checkpoints (run_id, checkpoint_path, eval_score, saved_at) VALUES (?, ?, ?, ?)",
                (run_id, checkpoint_path, eval_score, _now_iso()),
            )
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def get_run_history(self, run_id: str) -> dict:
        """Retrieve full history for a run.

        Returns:
            Dict with keys:
              - run_id, timestamp, algorithm, config, prompt_hash, notes
              - metrics_by_step: {step: {metric: value}}
              - checkpoints: list of {path, eval_score, saved_at}
        """
        conn = self._connect()
        try:
            run_row = conn.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()

            if run_row is None:
                raise KeyError(f"Run '{run_id}' not found in database")

            config = json.loads(run_row["config_json"])

            # Aggregate metrics by step
            metric_rows = conn.execute(
                "SELECT step, metric_name, metric_value FROM metrics WHERE run_id = ? ORDER BY step",
                (run_id,),
            ).fetchall()

            metrics_by_step: dict[int, dict[str, float]] = {}
            for row in metric_rows:
                step = row["step"]
                if step not in metrics_by_step:
                    metrics_by_step[step] = {}
                metrics_by_step[step][row["metric_name"]] = row["metric_value"]

            # Checkpoints
            ckpt_rows = conn.execute(
                "SELECT checkpoint_path, eval_score, saved_at FROM checkpoints WHERE run_id = ? ORDER BY saved_at",
                (run_id,),
            ).fetchall()
            checkpoints = [
                {
                    "path": r["checkpoint_path"],
                    "eval_score": r["eval_score"],
                    "saved_at": r["saved_at"],
                }
                for r in ckpt_rows
            ]

        finally:
            conn.close()

        return {
            "run_id": run_id,
            "timestamp": run_row["timestamp"],
            "algorithm": run_row["algorithm"],
            "config": config,
            "prompt_hash": run_row["prompt_hash"],
            "notes": run_row["notes"],
            "metrics_by_step": metrics_by_step,
            "checkpoints": checkpoints,
            "n_steps_logged": len(metrics_by_step),
        }

    def compare_runs(self, run_id_a: str, run_id_b: str) -> dict:
        """Side-by-side comparison of two runs: config diff + final metrics.

        Args:
            run_id_a: First run ID.
            run_id_b: Second run ID.

        Returns:
            Dict with keys: config_diff, metrics_comparison, run_a_summary, run_b_summary.
        """
        hist_a = self.get_run_history(run_id_a)
        hist_b = self.get_run_history(run_id_b)

        config_diff = _config_diff(hist_a["config"], hist_b["config"])

        # Final metric values (last step for each metric)
        def _final_metrics(history: dict) -> dict[str, float]:
            finals: dict[str, float] = {}
            for step_metrics in history["metrics_by_step"].values():
                finals.update(step_metrics)
            return finals

        metrics_a = _final_metrics(hist_a)
        metrics_b = _final_metrics(hist_b)
        all_metric_names = sorted(set(metrics_a) | set(metrics_b))

        metrics_comparison: dict[str, dict] = {}
        for name in all_metric_names:
            metrics_comparison[name] = {
                "run_a": metrics_a.get(name, None),
                "run_b": metrics_b.get(name, None),
            }

        def _best_checkpoint(history: dict) -> Optional[dict]:
            ckpts = history.get("checkpoints", [])
            if not ckpts:
                return None
            scored = [c for c in ckpts if c["eval_score"] is not None]
            if scored:
                return min(scored, key=lambda c: c["eval_score"])
            return ckpts[-1]

        return {
            "run_a": {
                "run_id": run_id_a,
                "algorithm": hist_a["algorithm"],
                "timestamp": hist_a["timestamp"],
                "best_checkpoint": _best_checkpoint(hist_a),
            },
            "run_b": {
                "run_id": run_id_b,
                "algorithm": hist_b["algorithm"],
                "timestamp": hist_b["timestamp"],
                "best_checkpoint": _best_checkpoint(hist_b),
            },
            "config_diff": config_diff,
            "metrics_comparison": metrics_comparison,
            "n_config_differences": len(config_diff),
        }

    def list_runs(self) -> list[dict]:
        """List all runs sorted by timestamp (newest first).

        Returns:
            List of run summary dicts.
        """
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT run_id, timestamp, algorithm, prompt_hash, notes FROM runs ORDER BY timestamp DESC"
            ).fetchall()

            results = []
            for row in rows:
                # Count metrics and checkpoints
                n_metrics = conn.execute(
                    "SELECT COUNT(DISTINCT step) FROM metrics WHERE run_id = ?",
                    (row["run_id"],),
                ).fetchone()[0]

                n_checkpoints = conn.execute(
                    "SELECT COUNT(*) FROM checkpoints WHERE run_id = ?",
                    (row["run_id"],),
                ).fetchone()[0]

                results.append(
                    {
                        "run_id": row["run_id"],
                        "timestamp": row["timestamp"],
                        "algorithm": row["algorithm"],
                        "prompt_hash": row["prompt_hash"],
                        "notes": row["notes"],
                        "n_steps_logged": n_metrics,
                        "n_checkpoints": n_checkpoints,
                    }
                )

        finally:
            conn.close()

        return results

    def export_run(self, run_id: str, output_path: str) -> None:
        """Export a full run (config + metrics + checkpoints) as JSON.

        Args:
            run_id: Run identifier.
            output_path: Destination .json file path.
        """
        history = self.get_run_history(run_id)

        # Convert step keys to strings for JSON serialization
        history["metrics_by_step"] = {
            str(step): metrics
            for step, metrics in history["metrics_by_step"].items()
        }
        history["exported_at"] = _now_iso()

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        print(f"Run {run_id} exported to {output_path}")

    # ------------------------------------------------------------------
    # Prompt versioning helpers
    # ------------------------------------------------------------------

    def list_prompt_versions(self) -> list[dict]:
        """List all tracked prompt templates and their usage counts.

        Returns:
            List of {prompt_hash, first_seen_at, times_used, preview} dicts.
        """
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT prompt_hash, prompt_template, first_seen_at, times_used "
                "FROM prompts ORDER BY first_seen_at"
            ).fetchall()
        finally:
            conn.close()

        results = []
        for row in rows:
            template = row["prompt_template"] or ""
            preview = template[:80] + "..." if len(template) > 80 else template
            results.append(
                {
                    "prompt_hash": row["prompt_hash"][:12] + "...",
                    "full_hash": row["prompt_hash"],
                    "first_seen_at": row["first_seen_at"],
                    "times_used": row["times_used"],
                    "preview": preview,
                }
            )
        return results

    def get_runs_by_prompt(self, prompt_hash: str) -> list[str]:
        """Find all run IDs that used a specific prompt template hash.

        Args:
            prompt_hash: Full or prefix SHA-256 hash.

        Returns:
            List of run_ids.
        """
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT run_id FROM runs WHERE prompt_hash LIKE ? ORDER BY timestamp",
                (prompt_hash + "%",),
            ).fetchall()
        finally:
            conn.close()
        return [row["run_id"] for row in rows]

    def delete_run(self, run_id: str) -> None:
        """Delete a run and all associated metrics and checkpoints.

        Args:
            run_id: Run identifier to delete.
        """
        conn = self._connect()
        try:
            conn.execute("DELETE FROM metrics WHERE run_id = ?", (run_id,))
            conn.execute("DELETE FROM checkpoints WHERE run_id = ?", (run_id,))
            conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
            conn.commit()
        finally:
            conn.close()
        print(f"Deleted run {run_id}")

    def __repr__(self) -> str:
        runs = self.list_runs()
        return f"ExperimentTracker(db={self.db_path!r}, n_runs={len(runs)})"
