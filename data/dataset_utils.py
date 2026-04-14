"""Utility functions for dataset loading and manipulation.

Provides fundamental I/O primitives and transformation helpers used
throughout the MiniAlign pipeline.
"""

import json
import random
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file and return a list of dicts.

    Args:
        path: Path to the .jsonl file. Each non-empty line must be valid JSON.

    Returns:
        List of parsed JSON objects.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If any line is malformed.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    records: list[dict] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise json.JSONDecodeError(
                    f"Malformed JSON on line {lineno} of {path}: {e.msg}",
                    e.doc,
                    e.pos,
                )

    return records


def save_jsonl(data: list[dict], path: str) -> None:
    """Save a list of dicts to a JSONL file (one JSON object per line).

    Args:
        data: List of serializable dicts.
        path: Destination file path. Parent directories are created if needed.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def split_dataset(
    data: list[dict],
    train_ratio: float = 0.9,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Randomly split data into train and validation sets.

    Args:
        data: List of data records.
        train_ratio: Fraction of data for training (0 < train_ratio < 1).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_data, val_data).
    """
    if not 0 < train_ratio < 1:
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")

    shuffled = data.copy()
    random.seed(seed)
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * train_ratio)
    return shuffled[:split_idx], shuffled[split_idx:]


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_chat_template(
    instruction: str,
    response: str,
    system_prompt: str = "You are a helpful assistant.",
) -> str:
    """Format an (instruction, response) pair as an Alpaca-style chat string.

    This produces a human-readable text string compatible with most
    causal LM SFT pipelines when a tokenizer chat template is unavailable.

    Args:
        instruction: The user's instruction or question.
        response: The assistant's response.
        system_prompt: System message placed at the top.

    Returns:
        Formatted string ready for tokenization.

    Example::

        >>> text = format_chat_template("What is 2+2?", "4.")
        >>> print(text)
        ### System:
        You are a helpful assistant.

        ### Instruction:
        What is 2+2?

        ### Response:
        4.<|endoftext|>
    """
    return (
        f"### System:\n{system_prompt}\n\n"
        f"### Instruction:\n{instruction}\n\n"
        f"### Response:\n{response}<|endoftext|>"
    )


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def deduplicate(data: list[dict], key: str = "instruction") -> list[dict]:
    """Remove duplicate records by exact string match on a specified key.

    Keeps the first occurrence of each unique value.

    Args:
        data: List of records.
        key: Dict key to deduplicate on.

    Returns:
        Deduplicated list preserving insertion order of first occurrences.
    """
    seen: set[Any] = set()
    unique: list[dict] = []

    for record in data:
        value = record.get(key)
        if value is None:
            # Keep records without the key (can't deduplicate them)
            unique.append(record)
            continue
        if value not in seen:
            seen.add(value)
            unique.append(record)

    removed = len(data) - len(unique)
    if removed:
        print(f"Deduplicated: removed {removed} duplicates ({len(unique)} remain)")

    return unique


# ---------------------------------------------------------------------------
# Length filtering
# ---------------------------------------------------------------------------


def filter_by_length(
    data: list[dict],
    min_len: int = 10,
    max_len: int = 2000,
    key: str = "response",
) -> list[dict]:
    """Filter records where the target field length is within [min_len, max_len].

    Length is measured in characters (not tokens) for speed.

    Args:
        data: List of records.
        min_len: Minimum character length (inclusive).
        max_len: Maximum character length (inclusive).
        key: The field whose length is checked. Falls back to "output" if key missing.

    Returns:
        Filtered list of records.
    """
    filtered: list[dict] = []
    too_short = 0
    too_long = 0

    for record in data:
        text = record.get(key, record.get("output", record.get("response", "")))
        char_len = len(str(text))

        if char_len < min_len:
            too_short += 1
        elif char_len > max_len:
            too_long += 1
        else:
            filtered.append(record)

    total_removed = len(data) - len(filtered)
    if total_removed:
        print(
            f"Length filter [{min_len}, {max_len}] chars on '{key}': "
            f"removed {too_short} too-short, {too_long} too-long "
            f"({len(filtered)}/{len(data)} remain)"
        )

    return filtered
