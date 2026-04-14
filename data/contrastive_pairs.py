"""Generate (chosen, rejected) pairs for preference learning.
Covers: Contrastive pair generation for preference learning (Cat 5)

Three complementary methods for building preference datasets:
  1. Degradation  — Use Claude to synthetically degrade good responses.
  2. Annotations  — Pull pairwise preferences from the Gradio annotation SQLite DB.
  3. RLAIF        — Use constitutional revision results (revised = chosen, original = rejected).
"""

import json
import sqlite3
from pathlib import Path
from typing import Optional

import anthropic
from tqdm import tqdm

from .dataset_utils import deduplicate, load_jsonl, save_jsonl

MODEL = "claude-haiku-4-5-20251001"

DEGRADATION_PROMPT = """You are helping create training data for AI safety research.

Below is a high-quality response to an instruction. Your task is to rewrite it to be WORSE by:
1. Introducing one subtle factual error (plausible but incorrect)
2. Making the response less complete (omit one useful point)
3. Slightly changing the tone to be less helpful or slightly condescending

Keep the degraded response plausible — it should look like a mediocre response, not an obviously bad one.

Original instruction:
{instruction}

Original response:
{response}

Write ONLY the degraded response (no explanation, no preamble):"""


# ---------------------------------------------------------------------------
# Method 1: Degradation
# ---------------------------------------------------------------------------


def degrade_response(
    instruction: str,
    good_response: str,
    api_key: str,
    max_tokens: int = 1024,
) -> str:
    """Use Claude to create a plausibly worse version of a good response.

    Args:
        instruction: The original instruction.
        good_response: The high-quality response to degrade.
        api_key: Anthropic API key.
        max_tokens: Max tokens for degraded output.

    Returns:
        Degraded response string.
    """
    client = anthropic.Anthropic(api_key=api_key)

    prompt = DEGRADATION_PROMPT.format(
        instruction=instruction,
        response=good_response,
    )

    message = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )

    return message.content[0].text.strip()


def pairs_from_degradation(
    dataset: list[dict],
    api_key: str,
    max_tokens: int = 1024,
) -> list[dict]:
    """Build preference pairs by degrading good responses.

    For each record with {instruction, response}, generates a degraded version
    as the rejected response and treats the original as the chosen response.

    Args:
        dataset: List of {instruction, response} dicts.
        api_key: Anthropic API key.
        max_tokens: Max tokens for degraded response.

    Returns:
        List of {prompt, chosen, rejected} dicts.
    """
    pairs: list[dict] = []
    errors = 0

    for item in tqdm(dataset, desc="Generating degradation pairs"):
        instruction = item.get("instruction", item.get("prompt", ""))
        good_response = item.get("response", item.get("output", ""))

        if not instruction or not good_response:
            continue

        try:
            degraded = degrade_response(
                instruction=instruction,
                good_response=good_response,
                api_key=api_key,
                max_tokens=max_tokens,
            )
            pairs.append(
                {
                    "prompt": instruction,
                    "chosen": good_response,
                    "rejected": degraded,
                    "source": "degradation",
                }
            )
        except Exception as e:
            errors += 1
            print(f"\nError degrading response: {e}")

    print(f"Degradation pairs: {len(pairs)} generated, {errors} errors")
    return pairs


# ---------------------------------------------------------------------------
# Method 2: From pairwise annotations in SQLite
# ---------------------------------------------------------------------------


def pairs_from_annotations(db_path: str) -> list[dict]:
    """Read pairwise preference annotations from the annotation app's SQLite store.

    Expects a table named 'pairwise_annotations' with columns:
      - instruction (or prompt): str
      - response_a: str
      - response_b: str
      - preferred: 'A' | 'B' | 'tie'
      - annotator_id: str (optional)

    Falls back gracefully if the table schema differs slightly.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        List of {prompt, chosen, rejected, source, annotator_id} dicts.
        Tie annotations are skipped.
    """
    db = Path(db_path)
    if not db.exists():
        raise FileNotFoundError(f"Annotation database not found: {db_path}")

    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Discover table name — support both 'pairwise_annotations' and 'annotations'
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]

    table_name = None
    for candidate in ("pairwise_annotations", "pairwise", "annotations"):
        if candidate in tables:
            table_name = candidate
            break

    if table_name is None:
        conn.close()
        raise RuntimeError(
            f"Could not find pairwise annotations table in {db_path}. "
            f"Available tables: {tables}"
        )

    # Fetch rows
    cursor.execute(f"SELECT * FROM {table_name}")  # noqa: S608
    rows = cursor.fetchall()
    conn.close()

    pairs: list[dict] = []
    skipped_ties = 0

    for row in rows:
        row_dict = dict(row)
        preferred = str(row_dict.get("preferred", "")).strip().upper()

        if preferred == "TIE" or preferred == "":
            skipped_ties += 1
            continue

        prompt = row_dict.get("instruction") or row_dict.get("prompt") or ""
        response_a = row_dict.get("response_a", "")
        response_b = row_dict.get("response_b", "")

        if preferred == "A":
            chosen, rejected = response_a, response_b
        elif preferred == "B":
            chosen, rejected = response_b, response_a
        else:
            skipped_ties += 1
            continue

        pairs.append(
            {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "source": "human_annotation",
                "annotator_id": row_dict.get("annotator_id", "unknown"),
            }
        )

    print(
        f"Annotation pairs: {len(pairs)} loaded from '{table_name}', "
        f"{skipped_ties} ties skipped"
    )
    return pairs


# ---------------------------------------------------------------------------
# Method 3: From constitutional RLAIF
# ---------------------------------------------------------------------------


def pairs_from_rlaif(rlaif_results_path: str) -> list[dict]:
    """Build preference pairs from constitutional AI revision results.

    RLAIF output format (JSONL):
      {prompt, original_response, revised_response, ...}

    The revised (constitutionally improved) response is the chosen response;
    the original is the rejected response.

    Args:
        rlaif_results_path: Path to RLAIF output JSONL.

    Returns:
        List of {prompt, chosen, rejected, source} dicts.
    """
    records = load_jsonl(rlaif_results_path)
    pairs: list[dict] = []
    skipped = 0

    for rec in records:
        prompt = rec.get("prompt", rec.get("instruction", ""))
        original = rec.get("original_response", rec.get("response", ""))
        revised = rec.get("revised_response", rec.get("revision", ""))

        if not prompt or not original or not revised:
            skipped += 1
            continue

        # Only include if the revised response is meaningfully different
        if revised.strip() == original.strip():
            skipped += 1
            continue

        pairs.append(
            {
                "prompt": prompt,
                "chosen": revised,
                "rejected": original,
                "source": "rlaif_constitutional",
            }
        )

    print(f"RLAIF pairs: {len(pairs)} loaded, {skipped} skipped")
    return pairs


# ---------------------------------------------------------------------------
# Export and combining
# ---------------------------------------------------------------------------


def export_dpo_dataset(pairs: list[dict], output_path: str) -> None:
    """Save preference pairs as JSONL and print statistics.

    Args:
        pairs: List of {prompt, chosen, rejected} dicts.
        output_path: Destination JSONL path.
    """
    if not pairs:
        print("Warning: No pairs to export.")
        return

    save_jsonl(pairs, output_path)

    # Statistics
    chosen_lengths = [len(p.get("chosen", "")) for p in pairs]
    rejected_lengths = [len(p.get("rejected", "")) for p in pairs]
    prompt_lengths = [len(p.get("prompt", "")) for p in pairs]

    avg_chosen = sum(chosen_lengths) / len(chosen_lengths)
    avg_rejected = sum(rejected_lengths) / len(rejected_lengths)
    avg_prompt = sum(prompt_lengths) / len(prompt_lengths)

    sources: dict[str, int] = {}
    for p in pairs:
        src = p.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1

    print(f"\nDPO Dataset Export Statistics")
    print(f"  Total pairs:         {len(pairs)}")
    print(f"  Avg prompt length:   {avg_prompt:.0f} chars")
    print(f"  Avg chosen length:   {avg_chosen:.0f} chars")
    print(f"  Avg rejected length: {avg_rejected:.0f} chars")
    print(f"  Source breakdown:")
    for src, count in sorted(sources.items()):
        print(f"    {src}: {count}")
    print(f"  Saved to: {output_path}")


def combine_all_sources(
    degradation_path: Optional[str],
    annotations_db: Optional[str],
    rlaif_path: Optional[str],
    output_path: str,
) -> list[dict]:
    """Combine preference pairs from all three sources, deduplicate, and save.

    Any source path that is None or does not exist is skipped gracefully.

    Args:
        degradation_path: Path to JSONL of degradation-generated pairs (or None).
        annotations_db: Path to SQLite annotation DB (or None).
        rlaif_path: Path to RLAIF results JSONL (or None).
        output_path: Destination for the combined JSONL.

    Returns:
        Combined and deduplicated list of preference pairs.
    """
    all_pairs: list[dict] = []

    # Load degradation pairs
    if degradation_path and Path(degradation_path).exists():
        try:
            deg_pairs = load_jsonl(degradation_path)
            all_pairs.extend(deg_pairs)
            print(f"Loaded {len(deg_pairs)} degradation pairs from {degradation_path}")
        except Exception as e:
            print(f"Warning: Could not load degradation pairs: {e}")

    # Load annotation pairs
    if annotations_db and Path(annotations_db).exists():
        try:
            ann_pairs = pairs_from_annotations(annotations_db)
            all_pairs.extend(ann_pairs)
        except Exception as e:
            print(f"Warning: Could not load annotation pairs: {e}")

    # Load RLAIF pairs
    if rlaif_path and Path(rlaif_path).exists():
        try:
            rlaif_pairs = pairs_from_rlaif(rlaif_path)
            all_pairs.extend(rlaif_pairs)
        except Exception as e:
            print(f"Warning: Could not load RLAIF pairs: {e}")

    if not all_pairs:
        print("Warning: No pairs loaded from any source.")
        return []

    print(f"\nTotal pairs before deduplication: {len(all_pairs)}")

    # Deduplicate on (prompt, chosen) to avoid exact duplicates
    seen: set[tuple] = set()
    unique_pairs: list[dict] = []
    for pair in all_pairs:
        key = (pair.get("prompt", ""), pair.get("chosen", ""))
        if key not in seen:
            seen.add(key)
            unique_pairs.append(pair)

    removed = len(all_pairs) - len(unique_pairs)
    if removed:
        print(f"Deduplication removed {removed} pairs ({len(unique_pairs)} remain)")

    export_dpo_dataset(unique_pairs, output_path)
    return unique_pairs
