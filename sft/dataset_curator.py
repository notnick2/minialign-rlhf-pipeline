"""
SFT Dataset Curator
====================
Reads annotation data from SQLite and filters/exports high-quality
instruction-response pairs for supervised fine-tuning.

Curation pipeline:
  1. Pull instruction_quality annotations with overall_score >= threshold
  2. Pull factuality annotations where ALL claims are labeled "True"
  3. De-duplicate by exact response string match
  4. Export as JSONL: {"instruction": ..., "response": ..., "quality_score": ..., "source": ...}
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from annotation.storage.annotation_store import AnnotationStore

logger = logging.getLogger(__name__)


class SFTDatasetCurator:
    """
    Curates SFT training data from human annotation signals.

    Args:
        db_path: Path to the SQLite annotations database.
        quality_threshold: Minimum overall_score for instruction_quality annotations.
    """

    def __init__(
        self,
        db_path: str = "annotations.db",
        quality_threshold: float = 4.0,
    ):
        self.store = AnnotationStore(db_path=db_path)
        self.quality_threshold = quality_threshold

    # ------------------------------------------------------------------ #
    #  Curation Sources                                                    #
    # ------------------------------------------------------------------ #

    def curate_from_quality_ratings(self) -> List[Dict]:
        """
        Extract high-quality pairs from instruction_quality annotations.

        Returns pairs where the weighted overall score >= quality_threshold.
        """
        rows = self.store.get_instruction_quality(
            min_score=self.quality_threshold, limit=100_000
        )
        curated = []
        for r in rows:
            if not r.get("prompt") or not r.get("response"):
                continue
            curated.append({
                "instruction": r["prompt"],
                "response": r["response"],
                "quality_score": r["overall_score"],
                "annotator": r["annotator"],
                "source": "instruction_quality",
                "metadata": {
                    "relevance": r["relevance"],
                    "completeness": r["completeness"],
                    "accuracy": r["accuracy"],
                    "format_adherence": r["format_adherence"],
                    "conciseness": r["conciseness"],
                },
            })
        logger.info(
            f"Curated {len(curated)} pairs from quality ratings "
            f"(threshold >= {self.quality_threshold})"
        )
        return curated

    def curate_from_factuality(self) -> List[Dict]:
        """
        Extract pairs from factuality annotations where ALL claims are "True".

        These represent responses that have been claim-by-claim verified.
        """
        rows = self.store.get_factuality(limit=100_000)
        curated = []
        for r in rows:
            claims = r.get("claims", [])
            if not claims:
                continue
            all_true = all(c.get("label") == "True" for c in claims)
            if not all_true:
                continue
            curated.append({
                "instruction": "",   # Factuality tab stores response only
                "response": r["response"],
                "quality_score": 5.0,   # Max score for fully verified responses
                "annotator": r["annotator"],
                "source": "factuality_verified",
                "metadata": {
                    "n_claims_verified": len(claims),
                    "claims": [c["claim"] for c in claims],
                },
            })
        logger.info(f"Curated {len(curated)} fully-factual responses")
        return curated

    def curate_from_pairwise(self) -> List[Dict]:
        """
        Extract the winning response from pairwise comparisons as high-quality SFT data.
        """
        rows = self.store.get_pairwise(limit=100_000)
        curated = []
        for r in rows:
            preference = r.get("preference")
            if preference == "A":
                chosen_response = r["response_a"]
            elif preference == "B":
                chosen_response = r["response_b"]
            else:
                # Tie: skip (no clear winner)
                continue

            if r.get("confidence", 1) >= 2:  # Only include medium/high confidence
                curated.append({
                    "instruction": r["prompt"],
                    "response": chosen_response,
                    "quality_score": 3.5 + 0.5 * r["confidence"],  # 4.0 or 4.5
                    "annotator": r["annotator"],
                    "source": "pairwise_winner",
                    "metadata": {
                        "preference": preference,
                        "confidence": r["confidence"],
                    },
                })
        logger.info(f"Curated {len(curated)} pairs from pairwise winners")
        return curated

    # ------------------------------------------------------------------ #
    #  De-duplication                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _response_hash(response: str) -> str:
        """SHA256 hash of the response string for deduplication."""
        return hashlib.sha256(response.strip().encode("utf-8")).hexdigest()

    def deduplicate(self, examples: List[Dict]) -> List[Dict]:
        """
        Remove duplicate examples based on exact response string match.
        Keeps the instance with the highest quality score.
        """
        # Group by response hash, keep highest score
        best: Dict[str, Dict] = {}
        for ex in examples:
            key = self._response_hash(ex.get("response", ""))
            if key not in best or ex["quality_score"] > best[key]["quality_score"]:
                best[key] = ex

        deduped = list(best.values())
        n_removed = len(examples) - len(deduped)
        logger.info(
            f"Deduplication: {len(examples)} → {len(deduped)} "
            f"({n_removed} duplicates removed)"
        )
        return deduped

    # ------------------------------------------------------------------ #
    #  Export                                                              #
    # ------------------------------------------------------------------ #

    def curate_and_export(
        self,
        output_path: str = "data/sft_curated.jsonl",
        include_factuality: bool = True,
        include_pairwise: bool = True,
        min_response_length: int = 20,
    ) -> int:
        """
        Run full curation pipeline and export to JSONL.

        Returns:
            Number of examples exported.
        """
        all_examples: List[Dict] = []

        # Source 1: Instruction quality ratings
        all_examples.extend(self.curate_from_quality_ratings())

        # Source 2: Fully-verified factuality annotations
        if include_factuality:
            all_examples.extend(self.curate_from_factuality())

        # Source 3: Pairwise winners
        if include_pairwise:
            all_examples.extend(self.curate_from_pairwise())

        # Filter by minimum response length
        before = len(all_examples)
        all_examples = [
            ex for ex in all_examples
            if len(ex.get("response", "")) >= min_response_length
        ]
        logger.info(
            f"Filtered by min_response_length={min_response_length}: "
            f"{before} → {len(all_examples)}"
        )

        # De-duplicate
        all_examples = self.deduplicate(all_examples)

        # Sort by quality score descending
        all_examples.sort(key=lambda x: x["quality_score"], reverse=True)

        # Export
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for ex in all_examples:
                # Flatten metadata into a string for compatibility
                record = {
                    "instruction": ex["instruction"],
                    "response": ex["response"],
                    "quality_score": ex["quality_score"],
                    "source": ex["source"],
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info(f"Exported {len(all_examples)} SFT examples to {output_path}")
        return len(all_examples)

    def get_statistics(self) -> Dict:
        """Return statistics about available annotation data for curation."""
        stats = self.store.get_statistics()
        quality_rows = self.store.get_instruction_quality(
            min_score=self.quality_threshold
        )
        return {
            "total_annotations": stats["total"],
            "instruction_quality_total": stats["instruction_quality"],
            "instruction_quality_above_threshold": len(quality_rows),
            "quality_threshold": self.quality_threshold,
            "pairwise_total": stats["pairwise"],
            "factuality_total": stats["factuality"],
        }


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Curate SFT dataset from annotations")
    parser.add_argument("--db-path", default="annotations.db")
    parser.add_argument("--output", default="data/sft_curated.jsonl")
    parser.add_argument("--quality-threshold", type=float, default=4.0)
    args = parser.parse_args()

    curator = SFTDatasetCurator(
        db_path=args.db_path,
        quality_threshold=args.quality_threshold,
    )
    n = curator.curate_and_export(output_path=args.output)
    print(f"Curated {n} examples → {args.output}")
