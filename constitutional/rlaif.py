"""
RLAIF Pipeline — Constitutional AI via Critique → Revise → Preference Pairs

For each (prompt, response) pair:
  1. For each constitutional principle, ask Claude to critique the response
  2. Ask Claude to revise the response to address the critique
  3. Output (original_response, revised_response) as a contrastive preference pair
     where revised = chosen (better) and original = rejected (worse)
  4. Also output the critique text as auxiliary training signal

Reference:
  Bai et al., "Constitutional AI: Harmlessness from AI Feedback" (2022)
  https://arxiv.org/abs/2212.06950
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, List, Optional

from constitutional.principles import CONSTITUTION, ConstitutionalPrinciple

logger = logging.getLogger(__name__)


@dataclass
class CritiqueResult:
    """Output of the critique phase for one principle."""
    principle_name: str
    principle_description: str
    critique: str


@dataclass
class RevisionResult:
    """Output of the revision phase for one principle."""
    principle_name: str
    original_response: str
    revised_response: str
    critique: str


@dataclass
class RLAIFPair:
    """A contrastive preference pair produced by constitutional revision."""
    prompt: str
    chosen: str           # revised response (better)
    rejected: str         # original response (worse)
    critique: str         # the critique that motivated the revision
    principle: str        # which principle triggered the revision
    improvement_score: float  # estimated improvement (0-1)


class RLAIFPipeline:
    """
    Constitutional AI RLAIF pipeline.

    Requires ANTHROPIC_API_KEY environment variable to be set.
    """

    def __init__(
        self,
        model: str = "claude-3-haiku-20240307",
        principles: Optional[List[ConstitutionalPrinciple]] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        min_revision_length_ratio: float = 0.5,
    ):
        """
        Args:
            model: Anthropic model to use for critique and revision.
            principles: List of principles to apply (defaults to full CONSTITUTION).
            max_retries: Number of API call retries on transient errors.
            retry_delay: Seconds to wait between retries.
            min_revision_length_ratio: Minimum ratio of revision to original length
                                       to consider the revision non-trivial.
        """
        try:
            import anthropic
            self.client = anthropic.Anthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY")
            )
        except ImportError:
            raise ImportError(
                "anthropic package is required for RLAIF. "
                "Install with: pip install anthropic"
            )

        self.model = model
        self.principles = principles or CONSTITUTION
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.min_revision_length_ratio = min_revision_length_ratio

    def _call_claude(self, prompt: str) -> str:
        """Call Claude API with retry logic."""
        import anthropic

        for attempt in range(self.max_retries):
            try:
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                )
                return message.content[0].text.strip()
            except anthropic.RateLimitError:
                if attempt < self.max_retries - 1:
                    wait = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Rate limited. Waiting {wait}s before retry {attempt+1}.")
                    time.sleep(wait)
                else:
                    raise
            except anthropic.APIError as e:
                logger.error(f"API error on attempt {attempt+1}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    raise

        return ""

    def critique(self, response: str, principle: ConstitutionalPrinciple) -> CritiqueResult:
        """Apply one principle's critique prompt to a response."""
        prompt = principle.critique_prompt.format(response=response)
        critique_text = self._call_claude(prompt)
        return CritiqueResult(
            principle_name=principle.name,
            principle_description=principle.description,
            critique=critique_text,
        )

    def revise(
        self,
        response: str,
        critique_result: CritiqueResult,
        principle: ConstitutionalPrinciple,
    ) -> RevisionResult:
        """Apply one principle's revision prompt to generate an improved response."""
        prompt = principle.revision_prompt.format(
            response=response,
            critique=critique_result.critique,
        )
        revised = self._call_claude(prompt)
        return RevisionResult(
            principle_name=principle.name,
            original_response=response,
            revised_response=revised,
            critique=critique_result.critique,
        )

    def _estimate_improvement(
        self, original: str, revised: str, critique: str
    ) -> float:
        """
        Heuristic improvement score (0-1).
        A real implementation would score with the reward model.
        Here we use length delta and non-trivial revision as proxy.
        """
        if not revised or revised.strip() == original.strip():
            return 0.0
        len_ratio = len(revised) / max(len(original), 1)
        # Too short a revision might be hallucinated / empty
        if len_ratio < self.min_revision_length_ratio:
            return 0.1
        # Reward non-trivial revisions
        word_overlap = len(
            set(original.lower().split()) & set(revised.lower().split())
        ) / max(len(set(original.lower().split())), 1)
        # Lower overlap = more substantial revision
        improvement = min(1.0, 0.3 + 0.7 * (1 - word_overlap) * len_ratio)
        return round(improvement, 3)

    def process_single(
        self,
        prompt: str,
        response: str,
        principles: Optional[List[ConstitutionalPrinciple]] = None,
    ) -> List[RLAIFPair]:
        """
        Apply all principles to one (prompt, response) pair.

        Returns a list of RLAIFPair objects — one per principle that produced
        a non-trivial revision.
        """
        active_principles = principles or self.principles
        pairs: List[RLAIFPair] = []

        for principle in active_principles:
            try:
                logger.info(f"Applying principle: {principle.name}")

                # Step 1: Critique
                critique_result = self.critique(response, principle)

                # Step 2: Revise
                revision_result = self.revise(response, critique_result, principle)

                revised = revision_result.revised_response
                if not revised or revised.strip() == response.strip():
                    logger.info(f"  No meaningful revision for principle {principle.name}")
                    continue

                # Step 3: Build preference pair
                improvement = self._estimate_improvement(
                    response, revised, critique_result.critique
                )
                pair = RLAIFPair(
                    prompt=prompt,
                    chosen=revised,
                    rejected=response,
                    critique=critique_result.critique,
                    principle=principle.name,
                    improvement_score=improvement,
                )
                pairs.append(pair)
                logger.info(
                    f"  Revision generated (improvement={improvement:.2f})"
                )

            except Exception as e:
                logger.error(f"Error processing principle {principle.name}: {e}")
                continue

        return pairs

    def process_batch(
        self,
        examples: List[dict],
        prompt_key: str = "prompt",
        response_key: str = "response",
        principles: Optional[List[ConstitutionalPrinciple]] = None,
        delay_between: float = 0.5,
    ) -> Iterator[RLAIFPair]:
        """
        Batch process a dataset of (prompt, response) dicts.

        Args:
            examples: List of dicts with at least prompt_key and response_key.
            prompt_key: Key for the prompt field.
            response_key: Key for the response field.
            principles: Subset of principles to apply (defaults to all).
            delay_between: Seconds to sleep between examples (rate limiting).

        Yields:
            RLAIFPair objects for each non-trivial revision produced.
        """
        total = len(examples)
        for i, example in enumerate(examples):
            prompt = example.get(prompt_key, "")
            response = example.get(response_key, "")

            if not prompt or not response:
                logger.warning(f"Skipping example {i}: missing prompt or response")
                continue

            logger.info(f"Processing example {i+1}/{total}")
            pairs = self.process_single(prompt, response, principles)
            for pair in pairs:
                yield pair

            if delay_between > 0 and i < total - 1:
                time.sleep(delay_between)

    def export_pairs(
        self,
        pairs: List[RLAIFPair],
        output_path: str,
        min_improvement: float = 0.2,
    ) -> int:
        """
        Export preference pairs to JSONL format.

        Args:
            pairs: List of RLAIFPair objects.
            output_path: Path to write JSONL file.
            min_improvement: Minimum improvement score to include a pair.

        Returns:
            Number of pairs written.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        n_written = 0
        with open(output_path, "w", encoding="utf-8") as f:
            for pair in pairs:
                if pair.improvement_score < min_improvement:
                    continue
                record = {
                    "prompt": pair.prompt,
                    "chosen": pair.chosen,
                    "rejected": pair.rejected,
                    "critique": pair.critique,
                    "principle": pair.principle,
                    "improvement_score": pair.improvement_score,
                    "source": "rlaif_constitutional",
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_written += 1

        logger.info(f"Exported {n_written} pairs to {output_path}")
        return n_written


# ======================================================================= #
#  CLI entry point                                                           #
# ======================================================================= #

def run_rlaif_on_dataset(
    input_path: str,
    output_path: str,
    model: str = "claude-3-haiku-20240307",
    min_improvement: float = 0.2,
):
    """
    Run the RLAIF pipeline on a JSONL dataset.

    Each line in input_path must be a JSON object with 'prompt' and 'response' fields.
    """
    logging.basicConfig(level=logging.INFO)

    pipeline = RLAIFPipeline(model=model)
    examples = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    logger.info(f"Loaded {len(examples)} examples from {input_path}")

    all_pairs: List[RLAIFPair] = []
    for pair in pipeline.process_batch(examples):
        all_pairs.append(pair)

    n = pipeline.export_pairs(all_pairs, output_path, min_improvement=min_improvement)
    print(f"Done: exported {n} constitutional preference pairs to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run RLAIF pipeline on a JSONL dataset")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output JSONL file for preference pairs")
    parser.add_argument("--model", default="claude-3-haiku-20240307", help="Anthropic model ID")
    parser.add_argument("--min-improvement", type=float, default=0.2)
    args = parser.parse_args()

    run_rlaif_on_dataset(args.input, args.output, args.model, args.min_improvement)
