"""Persona-conditioned instruction/response generation.
Covers: Persona-conditioned generation (Cat 5)

Generates responses for 6 distinct personas using the Anthropic SDK.
Each persona produces stylistically and linguistically distinct outputs
from the same instruction, enabling diverse SFT training data.
"""

import json
import os
from pathlib import Path
from typing import Optional

import anthropic
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Persona definitions
# ---------------------------------------------------------------------------

PERSONAS: dict[str, dict] = {
    "domain_expert": {
        "name": "domain_expert",
        "system_prompt": (
            "You are a seasoned domain expert with deep technical knowledge. "
            "Your responses are precise, use field-specific terminology correctly, "
            "cite relevant concepts and frameworks, assume the reader has graduate-level "
            "familiarity with the subject, and are structured and authoritative."
        ),
        "style_description": "Technical, precise, uses domain terminology, authoritative tone",
    },
    "curious_student": {
        "name": "curious_student",
        "system_prompt": (
            "You are an enthusiastic student who loves learning. "
            "Your responses build understanding from first principles, connect new ideas "
            "to things you already know, express genuine curiosity, ask follow-up questions "
            "at the end, and are warm and energetic in tone."
        ),
        "style_description": "Enthusiastic, builds from fundamentals, asks follow-up questions",
    },
    "layperson": {
        "name": "layperson",
        "system_prompt": (
            "You are a friendly explainer who communicates complex ideas in plain language. "
            "Never use jargon without immediately defining it. Use everyday analogies and "
            "relatable examples. Keep sentences short. Assume the reader has no background "
            "in the subject. Your goal is clarity above all else."
        ),
        "style_description": "Plain language, analogies, no jargon, clarity-focused",
    },
    "skeptic": {
        "name": "skeptic",
        "system_prompt": (
            "You are a careful, critical thinker. Always mention caveats, limitations, "
            "and where the evidence is uncertain or contested. Acknowledge alternative "
            "viewpoints. Flag when a claim is correlation vs causation. Your tone is measured "
            "and you prefer nuance over sweeping statements. You do not give false certainty."
        ),
        "style_description": "Mentions caveats, acknowledges uncertainty, critical mindset",
    },
    "non_native_speaker": {
        "name": "non_native_speaker",
        "system_prompt": (
            "You are a helpful assistant who writes in simple, clear English suitable for "
            "people who are still learning the language. Use short, direct sentences. "
            "Prefer common words over rare ones. Avoid idioms, phrasal verbs, and culturally "
            "specific references. Repeat key ideas to reinforce understanding."
        ),
        "style_description": "Simple sentences, common vocabulary, avoids idioms",
    },
    "child": {
        "name": "child",
        "system_prompt": (
            "You are explaining things to a curious 10-year-old. Use very simple words, "
            "fun comparisons, and short sentences. Make it exciting and approachable. "
            "Use relatable analogies from everyday life like toys, food, or games. "
            "Never use complex vocabulary. Keep the energy high and make learning feel like play."
        ),
        "style_description": "Very simple, fun analogies, for a 10-year-old, playful",
    },
}

MODEL = "claude-haiku-4-5-20251001"


# ---------------------------------------------------------------------------
# Core generation functions
# ---------------------------------------------------------------------------


def generate_persona_response(
    instruction: str,
    persona_name: str,
    api_key: str,
    max_tokens: int = 1024,
) -> dict:
    """Generate a response to an instruction conditioned on a specific persona.

    Args:
        instruction: The user instruction to respond to.
        persona_name: One of the 6 defined persona keys.
        api_key: Anthropic API key.
        max_tokens: Maximum tokens for the response.

    Returns:
        dict with keys: instruction, persona, response, tokens_used
    """
    if persona_name not in PERSONAS:
        raise ValueError(
            f"Unknown persona '{persona_name}'. "
            f"Choose from: {list(PERSONAS.keys())}"
        )

    persona = PERSONAS[persona_name]
    client = anthropic.Anthropic(api_key=api_key)

    message = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=persona["system_prompt"],
        messages=[{"role": "user", "content": instruction}],
    )

    response_text = message.content[0].text
    tokens_used = message.usage.input_tokens + message.usage.output_tokens

    return {
        "instruction": instruction,
        "persona": persona_name,
        "style_description": persona["style_description"],
        "response": response_text,
        "tokens_used": tokens_used,
        "model": MODEL,
    }


def generate_all_personas(
    instruction: str,
    api_key: str,
    personas: Optional[list[str]] = None,
    max_tokens: int = 1024,
) -> list[dict]:
    """Generate responses for all (or selected) personas for one instruction.

    Args:
        instruction: The user instruction.
        api_key: Anthropic API key.
        personas: Optional list of persona names. Defaults to all 6.
        max_tokens: Maximum tokens per response.

    Returns:
        List of response dicts (one per persona).
    """
    target_personas = personas if personas is not None else list(PERSONAS.keys())
    results = []

    for persona_name in target_personas:
        result = generate_persona_response(
            instruction=instruction,
            persona_name=persona_name,
            api_key=api_key,
            max_tokens=max_tokens,
        )
        results.append(result)

    return results


def process_dataset(
    instructions_path: str,
    output_path: str,
    api_key: str,
    personas: Optional[list[str]] = None,
    max_tokens: int = 1024,
) -> None:
    """Read instructions from JSONL, generate persona responses, write to output JSONL.

    Input JSONL format: one JSON object per line with at least an "instruction" key.
    Output JSONL format: one JSON object per line with instruction, persona, response, tokens_used.

    Args:
        instructions_path: Path to input JSONL file with instructions.
        output_path: Path to output JSONL file.
        api_key: Anthropic API key.
        personas: Optional list of persona names to use. Defaults to all 6.
        max_tokens: Maximum tokens per response.
    """
    instructions_file = Path(instructions_path)
    if not instructions_file.exists():
        raise FileNotFoundError(f"Instructions file not found: {instructions_path}")

    # Load instructions
    instructions = []
    with open(instructions_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                instructions.append(obj)

    if not instructions:
        raise ValueError(f"No instructions found in {instructions_path}")

    target_personas = personas if personas is not None else list(PERSONAS.keys())
    total_tasks = len(instructions) * len(target_personas)

    print(f"Processing {len(instructions)} instructions x {len(target_personas)} personas = {total_tasks} API calls")
    print(f"Personas: {target_personas}")

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    total_tokens = 0
    errors = 0

    with open(output_file, "w", encoding="utf-8") as out_f:
        with tqdm(total=total_tasks, desc="Generating persona responses") as pbar:
            for item in instructions:
                instruction = item.get("instruction", item.get("prompt", ""))
                if not instruction:
                    pbar.update(len(target_personas))
                    continue

                for persona_name in target_personas:
                    try:
                        result = generate_persona_response(
                            instruction=instruction,
                            persona_name=persona_name,
                            api_key=api_key,
                            max_tokens=max_tokens,
                        )
                        # Carry over any extra fields from input
                        for k, v in item.items():
                            if k not in result:
                                result[k] = v

                        out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                        total_tokens += result["tokens_used"]

                    except Exception as e:
                        errors += 1
                        print(f"\nError on instruction='{instruction[:60]}...', persona={persona_name}: {e}")

                    pbar.update(1)

    print(f"\nDone. Output written to: {output_path}")
    print(f"Total tokens used: {total_tokens:,}")
    if errors:
        print(f"Errors encountered: {errors}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    """Simple CLI for persona generation."""
    import argparse

    parser = argparse.ArgumentParser(description="Persona-conditioned response generation")
    parser.add_argument("--instructions", required=True, help="Input JSONL path")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--api-key", default=None, help="Anthropic API key (or set ANTHROPIC_API_KEY)")
    parser.add_argument(
        "--personas",
        nargs="+",
        choices=list(PERSONAS.keys()),
        default=None,
        help="Personas to use (default: all 6)",
    )
    parser.add_argument("--max-tokens", type=int, default=1024)

    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("Provide --api-key or set ANTHROPIC_API_KEY environment variable")

    process_dataset(
        instructions_path=args.instructions,
        output_path=args.output,
        api_key=api_key,
        personas=args.personas,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
