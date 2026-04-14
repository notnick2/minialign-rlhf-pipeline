"""Data utilities for MiniAlign — dataset loading, formatting, and contrastive pair generation."""

from .dataset_utils import (
    deduplicate,
    filter_by_length,
    format_chat_template,
    load_jsonl,
    save_jsonl,
    split_dataset,
)
from .contrastive_pairs import (
    combine_all_sources,
    degrade_response,
    export_dpo_dataset,
    pairs_from_annotations,
    pairs_from_degradation,
    pairs_from_rlaif,
)

__all__ = [
    # dataset_utils
    "load_jsonl",
    "save_jsonl",
    "split_dataset",
    "format_chat_template",
    "deduplicate",
    "filter_by_length",
    # contrastive_pairs
    "degrade_response",
    "pairs_from_degradation",
    "pairs_from_annotations",
    "pairs_from_rlaif",
    "export_dpo_dataset",
    "combine_all_sources",
]
