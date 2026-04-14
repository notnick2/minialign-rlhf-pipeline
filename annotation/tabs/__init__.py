from annotation.tabs.general_labeling import build_general_labeling_tab
from annotation.tabs.pairwise_preference import build_pairwise_preference_tab
from annotation.tabs.instruction_quality import build_instruction_quality_tab
from annotation.tabs.factuality import build_factuality_tab
from annotation.tabs.toxicity_bias import build_toxicity_bias_tab

__all__ = [
    "build_general_labeling_tab",
    "build_pairwise_preference_tab",
    "build_instruction_quality_tab",
    "build_factuality_tab",
    "build_toxicity_bias_tab",
]
