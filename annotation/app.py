"""
MiniAlign Annotation Interface
================================
Gradio web application with 5 labeling tabs:
  1. General Labeling  — text classification, NER, sentiment, intent
  2. Pairwise Preference — A vs B comparison
  3. Instruction Quality — 1-5 rubric ratings
  4. Factuality — claim-level true/false/uncertain labels
  5. Toxicity & Bias — harmful content and bias flagging

Each tab stores annotations to SQLite and shows real-time IAA scores
(Cohen's Kappa and Krippendorff's Alpha).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import gradio as gr

from annotation.storage.annotation_store import AnnotationStore
from annotation.tabs.general_labeling import build_general_labeling_tab
from annotation.tabs.pairwise_preference import build_pairwise_preference_tab
from annotation.tabs.instruction_quality import build_instruction_quality_tab
from annotation.tabs.factuality import build_factuality_tab
from annotation.tabs.toxicity_bias import build_toxicity_bias_tab


def build_app(db_path: str = "annotations.db") -> gr.Blocks:
    """Construct the full Gradio annotation interface."""

    store = AnnotationStore(db_path=db_path)

    with gr.Blocks(
        title="MiniAlign — Annotation Interface",
        theme=gr.themes.Soft(),
        css="""
        .main-header { text-align: center; margin-bottom: 1rem; }
        .stats-panel { background: #f0f4f8; padding: 1rem; border-radius: 8px; }
        """,
    ) as app:
        gr.Markdown(
            """
# MiniAlign Annotation Interface
**End-to-end RLHF data collection platform** — annotate texts across 5 task types
for alignment training data creation.

> All annotations are stored in SQLite. Inter-annotator agreement (Cohen's κ and
> Krippendorff's α) is computed live after each save.
            """,
            elem_classes=["main-header"],
        )

        # ---- Overview panel ------------------------------------------------
        with gr.Accordion("Dataset Overview (click to expand)", open=False):
            overview_output = gr.Textbox(
                label="Current Annotation Statistics",
                interactive=False,
                lines=12,
            )
            refresh_overview_btn = gr.Button("Refresh Overview")

            def _get_overview() -> str:
                stats = store.get_statistics()
                lines = [
                    "=== MiniAlign Annotation Statistics ===",
                    f"Total annotations:         {stats['total']}",
                    f"  General labeling:        {stats['general_labeling']}",
                    f"  Pairwise preferences:    {stats['pairwise']}",
                    f"  Instruction quality:     {stats['instruction_quality']}",
                    f"  Factuality labels:       {stats['factuality']}",
                    f"  Toxicity/Bias labels:    {stats['toxicity_bias']}",
                    "",
                    "Annotators:",
                ]
                for ann, cnt in (stats.get("annotators") or {}).items():
                    lines.append(f"  {ann}: {cnt} annotations")
                return "\n".join(lines)

            refresh_overview_btn.click(
                fn=_get_overview,
                inputs=[],
                outputs=[overview_output],
            )

        # ---- Five annotation tabs ------------------------------------------
        build_general_labeling_tab(store)
        build_pairwise_preference_tab(store)
        build_instruction_quality_tab(store)
        build_factuality_tab(store)
        build_toxicity_bias_tab(store)

    return app


def main():
    parser = argparse.ArgumentParser(description="MiniAlign Annotation Interface")
    parser.add_argument(
        "--db-path",
        default="annotations.db",
        help="Path to SQLite database file (default: annotations.db)",
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=7860, help="Port to listen on (default: 7860)"
    )
    parser.add_argument(
        "--share", action="store_true", help="Create a public Gradio share link"
    )
    args = parser.parse_args()

    app = build_app(db_path=args.db_path)
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
