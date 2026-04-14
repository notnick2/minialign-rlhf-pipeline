"""
Tab 1: General Labeling
Covers: text classification, NER span annotation, sentiment analysis, intent detection.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

import gradio as gr
import pandas as pd

from annotation.agreement.iaa import compute_all_agreement, interpret_kappa

if TYPE_CHECKING:
    from annotation.storage.annotation_store import AnnotationStore

CATEGORY_CHOICES = [
    "science", "technology", "politics", "sports", "entertainment",
    "health", "finance", "education", "other"
]

SENTIMENT_CHOICES = ["positive", "negative", "neutral", "mixed"]
INTENT_CHOICES = ["question", "command", "complaint", "greeting", "other"]
NER_LABELS = ["NAME", "ORG", "LOCATION", "DATE", "OTHER"]


def _parse_ner_input(raw: str) -> list[dict]:
    """
    Parse user-entered NER spans from a simple format:
      <span text> | <LABEL>
    One span per line.  Returns a list of dicts.
    """
    spans = []
    if not raw or not raw.strip():
        return spans
    for line in raw.strip().splitlines():
        line = line.strip()
        if "|" in line:
            parts = line.rsplit("|", 1)
            text = parts[0].strip()
            label = parts[1].strip().upper()
            if label not in NER_LABELS:
                label = "OTHER"
            spans.append({"text": text, "label": label})
        elif line:
            spans.append({"text": line, "label": "OTHER"})
    return spans


def build_general_labeling_tab(store: "AnnotationStore") -> gr.Tab:
    """Build and return the Gradio Tab component for general labeling."""

    def _save(
        annotator: str,
        text: str,
        category: str,
        ner_raw: str,
        sentiment: str,
        intent: str,
        notes: str,
    ) -> tuple[str, str]:
        if not annotator.strip():
            return "Error: annotator name is required.", ""
        if not text.strip():
            return "Error: text is required.", ""

        ner_spans = _parse_ner_input(ner_raw)
        row_id = store.save_general_label(
            annotator=annotator.strip(),
            text=text.strip(),
            category=category or None,
            ner_spans=ner_spans if ner_spans else None,
            sentiment=sentiment or None,
            intent=intent or None,
            notes=notes,
        )

        # Recompute IAA
        rows = store.get_general_labels(limit=5000)
        if rows:
            df = pd.DataFrame(rows)
            df = df.rename(columns={"id": "item_id"})
            # Use sentiment as the label for agreement
            df["label"] = df["sentiment"].fillna("")
            iaa = compute_all_agreement(df[["item_id", "annotator", "label"]])
            kappa_str = (
                f"{iaa['cohen_kappa']:.3f} ({interpret_kappa(iaa['cohen_kappa'])})"
                if iaa["n_annotators"] == 2 and iaa["n_annotations"] >= 2
                else "N/A (need 2 annotators on same items)"
            )
            alpha_str = f"{iaa['krippendorff_alpha']:.3f}" if iaa["n_annotations"] >= 2 else "N/A"
            iaa_display = (
                f"Annotations: {iaa['n_annotations']} | "
                f"Annotators: {iaa['n_annotators']} | "
                f"Cohen's κ: {kappa_str} | "
                f"Krippendorff's α: {alpha_str}"
            )
        else:
            iaa_display = "No annotations yet."

        return f"Saved annotation #{row_id}", iaa_display

    def _load_recent() -> str:
        rows = store.get_general_labels(limit=10)
        if not rows:
            return "No annotations yet."
        lines = []
        for r in rows:
            lines.append(
                f"[{r['created_at'][:19]}] {r['annotator']} | "
                f"cat={r['category']} | sent={r['sentiment']} | "
                f"intent={r['intent']}"
            )
        return "\n".join(lines)

    with gr.Tab("1. General Labeling") as tab:
        gr.Markdown("""
## Text Classification, NER, Sentiment & Intent Labeling
Enter a text snippet and annotate its category, named entities, sentiment, and intent.
        """)

        with gr.Row():
            annotator_input = gr.Textbox(label="Annotator ID", placeholder="e.g. annotator_01", scale=1)

        text_input = gr.Textbox(
            label="Text to Annotate",
            placeholder="Paste the text you want to label here...",
            lines=5,
        )

        with gr.Row():
            category_dd = gr.Dropdown(
                choices=CATEGORY_CHOICES,
                label="Category",
                value=None,
                allow_custom_value=True,
            )
            sentiment_radio = gr.Radio(
                choices=SENTIMENT_CHOICES,
                label="Sentiment",
                value=None,
            )
            intent_radio = gr.Radio(
                choices=INTENT_CHOICES,
                label="Intent",
                value=None,
            )

        with gr.Accordion("Named Entity Recognition (NER)", open=True):
            gr.Markdown(
                "Enter one entity per line in the format: `entity text | LABEL`\n\n"
                f"Valid labels: {', '.join(NER_LABELS)}"
            )
            ner_input = gr.Textbox(
                label="NER Spans",
                placeholder="Apple Inc. | ORG\nJohn Smith | NAME\nNew York | LOCATION",
                lines=4,
            )

        notes_input = gr.Textbox(label="Notes (optional)", lines=2)

        with gr.Row():
            save_btn = gr.Button("Save Annotation", variant="primary")
            clear_btn = gr.Button("Clear Form")

        status_output = gr.Textbox(label="Status", interactive=False)
        iaa_output = gr.Textbox(label="Inter-Annotator Agreement", interactive=False)

        gr.Markdown("### Recent Annotations")
        recent_output = gr.Textbox(
            label="Recent (last 10)", interactive=False, lines=8
        )
        refresh_btn = gr.Button("Refresh Recent")

        # Wire up events
        save_btn.click(
            fn=_save,
            inputs=[annotator_input, text_input, category_dd, ner_input,
                    sentiment_radio, intent_radio, notes_input],
            outputs=[status_output, iaa_output],
        )

        def _clear():
            return "", "", None, "", None, None, ""

        clear_btn.click(
            fn=_clear,
            inputs=[],
            outputs=[text_input, ner_input, category_dd, notes_input,
                     sentiment_radio, intent_radio, annotator_input],
        )

        refresh_btn.click(fn=_load_recent, inputs=[], outputs=[recent_output])

    return tab
