"""
Tab 2: Pairwise Preference Annotation
Annotators compare two model responses (A vs B) on a shared prompt.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import gradio as gr
import pandas as pd

from annotation.agreement.iaa import compute_all_agreement, interpret_kappa

if TYPE_CHECKING:
    from annotation.storage.annotation_store import AnnotationStore

PREFERENCE_CHOICES = ["A is better", "B is better", "Tie"]
REASON_CHOICES = [
    "more accurate",
    "safer / less harmful",
    "better formatted",
    "more helpful",
    "more concise",
    "better reasoning",
    "more honest",
]


def build_pairwise_preference_tab(store: "AnnotationStore") -> gr.Tab:
    """Build and return the Gradio Tab component for pairwise preference annotation."""

    def _save(
        annotator: str,
        prompt: str,
        response_a: str,
        response_b: str,
        preference: str,
        reasons: list[str],
        confidence: int,
        notes: str,
    ) -> tuple[str, str]:
        if not annotator.strip():
            return "Error: annotator name is required.", ""
        if not prompt.strip():
            return "Error: prompt is required.", ""
        if not response_a.strip() or not response_b.strip():
            return "Error: both Response A and B are required.", ""
        if not preference:
            return "Error: preference selection is required.", ""

        # Map display text to storage value
        pref_map = {"A is better": "A", "B is better": "B", "Tie": "tie"}
        pref_value = pref_map.get(preference, preference)

        row_id = store.save_pairwise(
            annotator=annotator.strip(),
            prompt=prompt.strip(),
            response_a=response_a.strip(),
            response_b=response_b.strip(),
            preference=pref_value,
            reasons=reasons or [],
            confidence=int(confidence),
            notes=notes,
        )

        # Recompute IAA on preference labels
        rows = store.get_pairwise(limit=5000)
        if rows:
            df = pd.DataFrame(rows)
            df = df.rename(columns={"id": "item_id"})
            df["label"] = df["preference"]
            iaa = compute_all_agreement(df[["item_id", "annotator", "label"]])
            kappa_str = (
                f"{iaa['cohen_kappa']:.3f} ({interpret_kappa(iaa['cohen_kappa'])})"
                if iaa["n_annotators"] == 2 and iaa["n_annotations"] >= 2
                else "N/A (need 2 annotators)"
            )
            alpha_str = (
                f"{iaa['krippendorff_alpha']:.3f}"
                if iaa["n_annotations"] >= 2
                else "N/A"
            )
            iaa_display = (
                f"Annotations: {iaa['n_annotations']} | "
                f"Annotators: {iaa['n_annotators']} | "
                f"Cohen's κ: {kappa_str} | "
                f"Krippendorff's α: {alpha_str}"
            )
        else:
            iaa_display = "No annotations yet."

        return f"Saved pairwise preference #{row_id}", iaa_display

    def _load_stats() -> str:
        rows = store.get_pairwise(limit=5000)
        if not rows:
            return "No annotations yet."
        df = pd.DataFrame(rows)
        counts = df["preference"].value_counts().to_dict()
        total = len(df)
        lines = [f"Total comparisons: {total}"]
        for pref, cnt in counts.items():
            lines.append(f"  {pref}: {cnt} ({100*cnt/total:.1f}%)")
        # Most common reasons
        import json
        all_reasons: list[str] = []
        for r in rows:
            try:
                all_reasons.extend(json.loads(r["reasons"]) if r["reasons"] else [])
            except Exception:
                # Keep UI responsive even if legacy rows contain invalid JSON.
                continue
        if all_reasons:
            from collections import Counter
            top = Counter(all_reasons).most_common(5)
            lines.append("\nTop reasons:")
            for reason, cnt in top:
                lines.append(f"  {reason}: {cnt}")
        return "\n".join(lines)

    with gr.Tab("2. Pairwise Preference") as tab:
        gr.Markdown("""
## Pairwise Preference Annotation (A vs B)
Given a prompt and two model responses, choose which response is better.
This data is used for reward model training.
        """)

        annotator_input = gr.Textbox(
            label="Annotator ID", placeholder="e.g. annotator_01"
        )

        prompt_input = gr.Textbox(
            label="Prompt",
            placeholder="The prompt / instruction shown to both models...",
            lines=4,
        )

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Response A")
                response_a_input = gr.Textbox(
                    label="Response A",
                    placeholder="First model response...",
                    lines=8,
                )
            with gr.Column():
                gr.Markdown("### Response B")
                response_b_input = gr.Textbox(
                    label="Response B",
                    placeholder="Second model response...",
                    lines=8,
                )

        preference_radio = gr.Radio(
            choices=PREFERENCE_CHOICES,
            label="Which response is better?",
            value=None,
        )

        reasons_check = gr.CheckboxGroup(
            choices=REASON_CHOICES,
            label="Why did you prefer it? (select all that apply)",
        )

        confidence_slider = gr.Slider(
            minimum=1, maximum=3, step=1, value=2,
            label="Confidence (1=low, 2=medium, 3=high)",
        )

        notes_input = gr.Textbox(label="Notes (optional)", lines=2)

        with gr.Row():
            save_btn = gr.Button("Save Preference", variant="primary")
            clear_btn = gr.Button("Clear Form")

        status_output = gr.Textbox(label="Status", interactive=False)
        iaa_output = gr.Textbox(label="Inter-Annotator Agreement", interactive=False)

        gr.Markdown("### Dataset Statistics")
        stats_output = gr.Textbox(label="Preference Distribution", interactive=False, lines=10)
        refresh_btn = gr.Button("Refresh Statistics")

        save_btn.click(
            fn=_save,
            inputs=[
                annotator_input, prompt_input, response_a_input, response_b_input,
                preference_radio, reasons_check, confidence_slider, notes_input,
            ],
            outputs=[status_output, iaa_output],
        )

        def _clear():
            return "", "", "", "", None, [], 2, ""

        clear_btn.click(
            fn=_clear,
            inputs=[],
            outputs=[
                annotator_input, prompt_input, response_a_input, response_b_input,
                preference_radio, reasons_check, confidence_slider, notes_input,
            ],
        )

        refresh_btn.click(fn=_load_stats, inputs=[], outputs=[stats_output])

    return tab
