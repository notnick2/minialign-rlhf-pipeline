"""
Tab 3: Instruction-Following Quality Rating
Rubric-based 1-5 scale rating on five dimensions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import gradio as gr
import pandas as pd

from annotation.agreement.iaa import compute_all_agreement, interpret_kappa

if TYPE_CHECKING:
    from annotation.storage.annotation_store import AnnotationStore

RUBRIC = """
### Scoring Rubric

**Relevance (1-5)**
- 5: Directly and completely addresses the prompt
- 3: Partially addresses the prompt
- 1: Off-topic or ignores the prompt

**Completeness (1-5)**
- 5: Covers all aspects the prompt requested
- 3: Covers main aspects but misses some details
- 1: Very incomplete or superficial

**Accuracy (1-5)**
- 5: All facts/reasoning are correct and verifiable
- 3: Mostly correct with minor errors
- 1: Contains significant factual errors

**Format Adherence (1-5)**
- 5: Format matches instructions exactly (lists, length, style)
- 3: Partially follows formatting instructions
- 1: Ignores format instructions entirely

**Conciseness (1-5)**
- 5: Optimally concise — no unnecessary content
- 3: Some padding or repetition
- 1: Severely padded or far too brief
"""

# Dimension weights for overall score computation
WEIGHTS = {
    "relevance": 0.20,
    "completeness": 0.25,
    "accuracy": 0.30,
    "format_adherence": 0.15,
    "conciseness": 0.10,
}


def build_instruction_quality_tab(store: "AnnotationStore") -> gr.Tab:
    """Build the instruction quality rating tab."""

    def _compute_overall(relevance, completeness, accuracy, format_adherence, conciseness):
        score = (
            WEIGHTS["relevance"] * relevance
            + WEIGHTS["completeness"] * completeness
            + WEIGHTS["accuracy"] * accuracy
            + WEIGHTS["format_adherence"] * format_adherence
            + WEIGHTS["conciseness"] * conciseness
        )
        return round(score, 2)

    def _save(
        annotator: str,
        prompt: str,
        response: str,
        relevance: int,
        completeness: int,
        accuracy: int,
        format_adherence: int,
        conciseness: int,
        notes: str,
    ) -> tuple[str, str, float]:
        if not annotator.strip():
            return "Error: annotator name is required.", "", 0.0
        if not prompt.strip() or not response.strip():
            return "Error: both prompt and response are required.", "", 0.0

        overall = _compute_overall(relevance, completeness, accuracy, format_adherence, conciseness)

        row_id = store.save_instruction_quality(
            annotator=annotator.strip(),
            prompt=prompt.strip(),
            response=response.strip(),
            relevance=int(relevance),
            completeness=int(completeness),
            accuracy=int(accuracy),
            format_adherence=int(format_adherence),
            conciseness=int(conciseness),
            notes=notes,
        )

        # Recompute IAA on overall scores
        rows = store.get_instruction_quality(limit=5000)
        iaa_display = "No annotations yet."
        if rows:
            df = pd.DataFrame(rows)
            df = df.rename(columns={"id": "item_id"})
            df["label"] = df["overall_score"]
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
                f"Krippendorff's α (ordinal): {alpha_str}"
            )

        return f"Saved quality annotation #{row_id} (overall={overall:.2f})", iaa_display, overall

    def _update_preview(r, c, a, f, cc):
        return _compute_overall(r, c, a, f, cc)

    with gr.Tab("3. Instruction Quality") as tab:
        gr.Markdown("""
## Instruction-Following Quality Rating
Rate the response on five dimensions using 1-5 sliders.
The overall score is a weighted average (accuracy weighted most heavily).
        """)

        annotator_input = gr.Textbox(
            label="Annotator ID", placeholder="e.g. annotator_01"
        )

        with gr.Row():
            prompt_input = gr.Textbox(
                label="Prompt / Instruction",
                placeholder="The instruction given to the model...",
                lines=4,
                scale=1,
            )
            response_input = gr.Textbox(
                label="Model Response",
                placeholder="The response to rate...",
                lines=4,
                scale=1,
            )

        with gr.Accordion("Scoring Rubric (click to expand)", open=False):
            gr.Markdown(RUBRIC)

        gr.Markdown("### Dimension Ratings")
        with gr.Row():
            relevance_slider = gr.Slider(1, 5, step=1, value=3, label="Relevance (weight 20%)")
            completeness_slider = gr.Slider(1, 5, step=1, value=3, label="Completeness (weight 25%)")

        with gr.Row():
            accuracy_slider = gr.Slider(1, 5, step=1, value=3, label="Accuracy (weight 30%)")
            format_slider = gr.Slider(1, 5, step=1, value=3, label="Format Adherence (weight 15%)")

        conciseness_slider = gr.Slider(1, 5, step=1, value=3, label="Conciseness (weight 10%)")

        overall_display = gr.Number(
            label="Overall Score (auto-computed)", value=3.0, interactive=False
        )

        # Auto-update overall score when sliders change
        for slider in [relevance_slider, completeness_slider, accuracy_slider,
                        format_slider, conciseness_slider]:
            slider.change(
                fn=_update_preview,
                inputs=[relevance_slider, completeness_slider, accuracy_slider,
                         format_slider, conciseness_slider],
                outputs=[overall_display],
            )

        notes_input = gr.Textbox(label="Notes (optional)", lines=2)

        with gr.Row():
            save_btn = gr.Button("Save Rating", variant="primary")
            clear_btn = gr.Button("Clear Form")

        status_output = gr.Textbox(label="Status", interactive=False)
        iaa_output = gr.Textbox(label="Inter-Annotator Agreement", interactive=False)

        save_btn.click(
            fn=_save,
            inputs=[
                annotator_input, prompt_input, response_input,
                relevance_slider, completeness_slider, accuracy_slider,
                format_slider, conciseness_slider, notes_input,
            ],
            outputs=[status_output, iaa_output, overall_display],
        )

        def _clear():
            return "", "", "", 3, 3, 3, 3, 3, 3.0, ""

        clear_btn.click(
            fn=_clear,
            inputs=[],
            outputs=[
                annotator_input, prompt_input, response_input,
                relevance_slider, completeness_slider, accuracy_slider,
                format_slider, conciseness_slider, overall_display, notes_input,
            ],
        )

    return tab
