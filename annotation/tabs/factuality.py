"""
Tab 4: Factuality Annotation
Claim-level factuality labeling: True / False / Uncertain / Not verifiable.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import gradio as gr
import pandas as pd

from annotation.agreement.iaa import compute_all_agreement, interpret_kappa

if TYPE_CHECKING:
    from annotation.storage.annotation_store import AnnotationStore

CLAIM_LABELS = ["True", "False", "Uncertain", "Not verifiable"]


def _split_into_claims(text: str) -> list[str]:
    """
    Split a response into individual claim-level sentences.

    Uses a simple regex-based sentence splitter.  For production use,
    this could be replaced with spacy or NLTK sentence tokenization.
    """
    # Split on period/exclamation/question followed by space and capital letter
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())
    # Filter out very short fragments (< 10 chars)
    claims = [s.strip() for s in sentences if len(s.strip()) >= 10]
    return claims if claims else [text.strip()]


def build_factuality_tab(store: "AnnotationStore") -> gr.Tab:
    """Build the factuality annotation tab."""

    # State: holds list of claim dicts for the current response
    current_claims: list[dict] = []

    def _decompose(response_text: str) -> tuple[str, list[dict]]:
        """Decompose response into claims and return HTML + state."""
        if not response_text.strip():
            return "Error: response text is required.", []

        claims = _split_into_claims(response_text)
        claim_state = [
            {"claim": c, "label": "Uncertain", "citation": ""}
            for c in claims
        ]
        preview = f"Found {len(claims)} claim(s):\n\n"
        for i, c in enumerate(claims, 1):
            preview += f"{i}. {c}\n"
        return preview, claim_state

    def _build_claim_components(claim_state: list[dict]):
        """Build dynamic claim labeling interface."""
        return claim_state

    def _save(
        annotator: str,
        response_text: str,
        claim_labels_json: str,  # JSON string of [{claim, label, citation}]
        notes: str,
    ) -> tuple[str, str]:
        if not annotator.strip():
            return "Error: annotator name is required.", ""
        if not response_text.strip():
            return "Error: response text is required.", ""

        import json as _json
        try:
            claims = _json.loads(claim_labels_json) if claim_labels_json else []
        except Exception:
            claims = []

        if not claims:
            return "Error: decompose the response and label at least one claim.", ""

        row_id = store.save_factuality(
            annotator=annotator.strip(),
            response=response_text.strip(),
            claims=claims,
            notes=notes,
        )

        # IAA on factuality: use proportion of True claims as a numeric label
        rows = store.get_factuality(limit=5000)
        iaa_display = "No annotations yet."
        if rows:
            records = []
            for r in rows:
                claims_list = r.get("claims", [])
                if claims_list:
                    n_true = sum(1 for c in claims_list if c.get("label") == "True")
                    prop_true = n_true / len(claims_list)
                    records.append({
                        "item_id": r["id"],
                        "annotator": r["annotator"],
                        "label": prop_true,
                    })
            if records:
                df = pd.DataFrame(records)
                iaa = compute_all_agreement(df)
                alpha_str = (
                    f"{iaa['krippendorff_alpha']:.3f}"
                    if iaa["n_annotations"] >= 2
                    else "N/A"
                )
                iaa_display = (
                    f"Annotations: {iaa['n_annotations']} | "
                    f"Annotators: {iaa['n_annotators']} | "
                    f"Krippendorff's α (proportion true): {alpha_str}"
                )

        return f"Saved factuality annotation #{row_id}", iaa_display

    with gr.Tab("4. Factuality Annotation") as tab:
        gr.Markdown("""
## Factuality Annotation
Paste a model response, decompose it into individual claims,
then label each claim as True / False / Uncertain / Not verifiable.
        """)

        annotator_input = gr.Textbox(
            label="Annotator ID", placeholder="e.g. annotator_01"
        )

        response_input = gr.Textbox(
            label="Model Response",
            placeholder="Paste the model response to fact-check...",
            lines=8,
        )

        decompose_btn = gr.Button("Decompose into Claims", variant="secondary")
        claims_preview = gr.Textbox(
            label="Extracted Claims",
            interactive=False,
            lines=6,
        )

        gr.Markdown("### Label Each Claim")
        gr.Markdown(
            "Edit the JSON below to assign labels. Valid labels: "
            + ", ".join(f"`{l}`" for l in CLAIM_LABELS)
        )

        claims_editor = gr.Textbox(
            label="Claims JSON (edit label and citation fields)",
            placeholder='[{"claim": "...", "label": "True", "citation": ""}]',
            lines=12,
        )

        import json as _json

        def _on_decompose(response_text: str) -> tuple[str, str]:
            preview, claim_state = _decompose(response_text)
            return preview, _json.dumps(claim_state, indent=2)

        decompose_btn.click(
            fn=_on_decompose,
            inputs=[response_input],
            outputs=[claims_preview, claims_editor],
        )

        notes_input = gr.Textbox(label="Notes (optional)", lines=2)

        with gr.Row():
            save_btn = gr.Button("Save Factuality Labels", variant="primary")
            clear_btn = gr.Button("Clear Form")

        status_output = gr.Textbox(label="Status", interactive=False)
        iaa_output = gr.Textbox(label="Inter-Annotator Agreement", interactive=False)

        save_btn.click(
            fn=_save,
            inputs=[annotator_input, response_input, claims_editor, notes_input],
            outputs=[status_output, iaa_output],
        )

        def _clear():
            return "", "", "", "", ""

        clear_btn.click(
            fn=_clear,
            inputs=[],
            outputs=[annotator_input, response_input, claims_preview, claims_editor, notes_input],
        )

        # Example / instructions
        with gr.Accordion("Instructions & Example", open=False):
            gr.Markdown("""
**Example claims JSON:**
```json
[
  {"claim": "Water boils at 100°C at sea level.", "label": "True", "citation": "basic chemistry"},
  {"claim": "The moon is made of cheese.", "label": "False", "citation": ""},
  {"claim": "This drug cures cancer.", "label": "Uncertain", "citation": ""},
  {"claim": "The prime minister smiled.", "label": "Not verifiable", "citation": ""}
]
```

**Label definitions:**
- **True**: Claim is factually correct and verifiable
- **False**: Claim is factually incorrect
- **Uncertain**: Claim might be true but cannot be easily verified
- **Not verifiable**: Claim cannot be fact-checked (opinions, subjective statements)
            """)

    return tab
