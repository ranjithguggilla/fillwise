"""Gradio front end: drop a CSV, get a missingness audit.

The report covers per-column missing rates, suspect sentinel zeros, and the
MCAR probe (can the gaps be predicted from the other columns?), plus a
plain-language recommendation per column.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .audit import audit_report
from .data import DATA_DIR, load_colic, load_pima


def _recommend(row: pd.Series) -> str:
    if row.suspect_zero_rate > 0.02:
        return "zeros look like sentinels; convert to NaN before imputing"
    if row.missing_rate == 0:
        return "complete"
    if row.missing_rate > 0.4:
        return "very sparse; add a missingness indicator, consider dropping"
    probe = row.mcar_probe_auc
    if probe is not None and not pd.isna(probe) and probe >= 0.65:
        return "gaps are predictable (not MCAR); impute + keep an indicator"
    return "moderate gaps; kNN or iterative imputation inside the CV fold"


def run_audit(file: str | None, demo: str) -> tuple[pd.DataFrame, object]:
    if file:
        df = pd.read_csv(file)
    elif demo == "horse colic":
        df = load_colic().X
    else:
        df = load_pima(repair_zeros=False).X
    rep = audit_report(df)
    rep["recommendation"] = rep.apply(_recommend, axis=1)

    fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * len(rep))))
    sub = rep.sort_values("missing_rate")
    ax.barh(sub.column, sub.missing_rate, color="#d95f02", label="missing")
    ax.barh(sub.column, sub.suspect_zero_rate, left=sub.missing_rate,
            color="#fc8d62", label="suspect zeros")
    ax.set_xlabel("fraction of rows")
    ax.set_title("Missing and suspect-zero rates by column")
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = Path(tempfile.mkdtemp()) / "audit.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return rep, str(out)


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="fillwise") as demo:
        gr.Markdown(
            "## fillwise — missing-data audit\n"
            "Upload any CSV, or pick a bundled clinical dataset. The audit finds "
            "missing cells, sentinel zeros, and gap patterns that predict other "
            "columns (which rules out MCAR)."
        )
        with gr.Row():
            file = gr.File(label="CSV file", file_types=[".csv"], type="filepath")
            demo_pick = gr.Radio(
                ["horse colic", "pima (raw, zeros kept)"],
                value="horse colic",
                label="or use a bundled dataset",
            )
        btn = gr.Button("Audit", variant="primary")
        table = gr.Dataframe(label="column audit")
        chart = gr.Image(label="missingness chart")
        btn.click(run_audit, inputs=[file, demo_pick], outputs=[table, chart])
    return demo


def main() -> None:
    build_demo().launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    _ = DATA_DIR  # bundled data lives next to the package
    main()
