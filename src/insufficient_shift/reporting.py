from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean
from typing import Any


def save_report(report: dict[str, Any], output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    rows = report["layers"]
    layers = [row["layer"] for row in rows]
    plt.figure(figsize=(8, 4.8))
    plt.plot(layers, [row["p_insufficient_complete"] for row in rows], marker="o", label="C+ complete")
    plt.plot(layers, [row["p_insufficient_incomplete"] for row in rows], marker="o", label="C- pseudo-sufficient")
    plt.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Transformer layer")
    plt.ylabel("P(Insufficient | A/B) from logit lens")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / "layer_shift.png", dpi=180)
    plt.close()


def save_batch_summary(reports: list[dict[str, Any]], output_dir: str | Path) -> dict[str, Any]:
    """Save aggregate metrics and a mean layer curve for a completed JSONL run."""
    if not reports:
        raise ValueError("Cannot summarize an empty batch")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    layer_ids = [row["layer"] for row in reports[0]["layers"]]
    for report in reports[1:]:
        if [row["layer"] for row in report["layers"]] != layer_ids:
            raise ValueError("Layer alignment differs between samples in the batch")

    layer_rows = []
    for index, layer in enumerate(layer_ids):
        complete = [report["layers"][index]["p_insufficient_complete"] for report in reports]
        incomplete = [report["layers"][index]["p_insufficient_incomplete"] for report in reports]
        contrasts = [report["layers"][index]["pair_contrast"] for report in reports]
        layer_rows.append(
            {
                "layer": layer,
                "mean_p_insufficient_complete": fmean(complete),
                "mean_p_insufficient_incomplete": fmean(incomplete),
                "mean_pair_contrast": fmean(contrasts),
            }
        )

    scalar_metrics = (
        "late_insufficient_margin_drop",
        "pair_contrast_collapse",
        "final_pair_contrast",
    )
    metrics_summary = {
        name: fmean(float(report["metrics"][name]) for report in reports)
        for name in scalar_metrics
    }
    shift_count = sum(bool(report["metrics"]["has_shift_signal"]) for report in reports)
    summary = {
        "model": reports[0]["model"],
        "method": reports[0]["method"],
        "sample_count": len(reports),
        "has_shift_signal_count": shift_count,
        "has_shift_signal_rate": shift_count / len(reports),
        "mean_metrics": metrics_summary,
        "layers": layer_rows,
    }
    (output / "batch_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return summary

    plt.figure(figsize=(8, 4.8))
    plt.plot(
        layer_ids,
        [row["mean_p_insufficient_complete"] for row in layer_rows],
        marker="o",
        label="Mean C+ complete",
    )
    plt.plot(
        layer_ids,
        [row["mean_p_insufficient_incomplete"] for row in layer_rows],
        marker="o",
        label="Mean C- pseudo-sufficient",
    )
    plt.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Transformer layer")
    plt.ylabel("Mean P(Insufficient | A/B) from logit lens")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / "mean_layer_shift.png", dpi=180)
    plt.close()
    return summary
