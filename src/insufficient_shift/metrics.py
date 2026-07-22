from __future__ import annotations

from typing import Any


def diagnose_shift(
    complete_probs: list[float],
    insufficient_probs: list[float],
    layers: list[int],
    *,
    threshold: float = 0.5,
    min_drop: float = 0.1,
) -> dict[str, Any]:
    if not complete_probs or len(complete_probs) != len(insufficient_probs) or len(layers) != len(complete_probs):
        raise ValueError("Probability arrays and layers must be non-empty and have equal length")

    best_idx = max(range(len(layers)), key=insufficient_probs.__getitem__)
    final_idx = len(layers) - 1
    pair_margins = [minus - plus for plus, minus in zip(complete_probs, insufficient_probs)]
    best_margin_idx = max(range(len(layers)), key=pair_margins.__getitem__)
    late_drop = insufficient_probs[best_idx] - insufficient_probs[final_idx]
    margin_collapse = pair_margins[best_margin_idx] - pair_margins[final_idx]
    flip = insufficient_probs[best_idx] >= threshold and insufficient_probs[final_idx] < threshold

    return {
        "best_insufficient_layer": layers[best_idx],
        "best_insufficient_probability": insufficient_probs[best_idx],
        "final_insufficient_probability": insufficient_probs[final_idx],
        "late_insufficient_drop": late_drop,
        "sufficiency_flip": flip,
        "best_pair_margin_layer": layers[best_margin_idx],
        "best_pair_margin": pair_margins[best_margin_idx],
        "final_pair_margin": pair_margins[final_idx],
        "pair_margin_collapse": margin_collapse,
        "has_shift_signal": bool(flip or late_drop >= min_drop or margin_collapse >= min_drop),
        "thresholds": {"classification": threshold, "minimum_effect": min_drop},
    }


def diagnose_margin_shift(
    complete_margins: list[float],
    insufficient_margins: list[float],
    layers: list[int],
    *,
    min_drop: float = 0.1,
) -> dict[str, Any]:
    """Diagnose late-layer loss using B-minus-A single-token logit margins."""
    if (
        not complete_margins
        or len(complete_margins) != len(insufficient_margins)
        or len(layers) != len(complete_margins)
    ):
        raise ValueError("Margin arrays and layers must be non-empty and have equal length")

    best_idx = max(range(len(layers)), key=insufficient_margins.__getitem__)
    final_idx = len(layers) - 1
    pair_contrasts = [
        minus - plus for plus, minus in zip(complete_margins, insufficient_margins)
    ]
    best_contrast_idx = max(range(len(layers)), key=pair_contrasts.__getitem__)
    late_drop = insufficient_margins[best_idx] - insufficient_margins[final_idx]
    contrast_collapse = pair_contrasts[best_contrast_idx] - pair_contrasts[final_idx]
    flip = insufficient_margins[best_idx] >= 0.0 and insufficient_margins[final_idx] < 0.0

    return {
        "best_insufficient_layer": layers[best_idx],
        "best_insufficient_margin": insufficient_margins[best_idx],
        "final_insufficient_margin": insufficient_margins[final_idx],
        "late_insufficient_margin_drop": late_drop,
        "sufficiency_flip": flip,
        "best_pair_contrast_layer": layers[best_contrast_idx],
        "best_pair_contrast": pair_contrasts[best_contrast_idx],
        "final_pair_contrast": pair_contrasts[final_idx],
        "pair_contrast_collapse": contrast_collapse,
        "has_shift_signal": bool(flip or late_drop >= min_drop or contrast_collapse >= min_drop),
        "thresholds": {"classification_margin": 0.0, "minimum_logit_effect": min_drop},
    }
