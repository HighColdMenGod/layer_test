from insufficient_shift.metrics import diagnose_margin_shift, diagnose_shift


def test_detects_hard_flip():
    result = diagnose_shift(
        complete_probs=[0.20, 0.10, 0.08],
        insufficient_probs=[0.60, 0.91, 0.40],
        layers=[30, 31, 32],
    )
    assert result["sufficiency_flip"] is True
    assert result["best_insufficient_layer"] == 31
    assert result["has_shift_signal"] is True


def test_detects_margin_collapse_without_flip():
    result = diagnose_shift(
        complete_probs=[0.10, 0.10, 0.10],
        insufficient_probs=[0.75, 0.65, 0.55],
        layers=[30, 31, 32],
    )
    assert result["sufficiency_flip"] is False
    assert result["pair_margin_collapse"] > 0.1
    assert result["has_shift_signal"] is True


def test_detects_single_token_margin_flip():
    result = diagnose_margin_shift(
        complete_margins=[-2.0, -2.5, -3.0],
        insufficient_margins=[0.5, 1.2, -0.4],
        layers=[30, 31, 32],
    )
    assert result["sufficiency_flip"] is True
    assert result["best_insufficient_layer"] == 31
    assert result["late_insufficient_margin_drop"] == 1.6
