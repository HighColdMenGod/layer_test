import json

from insufficient_shift.reporting import save_batch_summary


def _report(sample_id: str, offset: float) -> dict:
    return {
        "model": "test-model",
        "method": "test-method",
        "sample": {"id": sample_id},
        "metrics": {
            "has_shift_signal": offset > 0,
            "late_insufficient_margin_drop": 0.1 + offset,
            "pair_contrast_collapse": 0.2 + offset,
            "final_pair_contrast": 0.3 + offset,
        },
        "layers": [
            {
                "layer": 1,
                "p_insufficient_complete": 0.2,
                "p_insufficient_incomplete": 0.5 + offset,
                "pair_contrast": 0.3 + offset,
            },
            {
                "layer": 2,
                "p_insufficient_complete": 0.1,
                "p_insufficient_incomplete": 0.4 + offset,
                "pair_contrast": 0.3 + offset,
            },
        ],
    }


def test_saves_batch_summary(tmp_path):
    summary = save_batch_summary([_report("one", 0.0), _report("two", 0.2)], tmp_path)

    assert summary["sample_count"] == 2
    assert summary["has_shift_signal_count"] == 1
    saved = json.loads((tmp_path / "batch_summary.json").read_text(encoding="utf-8"))
    assert saved["layers"][0]["mean_p_insufficient_incomplete"] == 0.6
