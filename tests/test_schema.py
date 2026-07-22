import json

from insufficient_shift.schema import load_matched_pairs


def _record(sample_id: str) -> dict:
    return {
        "id": sample_id,
        "question": "What is the result?",
        "complete_context": "The result was significantly higher.",
        "insufficient_context": "The study measured the result.",
        "removed_critical_fact": "The result was significantly higher.",
        "metadata": {"pmcid": "123"},
    }


def test_loads_multiple_jsonl_records(tmp_path):
    path = tmp_path / "pairs.jsonl"
    path.write_text(
        "\n".join(json.dumps(_record(sample_id)) for sample_id in ("one", "two")),
        encoding="utf-8",
    )

    pairs = load_matched_pairs(path)

    assert [pair.id for pair in pairs] == ["one", "two"]
    assert pairs[0].metadata == {"pmcid": "123"}


def test_assigns_fallback_id_to_jsonl_record(tmp_path):
    record = _record("unused")
    del record["id"]
    path = tmp_path / "pairs.jsonl"
    path.write_text(json.dumps(record), encoding="utf-8")

    pairs = load_matched_pairs(path)

    assert pairs[0].id == "sample-0001"
