from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


DEFAULT_SYSTEM_PROMPT = (
    "You judge whether the context contains enough information to answer the "
    "question. Follow the label mapping in the user message and output exactly "
    "one label token."
)


@dataclass(frozen=True)
class MatchedPair:
    question: str
    complete_context: str
    insufficient_context: str
    removed_critical_fact: str
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, fallback_id: str | None = None) -> "MatchedPair":
        required = (
            "question",
            "complete_context",
            "insufficient_context",
            "removed_critical_fact",
        )
        missing = [key for key in required if not isinstance(payload.get(key), str)]
        if missing:
            raise ValueError(f"Matched pair is missing string fields: {', '.join(missing)}")

        metadata = payload.get("metadata", {})
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise ValueError("Matched-pair metadata must be a JSON object")

        pair_id = payload.get("id", fallback_id)
        return cls(
            question=payload["question"],
            complete_context=payload["complete_context"],
            insufficient_context=payload["insufficient_context"],
            removed_critical_fact=payload["removed_critical_fact"],
            system_prompt=payload.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
            id=str(pair_id) if pair_id is not None else None,
            metadata=metadata,
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "MatchedPair":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Single-pair JSON must contain one JSON object")
        return cls.from_dict(payload)


def load_matched_pairs(path: str | Path) -> list[MatchedPair]:
    """Load one JSON object, a JSON array, or newline-delimited JSON objects."""
    source = Path(path)
    if source.suffix.lower() == ".jsonl":
        pairs: list[MatchedPair] = []
        with source.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON on line {line_number} of {source}") from error
                if not isinstance(payload, dict):
                    raise ValueError(f"Line {line_number} of {source} is not a JSON object")
                pairs.append(
                    MatchedPair.from_dict(payload, fallback_id=f"sample-{line_number:04d}")
                )
    else:
        payload = json.loads(source.read_text(encoding="utf-8"))
        records = payload if isinstance(payload, list) else [payload]
        if not all(isinstance(record, dict) for record in records):
            raise ValueError("JSON input must contain an object or an array of objects")
        pairs = [
            MatchedPair.from_dict(record, fallback_id=f"sample-{index:04d}")
            for index, record in enumerate(records, start=1)
        ]

    if not pairs:
        raise ValueError(f"No matched pairs found in {source}")
    return pairs


@dataclass(frozen=True)
class LayerScore:
    layer: int
    sufficient_logit: float
    insufficient_logit: float
    insufficient_logit_margin: float
    p_insufficient_conditional: float
