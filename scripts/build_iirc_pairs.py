#!/usr/bin/env python3
"""Build hard IIRC matched pairs for the insufficient-information experiment.

The positive context is the human-selected minimal evidence from IIRC.  For a
comparison question, the negative context keeps the candidate-set bridge and
the gold candidate's evidence, but removes one competing candidate's operand.
The gold answer string therefore remains visible although the comparison can
no longer be completed.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


HARD_CUES = re.compile(
    r"\b(which|who|what|where|when|how|older|oldest|younger|youngest|first|earlier|later|most|more|"
    r"less|larger|largest|smaller|smallest|longer|longest|higher|highest|"
    r"lower|lowest|best|sooner|earliest|latest)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Candidate:
    pid: str
    title: str
    qid: str
    question: str
    answer_texts: tuple[str, ...]
    answer_passages: tuple[str, ...]
    main_facts: tuple[dict[str, Any], ...]
    linked_facts: tuple[dict[str, Any], ...]
    removed_fact: dict[str, Any]
    score: int

    @property
    def passage_count(self) -> int:
        return len({fact["passage"].casefold() for fact in self.linked_facts}) + 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/IIRC/dev.json"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/iirc_one_pairs/dev_one_pairs_hard_50.jsonl"),
    )
    parser.add_argument("--count", type=int, default=50)
    return parser.parse_args()


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalized(text: str) -> str:
    return re.sub(r"\W+", " ", text, flags=re.UNICODE).casefold().strip()


def deduplicate_facts(context: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse exact/overlapping annotation spans from the same passage."""
    kept: list[dict[str, Any]] = []
    for raw in context:
        text = clean_text(str(raw.get("text", "")))
        passage = str(raw.get("passage", "")).strip()
        if not text or not passage:
            continue
        fact = {"passage": passage, "text": text, "indices": raw.get("indices")}
        fact_norm = normalized(text)
        replaced = False
        for index, old in enumerate(kept):
            if old["passage"].casefold() != passage.casefold():
                continue
            old_norm = normalized(old["text"])
            if fact_norm in old_norm:
                replaced = True
                break
            if old_norm in fact_norm:
                kept[index] = fact
                replaced = True
                break
        if not replaced:
            kept.append(fact)
    return kept


def contains_answer(text: str, answer: str) -> bool:
    answer_norm = normalized(answer)
    return bool(answer_norm) and answer_norm in normalized(text)


def passage_matches_answer(passage: str, answer: str) -> bool:
    passage_norm = normalized(passage)
    answer_norm = normalized(answer)
    if not passage_norm or not answer_norm:
        return False
    if passage_norm in answer_norm or answer_norm in passage_norm:
        return True
    stop = {"the", "a", "an", "at", "of", "in", "and", "team", "club"}
    passage_tokens = set(passage_norm.split()) - stop
    answer_tokens = set(answer_norm.split()) - stop
    if not passage_tokens or not answer_tokens:
        return False
    overlap = len(passage_tokens & answer_tokens)
    return overlap / min(len(passage_tokens), len(answer_tokens)) >= 0.5


def make_candidate(passage: dict[str, Any], question: dict[str, Any]) -> Candidate | None:
    answer = question.get("answer", {})
    if answer.get("type") != "span":
        return None
    spans = answer.get("answer_spans") or []
    if len(spans) != 1:
        return None
    answer_texts = tuple(clean_text(str(span.get("text", ""))) for span in spans)
    answer_passages = tuple(str(span.get("passage", "")).strip() for span in spans)
    if not answer_texts or any(not text for text in answer_texts):
        return None
    facts = deduplicate_facts(question.get("context") or [])
    main_facts = [fact for fact in facts if fact["passage"].casefold() == "main"]
    linked_facts = [fact for fact in facts if fact["passage"].casefold() != "main"]
    linked_passages = {fact["passage"].casefold() for fact in linked_facts}
    if not main_facts or len(linked_passages) < 2:
        return None

    question_text = clean_text(str(question.get("question", "")))
    if not question_text or not HARD_CUES.search(question_text):
        return None
    answer_facts = [
        fact
        for fact in (*main_facts, *linked_facts)
        if contains_answer(fact["text"], answer_texts[0])
    ]
    if not answer_facts:
        return None

    removable = [
        fact
        for fact in linked_facts
        if 20 <= len(fact["text"]) <= 500
        and not contains_answer(fact["text"], answer_texts[0])
        and not passage_matches_answer(fact["passage"], answer_texts[0])
        and normalized(fact["text"]) != normalized(fact["passage"])
    ]
    if not removable:
        return None

    def removal_priority(fact: dict[str, Any]) -> tuple[int, int]:
        operand_cues = len(
            re.findall(
                r"\b(?:born|founded|formed|established|opened|released|sold|"
                r"years?|seasons?|albums?|copies|population|capacity|"
                r"debut|premiered|began|started|won|grossed)\b|\d",
                fact["text"],
                re.IGNORECASE,
            )
        )
        return (-operand_cues, len(fact["text"]))

    removed_fact = min(removable, key=removal_priority)
    retained_text = " ".join(
        fact["text"]
        for fact in (*main_facts, *linked_facts)
        if fact is not removed_fact
    )
    if not contains_answer(retained_text, answer_texts[0]):
        return None

    cue_count = len(HARD_CUES.findall(question_text))
    score = (
        20 * (len(linked_passages) + 1)
        + 3 * min(len(linked_facts), 6)
        + 2 * min(cue_count, 5)
        + min(len(re.findall(r"\d", removed_fact["text"])), 6)
    )
    return Candidate(
        pid=str(passage.get("pid", "")),
        title=clean_text(str(passage.get("title", ""))),
        qid=str(question.get("qid", "")),
        question=question_text,
        answer_texts=answer_texts,
        answer_passages=answer_passages,
        main_facts=tuple(main_facts),
        linked_facts=tuple(linked_facts),
        removed_fact=removed_fact,
        score=score,
    )


def render_fact(fact: dict[str, Any], main_title: str) -> str:
    source = (
        f"Main article — {main_title}"
        if fact["passage"].casefold() == "main"
        else fact["passage"]
    )
    return f"[Source: {source}]\n{fact['text']}"


def choose(candidates: list[Candidate], count: int) -> list[Candidate]:
    ranked = sorted(candidates, key=lambda item: (-item.score, item.qid))
    selected: list[Candidate] = []
    used_articles: set[str] = set()
    for candidate in ranked:
        if candidate.pid in used_articles:
            continue
        selected.append(candidate)
        used_articles.add(candidate.pid)
        if len(selected) == count:
            return selected
    for candidate in ranked:
        if candidate in selected:
            continue
        selected.append(candidate)
        if len(selected) == count:
            return selected
    raise ValueError(f"Only {len(selected)} eligible candidates; requested {count}")


def to_record(candidate: Candidate, rank: int) -> dict[str, Any]:
    complete_facts = candidate.main_facts + candidate.linked_facts
    insufficient_facts = tuple(
        fact for fact in complete_facts if fact is not candidate.removed_fact
    )
    return {
        "id": f"iirc-dev-hard-{rank:04d}",
        "question": candidate.question,
        "complete_context": "\n\n".join(
            render_fact(fact, candidate.title) for fact in complete_facts
        ),
        "insufficient_context": "\n\n".join(
            render_fact(fact, candidate.title) for fact in insufficient_facts
        ),
        "removed_critical_fact": candidate.removed_fact["text"],
        "metadata": {
            "dataset": "IIRC",
            "split": "dev",
            "pid": candidate.pid,
            "qid": candidate.qid,
            "main_article": candidate.title,
            "answer_type": "span",
            "gold_answers": list(candidate.answer_texts),
            "answer_passages": list(candidate.answer_passages),
            "removal_role": "comparison_operand",
            "removed_passage": candidate.removed_fact["passage"],
            "answer_text_retained_in_cminus": True,
            "context_passage_count": candidate.passage_count,
            "linked_fact_count": len(candidate.linked_facts),
            "selection_score": candidate.score,
        },
    }


def write_audit(path: Path, records: list[dict[str, Any]]) -> None:
    audit_path = path.with_name(path.stem + "_audit.csv")
    fields = [
        "id",
        "qid",
        "question",
        "gold_answers",
        "context_passage_count",
        "linked_fact_count",
        "removed_critical_fact",
    ]
    with audit_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            metadata = record["metadata"]
            writer.writerow(
                {
                    "id": record["id"],
                    "qid": metadata["qid"],
                    "question": record["question"],
                    "gold_answers": " | ".join(metadata["gold_answers"]),
                    "context_passage_count": metadata["context_passage_count"],
                    "linked_fact_count": metadata["linked_fact_count"],
                    "removed_critical_fact": record["removed_critical_fact"],
                }
            )


def main() -> None:
    args = parse_args()
    if args.count <= 0:
        raise ValueError("--count must be positive")
    passages = json.loads(args.input.read_text(encoding="utf-8"))
    candidates = [
        candidate
        for passage in passages
        for question in passage.get("questions", [])
        if (candidate := make_candidate(passage, question)) is not None
    ]
    selected = choose(candidates, args.count)
    records = [to_record(candidate, rank) for rank, candidate in enumerate(selected, 1)]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    write_audit(args.output, records)

    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "eligible_candidates": len(candidates),
        "selected": len(records),
        "unique_main_articles": len({record["metadata"]["pid"] for record in records}),
        "passage_count_distribution": {
            str(count): sum(
                record["metadata"]["context_passage_count"] == count for record in records
            )
            for count in sorted(
                {record["metadata"]["context_passage_count"] for record in records}
            )
        },
        "construction": (
            "C+ uses IIRC human-selected evidence; C- keeps the bridge and gold "
            "evidence but removes one competing comparison operand."
        ),
    }
    summary_path = args.output.with_name(args.output.stem + "_summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
