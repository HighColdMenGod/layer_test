#!/usr/bin/env python3
"""Create the manually audited 50-pair IIRC subset from generated candidates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


STRICT_PASS_QIDS = {
    "q_10893", "q_10962", "q_10969", "q_10985", "q_11188", "q_11315",
    "q_11319", "q_11330", "q_11331", "q_11348", "q_11361", "q_11383",
    "q_11384", "q_11497", "q_11529", "q_11551", "q_11616", "q_11655",
    "q_11734", "q_11738", "q_11784", "q_11793",
    "q_11881", "q_11884", "q_11900", "q_11928", "q_11938", "q_11945",
    "q_11979", "q_12046", "q_12049", "q_12133", "q_12134",
}

REPAIRED_QIDS = {
    "q_10914", "q_10961", "q_11050", "q_11137", "q_11193", "q_11339",
    "q_11410", "q_11437", "q_11466", "q_11507", "q_11756", "q_11877",
    "q_12024", "q_12039", "q_12069",
}

USABLE_WEAK_QIDS = {"q_11713", "q_11790"}


MAIN_REWRITES = {
    "q_11050": (
        "In 1863, Kane stayed in a hospital in Baltimore and later arrived in "
        "Gettysburg on July 2, 1863."
    ),
    "q_11137": (
        "In the 1969 Anglo-Italian League Cup final, Don Rogers scored once and "
        "Arthur Horsfield scored a hat-trick for Swindon."
    ),
    "q_11339": (
        "Alice Leigh's paternal grandfather was Sir Thomas Leigh, and her "
        "maternal grandfather was Sir John Spencer."
    ),
    "q_11756": "In 2011, Willie Mack defeated both Kevin Steen and Chris Hero.",
}


REMOVED_FACT_REWRITES = {
    "q_10914": "The Have a Nice Day Tour grossed $132 million.",
    "q_10961": "The Dallas Cowboys franchise was established in 1959.",
    "q_11410": "Mary I reigned from July 1553 until her death on 17 November 1558.",
    "q_11437": "Reading F.C. was established in 1871.",
    "q_11466": "Tom Aldred signed his first professional contract in December 2008.",
    "q_11507": "Charlie Wilson's War grossed $119 million worldwide.",
    "q_11877": "Ray Clemence began his professional career with Scunthorpe United in 1966.",
    "q_12024": "Real Time was originally webcast from 2 August to 6 September 2002.",
    "q_12039": "The San Diego Padres were founded in 1969.",
    "q_12069": "Boston College started its varsity football team in 1892.",
}


REVIEW_NOTES = {
    "q_11193": (
        "Restored the annotated Wikipedia-link alias Alexxis Nevaeh = Alisha "
        "Edwards in the source label."
    ),
    "q_11713": (
        "Usable but geographically coarse: compares a city population with a "
        "country population because the question asks which destination is larger."
    ),
    "q_11790": (
        "Usable but the age proxy is club founding versus the date the resort's "
        "first golf course was laid out."
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/IIRC/derived/dev_one_pairs_candidates_89.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/IIRC/derived/dev_one_pairs_curated_hard_50.jsonl"),
    )
    return parser.parse_args()


def blocks(context: str) -> list[str]:
    return [block for block in context.split("\n\n") if block.strip()]


def rewrite_main(record: dict[str, Any], new_text: str) -> None:
    title = record["metadata"]["main_article"]
    new_block = f"[Source: Main article — {title}]\n{new_text}"
    for key in ("complete_context", "insufficient_context"):
        items = blocks(record[key])
        first_main = next(
            (index for index, block in enumerate(items) if block.startswith("[Source: Main article —")),
            None,
        )
        if first_main is None:
            raise ValueError(f"{record['id']}: main evidence block not found")
        items = [block for block in items if not block.startswith("[Source: Main article —")]
        items.insert(first_main, new_block)
        record[key] = "\n\n".join(items)


def rewrite_removed_fact(record: dict[str, Any], new_text: str) -> None:
    old_text = record["removed_critical_fact"]
    if old_text not in record["complete_context"]:
        raise ValueError(f"{record['id']}: old removed fact not found in C+")
    record["complete_context"] = record["complete_context"].replace(old_text, new_text, 1)
    record["removed_critical_fact"] = new_text


def repair(record: dict[str, Any]) -> None:
    qid = record["metadata"]["qid"]
    if qid in MAIN_REWRITES:
        rewrite_main(record, MAIN_REWRITES[qid])
    if qid in REMOVED_FACT_REWRITES:
        rewrite_removed_fact(record, REMOVED_FACT_REWRITES[qid])
    if qid == "q_11193":
        old = "[Source: Alisha Edwards]"
        new = "[Source: Alexxis Nevaeh (Alisha Edwards)]"
        record["complete_context"] = record["complete_context"].replace(old, new)
        record["insufficient_context"] = record["insufficient_context"].replace(old, new)


def validate(record: dict[str, Any]) -> None:
    fact = record["removed_critical_fact"]
    if record["complete_context"].count(fact) != 1:
        raise ValueError(f"{record['id']}: removed fact must occur once in C+")
    if fact in record["insufficient_context"]:
        raise ValueError(f"{record['id']}: removed fact still occurs in C-")
    for answer in record["metadata"]["gold_answers"]:
        if answer.casefold() not in record["insufficient_context"].casefold():
            raise ValueError(f"{record['id']}: gold answer text disappeared from C-")


def main() -> None:
    args = parse_args()
    records = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_qid = {record["metadata"]["qid"]: record for record in records}
    wanted = STRICT_PASS_QIDS | REPAIRED_QIDS | USABLE_WEAK_QIDS
    missing = wanted - by_qid.keys()
    if missing:
        raise ValueError(f"Missing curated qids: {sorted(missing)}")

    selected = [record for record in records if record["metadata"]["qid"] in wanted]
    if len(selected) != 50:
        raise ValueError(f"Expected 50 selected records, got {len(selected)}")

    for rank, record in enumerate(selected, 1):
        qid = record["metadata"]["qid"]
        tier = (
            "strict_pass"
            if qid in STRICT_PASS_QIDS
            else "repaired"
            if qid in REPAIRED_QIDS
            else "usable_weak"
        )
        if tier == "repaired":
            repair(record)
        record["id"] = f"iirc-dev-curated-hard-{rank:04d}"
        record["metadata"]["quality_tier"] = tier
        record["metadata"]["manual_semantic_review"] = True
        record["metadata"]["evidence_rewritten"] = tier == "repaired"
        record["metadata"]["review_note"] = REVIEW_NOTES.get(qid, "")
        validate(record)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in selected:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    audit_path = args.output.with_name(args.output.stem + "_audit.csv")
    with audit_path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["id", "qid", "quality_tier", "question", "gold_answer", "removed_fact", "note"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in selected:
            metadata = record["metadata"]
            writer.writerow(
                {
                    "id": record["id"],
                    "qid": metadata["qid"],
                    "quality_tier": metadata["quality_tier"],
                    "question": record["question"],
                    "gold_answer": " | ".join(metadata["gold_answers"]),
                    "removed_fact": record["removed_critical_fact"],
                    "note": metadata["review_note"],
                }
            )

    counts = {
        tier: sum(record["metadata"]["quality_tier"] == tier for record in selected)
        for tier in ("strict_pass", "repaired", "usable_weak")
    }
    summary = {
        "output": str(args.output),
        "selected": len(selected),
        "quality_tiers": counts,
        "unique_main_articles": len({record["metadata"]["pid"] for record in selected}),
        "all_gold_answers_retained_in_cminus": True,
        "manual_semantic_review": True,
    }
    summary_path = args.output.with_name(args.output.stem + "_summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
