#!/usr/bin/env python3
"""Build high-confidence C+/C- matched pairs from Evidence Inference 2.0.

The script uses only the Python standard library. It joins prompts, annotations,
plain-text articles, and the official article-level split. For each selected
prompt it keeps a local abstract window as C+ and deletes one verified evidence
sentence to create C-.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
from typing import Iterable


LABEL_NAMES = {
    -1: "significantly decreased",
    0: "no significant difference",
    1: "significantly increased",
}

GENERIC_RESULT_CUE = re.compile(
    r"\b(?:significant(?:ly)?|no difference|did not differ|not differ|"
    r"higher|lower|greater|less|increase[ds]?|decrease[ds]?|reduc(?:e[ds]?|tion)|"
    r"improv(?:e[ds]?|ement)|superior|inferior|similar|comparable|"
    r"p\s*[<=>]|confidence interval|\bci\b)\b",
    re.IGNORECASE,
)

LABEL_RESULT_CUES = {
    -1: re.compile(
        r"\b(?:lower|less|smaller|shorter|fewer|decrease[ds]?|reduc(?:e[ds]?|tion)|"
        r"inferior|negative association|decline[ds]?|diminish(?:ed|es)?)\b",
        re.IGNORECASE,
    ),
    0: re.compile(
        r"\b(?:no (?:statistically )?significant difference|no difference|"
        r"did not differ|not (?:significantly )?different|not different|"
        r"similar|comparable|equivalent|unchanged|no (?:detectable )?effect)\b",
        re.IGNORECASE,
    ),
    1: re.compile(
        r"\b(?:higher|greater|larger|longer|more|increase[ds]?|"
        r"improv(?:e[ds]?|ement)|superior|positive association|rise|rose|enhanc(?:e[ds]?|ement))\b",
        re.IGNORECASE,
    ),
}

P_VALUE = re.compile(r"\bp\s*(?:value\s*)?([<=>])\s*(0?\.\d+|1(?:\.0+)?)", re.IGNORECASE)


@dataclass(frozen=True)
class SentenceSpan:
    start: int
    end: int  # exclusive
    text: str


@dataclass
class Candidate:
    prompt_id: str
    pmcid: str
    question: str
    complete_context: str
    insufficient_context: str
    removed_critical_fact: str
    label_code: int
    label: str
    outcome: str
    intervention: str
    comparator: str
    evidence_sentence_start: int
    evidence_sentence_end: int
    valid_annotator_count: int
    selected_sentence_support: int
    independent_evidence_sentence_count: int
    context_sentence_count: int
    cplus_chars: int
    cminus_chars: int
    explicit_result_cue: bool
    label_consistent_cue: bool
    cminus_result_cue_count: int
    annotation_alignment: float
    quality_score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/v2.0"))
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--count", type=int, default=150)
    parser.add_argument("--context-before", type=int, default=2)
    parser.add_argument("--context-after", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def is_true(value: str) -> bool:
    return value.strip().lower() in {"true", "1"}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalized_for_match(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def flagged_prompt_ids(readme_path: Path) -> set[str]:
    """Read all IDs in the three caveat lists from the dataset README."""
    text = readme_path.read_text(encoding="utf-8-sig")
    result: set[str] = set()
    for heading in ("Incorrect", "Questionable", "Somewhat malformed"):
        match = re.search(
            rf"^### {re.escape(heading)}:\s*(.*?)(?=^### |\Z)",
            text,
            flags=re.MULTILINE | re.DOTALL,
        )
        if match:
            result.update(re.findall(r"\b\d+\b", match.group(1)))
    return result


def abstract_end(text: str) -> int:
    match = re.search(r"(?m)^\s*BODY(?:[.:]|$)", text)
    return match.start() if match else len(text)


def sentence_spans_from_abstract(text: str) -> list[SentenceSpan]:
    """Return approximate sentence spans while retaining article offsets."""
    abstract = text[: abstract_end(text)]
    paragraph_boundaries = list(re.finditer(r"\n\s*\n+", abstract))
    chunks: list[tuple[int, int]] = []
    start = 0
    for boundary in paragraph_boundaries:
        chunks.append((start, boundary.start()))
        start = boundary.end()
    chunks.append((start, len(abstract)))

    results: list[SentenceSpan] = []
    header = re.compile(
        r"^\s*(?:(?:TITLE|ABSTRACT(?:\.[^:\r\n]+)?):+|"
        r"(?:BACKGROUND|OBJECTIVE|PURPOSE|METHODS?|RESULTS?|CONCLUSIONS?)\s*\r?\n)\s*",
        re.IGNORECASE,
    )
    # A digit after a period is often the continuation of a decimal confidence
    # interval (for example, "-0. 3" in the source conversion), not a new sentence.
    boundary = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[\"'])")

    for chunk_start, chunk_end in chunks:
        raw = abstract[chunk_start:chunk_end]
        if not raw.strip():
            continue
        header_match = header.match(raw)
        content_start = chunk_start + (header_match.end() if header_match else 0)
        content = abstract[content_start:chunk_end]
        local_start = 0
        for sentence_boundary in boundary.finditer(content):
            local_end = sentence_boundary.start()
            _append_sentence(results, content, content_start, local_start, local_end)
            local_start = sentence_boundary.end()
        _append_sentence(results, content, content_start, local_start, len(content))
    return results


def _append_sentence(
    output: list[SentenceSpan], content: str, absolute_start: int, local_start: int, local_end: int
) -> None:
    raw = content[local_start:local_end]
    left_trim = len(raw) - len(raw.lstrip())
    right_trimmed = raw.rstrip()
    if not right_trimmed.strip():
        return
    start = absolute_start + local_start + left_trim
    end = absolute_start + local_start + len(right_trimmed)
    text = normalize_text(content[local_start + left_trim : local_start + len(right_trimmed)])
    if text:
        output.append(SentenceSpan(start=start, end=end, text=text))


def best_overlapping_sentence(
    sentences: list[SentenceSpan], evidence_start: int, evidence_end_inclusive: int
) -> int | None:
    evidence_end = evidence_end_inclusive + 1
    overlaps = []
    for index, sentence in enumerate(sentences):
        overlap = max(0, min(sentence.end, evidence_end) - max(sentence.start, evidence_start))
        if overlap:
            overlaps.append((overlap, index))
    return max(overlaps)[1] if overlaps else None


def alignment_score(annotation: str, sentence: str) -> float:
    left = normalized_for_match(annotation)
    right = normalized_for_match(sentence)
    if not left or not right:
        return 0.0
    if left in right or right in left:
        return min(len(left), len(right)) / max(len(left), len(right))
    return SequenceMatcher(None, left, right).ratio()


def has_label_consistent_cue(label_code: int, sentence: str) -> bool:
    """Heuristic check that the evidence sentence states the annotated direction."""
    if LABEL_RESULT_CUES[label_code].search(sentence):
        return True
    if label_code != 0:
        return False
    # Non-significant findings are also commonly expressed only as p >= .05.
    for operator, value_text in P_VALUE.findall(sentence):
        value = float(value_text)
        if operator in {"=", ">"} and value >= 0.05:
            return True
    return False


def make_question(prompt: dict[str, str]) -> str:
    return (
        f"With respect to {normalize_text(prompt['Outcome'])}, what reported difference "
        f"does the study find between {normalize_text(prompt['Intervention'])} and "
        f"{normalize_text(prompt['Comparator'])}: significantly increased, "
        "significantly decreased, or no significant difference?"
    )


def build_candidates(
    data_dir: Path,
    split: str,
    before: int,
    after: int,
) -> tuple[list[Candidate], dict[str, int]]:
    split_ids = {
        value.strip()
        for value in (data_dir / "splits" / f"{split}_article_ids.txt").read_text().splitlines()
        if value.strip()
    }
    excluded = flagged_prompt_ids(data_dir / "README.md")
    prompts = {
        row["PromptID"]: row
        for row in read_csv(data_dir / "prompts_merged.csv")
        if row["PMCID"] in split_ids and row["PromptID"] not in excluded
    }

    annotations_by_prompt: dict[str, list[dict[str, str]]] = {}
    for row in read_csv(data_dir / "annotations_merged.csv"):
        if row["PromptID"] not in prompts:
            continue
        if not (is_true(row["Valid Label"]) and is_true(row["Valid Reasoning"])):
            continue
        annotations_by_prompt.setdefault(row["PromptID"], []).append(row)

    audit = {
        "split_prompts": len(prompts),
        "valid_annotation_groups": len(annotations_by_prompt),
        "excluded_caveat_ids": len(excluded),
        "rejected_label_disagreement": 0,
        "rejected_missing_text": 0,
        "rejected_no_abstract_offset": 0,
        "rejected_unmapped_evidence": 0,
        "rejected_short_context": 0,
        "rejected_other_gold_evidence_in_window": 0,
    }
    candidates: list[Candidate] = []

    for prompt_id, valid_rows in annotations_by_prompt.items():
        label_codes = {int(row["Label Code"]) for row in valid_rows if row["Label Code"].strip()}
        if len(label_codes) != 1 or next(iter(label_codes)) not in LABEL_NAMES:
            audit["rejected_label_disagreement"] += 1
            continue
        label_code = next(iter(label_codes))
        prompt = prompts[prompt_id]
        article_path = data_dir / "txt_files" / f"PMC{prompt['PMCID']}.txt"
        if not article_path.exists() or article_path.stat().st_size == 0:
            audit["rejected_missing_text"] += 1
            continue
        article = article_path.read_text(encoding="utf-8")
        sentences = sentence_spans_from_abstract(article)
        if not sentences:
            audit["rejected_missing_text"] += 1
            continue

        abstract_rows = []
        for row in valid_rows:
            try:
                start = int(row["Evidence Start"])
                end = int(row["Evidence End"])
            except ValueError:
                continue
            if is_true(row["In Abstract"]) and start >= 0 and end >= start:
                abstract_rows.append((row, start, end))
        if not abstract_rows:
            audit["rejected_no_abstract_offset"] += 1
            continue

        mapped: list[tuple[dict[str, str], int]] = []
        for row, evidence_start, evidence_end in abstract_rows:
            sentence_index = best_overlapping_sentence(sentences, evidence_start, evidence_end)
            if sentence_index is not None:
                mapped.append((row, sentence_index))
        if not mapped:
            audit["rejected_unmapped_evidence"] += 1
            continue

        rows_by_sentence: dict[int, list[dict[str, str]]] = {}
        for row, sentence_index in mapped:
            rows_by_sentence.setdefault(sentence_index, []).append(row)

        def sentence_rank(item: tuple[int, list[dict[str, str]]]) -> tuple[int, int, float, int]:
            index, rows = item
            sentence = sentences[index].text
            user_support = len({row["UserID"] for row in rows})
            explicit = int(bool(GENERIC_RESULT_CUE.search(sentence)))
            alignment = max(alignment_score(row["Annotations"], sentence) for row in rows)
            return user_support, explicit, alignment, -len(sentence)

        evidence_index, supporting_rows = max(rows_by_sentence.items(), key=sentence_rank)
        evidence = sentences[evidence_index]
        if not 25 <= len(evidence.text) <= 650:
            audit["rejected_unmapped_evidence"] += 1
            continue

        window_start = max(0, evidence_index - before)
        window_end = min(len(sentences), evidence_index + after + 1)
        window_indices = list(range(window_start, window_end))
        other_gold_indices = set(rows_by_sentence) - {evidence_index}
        if other_gold_indices.intersection(window_indices):
            audit["rejected_other_gold_evidence_in_window"] += 1
            continue

        complete_context = " ".join(sentences[index].text for index in window_indices)
        insufficient_context = " ".join(
            sentences[index].text for index in window_indices if index != evidence_index
        )
        if len(window_indices) < 3 or len(insufficient_context) < 160:
            audit["rejected_short_context"] += 1
            continue
        if evidence.text in insufficient_context:
            audit["rejected_unmapped_evidence"] += 1
            continue

        annotator_count = len({row["UserID"] for row in valid_rows})
        selected_support = len({row["UserID"] for row in supporting_rows})
        independent_evidence_count = len(rows_by_sentence)
        label_cue = has_label_consistent_cue(label_code, evidence.text)
        explicit = bool(GENERIC_RESULT_CUE.search(evidence.text)) or label_cue
        cminus_cues = len(GENERIC_RESULT_CUE.findall(insufficient_context))
        alignment = max(alignment_score(row["Annotations"], evidence.text) for row in supporting_rows)

        score = 0.0
        score += 5.0
        score += 4.0 if annotator_count >= 2 else 0.0
        score += 4.0 if selected_support >= 2 else 0.0
        score += 3.0 if independent_evidence_count == 1 else 0.0
        score += 3.0 if explicit else 0.0
        score += 2.0 if label_cue else 0.0
        score += 2.0 if cminus_cues == 0 else max(-2.0, 1.0 - cminus_cues)
        score += 2.0 if len(window_indices) >= 5 else 1.0
        score += 2.0 if 50 <= len(evidence.text) <= 350 else 0.0
        score += 2.0 * alignment

        candidates.append(
            Candidate(
                prompt_id=prompt_id,
                pmcid=prompt["PMCID"],
                question=make_question(prompt),
                complete_context=complete_context,
                insufficient_context=insufficient_context,
                removed_critical_fact=evidence.text,
                label_code=label_code,
                label=LABEL_NAMES[label_code],
                outcome=normalize_text(prompt["Outcome"]),
                intervention=normalize_text(prompt["Intervention"]),
                comparator=normalize_text(prompt["Comparator"]),
                evidence_sentence_start=evidence.start,
                evidence_sentence_end=evidence.end - 1,
                valid_annotator_count=annotator_count,
                selected_sentence_support=selected_support,
                independent_evidence_sentence_count=independent_evidence_count,
                context_sentence_count=len(window_indices),
                cplus_chars=len(complete_context),
                cminus_chars=len(insufficient_context),
                explicit_result_cue=explicit,
                label_consistent_cue=label_cue,
                cminus_result_cue_count=cminus_cues,
                annotation_alignment=round(alignment, 6),
                quality_score=round(score, 6),
            )
        )
    audit["candidate_pool"] = len(candidates)
    return candidates, audit


def select_balanced(candidates: Iterable[Candidate], count: int) -> list[Candidate]:
    if count % 3:
        raise ValueError("--count must be divisible by 3 for balanced three-label selection")
    target = count // 3
    buckets: dict[int, list[Candidate]] = {code: [] for code in LABEL_NAMES}
    for candidate in candidates:
        # Ambiguous or singly validated evidence is unsuitable for a one-pair
        # probe: the deleted sentence must state the annotated direction, and
        # the prompt/label must have at least two valid human annotations.
        if candidate.label_consistent_cue and candidate.valid_annotator_count >= 2:
            buckets[candidate.label_code].append(candidate)
    for bucket in buckets.values():
        bucket.sort(
            key=lambda row: (
                -row.quality_score,
                row.cminus_result_cue_count,
                -row.selected_sentence_support,
                row.prompt_id,
            )
        )

    selected: list[Candidate] = []
    used_pmcids: set[str] = set()
    used_evidence: set[str] = set()
    positions = {code: 0 for code in LABEL_NAMES}
    counts = {code: 0 for code in LABEL_NAMES}

    while len(selected) < count:
        made_progress = False
        for code in (-1, 0, 1):
            if counts[code] >= target:
                continue
            bucket = buckets[code]
            while positions[code] < len(bucket):
                candidate = bucket[positions[code]]
                positions[code] += 1
                evidence_key = normalized_for_match(candidate.removed_critical_fact)
                if candidate.pmcid in used_pmcids or evidence_key in used_evidence:
                    continue
                selected.append(candidate)
                used_pmcids.add(candidate.pmcid)
                used_evidence.add(evidence_key)
                counts[code] += 1
                made_progress = True
                break
        if not made_progress:
            break

    if len(selected) != count:
        raise RuntimeError(
            f"Could only select {len(selected)} unique-article balanced pairs; "
            f"label counts={counts}, targets={target}"
        )
    selected.sort(key=lambda row: (row.label_code, -row.quality_score, row.prompt_id))
    return selected


def write_outputs(
    selected: list[Candidate], audit: dict[str, int], output_dir: Path, split: str
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / f"{split}_one_pairs_{len(selected)}.jsonl"
    csv_path = output_dir / f"{split}_one_pairs_{len(selected)}_audit.csv"
    summary_path = output_dir / f"{split}_one_pairs_{len(selected)}_summary.json"

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for index, candidate in enumerate(selected, start=1):
            record = {
                "id": f"ei2-{split}-{index:04d}",
                "question": candidate.question,
                "complete_context": candidate.complete_context,
                "insufficient_context": candidate.insufficient_context,
                "removed_critical_fact": candidate.removed_critical_fact,
                "metadata": {
                    key: value
                    for key, value in asdict(candidate).items()
                    if key
                    not in {
                        "question",
                        "complete_context",
                        "insufficient_context",
                        "removed_critical_fact",
                    }
                },
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    audit_fields = [
        "id",
        "prompt_id",
        "pmcid",
        "label_code",
        "label",
        "quality_score",
        "valid_annotator_count",
        "selected_sentence_support",
        "independent_evidence_sentence_count",
        "explicit_result_cue",
        "label_consistent_cue",
        "cminus_result_cue_count",
        "annotation_alignment",
        "context_sentence_count",
        "cplus_chars",
        "cminus_chars",
        "evidence_sentence_start",
        "evidence_sentence_end",
        "outcome",
        "intervention",
        "comparator",
        "removed_critical_fact",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=audit_fields)
        writer.writeheader()
        for index, candidate in enumerate(selected, start=1):
            row = asdict(candidate)
            row["id"] = f"ei2-{split}-{index:04d}"
            writer.writerow({field: row[field] for field in audit_fields})

    label_counts = {str(code): sum(row.label_code == code for row in selected) for code in LABEL_NAMES}
    summary = {
        **audit,
        "selected": len(selected),
        "selected_unique_articles": len({row.pmcid for row in selected}),
        "selected_label_counts": label_counts,
        "selected_explicit_result_cue": sum(row.explicit_result_cue for row in selected),
        "selected_label_consistent_cue": sum(row.label_consistent_cue for row in selected),
        "selected_zero_cminus_result_cues": sum(row.cminus_result_cue_count == 0 for row in selected),
        "selected_two_or_more_valid_annotators": sum(
            row.valid_annotator_count >= 2 for row in selected
        ),
        "selected_evidence_supported_by_two_or_more_annotators": sum(
            row.selected_sentence_support >= 2 for row in selected
        ),
        "quality_score_min": min(row.quality_score for row in selected),
        "quality_score_mean": round(sum(row.quality_score for row in selected) / len(selected), 6),
        "quality_score_max": max(row.quality_score for row in selected),
        "construction": (
            "C+ is a local abstract window; C- deletes one verified evidence sentence. "
            "One prompt per article and 50 examples per effect label."
        ),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.data_dir / "derived"
    candidates, audit = build_candidates(
        data_dir=args.data_dir,
        split=args.split,
        before=args.context_before,
        after=args.context_after,
    )
    selected = select_balanced(candidates, args.count)
    write_outputs(selected, audit, output_dir, args.split)
    print(json.dumps({**audit, "selected": len(selected)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
