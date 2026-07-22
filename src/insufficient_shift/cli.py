from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any

from .metrics import diagnose_margin_shift
from .modeling import load_model, render_prompt
from .reporting import save_batch_summary, save_report
from .schema import MatchedPair, load_matched_pairs
from .scoring import score_labels_by_layer


METHOD_NAME = "single-token A/B raw logit lens"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose layer-wise insufficient-information shift")
    parser.add_argument("--model", required=True, help="HF model ID or local model path")
    parser.add_argument(
        "--data", required=True, help="One matched-pair JSON file or a multi-sample JSONL file"
    )
    parser.add_argument("--output", default="outputs/one_pair")
    parser.add_argument("--last-k", type=int, default=None, help="Only report the final K blocks")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument(
        "--min-drop", type=float, default=0.10, help="Minimum collapse in logit units"
    )
    parser.add_argument(
        "--sufficient-token",
        "--sufficient-label",
        dest="sufficient_token",
        default=" A",
        help="One-token verbalizer for Sufficient (leading space is intentional)",
    )
    parser.add_argument(
        "--insufficient-token",
        "--insufficient-label",
        dest="insufficient_token",
        default=" B",
        help="One-token verbalizer for Insufficient (leading space is intentional)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N input samples")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to an existing samples.jsonl and skip sample IDs already present",
    )
    return parser


def analyze_pair(args: argparse.Namespace, loaded, pair: MatchedPair) -> dict[str, Any]:
    common = {
        "sufficient_token": args.sufficient_token,
        "insufficient_token": args.insufficient_token,
        "last_k": args.last_k,
    }
    complete = score_labels_by_layer(
        loaded,
        render_prompt(
            loaded.tokenizer,
            pair.system_prompt,
            pair.question,
            pair.complete_context,
            sufficient_token=args.sufficient_token,
            insufficient_token=args.insufficient_token,
        ),
        **common,
    )
    incomplete = score_labels_by_layer(
        loaded,
        render_prompt(
            loaded.tokenizer,
            pair.system_prompt,
            pair.question,
            pair.insufficient_context,
            sufficient_token=args.sufficient_token,
            insufficient_token=args.insufficient_token,
        ),
        **common,
    )
    if [x.layer for x in complete] != [x.layer for x in incomplete]:
        raise RuntimeError("Layer alignment differs between the matched inputs")

    layers = [x.layer for x in complete]
    metrics = diagnose_margin_shift(
        [x.insufficient_logit_margin for x in complete],
        [x.insufficient_logit_margin for x in incomplete],
        layers,
        min_drop=args.min_drop,
    )
    rows = []
    for plus, minus in zip(complete, incomplete):
        rows.append(
            {
                "layer": plus.layer,
                "p_insufficient_complete": plus.p_insufficient_conditional,
                "p_insufficient_incomplete": minus.p_insufficient_conditional,
                "complete_logit_margin": plus.insufficient_logit_margin,
                "incomplete_logit_margin": minus.insufficient_logit_margin,
                "pair_contrast": (
                    minus.insufficient_logit_margin - plus.insufficient_logit_margin
                ),
                "complete_scores": asdict(plus),
                "incomplete_scores": asdict(minus),
            }
        )
    report = {
        "model": args.model,
        "method": METHOD_NAME,
        "label_tokens": {
            "sufficient": args.sufficient_token,
            "insufficient": args.insufficient_token,
        },
        "sample": asdict(pair),
        "metrics": metrics,
        "layers": rows,
        "interpretation_warning": (
            "Intermediate-layer A/B probabilities are constrained verbalizer preferences, "
            "not calibrated semantic probabilities. Treat the logit margin as primary. "
            "Raw logit-lens drift is correlational; confirm with interventions."
        ),
    }
    return report


def _read_existing_reports(path: Path) -> list[dict[str, Any]]:
    reports = []
    if not path.exists():
        return reports
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                report = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from error
            reports.append(report)
    return reports


def _sample_id(report: dict[str, Any]) -> str | None:
    value = report.get("sample", {}).get("id")
    return str(value) if value is not None else None


def run_batch(
    args: argparse.Namespace, loaded, pairs: list[MatchedPair]
) -> dict[str, Any]:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    results_path = output / "samples.jsonl"
    existing = _read_existing_reports(results_path) if args.resume else []
    completed_ids = {_sample_id(report) for report in existing}
    completed_ids.discard(None)

    input_ids = {pair.id for pair in pairs}
    if len(input_ids) != len(pairs) or None in input_ids:
        raise ValueError("Every JSONL sample must have a unique non-empty ID")
    unexpected = completed_ids - input_ids
    if unexpected:
        raise ValueError(
            "Existing samples.jsonl contains IDs not present in this input: "
            + ", ".join(sorted(unexpected)[:5])
        )
    if existing and any(report.get("model") != args.model for report in existing):
        raise ValueError("Existing samples.jsonl was produced with a different model")
    if existing and any(report.get("method") != METHOD_NAME for report in existing):
        raise ValueError("Existing samples.jsonl was produced with a different scoring method")
    expected_tokens = {
        "sufficient": args.sufficient_token,
        "insufficient": args.insufficient_token,
    }
    if existing and any(report.get("label_tokens") != expected_tokens for report in existing):
        raise ValueError("Existing samples.jsonl used different A/B label tokens")

    mode = "a" if args.resume else "w"
    reports = list(existing)
    with results_path.open(mode, encoding="utf-8") as handle:
        for index, pair in enumerate(pairs, start=1):
            if pair.id in completed_ids:
                print(f"[{index}/{len(pairs)}] skip {pair.id} (already complete)", file=sys.stderr)
                continue
            print(f"[{index}/{len(pairs)}] run {pair.id}", file=sys.stderr)
            report = analyze_pair(args, loaded, pair)
            handle.write(json.dumps(report, ensure_ascii=False) + "\n")
            handle.flush()
            reports.append(report)

    return save_batch_summary(reports, output)


def main() -> None:
    args = build_parser().parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be a positive integer")
    pairs = load_matched_pairs(args.data)
    if args.limit is not None:
        pairs = pairs[: args.limit]
    loaded = load_model(args.model, dtype=args.dtype)

    batch_mode = Path(args.data).suffix.lower() == ".jsonl" or len(pairs) > 1
    if batch_mode:
        summary = run_batch(args, loaded, pairs)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    report = analyze_pair(args, loaded, pairs[0])
    save_report(report, args.output)
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
