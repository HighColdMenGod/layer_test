# Evidence Inference 2.0 validation one-pair subset

This directory contains 150 matched context pairs constructed from the official
Evidence Inference 2.0 validation split for a layer-wise sufficiency probe.

## Files

- `validation_one_pairs_150.jsonl`: model-ready examples.
- `validation_one_pairs_150_audit.csv`: flat metadata for manual review.
- `validation_one_pairs_150_summary.json`: selection counts and quality summary.

## Pair construction

Each example uses one Evidence Inference prompt and its human-annotated abstract
evidence:

- `complete_context` (C+) is a local abstract window containing the selected
  evidence sentence, normally two sentences before and two after it.
- `insufficient_context` (C-) is the identical window with only that evidence
  sentence removed.
- `removed_critical_fact` records the exact deleted sentence.

The selection excludes prompt IDs listed as incorrect, questionable, or
malformed in the dataset README. It also requires valid label/reasoning flags,
agreement on the three-way label, at least two valid human annotators, valid
abstract offsets, and an evidence sentence that explicitly expresses the
annotated effect direction. A C+ window is rejected if it contains another
human-annotated evidence sentence for the same prompt.

The final subset is balanced across labels (50 decreased, 50 no-difference,
50 increased) and contains at most one prompt from each article.

## JSONL schema

Top-level fields are `id`, `question`, `complete_context`,
`insufficient_context`, `removed_critical_fact`, and `metadata`. Metadata keeps
the source PromptID/PMCID, label, intervention/comparator/outcome, source
offsets, annotator support, alignment score, and construction diagnostics.

## Limitation

C- is a controlled synthetic ablation, not a separately human-annotated
insufficient passage. Other sentences may contain results for different
outcomes. The construction removes every annotated evidence sentence for the
target prompt from the local C- window, but a small manual review is still
recommended before reporting individual examples as qualitative case studies.

Regenerate from the dataset root with:

```bash
python3 scripts/build_evidence_inference_pairs.py \
  --data-dir data/v2.0 --split validation --count 150
```
