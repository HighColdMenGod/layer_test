# Curated IIRC one-pair set

`dev_one_pairs_curated_hard_50.jsonl` contains the 50 matched pairs for the
insufficient-information experiment.

- 33 original evidence pairs passed strict semantic review.
- 15 pairs repair truncated/noisy IIRC snippets using the same source passage.
- 2 pairs are marked `usable_weak` with their caveats recorded in metadata.
- Every `C-` retains the gold-answer text but lacks one comparison operand.

Per-item decisions are recorded in
`dev_one_pairs_curated_hard_50_audit.csv`.

Run:

```bash
insufficient-shift \
  --model MODEL_PATH \
  --data data/iirc_one_pairs/dev_one_pairs_curated_hard_50.jsonl \
  --output output/iirc_curated_hard_50 \
  --last-k 12
```
