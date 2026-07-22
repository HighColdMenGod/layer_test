# One-pair Insufficient Shift Probe

这是一个最小实验架构，用一条 **matched pair** 快速检查 decoder-only LLM 的内部层是否出现信息不足信号的 late-layer shift。

输入不是单独一个 `C-`，而是一组只差一个关键事实的上下文：

- `C+`：足以回答问题的完整上下文；
- `C-`：仍高度相关、信息很多，但删除了决定答案成立的关键事实（pseudo-sufficient）。

代码不使用一条样本训练 probe。它要求模型在单 token 标签 `A`（Sufficient）和
`B`（Insufficient）之间选择，并用 final norm + LM head 读取每层最后位置的两个
logit。主要分析量是 `B-A` logit margin；A/B 归一化后的条件概率只用于辅助展示。

## 快速运行

```bash
python -m pip install -e .
insufficient-shift \
  --model Qwen/Qwen2.5-7B-Instruct \
  --data data/example_pair.json \
  --last-k 12 \
  --dtype bfloat16 \
  --output outputs/qwen_one_pair
```

本地模型可直接把 `--model` 换成本地目录。显存有限时先用 1.5B/3B 模型验证管线。

输出：

- `report.json`：逐层 A/B logits、条件概率、pair contrast 和诊断指标；
- `layer_shift.png`：`C+` 与 `C-` 的逐层 `P(Insufficient)` 曲线。

## 批量运行 JSONL

JSONL 文件每行放一个 matched-pair JSON 对象，可以包含额外的 `id` 和
`metadata` 字段：

```bash
insufficient-shift \
  --model Qwen/Qwen2.5-7B-Instruct \
  --data data/v2.0/derived/validation_one_pairs_150.jsonl \
  --last-k 12 \
  --dtype bfloat16 \
  --output outputs/qwen_validation_150
```

批量结果会逐条追加到 `samples.jsonl`，每条完成后立即落盘；最终生成
`batch_summary.json` 和 `mean_layer_shift.png`。如果运行中断，使用完全相同的
参数并增加 `--resume`，会按照样本 ID 跳过已经完成的数据。可先用
`--limit 3` 做小规模管线检查。

## 三个首要诊断量

1. `sufficiency_flip`：`C-` 的 B-A margin 在中间层大于等于 0，最终层却小于 0。
2. `late_insufficient_margin_drop`：`C-` 的最佳中间层 margin 减去最终层 margin。
3. `pair_contrast_collapse`：`[margin(C-) - margin(C+)]` 从最佳层到最终层缩小了多少。

`has_shift_signal` 默认在出现 flip，或后两项任一至少下降 0.10 logit 时为真。这个阈值只是单样本 smoke test 的观察标准，不是统计显著性标准。

## 实验边界

raw logit lens 能回答“内部 residual stream 经统一输出头后，标签倾向如何变化”，但不能单独证明该信号被模型因果使用。建议按以下顺序扩展：

1. 先确认单对样本的 prompt、tokenization 和曲线合理；
2. 扩到几十至几百个 matched pairs，报告 effect size 与 bootstrap CI；
3. 使用独立训练/验证数据拟合 layer-wise probe 或 tuned lens；
4. 对候选层做 activation patching/steering，验证能否改变最终 sufficiency decision。
