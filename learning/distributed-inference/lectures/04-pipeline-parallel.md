# L04 · Pipeline Parallel for Inference

## 1 · 朴素 idea
把 layer 1..L 分给 stages：
- GPU 0: layer 1-8
- GPU 1: layer 9-16
- GPU 2: layer 17-24
- GPU 3: layer 25-32

forward 流：GPU0 → 1 → 2 → 3

## 2 · 朴素 PP 的 bubble
单 batch 走完才能下一 batch → 大部分 GPU 在 idle。

```
GPU 0: [F1] [  ] [  ] [F2] [  ] [  ]
GPU 1: [  ] [F1] [  ] [  ] [F2] [  ]
GPU 2: [  ] [  ] [F1] [  ] [  ] [F2]
```
bubble = (N_stage - 1) / N_micro

## 3 · 1F1B (1 forward 1 backward) schedule
对推理：用 micro-batch 流水化。
对训练：F-B-F-B 交替，减 activation memory。

## 4 · Interleaved 1F1B
把 layer 进一步切：
- GPU 0: layer 1-4, 17-20
- GPU 1: layer 5-8, 21-24
- ...

bubble 减少到 (N_stage / N_chunks - 1) / N_micro。

## 5 · PP 与 TP 组合
- 7B 单 GPU 够
- 70B: TP=8 + PP=1 (单节点)
- 405B: TP=8 + PP=2 (2 节点)
- 671B (DeepSeek-V3): TP=8 + EP=64 + PP=2

## 6 · 推理 PP 痛点
- prefill 长 → 整流水线全卡住
- → 与 chunked prefill 必组合
- 或 → disaggregated（prefill 独立节点）

## 7 · 实现：[pp_demo.py](../src/pp_demo.py)
- PipelineStage class
- 朴素 PP + 1F1B mock
- 计算 bubble 比例
