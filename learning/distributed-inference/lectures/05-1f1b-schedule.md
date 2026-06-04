# L05 · 1F1B Schedule 详解

## 1 · Naive PP 时间线
```
n_stages=4, n_micro=4
GPU 0: F1 F2 F3 F4  -  -  -  -
GPU 1: -  F1 F2 F3 F4 -  -  -
GPU 2: -  -  F1 F2 F3 F4 -  -
GPU 3: -  -  -  F1 F2 F3 F4 -
```
bubble = 3 / 4 = 75%（巨大！）

## 2 · 推理优化：micro-batch overlap
推理只有 forward → 用更细 micro-batch 填 bubble。

```
n_micro=8:
GPU 0: F1 F2 F3 F4 F5 F6 F7 F8 -  -  -
GPU 1: -  F1 F2 F3 F4 F5 F6 F7 F8 -  -
...
```
bubble = 3/8 = 37.5%

## 3 · 训练：1F1B
为减 activation memory，F 和 B 交替：
```
GPU 0: F1 F2 F3 F4 B1 F5 B2 F6 B3 F7 ...
```
每 stage 只保留 N_stage activation。

## 4 · 推理 1F1B 不适用
因为推理只 F，没 B。但思想可借鉴：**连续 forward 不同 micro-batch**。

## 5 · vLLM/SGLang 怎么 PP
- continuous batching 已天然分摊 latency
- PP 在推理中较少用（TP + chunked prefill 已够）
- 仅在模型大到必须切 layer 时（>70B 单 stage 装不下）

## 6 · 实现
继承 [pp_demo.py](../src/pp_demo.py) 的 schedule 模拟。
