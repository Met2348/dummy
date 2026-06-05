# L04 · mini-HELM 设计

详见 `src/mini_helm.py`：
- 4 维度 score function
- run_mini_helm → 5×4 字典
- ascii_radar → 单 ckpt 雷达
- to_md → markdown 表格

跑：
```python
from mini_helm import run_mini_helm, to_md, ascii_radar
scores = run_mini_helm()
print(to_md(scores))
print(ascii_radar(scores["r1_tiny"]))
```

预期：vanilla 0.50, phi_tiny 0.93。

## 与真 HELM 区别

教学版数字、5 ckpt × 4 dim、~1s 跑完。
真 HELM 200+ scenario × 7 metric，几千 GPU-hour。

## 推荐：往真扩展

```python
# 加 SCORERS：用 lm-evaluation-harness 接 real bench
SCORERS["mmlu"] = lambda c: real_mmlu_score(c)
SCORERS["humaneval"] = lambda c: real_humaneval(c)
SCORERS["mt_bench"] = lambda c: real_mt_bench(c)
```

## 一句话

> mini-HELM = 4 维评测的极简实现，1s 出全 5 × 4 表格。
