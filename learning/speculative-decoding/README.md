# Topic 3: Speculative Decoding（投机解码全谱）

> Module 5 「用大模型」第 3 专题 · 12 lectures · 12 notebooks · ~13h

## 总览

| Lecture | 主题 | 代码 |
|---------|------|------|
| L01 | 直觉（猜+验） | `common.py` |
| L02 | Classic (Leviathan 2023) | `classic_spec_decode.py` |
| L03 | Medusa (Princeton 2024) | `medusa_heads.py` |
| L04 | EAGLE (THU 2024) | `eagle_minimal.py` |
| L05 | **EAGLE-2 dynamic tree** ⭐ | `eagle2.py` |
| L06 | EAGLE-3 (2025) | — |
| L07 | Lookahead (UCSD 2024) | `lookahead.py` |
| L08 | Self-Speculative | `self_spec.py` |
| L09 | Tree Attention | `tree_attention.py` |
| L10 | Dynamic draft tree | — |
| L11 | 评测方法 | `spec_eval.py` |
| L12 | **Capstone: 4 method × 5 task** ⭐ | `capstone_eagle3.py` |

## Tags

- `spec-decode` — 最终（含 capstone + README）

## 三轨代码策略

| 轨 | 工具 | 文件 |
|----|------|-----|
| minimal | 手写 8 个方法 | 各 `_minimal/_spec/_heads.py` |
| 库 | (留 placeholder：vllm spec / sglang spec) | — |
| kernel | tree attention | `tree_attention.py` |

## 算法关系图

```
classic spec (2023)              ← 单 draft model
   ↓ 共享 backbone
Medusa (2024)                    ← 多 LM head
   ↓ feature 投机
EAGLE-1 (2024) → EAGLE-2 (2024.06) → EAGLE-3 (2025)
                  ↑ dynamic tree
   ↓ 不需要 draft model
Lookahead / Self-Spec           ← Jacobi / skip layer
```

## Capstone 实测（synthetic 数据，不是真模型 benchmark）

| method | task | accept | MAU | sim_speedup |
|--------|------|--------|-----|-------------|
| classic | math | 1.00 | 4.00 | 3.57x |
| classic | json | 1.00 | 4.00 | 3.57x |
| eagle1  | code | 0.38 | 1.53 | 1.81x |
| eagle2  | math | 0.33 | 1.67 | 1.78x |
| ... |

> Synthetic 数据下分布噪声小 ⇒ classic 看起来反胜；真模型场景需 Qwen-1.5B+0.5B 跑：
> `python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-1.5B --speculative-model Qwen/Qwen2.5-0.5B --num-speculative-tokens 5`

## 环境

```powershell
python environment/verify_env.py
```

## 运行

```powershell
# 测试 (12/12 全绿)
python -c "import sys; sys.path.insert(0,'src'); sys.path.insert(0,'src/tests'); import test_spec_decode"

# Capstone
python src/capstone_eagle3.py
```

## 退出条件 checklist

- [x] 12 lecture + 12 notebook
- [x] 12 tests pass
- [x] 4 method × 5 task capstone 表
- [x] git tag `spec-decode` ✓

## 关键文献

- Leviathan et al., *Fast Inference from Transformers via Speculative Decoding* (2023)
- Chen et al., *Accelerating LLM Decoding with Speculative Sampling* (2023)
- Medusa (Cai 2024)
- EAGLE / EAGLE-2 / EAGLE-3 (Li 2024-2025)
- Lookahead (UCSD 2024)
- LayerSkip (Meta 2024.04)

## 一句话

> "draft k token + 并行 verify k+1 + rejection sample = **零 bias 加速**"。这套公式从 2023 到 2025 没变；变的只是 draft 模型从独立小模型 → Medusa head → EAGLE feature layer → EAGLE-2 dynamic tree → EAGLE-3 multi-feature。
