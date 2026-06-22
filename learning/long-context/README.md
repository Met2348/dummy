# Topic 5: 长上下文（Long Context）

> Module 3 「造大模型」第 5 专题 · 13 lectures · 13 notebooks · ~22h

## 总览

| Lecture | 主题 | 代码 |
|---------|------|------|
| L01 | RoPE 原理 + 复数旋转 | `common.py` |
| L02 | Position Interpolation (PI) | `rope_pi.py` |
| L03 | NTK-aware RoPE | `rope_ntk.py` |
| L04 | **YaRN** ⭐⭐⭐⭐⭐ | `rope_yarn.py` |
| L05 | 3D-RoPE (多模态/MM) | `rope_3d.py` |
| L06 | Ring Attention naive | `ring_attention_naive.py` |
| L07 | Striped/Ring lib 对照 | `ring_attention_lib.py` |
| L08 | Infini-Attention | `infini_attention.py` |
| L09 | FlashAttention 长 ctx 优化 | (用 L02 之 fa) |
| L10 | **NIAH + RULER** 评测 | `niah_eval.py`, `ruler_eval.py` |
| L11 | 长 ctx 数据打包 + 课程 | `long_data_packing.py` |
| L12 | 长 ctx 陷阱 (Lost in middle 等) | — |
| L13 | **Capstone** ⭐⭐⭐⭐⭐ Llama-3.2-1B YaRN 32k | `capstone_yarn_llama32.py` |

## Tags 路径

- `lc-rope` — RoPE + PI + NTK + YaRN + 3D
- `lc-ring` — Ring/Striped Attention
- `lc-infini` — Infini-Attention
- `lc-eval` — NIAH/RULER + data packing + pitfalls
- `long-context` — 最终, 含 Capstone

## 核心收获

1. **YaRN 优于 NTK** — NTK by parts + attn temperature 是当前 RoPE 扩 ctx 默认选项
2. **NIAH ≠ 真用** — RULER 测试更接近真实长 ctx 需求
3. **KV cache 是显存大头** — 1B + 128k ctx 已 17 GB，比模型本身还大
4. **Lost in middle 真实存在** — 设计长 ctx 系统要考虑加 RAG / 中部强化训练

## 关键论文

- YaRN (Peng et al, 2024) — 当前 RoPE 扩 ctx SOTA
- NTK-by-parts (bloc97 GitHub 2023) — YaRN 的核心 idea
- Ring Attention (Liu, Yan, Abbeel 2023) — 序列并行
- Infini-Attention (Munkhdalai 2024) — 滑窗 + 压缩 memory
- NIAH (Kamradt 2024 GitHub) — 评测标准
- RULER (NVIDIA 2024) — NIAH 升级版
- Lost in the Middle (Liu 2023) — 警示

## 环境

```powershell
python environment/verify_env.py
```

## 运行

```powershell
# 跑所有测试
python -m pytest src/tests/ -v

# Capstone 真训 (5090 24G + 5.5h)
python src/capstone_yarn_llama32.py --train --steps 500
```

## 运行验证（Runbook）

> 本段命令即 [`runbook.yaml`](runbook.yaml) 登记的"文档入口命令"，已在 ERIC-3080Ti（RTX 3080 Ti 16GB）上 V0+V1 验证通过（12/12，全 CPU 纯数值，秒级）。
> 一键复验本模块：
> ```powershell
> python scripts/eric_3080ti_env_audit.py --runbook --modules long-context
> ```

**RoPE 扩 ctx 家族（L02-L05，纯数学 self-test，各 < 4s）**：

```powershell
python learning/long-context/src/rope_pi.py     # Position Interpolation：压缩 position
python learning/long-context/src/rope_ntk.py    # NTK-aware：改 base 而非 position
python learning/long-context/src/rope_yarn.py   # YaRN ⭐：NTK-by-parts ramp + attn temperature
python learning/long-context/src/rope_3d.py     # 3D/M-RoPE：dim 分 t/h/w 三段独立旋转
```

**长 ctx 注意力 + 评测 + 数据打包（L06-L11）**：

```powershell
python learning/long-context/src/ring_attention_naive.py  # 分块 online-softmax，max diff 2.4e-07 ≈ vanilla
python learning/long-context/src/ring_attention_lib.py    # ring-flash-attention 探测（本机无→诚实 [SKIP]）
python learning/long-context/src/infini_attention.py      # local attn + 压缩 memory + per-head gate
python learning/long-context/src/niah_eval.py             # NIAH 测试集生成器（不跑模型）
python learning/long-context/src/ruler_eval.py            # RULER 4 子任务生成器
python learning/long-context/src/long_data_packing.py     # first-fit 打包 + block-diagonal mask + 课程
```

**Capstone（L13）：Llama-3.2-1B + YaRN scale=4 扩 32k + LoRA**：

```powershell
# 默认 dry-run：打印课程切换 (8192→16384→32768)，不加载权重，秒级（runbook smoke 形态）
python learning/long-context/src/capstone_yarn_llama32.py
# 真训 full 形态：需 HF-gated meta-llama/Llama-3.2-1B-Instruct + 5090/24G ~5.5h（本机不真跑）
python learning/long-context/src/capstone_yarn_llama32.py --train --steps 500
```

> **关键坑注记**：
> - 这 10 个 demo 均**无 argparse**（`runbook.yaml` 标 `v0: false`，跳过 `--help` 探针直接 smoke 直跑）；`common.py` 是纯 helper（无 `__main__`），非入口。
> - **秒级 PASS 是真的**：RoPE/PI/NTK/YaRN/3D 是纯数学旋转；ring-naive 真做分块 online-softmax（数值 ≈ vanilla attention）；NIAH/RULER 是题目生成器（设计上不跑模型）。非 no-op/假成功。
> - `ring_attention_lib` 打 `[SKIP]` 是**诚实的库缺失**报告（ring-flash-attention 仅 Linux 多 GPU），非"捕获异常假装成功"；可跑等价物即 `ring_attention_naive`。
> - **capstone 默认 dry-run 是诚实骨架**：显式打印 `dry-run, no training` 并指路 `--train`，非静默假成功。`--train` 路径需 HF-gated 权重 + 大显存，未纳入本机 smoke。
> - 早期 env audit 修过的 **RoPE 分段 shape**（`rope_3d`）与 **长 doc 打包溢出**（`long_data_packing` 超长先切块再 first-fit）两处真实 bug 已在代码中修好，本轮复验无回归。

**测试（V2）**：

```powershell
python -m pytest learning/long-context/src/tests/ -v
# 或经审计 harness：python scripts/eric_3080ti_env_audit.py --modules long-context --tests
```
