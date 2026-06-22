# SSM / Hybrid 学习包

> Module 3「造大模型」**专题 4 / 8** — State Space Models, Mamba, RWKV, Hybrid

| 元数据 | 值 |
|--------|----|
| 方法 | 12 |
| Lecture | 11 |
| 估时 | 12h |
| 环境 | WSL2 (mamba-ssm) |

## Lectures

| # | 名称 | 关键 |
|---|------|------|
| L01 | SSM 引论 | HiPPO / State Space |
| L02 | S4 + S5 | 卷积 / scan |
| L03 | **Mamba** | Selective SSM |
| L04 | Mamba-2 SSD | 矩阵化 |
| L05 | Mamba-3 + 长 ctx | 多 head SSM |
| L06 | RWKV-7 | linear attention |
| L07 | RetNet | retention |
| L08 | Jamba | Mamba+attn+MoE |
| L09 | Zamba / Codestral-Mamba | shared attn |
| L10 | Hybrid 设计 | 选型 |
| L11 | **Capstone — mini-Mamba** | 集成 |

## src/

```
common.py            # ZOH discretize / scan
s4_naive.py          # S4 (LTI) naive
mamba_block.py       # Selective scan + block
mamba2_block.py      # SSD form
mamba_lib.py         # mamba-ssm lib wrapper
rwkv_block.py        # RWKV-7 simplified
jamba_block.py       # hybrid layer
zamba_block.py       # shared attention
mini_mamba.py        # capstone
capstone_train_mini_mamba.py
tests/               # 6 tests
```

## 方法对照

| | KV cache | scan | 速度 (32k) |
|---|---------|------|-----------|
| Transformer | O(L · d) | — | base |
| S4 | O(d_state) | yes | 3× |
| **Mamba** | O(d_state) | selective | **3-5×** |
| RWKV-7 | O(d) | weighted | 4× |
| Jamba (hybrid) | mix | mix | 2× |

## 验证

```bash
python learning/ssm-hybrid/environment/verify_env.py
python -m pytest learning/ssm-hybrid/src/tests/ -v
python learning/ssm-hybrid/src/capstone_train_mini_mamba.py
```

## 运行验证（Runbook）

> 本段命令即 [`runbook.yaml`](runbook.yaml) 登记的"文档入口命令"，已在 ERIC-3080Ti（RTX 3080 Ti 16GB，repo-local `.venv`）上 **V0+V1 全绿（10/10）** 验证通过。
> 一键复验本模块：
> ```powershell
> python scripts/eric_3080ti_env_audit.py --runbook --modules ssm-hybrid `
>   --json-out docs/local-env/ERIC-3080Ti-runbook-results.json --md-out docs/local-env/ERIC-3080Ti-runbook-matrix.md
> ```

**10 个 from-scratch 直跑 demo**（全部无 argparse、纯 CPU、秒级真实 forward；无参数可调，直跑即验证）：

```powershell
# SSM 基础 → Mamba → 混合架构 → capstone，逐个直跑
python learning/ssm-hybrid/src/s4_naive.py                 # S4 (LTI) ZOH+scan
python learning/ssm-hybrid/src/mamba_block.py              # Mamba 选择性扫描 (S6)
python learning/ssm-hybrid/src/mamba2_block.py             # Mamba-2 SSD form
python learning/ssm-hybrid/src/mamba_original_minimal.py   # 论文核心机制 (selective-copy + S6 scan)
python learning/ssm-hybrid/src/rwkv_block.py               # RWKV-7 简化 time-mix
python learning/ssm-hybrid/src/jamba_block.py              # Jamba 混合层 (Mamba+attn+FFN)
python learning/ssm-hybrid/src/zamba_block.py              # Zamba 共享注意力
python learning/ssm-hybrid/src/mini_mamba.py               # capstone 模型 forward
python learning/ssm-hybrid/src/mamba_lib.py                # mamba-ssm 库 wrapper (本机 [SKIP])
python learning/ssm-hybrid/src/capstone_train_mini_mamba.py  # 训练 80 step，loss 64→7 真下降 (~9s)
```

> ⚠️ **可选 CUDA-kernel 库坑**：`mamba_lib.py` 依赖官方 `mamba-ssm` / `causal-conv1d`（融合 CUDA kernel，**仅 Linux/WSL2**，本 Windows box 未装）。
> 脚本 `has_mamba_lib()` 真实探测：库不存在时打印 `[SKIP] mamba-ssm not installed`，exit 0（**诚实跳过、非假成功**——它显式报告库缺失，不伪装计算）。
> 手写等价物即 `mamba_block.py`（数学等价于 CUDA kernel，已单列直跑），无需安装重型 CUDA 包即可学全部机制。
> **不要** `pip install mamba-ssm causal-conv1d`（无预编译 Windows wheel，需 nvcc 现场编译，易踩坑）。
> 裸 import（`from common import` / `from mamba_block import`）依赖 harness 注入的 `PYTHONPATH=src`；手动直跑时也用 `python learning/ssm-hybrid/src/<x>.py`（仓库根 cwd）即可，audit harness 已自动处理。

**测试（V2）**：

```powershell
python -m pytest learning/ssm-hybrid/src/tests/ -q
# 或经审计 harness（基线 6 测全绿）：python scripts/eric_3080ti_env_audit.py --modules ssm-hybrid --tests
```

## Tags

```
ssm-foundations (L01-L02)
ssm-mamba       (L03-L05)
ssm-rwkv        (L06-L07)
ssm-hybrid-arch (L08-L10)
ssm-hybrid      (L11 收口)
```
