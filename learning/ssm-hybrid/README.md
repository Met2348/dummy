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

## Tags

```
ssm-foundations (L01-L02)
ssm-mamba       (L03-L05)
ssm-rwkv        (L06-L07)
ssm-hybrid-arch (L08-L10)
ssm-hybrid      (L11 收口)
```
