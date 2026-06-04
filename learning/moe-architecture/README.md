# MoE Architecture 学习包

> Module 3「造大模型」**专题 3 / 8** — Mixture of Experts 从 Shazeer 2017 到 DeepSeek-V3 Aux-Free

| 元数据 | 值 |
|--------|----|
| 方法数 | 14 |
| Lecture | 13 + L00 setup |
| 估时 | 16h |
| 环境 | **WSL2** (Module 2 末已切) |
| Design | `docs/superpowers/specs/2026-06-04-moe-architecture-design.md` |
| Plan | `docs/superpowers/plans/2026-06-04-moe-architecture.md` |

---

## Lecture

| # | Lecture | 关键 |
|---|---------|------|
| L00 | WSL2 + megablocks setup | 环境 |
| L01 | Sparse MoE 起点 (Shazeer 2017) | 基础 |
| L02 | GShard top-2 + capacity | 工程化 |
| L03 | Switch top-1 极简 | 简化 |
| L04 | Expert Choice 反向路由 | encoder 友好 |
| L05 | Mixtral 8x7B | 开源主流 |
| L06 | DeepSeekMoE 细粒度 + shared | DS-V2 |
| L07 | **Aux-Loss-Free** ⭐⭐⭐⭐⭐ | DS-V3 核心 |
| L08 | Phi-MoE 小路线 | MS |
| L09 | Qwen3-MoE | 阿里 |
| L10 | MoR (Mixture of Recursions) | 实验性 |
| L11 | MoE training (z-loss + crash) | 稳定性 |
| L12 | MoE inference (grouped GEMM) | 优化 |
| L13 | **Capstone — mini-MoE** | 集成 |

---

## src/

```
common.py                # load metrics / capacity
moe_layer_naive.py       # Shazeer 教学版
gshard_router.py         # GShard top-2
switch_router.py         # Switch top-1
expert_choice.py         # 反向路由
mixtral_load.py          # Mixtral config
deepseekmoe_layer.py     # DeepSeekMoE
aux_loss_free.py         # ⭐ Aux-Free
phi_moe_load.py / qwen3_moe_load.py
router_z_loss.py         # z-loss + crash demo
grouped_gemm_demo.py     # 推理优化
mini_moe.py              # capstone 集成
capstone_train_mini_moe.py
tests/                   # 4 测试
```

---

## Routing 方法对照

| 方法 | top-k | aux | 推荐 |
|------|-------|-----|------|
| Shazeer | 2 | yes | 起点 |
| GShard | 2 | yes | 工程 baseline |
| Switch | 1 | yes | 极简 |
| Expert Choice | reverse | no | encoder |
| Mixtral | 2 | yes | 主流开源 |
| DeepSeekMoE | 6-8 | yes | 细粒度 |
| **DS-V3 Aux-Free** | 8 | **no** ⭐ | **未来主流** |

---

## 验证命令

```bash
python learning/moe-architecture/environment/verify_env.py
python -m pytest learning/moe-architecture/src/tests/ -v
jupyter nbconvert --execute --inplace learning/moe-architecture/notebooks/*.ipynb
python learning/moe-architecture/src/capstone_train_mini_moe.py
```

---

## Git Tags

```
chore: moe-architecture scaffold
moe-routing        (L01-L04)
moe-modern         (L05-L06)
moe-aux-free       (L07) ⭐
moe-new            (L08-L10)
moe-stability      (L11-L12)
moe-arch           (L13 收口) ⭐
```

---

## 与其他专题接口

```
本专题 → 把 GPT-mini MLP 替换为 MoE
       → 专题 7 真训时可选 dense vs MoE
       → 专题 8 graduation 五部曲对照
```
