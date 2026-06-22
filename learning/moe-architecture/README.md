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

## 运行验证（Runbook）

> 本段命令即 [`runbook.yaml`](runbook.yaml) 登记的"文档入口命令"，已在 ERIC-3080Ti（RTX 3080 Ti 16GB，repo-local `.venv`）上 V0+V1 验证通过（13/13 PASS）。
> 一键复验本模块：
> ```powershell
> .venv/Scripts/python.exe scripts/eric_3080ti_env_audit.py --runbook --modules moe-architecture
> ```

全部 13 个 demo 均为**无参直跑**（无 argparse，`v0: false`），CPU 即可，无需 GPU、零网络下载：

**路由算法（forward self-test，秒级，真打印张量/load 分布）**：

```powershell
python learning/moe-architecture/src/moe_layer_naive.py    # Shazeer top-2 + aux loss
python learning/moe-architecture/src/gshard_router.py      # GShard top-2 + capacity drop
python learning/moe-architecture/src/switch_router.py      # Switch top-1 极简
python learning/moe-architecture/src/expert_choice.py      # Expert Choice 反向路由
python learning/moe-architecture/src/deepseekmoe_layer.py  # 细粒度 + shared expert
```

**训练动态 / 稳定性（真跑 100~200 step，~10-20s）**：

```powershell
python learning/moe-architecture/src/aux_loss_free.py      # DS-V3 bias 控负载，load 真平衡
python learning/moe-architecture/src/router_z_loss.py      # z-loss + 有/无 aux 路由坍缩对照
```

**架构 summary（纯 config dict 打印，<1s，不下载权重）**：

```powershell
python learning/moe-architecture/src/mixtral_load.py       # Mixtral 8x7B / 8x22B
python learning/moe-architecture/src/phi_moe_load.py       # Phi-3.5-MoE
python learning/moe-architecture/src/qwen3_moe_load.py     # Qwen3-A3B / 235B
```

**推理优化 + 集成 + Capstone**：

```powershell
python learning/moe-architecture/src/grouped_gemm_demo.py        # naive vs block_diag fused 实测
python learning/moe-architecture/src/mini_moe.py                 # 4-routed/1-shared Aux-Free 集成
python learning/moe-architecture/src/capstone_train_mini_moe.py  # mini-MoE 训练，loss 60.6→12.2 真下降 (~6s)
```

> 📝 **坑注记**：
> - `mixtral/phi/qwen3_moe_load` 是**纯 config 打印**，秒级 PASS 属正常（本就无外部依赖，非"假成功"no-op）。
> - `grouped_gemm_demo` 在 CPU 小尺寸下 `block_diag` fused **比 naive 慢**（speedup<1×，打印真实测得数）；该融合在 GPU 大尺寸才显优势，属教学对照非 bug。
> - 全部脚本无 argparse → `runbook.yaml` 里 `v0: false`（跳过 `--help` 探针），`capstone_train_mini_moe.py` 直跑硬编码 100 step。
> - megablocks（L00 lecture）仅作背景，`src/` 不依赖；`grouped_gemm` 用纯 `torch.block_diag` 教学实现。
> - 文档微漂移（不影响运行）：README「方法数 14」实 13 个可跑脚本（+ `common.py` 工具）；论文 PDF 实际在 `paper/`（`papers/` 仅占位 README）；环境标注 WSL2，但本机 Windows 原生 `.venv` 即可跑通全部 demo。

**测试（V2）**：

```powershell
python -m pytest learning/moe-architecture/src/tests/ -v
# 或经审计 harness：python scripts/eric_3080ti_env_audit.py --modules moe-architecture --tests
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
