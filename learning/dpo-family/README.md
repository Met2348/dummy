# DPO Family — 去 RM 革命

> 专题 3 / 7 — 13 lecture / RainbowPO 统一 / 6 方法 capstone
>
> Design: [link](../../docs/superpowers/specs/2026-06-03-dpo-family-design.md)

## 已完成（本轮）

- L01 DPO 推导 lecture (28 slides)
- src/dpo_minimal.py: 手写 4 行核心
- src/tests/test_dpo_loss_equivalence.py
- environment + papers

## 待续

L02-L13 (IPO/KTO/ORPO/SimPO/CPO/DPOP/Step-DPO/Iterative/Online/Nash/RainbowPO/Capstone)

## 运行验证（Runbook）

> 本段命令即 [`runbook.yaml`](runbook.yaml) 登记的"文档入口命令"，已在 ERIC-3080Ti（RTX 3080 Ti 16GB）上 V0+V1 验证通过（10/10）。
> 一键复验本模块：
> ```powershell
> python scripts/eric_3080ti_env_audit.py --runbook --modules dpo-family
> ```

**手写 DPO 训练**（GPT-2 actor+ref，Anthropic-HH 偏好对，真实 log-ratio DPO，可用 GPU）：

```powershell
# 真实跑（200 偏好对，1 epoch）
python learning/dpo-family/src/dpo_minimal.py --n-train 200 --epochs 1 --max-length 256 --log-interval 20
# 快速 smoke（验证可跑通，~30s）
python learning/dpo-family/src/dpo_minimal.py --n-train 8 --epochs 1 --max-length 64 --log-interval 4
```

**6+ PO 变体 minimal demo + 统一公式 + capstone**（纯数值 self-test，无 argparse、无网络，秒级 CPU 可跑）：

```powershell
python learning/dpo-family/src/ipo_minimal.py               # IPO：squared loss 高 margin 饱和 vs DPO 持续推
python learning/dpo-family/src/kto_minimal.py               # KTO：单边 desired/undesired 标签，前景理论非对称
python learning/dpo-family/src/simpo_minimal.py             # SimPO：length-norm reward + 无 ref，去长度偏置
python learning/dpo-family/src/cpo_minimal.py               # CPO：SFT NLL + λ·对比（actor 自代 ref）
python learning/dpo-family/src/dpop_minimal.py              # DPOP：DPO + hinge 惩罚 chosen 概率下降
python learning/dpo-family/src/orpo_minimal.py              # ORPO：log-odds 比替代 log π/π_ref，无 ref
python learning/dpo-family/src/rainbowpo.py                 # RainbowPO：一个 unified_po_loss + POConfig = 6 变体全覆盖
python learning/dpo-family/src/capstone_dpo_comparison.py   # Capstone：6 变体 50 步横向 benchmark（loss/margin/Δchosen_logp）
```

> ✅ **无 trl 依赖**：DPO/IPO/KTO/ORPO/SimPO/CPO/DPOP/RainbowPO **全是手写张量级 loss**（log-ratio of policy vs ref），不踩 trl 1.5.x `DPOTrainer`/`DPOConfig` 漂移坑。要 trl 生产路径自行 `pip install trl` 另测。
> 📦 **数据集**：`dpo_minimal.py` 用命名空间 id `Anthropic/hh-rlhf`（datasets 5.x 可加载，本机已缓存离线可跑）；加载失败时回退内置 dummy 偏好对**仍真实训练**（打印 WARN，不静默假成功）。
> 📁 **子模块** `official/repos/direct-preference-optimization`（eric-mitchell 原实现）仅供源码对照，不在 runbook 内运行。

**测试（V2）**：

```powershell
pytest learning/dpo-family/src/tests/
# 或经 harness：python scripts/eric_3080ti_env_audit.py --modules dpo-family --tests
```

