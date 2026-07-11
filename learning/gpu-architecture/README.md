# GPU Architecture — GPU 体系结构基础（Roofline 为核心）

> Module 8（系统与 Infra）7 专题系列的第 1 站。核心论文：Samuel Williams / Andrew Waterman / David
> Patterson — *Roofline: An Insightful Visual Performance Model for Floating-Point Programs and
> Multicore Architectures*（2009）。8 篇 lecture + 8 个纯 CPU 可直跑 self-test + 1 篇 806 行中文论文导读。
>
> 论文导读：[`paper/guide_01_roofline_model.md`](paper/guide_01_roofline_model.md)（原文 PDF 同目录）

---

## 专题概览

| # | Lecture | 主题 | 核心公式 / idea | 对应代码 |
|---|---------|------|----------------|---------|
| 01 | gpu-overview | 4 代旗舰 GPU 参数对照 + Roofline 直觉 | peak TFLOPS 涨速 (~3×/代) > HBM 带宽涨速 (~1.5×/代) → ridge point 整体右移 | — （纯概念表，数据源同 `common.py`） |
| 02 | memory-hierarchy | 5 层存储金字塔 | registers→SMEM→L1→L2→HBM3，延迟 1→600 cycle 单调升，带宽单调降 | [`memory_hierarchy.py`](src/memory_hierarchy.py) |
| 03 | tensor-core | Tensor Core 4 代演进 | V100 FP16 → A100 BF16/TF32 → H100 wgmma FP8 → B200 tcgen05 FP4，accumulator 从 RF 挪到独立 tensor memory | [`tensor_core.py`](src/tensor_core.py) |
| 04 | sm-occupancy | SM 占用率四瓶颈 | `blocks_per_sm = min(threads, warps, regs, smem, max_blocks)`；occupancy 高 ≠ 性能好（Volkov） | [`sm_occupancy.py`](src/sm_occupancy.py) |
| 05 | nvlink-topology | NVLink/NVSwitch 4 代 + Ring All-Reduce | `time = 2(N-1)/N × size/BW`；GB200 NVL72 把 72 卡当一个超级 GPU 用 | [`nvlink_topology.py`](src/nvlink_topology.py) |
| 06 | roofline ⭐ | Roofline 模型实战 + 复现 2009 原始论文 | `achievable = min(peak, BW×AI)`；`ridge = peak/BW` | [`roofline.py`](src/roofline.py) + [`roofline_original_minimal.py`](src/roofline_original_minimal.py) + [`common.py`](src/common.py) |
| 07 | hbm-bandwidth | HBM 是推理 decode 的天花板 | `decode tok/s ≤ BW / (weight_bytes + KV_bytes)`；量化是最大杠杆 | — （纯公式，复用 `common.py` 的 `hbm_tb_s`） |
| 08 | capstone | Roofline Zoo：10 LLM op × 4 代 GPU | 40 组分析里 25 组 memory-bound，H100 大 GEMM util ≥ 80% | [`capstone_roofline_zoo.py`](src/capstone_roofline_zoo.py) |

**预计学时**：约 4 h（8 篇 lecture 精读 + 论文导读通读 + 8 个脚本全部跑一遍并读源码）。

---

## 学习路径

```
        L01 GPU 总览（4 代对照 + Roofline 直觉）
                |
        L02 存储层次（5 层金字塔）
                |
        L03 Tensor Core（4 代 MMA 演进）
        L04 SM 占用率（4 个瓶颈维度）
                |
        L05 NVLink 拓扑（Ring All-Reduce）
                |
        L06 Roofline 模型 ⭐ 核心（现代 GPU + 复现 2009 原论文）
        L07 HBM 带宽（推理 decode 天花板）
                |
        L08 Capstone：Roofline Zoo（10 op × 4 GPU = 40 组）
```

---

## 目录结构

```
gpu-architecture/
├── README.md
├── runbook.yaml
├── paper/
│   ├── README.md                        # 论文索引（1 篇）
│   ├── 01_roofline_model.pdf             # 原始论文 PDF
│   ├── guide_01_roofline_model.md        # 806 行中文导读
│   └── guide_01_roofline_model.pdf       # 导读渲染版 PDF
├── lectures/
│   └── 01..08-*.md                       # 8 篇 lecture markdown
└── src/
    ├── common.py                     # GPUSpec dataclass + A100/H100/H200/B200/GB200 参数表 + ridge point
    ├── memory_hierarchy.py           # 5 层存储金字塔 + tier 推荐
    ├── tensor_core.py                # MMA 形状选择（wgmma / tcgen05）
    ├── sm_occupancy.py               # SM 占用率四瓶颈计算器
    ├── nvlink_topology.py            # NVLink 拓扑 + Ring All-Reduce
    ├── roofline.py                   # 现代 GPU roofline 分析器（依赖 common.py）
    ├── roofline_original_minimal.py  # 复现 2009 原始 Roofline 论文模型（Opteron X2/X4 + 4 kernel + ceilings）
    ├── capstone_roofline_zoo.py      # Capstone：10 op × 4 GPU（依赖 common.py + roofline.py）
    └── tests/
        └── test_all.py               # 聚合 8 模块 _self_test()（脚本式，无 test_ 前缀函数）
```

**没有** `environment/`、`notebooks/`、`official/` 目录——本模块所有代码只依赖标准库，无需外部权重/框架，见下节。

---

## 环境配置

本模块**没有** `environment/` 目录、也没有 `verify_env.py`——不是遗漏，是因为 8 个 `src/*.py`
全部只 `import dataclasses` / `from __future__ import annotations`（+ 互相 import：`roofline.py`
依赖 `common.py`；`capstone_roofline_zoo.py` 依赖 `common.py` 和 `roofline.py`），**零第三方包、零网络、
零 GPU 依赖**（`grep -rn "add_argument\|ArgumentParser\|^import \|^from " src/*.py` 可自行核实）。

复用仓库根 `.venv`（Python 3.13）即可，不需要额外 `pip install`。跑起来本身就是"验证"：

```powershell
python learning/gpu-architecture/src/common.py
# [OK] gpu_architecture.common (H100 ridge 295 FLOP/byte)
```

不需要设置 `PYTHONPATH`，也不依赖当前工作目录：Python 解释器会自动把脚本所在目录插进
`sys.path[0]`，所以从 repo 根目录直接 `python learning/gpu-architecture/src/roofline.py` 就能
解析 `from common import ...`（已实测：unset `PYTHONPATH` 后从 repo 根跑 `roofline.py` /
`capstone_roofline_zoo.py` 均正常）。

---

## 横向对比：5 代旗舰 GPU（数据源 `src/common.py`）

| GPU | BF16 TFLOPS | FP8 TFLOPS | FP4 TFLOPS | HBM | HBM BW | NVLink | TDP | Ridge Point (BF16) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A100-80G | 312 | — | — | 80 GB | 2.04 TB/s | 0.6 TB/s | 400 W | 153.0 FLOP/byte |
| H100-SXM5 | 989 | 1979 | — | 80 GB | 3.35 TB/s | 0.9 TB/s | 700 W | 295.2 FLOP/byte |
| H200-SXM | 989 | 1979 | — | 141 GB | 4.80 TB/s | 0.9 TB/s | 700 W | 206.0 FLOP/byte |
| B200 | 2250 | 4500 | 9000 | 192 GB | 8.00 TB/s | 1.8 TB/s | 1000 W | 281.2 FLOP/byte |
| GB200-NVL | 2500 | 5000 | 10000 | 192 GB | 8.00 TB/s | 1.8 TB/s | 1200 W | 312.5 FLOP/byte |

Ridge Point 列由 `GPUSpec.ridge_point_bf16()` 实跑算出（非手算，可用
`python -c "from common import GPUS; print(GPUS['H100'].ridge_point_bf16())"` 复验，需先
`cd learning/gpu-architecture/src`）。

**反直觉细节**：ridge point **不是单调随代数上升**。H200 相对 H100 反而**降到 206**——因为 H200
只加了 HBM 带宽（3.35→4.8 TB/s），BF16 算力完全不变（989→989），是"同代内存刷新"而非新一代算力
跃升。B200 相对 H100 也略降至 281.2：B200/H100 的算力涨幅是 2.275×，但 HBM 带宽涨幅是 2.388×，
带宽反而涨得比算力快一点点。"堆算力"不等于 ridge point 会右移——这比"越新的卡 ridge 越高"的
直觉更准确，也是本模块用代码算出来、而非只靠肉眼看表格能得出的结论。

---

## 关键公式（cheatsheet）

```
achievable_TFLOPS = min(peak_TFLOPS, HBM_TB/s × AI)         ← Roofline 核心不等式
AI = FLOPs / bytes_moved                                     ← operational / arithmetic intensity
ridge_point = peak_TFLOPS / HBM_TB/s                          ← compute-bound / memory-bound 分界线

GEMM(m,n,k):       flops = 2mnk            bytes = dtype_bytes·(mk + kn + mn)
Attention(b,h,s,d): flops = 4·b·h·s²·d      bytes = dtype_bytes·(3bhsd + bhs² + bhsd)
LayerNorm(n,hidden): flops = 8·n·hidden     bytes = 2·dtype_bytes·n·hidden

blocks_per_sm = min(max_threads/tpb, max_warps/wpb, max_regs/(regs·tpb), max_smem/smem, max_blocks)
occupancy = active_warps / max_warps_per_sm                   ← 五路瓶颈取最小

allreduce_time_ms = 2(N-1)/N × bytes / per_gpu_BW              ← Ring All-Reduce

decode_tokens_per_sec ≤ HBM_BW / (sizeof_weights + sizeof_KV_cache)   ← 推理 decode 天花板（L07）
```

---

## 自测 10 题

1. Roofline 公式 `min(peak, BW×AI)` 里两个分支各对应什么瓶颈？ridge point 的物理含义是什么？
2. 为什么 GEMV（m=1 的 GEMM）几乎总是 memory-bound，方阵大 GEMM 几乎总是 compute-bound？用 AI 公式推导（GEMV ai≈1 vs 8k³ GEMM ai≈2700）。
3. H100→H200 只提了 HBM 带宽（989→989 TFLOPS 不变，3.35→4.8 TB/s），对 ridge point 的影响是升还是降？这为什么依然是一次"有意义"的升级？
4. 5 层存储金字塔（register/SMEM/L1/L2/HBM）里，延迟和带宽各如何单调变化？为什么 kernel 优化第一原则是"让数据尽量留在上层"？
5. wgmma（H100）和 tcgen05（B200）的核心差异是什么？accumulator 挪到独立 tensor memory 对 occupancy 有什么帮助？
6. SM occupancy 的四个瓶颈维度是什么？为什么"occupancy=1.0"不等于"性能最优"（提示：Volkov "Better Performance at Lower Occupancy"）？
7. Ring All-Reduce 系数 `2(N-1)/N` 在 N 很大时趋近多少？GB200 NVL72（N=72）为什么仍比 8-GPU H100 集群快，尽管系数已接近上限？
8. LLM 推理 decode 阶段 token/s 上限公式是什么？为什么"量化"是这个阶段最大的杠杆，而不是升级算力？
9. Roofline 原论文（2009）用 *operational intensity* 而非泛泛的 *arithmetic intensity*，两者区别是什么？这个区别对今天分析 LLM kernel 为什么依然重要？
10. capstone 40 组结果里 25 组是 memory-bound。如果把所有 GEMM 换成 FP8（bytes 减半、FLOPs 不变），AI 会怎么变？这对 memory-bound op 的 achievable TFLOPS 有什么影响？

---

## Git 里程碑

| Tag | 内容 |
|-----|------|
| `gpu-arch-overview` | L01-02：4 代 GPU 对照 + 存储层次 |
| `gpu-arch-compute` | L03-05：Tensor Core + SM 占用率 + NVLink 拓扑 |
| `gpu-arch-roofline` | L06-07：Roofline 模型（现代 + 复现 2009 原论文）+ HBM 带宽 |
| `gpu-architecture` | L08 Capstone + 模块完结 |

---

## 运行验证（Runbook）

> 本段命令即 [`runbook.yaml`](runbook.yaml) 登记的"文档入口命令"，已在 ERIC-3080Ti（RTX 3080 Ti
> 16GB）上 V0+V1 验证通过（8/8，纯 CPU 秒级，无需 GPU）。
> 一键复验本模块：
> ```powershell
> python scripts/eric_3080ti_env_audit.py --runbook --modules gpu-architecture
> ```

8 个脚本全部**无 argparse**（纯 `_self_test()` 直跑，无可调预算/flag，跑一次就是全部）：

```powershell
python learning/gpu-architecture/src/common.py                    # [OK] ...(H100 ridge 295 FLOP/byte)
python learning/gpu-architecture/src/memory_hierarchy.py          # [OK] memory_hierarchy
python learning/gpu-architecture/src/tensor_core.py                # [OK] ...(FP4 8388608 FLOP/cycle)
python learning/gpu-architecture/src/sm_occupancy.py                # [OK] ...(good kernel occ 1.00)
python learning/gpu-architecture/src/nvlink_topology.py             # [OK] ...(H100 1GB allreduce 1.944ms)
python learning/gpu-architecture/src/roofline.py                    # [OK] ...(big GEMM util 100.0%)
python learning/gpu-architecture/src/roofline_original_minimal.py   # [OK] ...(X4 ridge 4.46)
```

**Capstone（L08）：10 LLM op × 4 代 GPU = 40 组 roofline 分析**

```powershell
python learning/gpu-architecture/src/capstone_roofline_zoo.py
# [OK] capstone_roofline_zoo: 25/40 mem-bound, H100 big GEMM 100.0% util
```

**关键坑注记**

- 全部 8 个脚本零第三方依赖（只 `dataclasses`/`__future__` + 互相 import），CPU 秒级 PASS 属正常——
  这些是纯数值/dataclass self-test（非训练 demo），不存在"假成功"风险：没有网络、没有权重、没有
  mock 捷径可走，每个 `[OK]` 后面的数字都是真算出来的（已逐个直跑核对，含 `capstone_roofline_zoo`
  的 `25/40 mem-bound` 与 `nvlink_topology` 的 `1.944ms` 均与 lecture 文档里的期望值一致）。
- 不需要设置 `PYTHONPATH`，也不依赖 CWD：Python 自动把脚本所在目录插进 `sys.path[0]`，从 repo 根
  直接 `python learning/gpu-architecture/src/xxx.py` 即可（已实测 unset `PYTHONPATH` 后依然可跑）。
- `memory_hierarchy.py` 曾有一处死代码：`_self_test()` 里写的是
  `recommend_tier(1024, reuse=100) if False else recommend_tier(1024, 100)`——三元表达式的
  `if False else` 恒走后半支，前半支用了错误关键字名 `reuse`（真实形参名是 `reuse_count`），
  **如果真被执行会 `TypeError`**，只是 Python 三元表达式的惰性求值让它从未真正跑到，所以模块本身
  一直能跑通。已清理为直接调用真实分支，行为/输出不变（本次唯一代码改动，见下）。

**测试（V2）**

```powershell
python learning/gpu-architecture/src/tests/test_all.py    # 预期：=== 8/8 passed ===
# 或经审计 harness：python scripts/eric_3080ti_env_audit.py --modules gpu-architecture --tests
```

> 注：`test_all.py` 是脚本式聚合器（只有 `def main()`，没有任何 `test_` 前缀函数），pytest 收集会
> 得到 `no tests ran`（rc=5）；audit harness 在这种情况下会**自动回退**成
> `python tests/test_all.py` 直接跑（已实测：直跑输出 `=== 8/8 passed ===`，是真实 assert 执行，
> 不是空心 PASS）。

---

## 跨专题衔接

| 下一专题 | 衔接点 |
|---------|-------|
| `cuda-essentials` | 从"看懂 roofline/occupancy 数字"进到"写一个真 CUDA kernel"（execution-model → vector-add → warp-primitives → shared-memory → coalescing → reduce-gemm → capstone-softmax） |
| `kernel-engineering` | Tensor Core MMA 形状 + occupancy 直觉，延伸到 Triton / CUTLASS / FlashAttention 的真实 kernel 工程 |
| `cluster-networking` | NVLink / Ring All-Reduce 从单机 8 卡扩展到 fat-tree / NCCL / SHARP 集群互联 |
| `storage-dataops` | HBM 是"卡内"天花板；这个专题接着看"卡外"存储层级（storage tiers / dataloader / sharding / checkpoint） |
| `training-orchestration` | 多卡 roofline / allreduce 数字最终要跑在 Slurm / gang-scheduling / Ray 编排出来的真实集群上 |
| `infra-graduation` | M8 毕业模块：mini-cluster 仿真 + topology 选型器，把本专题的 ridge point / allreduce 公式串成端到端系统设计 |

---

## 完成验收（自查）

- [ ] 8 篇 lecture 全过（01 总览 → 08 capstone）
- [ ] `paper/guide_01_roofline_model.md` 通读一遍，能回答文末「新手复习问题」10 题
- [ ] 8 个 `src/*.py` self-test 全部亲自跑过一遍
- [ ] 不看代码也能默写 `achievable = min(peak, BW×AI)` 和 `ridge = peak/BW`
- [ ] 能解释为什么 H200 的 ridge point 反而比 H100 低
- [ ] capstone：能说出 40 组里 memory-bound 有多少组、H100 大 GEMM util 是多少
- [ ] `python scripts/eric_3080ti_env_audit.py --runbook --modules gpu-architecture` 全绿（8/8）
- [ ] `python learning/gpu-architecture/src/tests/test_all.py` 显示 `8/8 passed`

---

🎓 **Module 8 第 1 专题完成 → 进入 `cuda-essentials`：从"看懂数字"到"写第一个 CUDA kernel"。**
