# Kernel Engineering — Triton / CUTLASS / FlashAttention 三家 Kernel 工程实践

> Module 8（系统与 Infra）7 专题系列的第 3 站。核心论文：Tri Dao et al. — *FlashAttention: Fast and
> Memory-Efficient Exact Attention with IO-Awareness*（2022）。6 篇 lecture + 6 个纯 CPU 可直跑
> self-test + 1 篇 472 行中文论文导读 + 1 个官方 `flash-attention` 仓库只读 submodule + 1 个教学 notebook。
>
> ⚠️ **模块名带 "kernel"，但和 `gpu-architecture`/`cuda-essentials` 一样，`src/` 下的 6 个脚本不是可编译的
> Triton/CUDA/CUTLASS kernel**，而是用可断言验证的纯 Python 数值/决策逻辑去模拟这些 kernel 机制的行为
> （Triton autotune 打分、CUTLASS layout 偏移代数、FlashAttention online softmax、kernel fusion 的 HBM
> 流量核算）。不同于前两站的是：本模块额外带了一个**真实官方仓库只读 submodule**
> （[`official/repos/flash-attention`](official/repos/flash-attention)，Dao-AILab 官方实现，约 23MB 真实
> CUDA/C++ 源码，pinned `bc58abc`）供源码阅读对照——但编译/安装它已被 [`official/README.md`](official/README.md)
> 明确文档化为"Windows 工作站默认走 CPU-safe 复现，编译官方包留到以后的 WSL2/Linux CUDA 任务"，**不在本模块/
> 本轮 runbook 验证范围内，不要在共享 `.venv` 里装/编译 flash-attn**。
>
> 论文导读：[`paper/guide_01_flashattention.md`](paper/guide_01_flashattention.md)（原文 PDF 同目录）

---

## 专题概览

| # | Lecture | 主题 | 核心公式 / idea | 对应代码 |
|---|---------|------|----------------|---------|
| 01 | overview | Triton / CUTLASS / FlashInfer-FA3 三家选型 | Triton=Python DSL 易用+autotune；CUTLASS=C++ 模板对 Tensor Core 全 ISA 控制；FlashInfer/FA3=LLM 推理专用（paged KV/MQA/GQA） | — （纯概念选型表，无独立代码） |
| 02 | triton | Triton 实战：block-pointer + autotune | `@triton.autotune` 按 `(M,N,K)` key 缓存最优 `(BLOCK_M,BLOCK_N,BLOCK_K,num_warps,num_stages)` | [`triton_style.py`](src/triton_style.py) |
| 03 | cutlass | CUTLASS/CuTe Layout 代数 | `offset = Σ idx_d × stride_d`（线性/仿射）；真正的 swizzle 需要 XOR（非线性），无法用纯 stride 表示 | [`cutlass_layout.py`](src/cutlass_layout.py)（⚠️ `swizzle_32b()` 是**诚实标注的 stub**，见下「关键坑注记」） |
| 04 | flashattention ⭐ | FlashAttention 三代演进 + online softmax | `m_new=max(m,max(s_block))`；旧的 `l,o` 按 `exp(m_old-m_new)` 重新缩放后再累加新 block | [`flash_attention.py`](src/flash_attention.py) |
| 05 | fusion-patterns | 常见 Fusion 模式（pointwise/matmul）+ 反例 | 融合省 1 次 HBM 往返；但中间 buffer 撑爆 SMEM/阻碍并行调度时不该融合 | [`fused_mlp.py`](src/fused_mlp.py) + [`rmsnorm_kernel.py`](src/rmsnorm_kernel.py) |
| 06 | capstone | Capstone：Attention HBM 节省 | naive `O(N²)` vs flash `O(N·d)` HBM 流量；128k seq → 1025× 节省 | [`capstone_attn_speedup.py`](src/capstone_attn_speedup.py) |

**预计学时**：约 3.5–4 h（6 篇 lecture 精读 + 472 行论文导读通读 + 6 个脚本全部跑一遍并读源码 + 走一遍
`notebooks/01_flashattention_walkthrough.ipynb` + 大致浏览 `official/repos/flash-attention` 的官方源码定位）。

---

## 学习路径

```
        L01 三家概览（Triton vs CUTLASS vs FlashInfer/FA3 选型）
                |
        L02 Triton 实战（autotune 决策 + block-pointer idiom）
        L03 CUTLASS/CuTe Layout 代数（shape+stride 偏移 + swizzle 概念）
                |
        L04 FlashAttention 三代演进 ⭐ 核心（online softmax + IO-aware）
                |
        L05 常见 Fusion 模式（pointwise/matmul 融合 + 不该融合的反例）
                |
        L06 Capstone：Attention HBM 节省（5 个序列长度 naive vs flash 实测）

   延伸阅读（不参与 runbook 验证，量力而行）：
   official/repos/flash-attention/ — 官方真实 CUDA/C++ 源码，只读对照
   notebooks/01_flashattention_walkthrough.ipynb — naive vs flash 交互式对拍 + 官方源码存在性检查
```

---

## 目录结构

```
kernel-engineering/
├── README.md
├── runbook.yaml
├── paper/
│   ├── README.md                          # 论文索引（1 篇）
│   ├── 01_flashattention.pdf              # 原始论文 PDF（Dao et al. 2022）
│   ├── guide_01_flashattention.md         # 472 行中文导读
│   └── guide_01_flashattention.pdf        # 导读渲染版 PDF
├── lectures/
│   └── 01..06-*.md                        # 6 篇 lecture markdown
├── notebooks/
│   └── 01_flashattention_walkthrough.ipynb  # naive vs flash 对拍 + 官方源码路径存在性检查
├── official/
│   ├── README.md                          # 官方仓库定位说明（只读参考，不编译，见上方警示块）
│   └── repos/
│       └── flash-attention/               # Dao-AILab/flash-attention git submodule（pinned bc58abc，约 23MB 真实 CUDA/C++ 源码）
└── src/
    ├── triton_style.py                    # Triton 风格 autotune 决策（SMEM 预算估算 + 启发式打分）
    ├── cutlass_layout.py                  # CUTLASS/CuTe layout（row/col-major；swizzle_32b 为 stub）
    ├── flash_attention.py                 # FlashAttention-2 online softmax vs naive attention
    ├── fused_mlp.py                       # GeLU+matmul kernel fusion 的 HBM 流量核算
    ├── rmsnorm_kernel.py                  # RMSNorm + Linear fusion
    ├── capstone_attn_speedup.py           # Capstone：5 个序列长度 HBM 流量/speedup 曲线
    └── tests/
        └── test_all.py                    # 聚合 6 模块 _self_test()（脚本式，无 test_ 前缀函数）
```

**没有** `environment/` 目录（也没有 `verify_env.py`）——本模块 `src/` 下所有代码只依赖标准库，见下节。
`official/repos/flash-attention/` 内部结构不在此展开（它是完整的第三方 git 仓库，见其自身 `README.md`）。

---

## 环境配置

`src/` 下 6 个脚本**全部只 `import dataclasses` / `import math` / `from __future__ import annotations`**，
**零第三方包、零网络、零 GPU 依赖、也互相零 import**（与 `gpu-architecture`/`cuda-essentials` 不同——那两个
模块内部脚本会互相 `from common import ...`，本模块的 6 个脚本彼此完全独立，任意顺序单独跑都行）。已用
`grep -rnE "add_argument|ArgumentParser|^import |^from " src/*.py` 核实，命中的 import 只有
`dataclasses`/`math`/`__future__`：

```text
src/capstone_attn_speedup.py:2:from __future__ import annotations
src/cutlass_layout.py:2:from __future__ import annotations
src/cutlass_layout.py:3:from dataclasses import dataclass
src/flash_attention.py:2:from __future__ import annotations
src/flash_attention.py:3:import math
src/fused_mlp.py:2:from __future__ import annotations
src/fused_mlp.py:3:import math
src/rmsnorm_kernel.py:2:from __future__ import annotations
src/rmsnorm_kernel.py:3:import math
src/triton_style.py:2:from __future__ import annotations
src/triton_style.py:3:from dataclasses import dataclass, field
```

复用仓库根 `.venv`（Python 3.13）即可，不需要额外 `pip install`，也不需要设置 `PYTHONPATH`：

```powershell
python learning/kernel-engineering/src/flash_attention.py
# [OK] flash_attention (online == naive)
```

**两个例外，均不在本轮 runbook 验证范围内（V0/V1/V2 只覆盖 `src/*.py` 六个脚本）**：

1. **`official/repos/flash-attention/`** 是真实的 Dao-AILab 官方 git submodule（`git submodule status`
   实测 pinned `bc58abc67bd...`，与 `official/README.md` 记录的 `bc58abc` 一致）。这是**只读源码阅读参考**，
   `official/README.md` 已列出 5 个建议先读的官方文件（`README.md`/`flash_attn_interface.py`/
   `flash_api.cpp`/`hopper/softmax.h`/`hopper/tile_size.h`，已逐一核实 5 个路径均存在）并映射到本地教学
   文件。**不要在共享 `.venv` 里安装/编译 `flash-attn`**——它需要 Linux CUDA 工具链，本 Windows 工作站的
   默认路径是源码阅读 + 本地 CPU-safe 机制复现（即 `src/flash_attention.py`），真正编译留给以后的
   WSL2/Linux CUDA 任务。
2. **`notebooks/01_flashattention_walkthrough.ipynb`** 是本模块唯一 `import torch` 的地方（仅用来
   `print(torch.__version__)` / `torch.cuda.is_available()`，本机实测 `2.11.0+cu128` / `True`；其余全部
   计算仍是纯 Python）。notebook 端到端执行属于 spec 里明确划为**后续单列 pass 的 V3**（全仓 410 个
   notebook 统一处理），本轮不执行；但已静态核实它 import 的
   `learning/kernel-engineering/src/flash_attention.py` 存在、它断言存在性的 3 个官方源码路径
   （`flash_attn/flash_attn_interface.py`/`csrc/flash_attn/flash_api.cpp`/`hopper/softmax.h`）也确实存在。

---

## 横向对比：6 个脚本实测结果（本模块脚本直接产出，非手算）

| 脚本 | 优化对象 | 关键机制 | 本模块实测结果 |
|---|---|---|---|
| `triton_style.py` | Tile 配置选择 | SMEM 预算约束 + `reuse×num_stages − parallelism_penalty` 打分 | 4096³ 大 GEMM 选中 **128×256** tile；`smem_limit_kb=1` 时无合法配置正确抛 `ValueError` |
| `cutlass_layout.py` | 内存访问模式 | `(shape,stride)` 线性偏移代数 | row-major/col-major 的 coalescing 轴（`axis=1`/`axis=0`）已验证方向相反；`swizzle_32b` 现为**诚实标注的 stub**（见下） |
| `flash_attention.py` | Attention HBM 往返 | 分块递推 `m/l/o`，不 materialize `N×N` | naive vs flash 输出**数值一致**（`block_n=3` 不整除 `N=8` 的边界情况下 max diff < 1e-9） |
| `fused_mlp.py` | HBM 往返次数 | 激活值留寄存器，不写中间 `h` | `N=2048,D=4096,H=16384` 配置下**省 30.8% HBM 流量**；unfused/fused 数值一致 |
| `rmsnorm_kernel.py` | HBM 往返次数 | norm 统计量 + matmul 同 kernel 内完成 | unfused vs fused 输出数值一致（diff < 1e-9）；单行结果与手算 RMS 对拍一致 |
| `capstone_attn_speedup.py` | Attention 总 HBM 流量 | naive `O(N²)` vs flash `O(N·d)` scaling | 512→131072 序列长度，speedup **5.0× → 1025.0×** 单调上升（详见下方 Runbook） |

---

## 关键公式（cheatsheet）

```
Triton autotune:
  smem_bytes(cfg) = dtype_bytes × num_stages × (block_m×block_k + block_k×block_n)   ← SMEM 预算
  score(cfg) = (block_m×block_n)/(block_m+block_n) × num_stages − parallelism_penalty ← 选 tile 启发式

CuTe Layout：
  offset(idx) = Σ idx_d × stride_d              ← 线性/仿射偏移，row_major=(cols,1)，col_major=(1,rows)
  swizzle 的真实公式（未在本模块实现，见 stub 说明）：
  offset = row×cols + (col XOR (row×cols mod 32))   ← 非线性，突破了 (shape,stride) 能表示的范围

FlashAttention-2 online softmax（每来一个新 K/V block）：
  m_new = max(m, max(s_block))
  rescale = exp(m − m_new)
  l_new = l×rescale + Σ exp(s_block − m_new)
  o_new = o×rescale + Σ exp(s_block − m_new) × V_block
  最终 O = o_final / l_final                    ← 与标准 softmax(QK^T/√d)V 数学等价（exact，非近似）

HBM 流量（capstone，dtype_bytes=2）：
  naive_bytes  = dtype_bytes × (3Nd + 4N² + Nd)   ← S/P 各计"写入+读回" 2 次往返
  flash_bytes  = dtype_bytes × (3Nd + Nd) = 4×dtype_bytes×N×d
  speedup = naive_bytes / flash_bytes             ← N 越大增长越快（N² vs N 项）

Kernel fusion 流量：
  fused   = weights + io
  unfused = weights + io + 2×dtype_bytes×N×H      ← 多出的一项 = 中间激活 h 的写入+读回

RMSNorm：
  y = x / sqrt(mean(x²) + eps) × weight
```

---

## 自测 10 题

1. Triton `autotune()` 的打分函数 `score(c) = reuse×num_stages − parallelism_penalty` 里，`reuse` 奖励
   什么、`parallelism_penalty` 惩罚什么？为什么 4096³ 大 GEMM 会选中大 tile（128×256）？
2. `TritonConfig.smem_bytes()` 为什么要乘 `num_stages`？如果 SMEM 预算收紧到 1KB，`autotune()` 会怎样
   （提示：看 `_self_test()` 里 `huge` 配置那段）？
3. CuTe `Layout.offset(idx) = Σ idx_d × stride_d` 是一个线性/仿射函数。为什么真正的 bank-conflict-free
   swizzle（`col XOR (row×cols mod 32)`）无法用这个 `(shape, stride)` 表示？
4. `cutlass_layout.py` 的 `swizzle_32b()` 目前返回的 `Layout` 和 `row_major()` 完全相同——这是 bug 还是
   诚实标注的 stub？你怎么从代码（docstring + `_self_test()` 的断言）判断出这一点，而不是被函数名和
   docstring 的开头两行误导？
5. FlashAttention 的 online softmax 递推里，为什么每来一个新 block 后，旧的 `l`（分母）和 `o`（分子）都
   要先乘 `exp(m_old − m_new)` 再累加新 block 的贡献？
6. `flash_attention.py` 的 `_self_test()` 特意用 `block_n=3` 处理 `N=8`（3 不整除 8）。这在测什么边界
   情况？如果只用能整除的 `block_n`（比如 4）测试，可能漏掉什么潜在 bug？
7. `fused_mlp.py` 里 `hbm_traffic(fused=True)` 比 `fused=False` 少算了哪一项？为什么"融合"能省掉这一项，
   而权重 `W1`/`W2` 的读取流量两种情况下不变？
8. `fused_rmsnorm_linear` 和"先调用 `rmsnorm_batch` 再做矩阵乘"两种写法，数学上为什么结果必须**完全
   一致**（而不是近似）？这说明 kernel fusion 一般改变的是什么，不改变的是什么？
9. Capstone 的 5 个序列长度里，speedup 从 512 的 5.0× 涨到 131072 的 1025.0×。这个增长率背后的渐近关系
   是什么（`naive` 流量里哪一项是 `O(N²)`，`flash` 流量里哪一项是 `O(N)`）？
10. 本模块 6 个脚本全部零第三方依赖、CPU 秒级跑完——这种"纯 Python 数值模拟"能验证 Triton/CUTLASS/
    FlashAttention 的哪些性质（算法正确性、相对流量比例、决策逻辑），又不能验证哪些（真实 wall-clock、
    真实 SM occupancy、真实 shared-memory bank conflict 延迟）？`official/repos/flash-attention/` 的存在
    如何补上这块拼图？

---

## Git 里程碑

| Tag | 内容 |
|-----|------|
| `kernel-eng-overview` | L01-02：三家选型概览 + Triton autotune 实战 |
| `kernel-eng-cutlass` | L03：CUTLASS/CuTe layout 代数 |
| `kernel-eng-flashattn` | L04：FlashAttention 三代演进（核心） |
| `kernel-engineering` | L05-06：Fusion 模式 + Capstone HBM 节省，模块完结 |

---

## 运行验证（Runbook）

> 本段命令即 [`runbook.yaml`](runbook.yaml) 登记的"文档入口命令"，已在 ERIC-3080Ti（RTX 3080 Ti 16GB）上
> V0+V1 验证通过（6/6，纯 CPU 秒级，无需 GPU；V0 全部 `v0: false` 因为脚本无 argparse，跳过 `--help`
> 探针）。
> 一键复验本模块：
> ```powershell
> python scripts/eric_3080ti_env_audit.py --runbook --modules kernel-engineering
> ```

6 个脚本全部**无 argparse**（纯 `_self_test()` 直跑，无可调预算/flag，跑一次就是全部）：

```powershell
python learning/kernel-engineering/src/triton_style.py           # [OK] triton_style (big GEMM picked 128x256)
python learning/kernel-engineering/src/cutlass_layout.py         # [OK] cutlass_layout (swizzle_32b is a stub — see docstring)
python learning/kernel-engineering/src/flash_attention.py        # [OK] flash_attention (online == naive)
python learning/kernel-engineering/src/fused_mlp.py              # [OK] fused_mlp (saved 30.8% HBM)
python learning/kernel-engineering/src/rmsnorm_kernel.py         # [OK] rmsnorm_kernel
```

**Capstone（L06）：5 个序列长度 naive vs flash HBM 流量/speedup**

```powershell
python learning/kernel-engineering/src/capstone_attn_speedup.py
# [OK] capstone_attn_speedup (128k seq: 1025.0x HBM saved)
#
# Seq len | Naive MB | Flash MB | Speedup
# --------|----------|----------|--------
#     512 |      2.6 |      0.5 |    5.0x
#    2048 |     35.7 |      2.1 |   17.0x
#    8192 |    545.3 |      8.4 |   65.0x
#   32768 |   8623.5 |     33.6 |  257.0x
#  131072 | 137573.2 |    134.2 | 1025.0x
```

**关键坑注记**

- **`cutlass_layout.py` 的 `swizzle_32b()` 是诚实标注的 stub，不是 bug 但曾经是"文档说了、代码没做到"的
  gap**：docstring 写了 "XOR-based swizzle to break bank conflicts" 和公式说明，但函数体在本次验证前
  **直接返回和 `row_major()` 完全相同的 `Layout`**，没有任何 XOR 变换，`_self_test()` 也从未调用/断言过它
  ——这是本次验证独立核实出的真实 gap（非用户误判）。**处理方式**：没有去实现真正的 XOR swizzle 算法
  （超出"验证文档命令能否跑通"的范畴，且 `Layout` 当前的 `(shape,stride)` 线性偏移表示本身就装不下 XOR
  这种非仿射变换，需要更大的架构改动）；而是把 docstring 改成显式的 STUB 说明 + 解释为什么这个表示装
  不下真 swizzle，并在 `_self_test()` 里新增一条断言**显式锁定当前（未 swizzle）行为**，让这个 gap
  从"静默未测试"变成"诚实且可测试"。真正实现留作后续工作。
- `06-capstone.md` 的"结果"表格**原文档漂移**：文档里写的 `naive_mb`/`speedup` 数字（如 512 序列长度写
  `1.6MB/3.2x`）与 `capstone_attn_speedup.py` 当前 `hbm_naive_attn()` 实际算出的数字（`2.6MB/5.0x`）不
  一致——当前代码把 `S`/`P` 矩阵各计"写入 HBM + 读回"2 次往返（更贴合 paper guide §3 描述的 9 步 HBM
  路径），推测是公式在早期迭代过、lecture 表格没有同步重新生成。已用**实测 stdout 逐字替换**该表格；
  `04-flashattention.md` 引用的"1025×"headline 数字和"68 GB S+P footprint"这两个独立说法本身仍然
  准确（已核实），未受影响。
- 全部 6 个脚本零第三方依赖、CPU 秒级 PASS 属正常——这些是纯数值/dataclass self-test（非训练 demo），
  不存在"假成功"风险：没有网络、没有权重、没有 mock 捷径可走，每个 `[OK]` 后面的数字都是真算出来的。
- 不需要设置 `PYTHONPATH`，也不依赖 CWD：Python 自动把脚本所在目录插进 `sys.path[0]`；且本模块 6 个
  脚本彼此零 import（比 gpu-architecture/cuda-essentials 更简单，那两个模块内部脚本会互相
  `from common import ...`）。
- `official/repos/flash-attention/` submodule **不参与本轮 V0/V1/V2**，也**不要尝试编译/安装**它——
  Windows 工作站的默认学习路径是读源码 + 跑本地 CPU-safe 复现（`src/flash_attention.py`）。
- `notebooks/01_flashattention_walkthrough.ipynb` 同理不参与本轮验证（V3，留作后续 410-notebook 统一
  pass），已静态核实其 import 的本地脚本路径和断言存在性的 3 个官方源码路径均真实存在。

**测试（V2）**

```powershell
python learning/kernel-engineering/src/tests/test_all.py    # 预期：=== 6/6 passed ===
# 或经审计 harness：python scripts/eric_3080ti_env_audit.py --modules kernel-engineering --tests
```

> 注：`test_all.py` 是脚本式聚合器（只有 `def main()`，没有任何 `test_` 前缀函数），pytest 收集会得到
> `no tests ran`（rc=5）；audit harness 在这种情况下会**自动回退**成 `python tests/test_all.py` 直接跑
> （已实测：直跑输出 `=== 6/6 passed ===`，是真实 assert 执行，不是空心 PASS；输出里能看到
> `cutlass_layout (swizzle_32b is a stub — see docstring)` 字样，即上面那条坑注记落到了可执行断言上）。

---

## 跨专题衔接

| 专题 | 衔接点 |
|---------|-------|
| ← `cuda-essentials` | 上一站用纯 Python 数值模拟练执行模型/内存访问（execution model、bank conflict、coalescing、reduce、tiled GEMM、online softmax 的雏形）；本站把 online softmax 落到完整 attention 算法（FlashAttention），并引入 Triton/CUTLASS 两个真实生产级 kernel 框架的设计语言 |
| → `cluster-networking` | 本站的 kernel 优化是单 GPU 内部的计算/访存问题；`cluster-networking` 把优化粒度升到多机通信（接续 `gpu-architecture` 的 NVLink 拓扑，走向 fat-tree / NCCL / SHARP 集群互联） |
| → `storage-dataops` | 本站 fusion/HBM 流量分析是"计算 kernel 内部"的优化；`storage-dataops` 接着看"kernel 外部"的存储链路（storage tiers / dataloader / sharding / checkpoint） |
| → `training-orchestration` | Triton/CUTLASS 手写的 kernel 最终要跑在 `training-orchestration` 编排出来的真实分布式训练任务里，才能体现出 wall-clock 收益 |
| → `infra-graduation` | M8 毕业模块：把本站的 kernel fusion / HBM 流量直觉和其余 5 站串成端到端系统设计 |

---

## 完成验收（自查）

- [ ] 6 篇 lecture 全过（01 三家概览 → 06 capstone）
- [ ] `paper/guide_01_flashattention.md` 通读一遍，能回答文末「AI 学习者最容易卡住的点」5 条 + 「闭卷掌握
      检查」12 题
- [ ] 6 个 `src/*.py` self-test 全部亲自跑过一遍
- [ ] 不看代码也能默写 online softmax 的 `m_new/rescale/l_new/o_new` 四步递推
- [ ] 能解释为什么 `swizzle_32b()` 当前是 stub、真正的 XOR swizzle 为什么装不进 `(shape,stride)` 线性
      表示
- [ ] 能说出 capstone 5 个序列长度里 speedup 从多少涨到多少，以及背后 `O(N²)` vs `O(N)` 的渐近关系
- [ ] 大致浏览过 `official/repos/flash-attention/` 里 `official/README.md` 列出的 5 个官方文件，知道
      Python 接口和 CUDA kernel 分别放在哪
- [ ] （可选）跑一遍 `notebooks/01_flashattention_walkthrough.ipynb`，感受交互式 naive vs flash 对拍
- [ ] `python scripts/eric_3080ti_env_audit.py --runbook --modules kernel-engineering` 全绿（6/6）
- [ ] `python learning/kernel-engineering/src/tests/test_all.py` 显示 `6/6 passed`

---

🎓 **Module 8 第 3 专题完成 → 进入 `cluster-networking`：从单 GPU kernel 工程扩展到多机集群互联。**
