# CUDA Essentials — CUDA 执行模型与内存优化基础（Python 数值模拟）

> Module 8（系统与 Infra）7 专题系列的第 2 站。核心文档：NVIDIA Corporation — *CUDA C++ Programming
> Guide*, Release 13.3（官方编程模型手册，非传统论文）。7 篇 lecture + 9 个纯 CPU 可直跑 self-test +
> 1 篇 878 行中文论文导读。
>
> ⚠️ **本模块名为 "cuda-essentials"，但 `src/` 下的代码不是可编译的 `.cu` kernel，也不需要
> nvcc / CUDA Toolkit / 真实 GPU**——9 个脚本全部是 CUDA 执行模型（grid/block/warp/thread 映射、
> shared memory bank、全局内存 coalescing、reduce、tiled GEMM、online softmax）的**纯 Python 数值
> 模拟**，用整数/浮点运算复现同样的索引公式和数值行为。这是论文导读自己写明的设计选择（见
> `paper/guide_01_cuda_cpp_programming_guide.md` §20："本专题代码都是 Python mock，目标是帮助你
> 在没有真实 CUDA 编译环境时也能练机制"）。想写/编译/跑真正的 kernel，下一站 `kernel-engineering`
> 见。
>
> 论文导读：[`paper/guide_01_cuda_cpp_programming_guide.md`](paper/guide_01_cuda_cpp_programming_guide.md)（原文 PDF 同目录）

---

## 专题概览

| # | Lecture | 主题 | 核心公式 / idea | 对应代码 |
|---|---------|------|----------------|---------|
| 01 | execution-model | 三层并行执行模型 | `gid = blockIdx.x*blockDim.x + threadIdx.x`；warp 才是真正的执行单位（SIMT lock-step），256-thread block = 8 warp | [`common.py`](src/common.py) |
| 02 | vector-add | CUDA Hello World | ceil-div 防漏尾 `(n+bs-1)/bs` + 边界检查 `if(gid<n)` + cudaCheck；memory-bound（ai=1，util 100% 时 ≤0.5% peak FLOPS） | [`vector_add.py`](src/vector_add.py) |
| 03 | warp-primitives | Warp 级原语 | `__shfl_down_sync` 树规约 5 步（log2 32）完全不用 shared memory，是 Triton `tl.sum(...,axis=0)` 的底层实现 | [`warp_primitives.py`](src/warp_primitives.py) |
| 04 | shared-memory | Bank Conflict | 32 个 4B bank 循环映射 `(byte_offset/4)%32`；同 warp 撞同 bank 不同字 → N-way 冲突串行；padding / swizzle / warp-shuffle 三解法 | [`shared_memory.py`](src/shared_memory.py) |
| 05 | coalescing | 全局内存合并访问 | 128B sector 粒度；stride 1/2/8/32 → 效率 100%/50%/12.5%/3.1%；AoS vs SoA（LLM 张量天然 SoA） | [`coalescing.py`](src/coalescing.py) |
| 06 | reduce-gemm | Reduce 三代 + Tiled GEMM | naive 串行 → Brent-Kung 树（log2 步）→ warp-shuffle；tile=32 分块复用（Volkov 2008）→ HBM 流量减少 32× | [`reduce_kernel.py`](src/reduce_kernel.py) + [`gemm_tiled.py`](src/gemm_tiled.py) |
| 07 | capstone-softmax ⭐ | Online Softmax | 1-pass 递推 `m_new=max(m,x)`，`d_new=d·exp(m-m_new)+exp(x-m_new)`（Milakov & Gimelshein 2018）；FlashAttention 的核心机制 | [`capstone_softmax.py`](src/capstone_softmax.py) |
| — | *(paper 导读专属，无独立 lecture)* | 官方 Guide 自身例子复现 | thread linearization、block 调度 waves、CUDA streams 顺序、CUDA Graphs 摊薄 launch overhead、`occupancy=active_threads/max_threads_per_sm` | [`cuda_original_minimal.py`](src/cuda_original_minimal.py) |

**预计学时**：约 3.5–4 h（7 篇 lecture 精读 + 878 行论文导读通读 + 9 个脚本全部跑一遍并读源码）。

---

## 学习路径

```
        L01 执行模型（Grid → Block → Warp → Thread + 全局线程 id 公式）
                |
        L02 Vector-Add（CUDA Hello World：ceil-div + 边界检查）
                |
        L03 Warp 级原语（shuffle 树规约，替代 shared memory）
        L04 Shared Memory（bank conflict 模型）
                |
        L05 Coalescing（全局内存合并访问，128B sector）
                |
        L06 Reduce 三代 + Tiled GEMM ⭐ 核心优化案例（HBM 流量分析）
                |
        L07 Capstone：Online Softmax（FlashAttention 前置知识）

   番外（无独立 lecture，随论文导读学）：
   cuda_original_minimal.py — 官方 Guide 自身例子：streams / CUDA Graphs / occupancy / block 调度
```

---

## 目录结构

```
cuda-essentials/
├── README.md
├── runbook.yaml
├── paper/
│   ├── README.md                                   # 论文索引（1 篇）
│   ├── 01_cuda_cpp_programming_guide.pdf            # 官方原文 PDF（Release 13.3）
│   ├── guide_01_cuda_cpp_programming_guide.md       # 878 行中文导读
│   └── guide_01_cuda_cpp_programming_guide.pdf      # 导读渲染版 PDF
├── lectures/
│   └── 01..07-*.md                                  # 7 篇 lecture markdown
└── src/
    ├── common.py                    # Grid/Block/Thread dataclass + 1D launch_config
    ├── vector_add.py                # CUDA hello world kernel（依赖 common.py）
    ├── warp_primitives.py           # shfl_down_sync 树规约 + ballot_sync
    ├── shared_memory.py             # bank conflict 模型（32 bank）
    ├── coalescing.py                # 128B sector 合并访问模型
    ├── reduce_kernel.py             # reduce 三代（依赖 warp_primitives.py）
    ├── gemm_tiled.py                # naive vs tiled GEMM + HBM 流量对比
    ├── cuda_original_minimal.py     # 复现官方 Guide 自身例子（streams/graphs/occupancy）
    ├── capstone_softmax.py          # Capstone：naive vs online softmax
    └── tests/
        └── test_all.py              # 聚合 9 模块 _self_test()（脚本式，无 test_ 前缀函数）
```

**没有** `environment/`、`notebooks/`、`official/` 目录——同 `gpu-architecture`，本模块所有代码只依赖
标准库，无需外部权重/框架/GPU，见下节。

---

## 环境配置

本模块**没有** `environment/` 目录、也没有 `verify_env.py`——不是遗漏，是因为 9 个 `src/*.py`
全部只 `import dataclasses` / `import math` / `from __future__ import annotations`（+ 互相
import：`vector_add.py` 依赖 `common.py`；`reduce_kernel.py` 依赖 `warp_primitives.py`），**零第三方
包、零网络、零 GPU 依赖**（`grep -rnE "add_argument|ArgumentParser|^import |^from " src/*.py` 可自行
核实，命中的 import 只有 `dataclasses`/`math`/`__future__`/模块间互相引用）。

复用仓库根 `.venv`（Python 3.13）即可，不需要额外 `pip install`。跑起来本身就是"验证"：

```powershell
python learning/cuda-essentials/src/common.py
# [OK] cuda_essentials.common
```

不需要设置 `PYTHONPATH`，也不依赖当前工作目录：Python 解释器会自动把脚本所在目录插进
`sys.path[0]`，所以从 repo 根目录直接 `python learning/cuda-essentials/src/vector_add.py` 就能
解析 `from common import launch_config`（已实测：unset `PYTHONPATH` 后从 repo 根跑 `vector_add.py`
/ `reduce_kernel.py` 均正常，两者分别依赖 `common.py` / `warp_primitives.py` 的模块内 import）。

---

## 横向对比：5 种执行/内存优化技术（实测数据，本模块脚本直接产出）

| 技术 | 优化对象 | 关键机制 | 本模块实测结果 |
|---|---|---|---|
| Coalescing | 全局内存事务数 | 128B sector 粒度 | stride 1 → 100% 效率；stride 32 → **3.12%**（`efficiency` 实算，32× 劣化） |
| Bank-conflict-free 访问 | Shared memory 访问延迟 | 32 bank 循环映射 `(offset/4)%32` | stride 32 → 31-way conflict；stride 2 → 16-way conflict（`count_conflicts` 实算） |
| Warp shuffle reduce | Reduce 延迟 + SMEM 占用 | `__shfl_down_sync` 5 步树，0 SMEM | 与 naive 串行、Brent-Kung 树三法结果完全一致（sum 1024 = 523776） |
| Tiled GEMM | HBM 流量 | shared-memory-style 分块复用，tile=32 | **32.0×** HBM 流量削减（`hbm_traffic_naive/tiled` 实算，非手算常数） |
| Online softmax | HBM 往返次数 | 1-pass 递推 rescale，无需 materialize 中间矩阵 | 数值上与 3-pass naive **完全一致**（diff < 1e-9），大值（1e3+）下依然稳定 |

---

## 关键公式（cheatsheet）

```
gid = blockIdx.x * blockDim.x + threadIdx.x                        ← 全局线程 id（1D）
n_warps = ceil(n_threads / 32)                                      ← 一个 block 的 warp 数

bank   = (byte_offset // 4) % 32                                    ← shared memory bank 映射
n_sectors  = |{addr // 128 : addr in addresses}|                    ← coalescing 触及的 128B sector 数
efficiency = bytes_used / (n_sectors * 128)                         ← 合并访问效率，理想值 1.0

reduce(warp shuffle):  v[i] += shfl_down_sync(v[i], d)，d = 16,8,4,2,1   ← 5 步 = log2(32)

GEMM naive HBM 字节:  2 * M * N * K * dtype_bytes                    ← 每个 (i,j) 重读 K 个 A + K 个 B
GEMM tiled HBM 字节:  2 * ceil(M/T) * ceil(N/T) * ceil(K/T) * T² * dtype_bytes  ← tile=T 复用后

softmax online 递推:  m_new = max(m, x_i)
                       d_new = d * exp(m - m_new) + exp(x_i - m_new)  ← FlashAttention 核心机制

occupancy = active_threads / max_threads_per_sm                      ← 资源受限占用率估算（cuda_original_minimal）
```

---

## 自测 12 题

1. 全局线程 id 公式是什么？给定 `n=10000, blockDim=256`，最后一个 block 里有多少个线程会命中
   `if (gid < n)` 的边界检查、多少个不会？
2. warp 是多少个线程？为什么说它才是"真正的执行单位"而不是 block？warp divergence（if/else）会
   带来什么代价？
3. `__shfl_down_sync` 树规约需要几步完成 32 个 lane 的求和？为什么它能完全不用 shared memory？
   跨 warp 的规约还需要什么？
4. shared memory 的 32 个 bank 是怎么由字节地址映射的？stride 32 访问为什么会产生 31-way
   conflict？padding（`[32][33]`）为什么能解决？
5. 全局内存合并访问以多少字节为一个 sector？stride 1/2/8/32 各自的合并效率是多少（可用
   `coalescing.py` 验证）？
6. AoS 和 SoA 哪种内存布局对合并访问更友好？为什么 LLM 的 `[B,S,D]` 张量天然是 SoA？
7. Brent-Kung 树规约和 warp-shuffle 规约分别是多少步复杂度？为什么两者的最终结果必须与 naive
   串行加法完全一致（误差 < 1e-3）？
8. Tiled GEMM 用 shared-memory-style 复用数据后，tile=32 时 HBM 流量相对 naive 减少多少倍？这个
   倍数是手算出来的还是脚本实算出来的？
9. Online softmax 维护的两个不变量 `m_i`、`d_i` 分别是什么？它和 3-pass naive softmax 在数值上
   是什么关系？为什么它是 FlashAttention 能做 kernel fusion 的关键？
10. CUDA Graphs 解决了什么开销问题？`graph_submission_overhead_us` 里 80us 的 instantiate 开销
    为什么能被 `repeats` 摊薄到可忽略？
11. `occupancy_from_threads(768)` 为什么算出 0.75？如果 `threads_per_block` 改成 32 又会是多少、
    为什么（提示：`max_blocks_per_sm` 的硬上限先触发）？
12. 本模块的 9 个 `src/*.py` 都不需要真实 GPU 就能"验证"CUDA 执行模型——这种纯数值模拟方法能
    验证什么（索引公式、相对倍数关系、算法正确性），又不能验证什么（真实延迟、真实带宽、真实
    occupancy 上限）？

---

## Git 里程碑

| Tag | 内容 |
|-----|------|
| `cuda-ess-execution` | L01-02：执行模型 + Vector-Add |
| `cuda-ess-primitives` | L03-04：Warp 级原语 + Shared Memory |
| `cuda-ess-memory` | L05-06：Coalescing + Reduce/Tiled GEMM |
| `cuda-essentials` | L07 Capstone + 模块完结 |

---

## 运行验证（Runbook）

> 本段命令即 [`runbook.yaml`](runbook.yaml) 登记的"文档入口命令"，已在 ERIC-3080Ti（RTX 3080 Ti
> 16GB）上 V0+V1 验证通过（9/9，纯 CPU 秒级，无需 GPU）。
> 一键复验本模块：
> ```powershell
> python scripts/eric_3080ti_env_audit.py --runbook --modules cuda-essentials
> ```

9 个脚本全部**无 argparse**（纯 `_self_test()` 直跑，无可调预算/flag，跑一次就是全部）：

```powershell
python learning/cuda-essentials/src/common.py                  # [OK] cuda_essentials.common
python learning/cuda-essentials/src/vector_add.py               # [OK] vector_add
python learning/cuda-essentials/src/warp_primitives.py          # [OK] warp_primitives (sum=496, ballot popcount=16)
python learning/cuda-essentials/src/shared_memory.py            # [OK] shared_memory (stride2 conflicts 16)
python learning/cuda-essentials/src/coalescing.py                # [OK] coalescing (stride32 eff 0.0312)
python learning/cuda-essentials/src/reduce_kernel.py              # [OK] reduce_kernel (sum 1024 = 523776)
python learning/cuda-essentials/src/gemm_tiled.py                 # [OK] gemm_tiled (HBM traffic 32.0x reduction)
python learning/cuda-essentials/src/cuda_original_minimal.py      # [OK] cuda_original_minimal (launch, streams, graphs, occupancy)
```

**Capstone（L07）：数值稳定 softmax，naive 3-pass vs online 1-pass**

```powershell
python learning/cuda-essentials/src/capstone_softmax.py
# [OK] capstone_softmax (online == naive)
```

**关键坑注记**

- **本模块虽名为 "cuda-essentials"，但不需要真实 CUDA 编译环境**：9 个脚本零第三方依赖（只
  `dataclasses`/`math`/`__future__` + 互相 import），是 CUDA 执行模型/内存访问模式的纯 Python
  **数值模拟**（不是 `.cu`/Triton 编译产物）——这是论文导读 §20 自己写明的设计选择（"本专题代码
  都是 Python mock，目标是帮助你在没有真实 CUDA 编译环境时也能练机制"）。想写/编译真正的 kernel，
  下一站 `kernel-engineering` 见。
- 全部 9 个脚本 CPU 秒级 PASS 属正常——这些是纯数值/dataclass self-test（非训练 demo），不存在
  "假成功"风险：没有网络、没有权重、没有 mock 捷径可走，每个 `[OK]` 后面的数字都是真算出来的
  （已逐个直跑核对：`stride2 conflicts 16`、`stride32 eff 0.0312`、`HBM traffic 32.0x reduction`、
  `sum 1024 = 523776` 均与 lecture 文档里的期望值/公式一致）。
- `cuda_original_minimal.py` 是**唯一没有对应 lecture 的脚本**：它复现的是官方 Guide 自身的例子
  （thread linearization、block 调度 waves、CUDA streams 顺序、CUDA Graphs 摊薄 launch overhead、
  occupancy 估算），这些主题只出现在 878 行论文导读里（§ "20. 本仓库代码怎么对应 Guide" 逐一点名
  对应函数），7 篇 lecture 都没单独讲——已逐一 grep 确认（`stream|graph|occupancy` 在
  `lectures/*.md` 零命中）。
- 不需要设置 `PYTHONPATH`，也不依赖 CWD：Python 自动把脚本所在目录插进 `sys.path[0]`，从 repo 根
  直接 `python learning/cuda-essentials/src/xxx.py` 即可（已实测 unset `PYTHONPATH` 后
  `vector_add.py`/`reduce_kernel.py` 的模块内 import 依然正常解析）。

**测试（V2）**

```powershell
python learning/cuda-essentials/src/tests/test_all.py    # 预期：=== 9/9 passed ===
# 或经审计 harness：python scripts/eric_3080ti_env_audit.py --modules cuda-essentials --tests
```

> 注：`test_all.py` 是脚本式聚合器（只有 `def main()`，没有任何 `test_` 前缀函数），pytest 收集会
> 得到 `no tests ran`（rc=5）；audit harness 在这种情况下会**自动回退**成
> `python tests/test_all.py` 直接跑（已实测：直跑输出 `=== 9/9 passed ===`，是真实 assert 执行，
> 不是空心 PASS）。

---

## 跨专题衔接

| 专题 | 衔接点 |
|---------|-------|
| ← `gpu-architecture` | 上一站教会你"看懂/算出" roofline、occupancy、ridge point 等数字；本站把其中最基础的一类（occupancy、warp、shared memory、coalescing）落到"逐 thread/逐 byte 怎么映射"的执行模型细节 |
| → `kernel-engineering` | 从"用 Python 数值模型验证执行逻辑"进到接触真实 kernel 工程（Triton / CUTLASS / FlashAttention 源码级） |
| → `cluster-networking` | 本站的 warp/block/SM 是单卡内部并行；`cluster-networking` 把并行粒度升到多机多卡互联（在 `gpu-architecture` 的 NVLink 拓扑基础上展开） |
| → `storage-dataops` | 本站的 coalescing/HBM 流量分析是"卡内"访存优化；`storage-dataops` 接着看"卡外"存储层级（storage tiers / dataloader / sharding / checkpoint） |
| → `training-orchestration` | Tiled GEMM/reduce 这类 kernel 级优化直觉，最终要跑在 `training-orchestration` 编排出来的真实分布式训练任务里才有意义 |
| → `infra-graduation` | M8 毕业模块：把本站的执行模型直觉和其余 5 站串成端到端系统设计 |

---

## 完成验收（自查）

- [ ] 7 篇 lecture 全过（01 执行模型 → 07 capstone）
- [ ] `paper/guide_01_cuda_cpp_programming_guide.md` 通读一遍，能回答文末「新手复习问题」15 题
- [ ] 9 个 `src/*.py` self-test 全部亲自跑过一遍
- [ ] 不看代码也能默写 `gid = blockIdx.x*blockDim.x + threadIdx.x` 和 shared memory bank 映射
      `(offset/4) % 32`
- [ ] 能解释为什么 stride 32 的合并访问效率只有 3.1%，以及这和 bank conflict 是两个不同的模型
- [ ] 能说出 tiled GEMM 在 tile=32 时 HBM 流量比 naive 减少多少倍
- [ ] 能默写 online softmax 的两个递推不变量，并解释它为什么是 FlashAttention 的前置知识
- [ ] 能说清楚本模块的代码是"数值模拟"而不是可编译 CUDA kernel，以及这个设计选择的取舍
- [ ] `python scripts/eric_3080ti_env_audit.py --runbook --modules cuda-essentials` 全绿（9/9）
- [ ] `python learning/cuda-essentials/src/tests/test_all.py` 显示 `9/9 passed`

---

🎓 **Module 8 第 2 专题完成 → 进入 `kernel-engineering`：从"数值模拟"到真实 Triton/CUTLASS kernel 工程。**
