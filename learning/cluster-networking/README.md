# Cluster Networking — Fat-Tree / NCCL Collectives / SHARP 集群网络与通信原语

> Module 8（系统与 Infra）7 专题系列的第 4 站。核心论文：Hu et al. — *Demystifying NCCL: An In-depth
> Analysis of GPU Communication Protocols and Algorithms*（arXiv 2507.04786v3，分析对象 NCCL 2.19.1）。
> 6 篇 lecture + 7 个纯 CPU 可直跑 self-test + 1 篇 888 行中文论文导读。
>
> ⚠️ **模块名带 "cluster"/"nccl"，但和 `gpu-architecture`/`cuda-essentials`/`kernel-engineering` 一样，
> `src/` 下的 7 个脚本不是真实的分布式训练代码**（没有 `torch.distributed`、没有 socket/多进程/多机通信，
> grep 全仓核实零 `torch`/`socket`/`multiprocessing`/`subprocess` 命中），而是用可断言验证的纯 Python
> 时延/带宽解析模型去复现集群网络的关键决策：链路时延带宽建模、fat-tree/dragonfly 拓扑的 bisection 带宽、
> ring/tree/halving-doubling 三种 all-reduce 算法的时延公式、NCCL 五个核心 collective 的成本关系、SHARP
> 交换机内聚合的加速比，以及《Demystifying NCCL》论文里 protocol(Simple/LL/LL128) + channel + algorithm
> 的选择启发式。
>
> 论文导读：[`paper/guide_01_demystifying_nccl.md`](paper/guide_01_demystifying_nccl.md)（888 行，原文 PDF
> 同目录）

---

## 专题概览

| # | Lecture | 主题 | 核心公式 / idea | 对应代码 |
|---|---------|------|----------------|---------|
| 01 | fabric-overview | 三层网络(intra-node/intra-rack/inter-node) + 4 类互连协议选型 | NVLink5 900GB/s / IB NDR-XDR 400-800Gbps / RoCEv2 / Eth | [`common.py`](src/common.py)（`LINKS` 表 7 条链路） |
| 02 | allreduce ⭐ | Ring / Tree / Halving-doubling 三种 all-reduce 算法 | Ring `T=2(N-1)·(lat+size/N/BW)`；Tree `T=2⌈log2N⌉·(lat+size/2BW)`；HD 步数 O(logN) 且 BW-optimal | [`allreduce_algos.py`](src/allreduce_algos.py) |
| 03 | fat-tree | Fat-tree 3-tier 拓扑 + Dragonfly | bisection BW=(N/2)·BW/oversubscription；Dragonfly 平均 3 hop | [`fabric_topology.py`](src/fabric_topology.py) |
| 04 | nccl | NCCL 5 个核心 collective op | AllReduce ≈ ReduceScatter + AllGather（成本可加性，±5%内） | [`nccl_collectives.py`](src/nccl_collectives.py) |
| 05 | sharp | SHARP 交换机内聚合(in-network reduction) | `T_SHARP=2·(lat+(size/N)/BW)`，恒 2 步，与 N 无关 | [`sharp_inline.py`](src/sharp_inline.py) |
| 06 | capstone | Capstone：4 fabric × 4 集群规模选型 | 70B 模型 140GB BF16 梯度，逐场景挑 ring/HD/SHARP 中最快者 | [`capstone_cluster_sim.py`](src/capstone_cluster_sim.py) |
| — | （无对应 lecture，仅见论文导读 §16） | 复现《Demystifying NCCL》的 protocol/channel/algorithm 选择模型 | Simple/LL/LL128 三协议的 bandwidth_fraction×payload_fraction 复合有效带宽；channel 数按 FIFO 512KiB 填充率收缩；ring/tree 按消息大小+GPU 数切换 | [`nccl_original_minimal.py`](src/nccl_original_minimal.py)（212 行，本模块最长脚本） |

**预计学时**：约 3–3.5 h（6 篇 lecture 精读 + 888 行论文导读通读 + 7 个脚本全部跑一遍并读源码）。

---

## 学习路径

```
        L01 三层网络 + 4 类互连协议选型（common.py 的 7 条 LINKS）
                |
        L02 All-reduce 算法家族 ⭐（ring/tree/halving-doubling 时延公式）
                |
        L03 Fat-tree / Dragonfly 拓扑（bisection 带宽 + switch 数）
                |
        L04 NCCL 5 个核心 collective（AllReduce/ReduceScatter/AllGather/Broadcast/AllToAll）
                |
        L05 SHARP 交换机内聚合（恒 2 步，大 N 下对 ring 加速比精确收敛到 N-1×）
                |
        L06 Capstone：4 fabric × 4 集群规模，综合选型

   延伸阅读（论文导读 §16 专属，无对应 lecture）：
   nccl_original_minimal.py — 复现论文 protocol(Simple/LL/LL128) + channel + algorithm 选择模型
```

---

## 目录结构

```
cluster-networking/
├── README.md
├── runbook.yaml
├── paper/
│   ├── README.md                              # 论文索引（1 篇）
│   ├── 01_demystifying_nccl.pdf                # 原始论文 PDF
│   ├── guide_01_demystifying_nccl.md           # 888 行中文导读
│   └── guide_01_demystifying_nccl.pdf          # 导读渲染版 PDF
├── lectures/
│   └── 01..06-*.md                             # 6 篇 lecture markdown
└── src/
    ├── common.py                               # Link 数据类 + 7 种互连链路时延/带宽表
    ├── fabric_topology.py                      # Fat-tree(3-tier) + Dragonfly 拓扑
    ├── allreduce_algos.py                      # ring/tree/halving-doubling + NCCL 启发式算法选择器
    ├── nccl_collectives.py                     # all_gather/reduce_scatter/all_reduce/broadcast/all_to_all
    ├── sharp_inline.py                         # SHARP in-network reduction vs ring 加速比
    ├── nccl_original_minimal.py                # 论文 protocol/channel/algorithm 选择模型（212 行，最长）
    ├── capstone_cluster_sim.py                  # Capstone：4 fabric × 4 集群规模选型
    └── tests/
        └── test_all.py                         # 聚合 7 模块 _self_test()（脚本式，无 test_ 前缀函数）
```

**没有** `environment/` 目录（也没有 `verify_env.py`）——本模块 `src/` 下所有代码只依赖标准库，见下节。

---

## 环境配置

`src/` 下 7 个脚本全部只依赖标准库（`dataclasses` / `math` / `from __future__ import annotations`），
**零第三方包、零网络、零 GPU 依赖**。已用 `grep -nE "^import |^from " src/*.py` 核实：

```text
common.py / fabric_topology.py      → 只 dataclasses + __future__（不 import 同目录任何模块）
allreduce_algos.py                  → + from common import Link
nccl_collectives.py                 → + from common import Link，+ from allreduce_algos import ring_allreduce, tree_allreduce
sharp_inline.py                     → + from common import Link（speedup_vs_ring() 函数体内还有局部 import allreduce_algos）
nccl_original_minimal.py            → + dataclasses/math + from common import Link, LINKS
capstone_cluster_sim.py             → + from common import LINKS，+ from allreduce_algos import ring_allreduce, halving_doubling，+ from sharp_inline import sharp_allreduce
```

和 `gpu-architecture`/`cuda-essentials` 同款模式（不同于零 import 的 `kernel-engineering`）：5 个脚本会
`from common import ...` 等跨文件 import。**已实测**直接 `python learning/cluster-networking/src/X.py`
（不设 `PYTHONPATH`、从 repo 根目录跑）对全部 7 个脚本都可行——Python 会自动把脚本所在目录
（`learning/cluster-networking/src/`）插入 `sys.path[0]`，同目录下的 `common.py`/`allreduce_algos.py`/
`sharp_inline.py` 都能被正常 import，不需要额外配置。

复用仓库根 `.venv`（Python 3.13）即可，不需要额外 `pip install`：

```powershell
python learning/cluster-networking/src/allreduce_algos.py
# [OK] allreduce_algos (8-GPU 1GB ring 3896us)
```

---

## 横向对比：7 个脚本实测结果（本模块脚本直接产出，非手算）

| 脚本 | 建模对象 | 关键机制 | 本模块实测结果 |
|---|---|---|---|
| `common.py` | 链路时延/带宽 | `time_to_send = latency + size/BW` | 1GB 在 NVLink4 ≈2223us，IB NDR ≈20002us（IB 慢约 9 倍，符合断言 `t_ib > 8*t`） |
| `fabric_topology.py` | Fat-tree/Dragonfly 拓扑 | bisection BW、3-tier switch 数量、Dragonfly 平均 hop | 1024-node fat-tree、50GB/s 链路 → bisection 25600 GB/s；leaf switch 数=32 |
| `allreduce_algos.py` | Ring/Tree/HD 时延 + 算法选择 | 步数 ring=2(N-1)，tree=HD=2⌈log2N⌉；tree 不除 N，HD 除 N | 8-GPU 1GB NVLink4 ring ≈3896us；tiny(1KB,64GPU) 下 tree<ring；`pick_algorithm` 三分支全过 |
| `nccl_collectives.py` | 5 个 collective 成本 | AllReduce = ReduceScatter+AllGather（±5%内） | 8-GPU AllReduce(100MB,NVLink4)=396us，与 RS+AG 之和吻合；broadcast 随 N 严格递增 |
| `sharp_inline.py` | SHARP vs ring 加速比 | SHARP 恒 2 步 vs ring 2(N-1) 步 | 64-GPU 100MB IB-NDR：**63.0× 加速**；512-GPU：**511.0× 加速**（两个比值都精确等于 `n_gpus-1`） |
| `nccl_original_minimal.py` | NCCL protocol/channel/algorithm 选择 | 消息大小+intra/inter-node 决定 Simple/LL/LL128；FIFO 512KiB 决定 channel 数 | huge(512MiB)+RoCE+inter-node → `Simple` protocol + `ring` + 16 channels |
| `capstone_cluster_sim.py` | 70B 模型 140GB 梯度 all-reduce 选型 | 4 场景各跑 ring/HD(+SHARP if IB/RoCE)，挑最快者 | NVLink-8GPU 选 halving_doubling(0.23s)；IB-512GPU 选 sharp(0.01s，比 ring 快 511×，完整 4 行表见下方 Runbook） |

**独立发现的洞察**（本次验证实测复算，非文档抄录）：

1. **SHARP 加速比精确收敛到 `n_gpus-1`，与消息大小无关**——`sharp_inline.py` 自测用 64/512-GPU × 100MB，
   `capstone_cluster_sim.py` 用 64/512-GPU × 140GB，两个完全不同的消息量级下算出的加速比**都精确等于**
   63.0×/511.0×。原因：大消息下两种算法的时延都由带宽项主导、latency 项可以忽略，比值退化成纯步数比
   `ring_steps/sharp_steps = 2(N-1)/2 = N-1`，恰好与传输了多少字节无关。
2. **`nccl_original_minimal.py` 的 `ProtocolProfile` 有两个容易混淆的字段**：`bandwidth_fraction`（协议
   能利用链路原始带宽的比例）和 `payload_fraction`（每个传输单元里有效数据占比，由 `data_bytes_per_unit`/
   `flag_bytes_per_unit` 算出）。`protocol_transfer_time_us()` 里真正生效的有效带宽是**两者相乘**
   （`effective_bw = link_BW × bandwidth_fraction × payload_fraction`），不是单看 `bandwidth_fraction`。
   把 3 个协议的字段值代入算出的复合有效带宽：Simple=98%×100%=**98.0%**，LL=35%×50%=**17.5%**，
   LL128=95%×93.75%=**89.06%**——LL 的真实有效带宽（17.5%）比只看 `bandwidth_fraction=35%` 这一个数字
   直觉上更差，这是读代码时容易漏看的复合效应，见下方 cheatsheet。

---

## 关键公式（cheatsheet）

```
链路时延：
  T = latency_us + (bytes/1e9) / BW_gb_s × 1e6

Ring all-reduce（BW-optimal）：
  T = 2(N-1) × (latency + (size/N)/BW)        ← 步数随 N 线性增长

Tree all-reduce（latency-optimal）：
  T = 2⌈log2 N⌉ × (latency + size/(2·BW))     ← 步数 O(logN)，但 BW 项不除 N，大消息不划算

Halving-doubling / Rabenseifner（大 N + 大消息最优）：
  T = 2⌈log2 N⌉ × (latency + (size/N)/BW)     ← O(logN) 步 + BW 项除 N，两头都占

SHARP in-network reduction：
  T = 2 × (latency + (size/N)/BW)             ← 恒 2 步，与 N 无关；大消息下 vs ring 加速比 → N-1

Fat-tree bisection 带宽：
  BW_bisect = (N/2) × link_BW / oversubscription

NCCL collective 成本关系：
  AllReduce ≈ ReduceScatter + AllGather       （ZeRO-3 把两步拆开放不同 phase，通信量不变但可隐藏延迟）

NCCL protocol 复合有效带宽（《Demystifying NCCL》论文核心，nccl_original_minimal.py）：
  payload_fraction = data_bytes_per_unit / (data_bytes_per_unit + flag_bytes_per_unit)
  effective_bw     = link_BW × bandwidth_fraction × payload_fraction

  Simple: lat=6.0us, bandwidth_fraction=98%, payload_fraction=100%(0B flag)   → 复合有效带宽 98.0%，大消息最优
  LL:     lat=1.0us, bandwidth_fraction=35%, payload_fraction=50%(4B+4B)     → 复合有效带宽仅 17.5%，但延迟最低
  LL128:  lat=2.0us, bandwidth_fraction=95%, payload_fraction=93.75%(120B+8B) → 复合有效带宽 89.06%，intra-node 甜点
```

---

## 自测题

1. Ring / Tree / Halving-doubling 三种算法的步数公式不同（`2(N-1)` vs `2⌈log2N⌉` vs `2⌈log2N⌉`），BW 项
   是否除以 N 也不同——在"大 N + 大消息"场景下，为什么 halving-doubling 通常比 ring 和 tree 都好？
2. `sharp_inline.py` 里 SHARP 相对 ring 的加速比在 64-GPU 是 63.0×，512-GPU 是 511.0×——这两个数字和
   `n_gpus` 之间有什么精确关系？为什么与传输的字节数（100MB vs 140GB）无关？
3. `nccl_original_minimal.py` 的 `choose_protocol()` 为什么在 intra-node 且消息 <32MB 时选 LL128，但
   跨节点大消息一律退回 Simple？LL128 的硬件前提（128-byte atomic write）为什么在某些 PCIe 路径上无法
   保证？
4. `ProtocolProfile.bandwidth_fraction` 和 `payload_fraction` 是两个不同的字段——为什么 LL 的真实有效
   带宽（17.5%）比只看 `bandwidth_fraction=35%` 这一个数字直觉上更差？两者相乘的物理含义分别对应什么
   开销来源？
5. `choose_channels()` 为什么消息越小、channel 数越少？如果不做这个收缩，会发生什么（提示：想一想 FIFO
   512KiB 填不满意味着什么）？
6. `nccl_collectives.py` 里 `all_reduce == reduce_scatter + all_gather`（±5%内）这个断言在教你什么？
   ZeRO-3 为什么要把这两步拆开放在不同 phase？
7. `capstone_cluster_sim.py` 的 4 个场景里，PCIe-only 的 8-GPU 配置为什么被判定为"训练 70B 完全不可行"？
   如果把 `GRADIENT_BYTES` 换成一个 7B 模型的量级，PCIe 场景的结论会变吗？
8. Fat-tree 的 bisection 带宽公式里 `oversubscription` 参数是什么意思？2:1 oversubscription 为什么恰好
   让 bisection 带宽减半而不是任意比例？
9. `nccl_original_minimal.py` 是本模块唯一没有对应 lecture 的脚本，只在论文导读 §16 出现——它的
   `choose_algorithm()`（按消息大小+GPU数在 tree/ring 间选）和 `allreduce_algos.py` 的
   `pick_algorithm()`（按消息大小+GPU数在 tree/ring/halving_doubling 间选）除了都叫"算法选择"，
   在阈值和可选算法集合上有什么具体不同？
10. 本模块 7 个脚本全部零第三方依赖、CPU 秒级跑完——这种"纯解析时延模型"能验证 NCCL 的哪些性质
    （算法选择的相对趋势、协议 crossover 的方向），又不能验证哪些（真实 kernel launch overhead、真实
    GPUDirect RDMA 带宽、真实 topology-aware ring 构建算法、真实拥塞控制）？

---

## Git 里程碑

| Tag | 内容 |
|-----|------|
| `cluster-net-overview` | L01-02：fabric 总览 + all-reduce 算法家族 |
| `cluster-net-topology` | L03-04：fat-tree/dragonfly 拓扑 + NCCL collectives |
| `cluster-net-sharp` | L05：SHARP in-network reduction |
| `cluster-networking` | L06：Capstone 选型，模块完结 |

---

## 运行验证（Runbook）

> 本段命令即 [`runbook.yaml`](runbook.yaml) 登记的"文档入口命令"，已在 ERIC-3080Ti（RTX 3080 Ti 16GB）上
> V0+V1 验证通过（7/7，纯 CPU 秒级，无需 GPU；V0 全部 `v0: false` 因为脚本无 argparse，跳过 `--help`
> 探针）。
> 一键复验本模块：
> ```powershell
> python scripts/eric_3080ti_env_audit.py --runbook --modules cluster-networking
> ```

7 个脚本全部**无 argparse**（纯 `_self_test()` 直跑，无可调预算/flag，跑一次就是全部）：

```powershell
python learning/cluster-networking/src/common.py                  # [OK] cluster_networking.common (1GB NVLink 2223us, IB 20002us)
python learning/cluster-networking/src/fabric_topology.py         # [OK] fabric_topology (1024-node FT bisection 25600 GB/s)
python learning/cluster-networking/src/allreduce_algos.py         # [OK] allreduce_algos (8-GPU 1GB ring 3896us)
python learning/cluster-networking/src/nccl_collectives.py        # [OK] nccl_collectives (8-GPU AR=396us = RS+AG)
python learning/cluster-networking/src/sharp_inline.py            # [OK] sharp_inline (64-GPU speedup 63.0x, 512-GPU 511.0x)
python learning/cluster-networking/src/nccl_original_minimal.py   # [OK] nccl_original_minimal (Simple ring, 16 channels)
```

**Capstone（L06）：4 fabric × 4 集群规模选型**

```powershell
python learning/cluster-networking/src/capstone_cluster_sim.py
# [OK] capstone_cluster_sim (NVLink 8-GPU 0.23s, IB 512 SHARP 0.01s)
#
# Cluster | Link            | Best algo       | Best (s) | Ring (s)
# --------|-----------------|-----------------|----------|---------
#       8 | NVLink 4 (per GPU) | halving_doubling |     0.23 |    0.54
#       8 | PCIe Gen5 x16   | halving_doubling |     1.64 |    3.83
#      64 | IB NDR 400G     | sharp           |     0.09 |    5.51
#     512 | IB XDR 800G     | sharp           |     0.01 |     2.8
```

**关键坑注记**

- **`lectures/06-capstone.md` 和 `lectures/05-sharp.md` 原有文档漂移，本次验证已修正**：06 的"输出
  (近似)"表格第 3/4 行（64/512-GPU）的 `ring_time_s`（原写 27.50s/175.00s）和第 3 行 `sharp` 时间
  （原写 0.04s）与 `capstone_cluster_sim.py` 当前实际算出的数字（5.51s/2.8s、0.09s）不一致，"教学结论"
  里"175s→10ms=17500×加速"也随之不成立；05 的"64 GPU 100MB ring~30ms/SHARP~0.5ms→60×"同样与
  `sharp_inline.py` 的 `_self_test()` 实测（4.13ms/0.066ms→63×）不符。**已用实测 stdout 逐字替换**两处
  数字，并补充"两个比值都精确等于 `n_gpus-1`"的推导说明（非只改数字，也解释了为什么）。**未改动任何
  `src/*.py` 代码**——公式本身与断言、自测结果都自洽，判定为纯文档漂移（历史上公式或 LINKS 数值调整过，
  lecture 里的示例数字没有同步重新生成），不是代码 bug。
- 全部 7 个脚本零第三方依赖、CPU 秒级 PASS 属正常——这些是纯数值/dataclass self-test（非训练 demo），
  不存在"假成功"风险：没有网络、没有权重、没有 mock 捷径可走，每个 `[OK]` 后面的数字都是真算出来的。
- `capstone_cluster_sim.py` **真的调用**了 `allreduce_algos.ring_allreduce`/`halving_doubling` 和
  `sharp_inline.sharp_allreduce`（而不是硬编码结果）——4 个场景各自现算 2-3 个算法的时延，用 `min()`
  选最快者，逐行核对过公式代入后数字吻合，不是摆样子的"综合 capstone"。
- 6 篇 lecture 逐一核查过，**没有**发现其它 M8 模块出现过的"CWD 依赖的 `python -c "import sys;
  sys.path.insert(...)"` 一行流入口"写法；论文导读 §16 给出的两个 PowerShell 命令
  （`nccl_original_minimal.py` 和 `tests/test_all.py`）都是仓库根相对路径直跑，已核实等价可行。
- 不需要设置 `PYTHONPATH`，也不依赖 CWD：Python 自动把脚本所在目录插进 `sys.path[0]`；本模块 5 个脚本
  （除 common/fabric_topology 外）会互相 `from common import ...` 等跨文件 import（同 gpu-architecture/
  cuda-essentials 模式，比零 import 的 kernel-engineering 更接近真实工程代码结构）。

**测试（V2）**

```powershell
python learning/cluster-networking/src/tests/test_all.py    # 预期：=== 7/7 passed ===
# 或经审计 harness：python scripts/eric_3080ti_env_audit.py --modules cluster-networking --tests
```

> 注：`test_all.py` 是脚本式聚合器（只有 `def main()`，没有任何 `test_` 前缀函数），pytest 收集会得到
> `no tests ran`（rc=5）；audit harness 在这种情况下会**自动回退**成 `python tests/test_all.py` 直接跑
> （已实测：直跑输出 `=== 7/7 passed ===`，是真实 assert 执行，不是空心 PASS）。本轮验证**未修改任何
> `src/*.py` 代码**，只改了 2 篇 lecture 的示例数字，因此 V2 视为基线绿，未重跑完整 harness `--tests`。

---

## 跨专题衔接

| 专题 | 衔接点 |
|---------|-------|
| ← `kernel-engineering` | 上一站的 kernel 优化是单 GPU 内部的计算/访存问题；本站把优化粒度升到多机通信（接续 `gpu-architecture` 的 NVLink 拓扑，走向 fat-tree/NCCL/SHARP 集群互联） |
| → `storage-dataops` | 本站关心梯度/激活值怎么在 GPU 之间搬；`storage-dataops` 接着看训练数据/checkpoint 怎么在存储和 GPU 之间搬（storage tiers/dataloader/sharding/checkpoint） |
| → `training-orchestration` | 本站的 all-reduce/SHARP 选型最终要在 `training-orchestration` 编排的真实分布式训练任务里生效，才能体现出通信优化的 wall-clock 收益 |
| → `infra-graduation` | M8 毕业模块：把本站的集群网络直觉和其余 6 站串成端到端系统设计 |

---

## 完成验收（自查）

- [ ] 6 篇 lecture 全过（01 fabric 总览 → 06 capstone）
- [ ] `paper/guide_01_demystifying_nccl.md` 通读一遍，能回答文末「新手友好的复习问题」10 条
- [ ] 7 个 `src/*.py` self-test 全部亲自跑过一遍
- [ ] 不看代码也能默写 ring/tree/halving-doubling/SHARP 四种 all-reduce 的时延公式
- [ ] 能解释为什么 SHARP 相对 ring 的加速比精确等于 `n_gpus-1`、与消息大小无关
- [ ] 能说出 NCCL 三种 protocol(Simple/LL/LL128) 分别在什么消息大小/拓扑下胜出，以及
      `bandwidth_fraction`×`payload_fraction` 复合有效带宽是怎么算的
- [ ] `python scripts/eric_3080ti_env_audit.py --runbook --modules cluster-networking` 全绿（7/7）
- [ ] `python learning/cluster-networking/src/tests/test_all.py` 显示 `7/7 passed`

---

🎓 **Module 8 第 4 专题完成 → 进入 `storage-dataops`：从集群网络通信扩展到存储与数据管线。**
