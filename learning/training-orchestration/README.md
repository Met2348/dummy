# Training Orchestration — Slurm / Gang Scheduling / 故障容忍 / Ray Actor / Elastic Training 训练编排

> Module 8（系统与 Infra）7 专题系列的第 6 站。核心论文：Moritz et al. — *Ray: A Distributed Framework
> for Emerging AI Applications*（arXiv:1712.05889，OSDI 2018）。
> 6 篇 lecture + 8 个纯 CPU 可直跑 self-test + 1 篇 685 行中文论文导读。
>
> ⚠️ **`04-ray.md` 和两个 `ray_*.py` 文件名带 "ray"，但都不依赖真实 Ray 框架**——已用 grep 核实
> `ray_actors.py`/`ray_original_minimal.py` 的 import 语句里只有 `dataclasses`/`__future__`，零
> `import ray`；`.venv` 里也确认未装 ray（`import ray` 报 `ModuleNotFoundError`）。这两个脚本用纯
> Python dataclass **手写模拟** Ray 论文里的机制，而不是调用真实 Ray API，见下方「环境配置」段的详细
> 拆解（`ray_actors.py` 模拟 actor 编程模型本身；`ray_original_minimal.py` 模拟 GCS+调度器+对象存储+
> lineage 容错这层更深的系统架构；两者都**不**实现论文/lecture 提到的 Placement Group，grep 全模块核实
> 零 `Placement`/`STRICT_PACK` 命中）。
>
> 论文导读：[`paper/guide_01_ray.md`](paper/guide_01_ray.md)（685 行，原文 PDF 同目录）

---

## 专题概览

| # | Lecture | 主题 | 核心公式 / idea | 对应代码 |
|---|---------|------|----------------|---------|
| 01 | slurm | FIFO + priority + fairshare 调度，backfill 让小 job 插队空窗，reservation 预留时间窗 | `fairshare=(used_recently/fair_share_limit)^-1` | [`common.py`](src/common.py)（Job/Node/JobState）+ [`slurm_scheduler.py`](src/slurm_scheduler.py) |
| 02 | gang-scheduling | 分布式训练所有 rank 必须同时启动，否则半数等 timeout 或 ckpt 不一致 | 原子 all-or-nothing 分配；`n_gpus > 集群总量` → 永久 starvation | [`gang_scheduling.py`](src/gang_scheduling.py) |
| 03 | fault-tolerance | 集群 MTBF 随规模变小；Young's Formula(1974) 求最优 ckpt 间隔 | `1/MTBF_cluster=N_gpu/MTBF_gpu+N_fabric/MTBF_fabric`；`T_opt=√(2·C·M)`；`wasted%=C/T+T/(2M)` | [`fault_tolerance.py`](src/fault_tolerance.py) |
| 04 | ray ⭐ | Actor(有状态)/Task(无状态)/Placement Group 三大概念；Ray 论文 GCS + bottom-up scheduler + lineage 容错 | RLHF actor/critic/RM/ref 布局；`score(node)=queue_wait+remote_input/BW` | [`ray_actors.py`](src/ray_actors.py) + [`ray_original_minimal.py`](src/ray_original_minimal.py)（⚠️均为纯 dataclass 模拟，见上方澄清） |
| 05 | elastic | torchrun rendezvous：节点动态 join/leave，`--nnodes=min:max` | quorum 判定；membership 变化触发 `generation+1` | [`elastic_training.py`](src/elastic_training.py) |
| 06 | capstone | 64 节点×8GPU=512GPU 集群单点调度快照：FIFO+backfill 排 8 个 job + 同规模 Young's formula ckpt 开销 | util=已用GPU/总GPU；T_opt 随集群变大而变短 | [`capstone_cluster_run.py`](src/capstone_cluster_run.py) |

**唯一带论文的一站**：本模块只有 L04 Ray 配了完整学术论文导读（其余 5 篇 lecture 是工程实践笔记，无对应
论文），因此标 ⭐。

**预计学时**：约 2.5–3 h（6 篇 lecture 精读约 217 行 + 685 行论文导读通读 + 8 个脚本全部跑一遍并读源码）。

---

## 学习路径

```
        L01 Slurm FIFO/priority/fairshare + backfill（common.py 的 Job/Node + slurm_scheduler.py）
                |
        L02 Gang Scheduling（分布式训练 all-or-nothing 原子分配 + starvation 监控）
                |
        L03 故障容忍与 ckpt 频率（MTBF 可加性 + Young's formula 最优 ckpt 间隔）
                |
        L04 Ray & Actor 模型 ⭐（唯一带论文的一站：actor 编程模型 + GCS/调度器/lineage 系统架构）
                |
        L05 Elastic Training（torchrun rendezvous：节点动态 join/leave + quorum/generation）
                |
        L06 Capstone：64节点×512GPU 集群单点调度快照，综合 FIFO+backfill 调度 + Young's formula 容错开销
```

---

## 目录结构

```
training-orchestration/
├── README.md
├── runbook.yaml
├── paper/
│   ├── README.md                      # 论文索引（1 篇）
│   ├── 01_ray.pdf                     # 原始论文 PDF
│   ├── guide_01_ray.md                # 685 行中文导读
│   └── guide_01_ray.pdf               # 导读渲染版 PDF
├── lectures/
│   └── 01..06-*.md                    # 6 篇 lecture markdown
└── src/
    ├── common.py                      # Job/Node/JobState 数据类 + make_cluster()
    ├── slurm_scheduler.py             # try_assign/release/fifo_with_backfill（真·backfill，见下方说明）
    ├── gang_scheduling.py             # gang_assign(原子分配) + starvation_check
    ├── fault_tolerance.py             # FaultModel(MTBF 可加性) + optimal_ckpt_interval(Young's formula)
    ├── ray_actors.py                  # Ray actor 编程模型模拟（TrainerActor/ParameterServer）
    ├── ray_original_minimal.py        # Ray 论文系统架构模拟（GCS/调度器/对象存储/lineage，253行，本模块最长）
    ├── elastic_training.py            # torchrun 风格 rendezvous（join/leave/quorum/generation）
    ├── capstone_cluster_run.py        # Capstone：综合调度器 + 故障容忍模型
    └── tests/
        └── test_all.py                # 聚合 8 模块 _self_test()（脚本式，无 test_ 前缀函数）
```

**没有** `environment/` 目录（也没有 `verify_env.py`）——本模块 `src/` 下所有代码只依赖标准库，见下节。

---

## 环境配置

`src/` 下 8 个脚本全部只依赖标准库（`dataclasses` / `enum` / `math` / `from __future__ import
annotations`），**零第三方包、零网络、零 GPU 依赖**。已用 `grep -nE "^import |^from " src/*.py` 核实：

```text
common.py / fault_tolerance.py / ray_actors.py / ray_original_minimal.py / elastic_training.py
                                     → 只 dataclasses/enum/math + __future__（不 import 同目录任何模块）
slurm_scheduler.py / gang_scheduling.py
                                     → + from common import Job, Node, JobState
capstone_cluster_run.py             → + from common import make_cluster, Job, JobState，
                                       + from slurm_scheduler import fifo_with_backfill, release，
                                       + from fault_tolerance import FaultModel, optimal_ckpt_interval
```

和 `gpu-architecture`/`cuda-essentials`/`cluster-networking`/`storage-dataops` 同款模式（不同于零 import
的 `kernel-engineering`）：3 个脚本会 `from common import ...` 等跨文件 import。**已实测**直接
`python learning/training-orchestration/src/X.py`（不设 `PYTHONPATH`、从 repo 根目录跑）对全部 8 个脚本
都可行——Python 会自动把脚本所在目录（`learning/training-orchestration/src/`）插入 `sys.path[0]`，同目录
下的 `common.py`/`slurm_scheduler.py`/`fault_tolerance.py` 都能被正常 import，不需要额外配置。

复用仓库根 `.venv`（Python 3.13）即可，不需要额外 `pip install`：

```powershell
python learning/training-orchestration/src/common.py
# [OK] training_orchestration.common
```

### "ray" 命名澄清：两个 `ray_*.py` 具体在模拟什么

`ray_actors.py`（68 行）和 `ray_original_minimal.py`（253 行）都不 `import ray`，是用纯 Python
dataclass **手写模拟** Ray 论文（Moritz et al. 2018 OSDI, arXiv:1712.05889）里的机制，分工不同：

- **`ray_actors.py` 模拟 Ray 的 actor 编程模型（API 层）**：`ActorSystem.spawn(cls, name)` 对应
  `Class.remote()`（返回带唯一 `actor_id` 的句柄），`ActorSystem.call(actor_id, method, ...)` 对应
  `actor_handle.method.remote()`（这里是同步直接方法分发，非真异步/跨进程 RPC）。要验证的核心概念是
  **"actor 方法调用之间共享、持久化的内部状态"**——`TrainerActor.step()` 的 `step`/`tokens_seen` 计数器
  跨调用累加、`ParameterServer.push()/pull()` 的梯度累积到同一个 `params` 列表，体现了论文里 actor
  （相对于无状态 task）"状态常驻进程内、可被连续方法调用读写"这一核心区分（对应论文导读 §6"Actor 的
  优点：状态常驻，适合 trainer、parameter server"，`ParameterServer` 正是论文举的经典例子）。

- **`ray_original_minimal.py` 模拟 Ray 论文更深一层的系统架构**，覆盖论文导读 §8-13 描述的四个组件：
  - `GlobalControlStore`（GCS）：object table / task table / function table / actor 的"上一次调用"
    链，对应论文的元数据中心（不是"中央调度器"，是控制状态存储）
  - `choose_node_bottom_up()`：本地调度器优先（本地队列 `queued_ms` 不超阈值就本地跑），本地过载才转交
    全局调度器；全局调度器用 `queue_wait_ms(node) + remote_input_mb(task,node)/bandwidth_mb_per_ms`
    打分选节点——逐字对应论文导读 §10 的 bottom-up scheduler 简化代价公式
  - `ObjectRef`/`ObjectRecord` + `ray_get()`/`ray_wait()`：不可变 object store 的位置追踪 + 阻塞
    get(等全部) / 部分 wait(前 k 个 ready) 两种语义
  - `actor_method_call()` + `reconstruct_lineage()`：把 actor 方法调用记成带 `stateful_dep`（指向该
    actor 上一次调用）的 task，对象丢失时递归回溯"要 replay 哪些 task"——对应论文导读 §13 的
    lineage-based fault tolerance（task replay + actor stateful edge，即"Figure 11"那套机制）

  两个文件合起来覆盖了论文从"编程模型"（actor API，`ray_actors.py`）到"运行时架构"（GCS/调度器/对象
  存储/lineage 容错，`ray_original_minimal.py`）的两层，但都是**单进程同步 toy 实现**（无真实网络
  RPC、无真实分布式 object store、无真实 fault injection）。**两者都不实现 Placement Group**（04-ray.md
  概念表和 02-gang-scheduling.md 都提到过 Ray 的 `PlacementGroup + STRICT_PACK`，但那只是"真实 Ray 怎么
  做 gang scheduling"的参考说明，本模块代码里 grep 全 `src/` 零 `Placement`/`STRICT_PACK` 命中——
  `gang_scheduling.py` 的 `gang_assign()` 是一个调度器无关的通用原子分配器，不特定绑定 Ray API）。

  这和 M5 `sglang-radixattention`/`inference-engine-core`、M8 `cuda-essentials` 等模块"本地无对应重型
  框架、但 `src/` 是可断言验证的纯 Python 机制复现"是同一模式——学习目的是理解机制，不是替代真实 Ray
  部署。

---

## 横向对比：8 个脚本实测结果（本模块脚本直接产出，非手算）

| 脚本 | 建模对象 | 关键机制 | 本模块实测结果 |
|---|---|---|---|
| `common.py` | Job/Node/JobState 数据类 | `make_cluster(n_nodes, gpus_per_node)` | `[OK] training_orchestration.common`（16节点×8GPU=128 free GPU） |
| `slurm_scheduler.py` | Slurm 风格 FIFO + backfill | `try_assign` 贪心按剩余量降序装箱；`fifo_with_backfill` 让排队小 job 不被队头卡住的大 job 阻塞 | `[OK] slurm_scheduler (2/3 initial + backfill carol)`——alice(24)+bob(8) 首轮填满 32-GPU 集群，carol(4) 排不下；alice 释放后单独重新调度 carol 成功入队 |
| `gang_scheduling.py` | Gang scheduling 原子分配 | `gang_assign` 全部 GPU 一次性分配或全不分配 | `[OK] gang_scheduling`（64-GPU job 独占集群后，16-GPU job 分配失败；128-GPU 需求触发 starvation 检测） |
| `fault_tolerance.py` | 集群 MTBF + Young's formula | `1024GPU MTBF≈8.5h`；`T_opt=√(2·1·8.5·3600)≈247s` | `[OK] fault_tolerance (1024 GPU MTBF 8.5h, T_opt 247s, wasted 0.81%)` |
| `ray_actors.py` | Ray actor 编程模型（有状态方法调用） | `TrainerActor.step()` 跨调用计数器累加；`ParameterServer.push/pull` 梯度累积 | `[OK] ray_actors (trainer + PS round-trip)`（2 次 step 后 `step==2`；push `[1,2,3]` 后 `params[0]==-0.001`） |
| `ray_original_minimal.py` | Ray 论文系统架构（GCS/调度器/对象存储/lineage） | 本地优先+全局兜底调度；lineage 递归回溯 replay 链 | `[OK] ray_original_minimal (GCS, scheduler, actors, lineage)`（大对象在远端时调度器正确选中数据所在节点；200-node×32-core×5ms task 吞吐量算出恰好 1,280,000 tasks/s，与论文导读 §15 估算一致） |
| `elastic_training.py` | torchrun rendezvous（节点动态 join/leave） | `min_nodes=4,max_nodes=16`；quorum 判定；每次 join/leave `generation+1` | `[OK] elastic_training (final world 2, gen 30)`（16 节点扩容到上限后批量退出 14 个，quorum 丢失；30 次成功的 join/leave 各 +1 gen） |
| `capstone_cluster_run.py` | Capstone：调度器+故障容忍综合快照 | 64节点×8GPU=512GPU，8 个混合优先级 job 跑 FIFO+backfill，同规模算 Young's formula | `[OK] capstone_cluster_run (8/8 jobs, util 0.69, MTBF 16.71h, T_ckpt 347.0s)`，与 `lectures/06-capstone.md` 文档表格逐字一致（**无文档漂移**，验证时核对未发现需要订正之处） |

**独立核实的两个关键点**（本次验证逐一读码+手算确认，非文档抄录）：

1. **`fifo_with_backfill` 是真 backfill，不是伪装 FIFO**：算法先按 `submitted_at` 排序，但在
   `for j in queue: try_assign(j)` 这个主循环里，**后面提交的小 job 并不会因为前面某个大 job 排不下而被
   挡住**——只要它当时能装进剩余容量就会立刻和大 job 一起被调度（这正是 backfill 的定义：不让队头卡住的
   job 阻塞后面能塞进空隙的小 job）。额外的 `for j in blocked: try_assign(j)` 二次遍历在单次静态调用内
   实际是 no-op（容量不会在两次遍历之间变化），真正的跨时刻 backfill 效果由 self-test 里"`release(alice)`
   后单独重新调用 `fifo_with_backfill([carol])`"来体现——这更贴近真实调度器的行为：**backfill 不是一次
   静态快照内的技巧，而是调度器在资源变化后重新评估队列时自然产生的效果**（对应 `capstone_cluster_run.py`
   注释"A real Slurm loop runs continuously"——真实 Slurm 是持续轮询重新调度，此处只做单点快照）。
2. **`capstone_cluster_run.py` 不是逐小时步进的 24h 离散事件仿真**：文件 docstring 写"24h cluster
   simulation... with faults"，但 `simulate_24h()` 函数体内明确注释"Single scheduling pass at t=0"——
   它只做**一次**静态调度快照（8 个 job 一次性提交，跑一遍 `fifo_with_backfill` 看谁被排上），加上**另一
   个独立的、闭式公式**算出的同规模集群故障容忍数字（`FaultModel(n_gpus=512).cluster_mtbf_hours()` +
   `optimal_ckpt_interval`），两者共享同一个集群规模参数但彼此不是相互作用的时间序列——代码里没有任何
   "24 小时"作为仿真时长的参数，也没有真的按时间步进注入故障事件让某个 job 中途失败重调度。"24h"是叙事
   标签（对应 lecture 标题"24h Cluster 调度仿真"，代表"一个典型的集群运营切片"），不是被仿真的真实时长
   ——这不算文档造假（lecture 06 的"教学结论"本身只用这两个数字做定性推理，没有声称模拟了故障真实发生的
   时间过程），但读者需要清楚它是"调度快照 + 故障容忍闭式估算"的组合，不是"跑了 24 小时、中途死了几张卡
   然后从 ckpt 恢复"的动态仿真。

---

## 关键公式（cheatsheet）

```
Fairshare 优先级衰减（L01）：
  fairshare = (used_recently / fair_share_limit)^-1        ← 用得越多，优先级衰减越快

集群 MTBF 可加性（L03，fault_tolerance.py::FaultModel）：
  1/MTBF_cluster = N_gpu/MTBF_gpu + N_fabric/MTBF_fabric    ← 失效率可加，MTBF 是失效率的倒数
  1024 GPU (per-GPU MTBF≈8760h) + 1 fabric(720h) → MTBF≈8.5h
  16384 GPU (Llama-3 训练规模) → MTBF≈0.5h                  ← 万卡训练约 30 分钟一次故障

Young's Formula(1974) 最优 ckpt 间隔（L03，fault_tolerance.py::optimal_ckpt_interval）：
  T_opt = √(2 · C · M)        C=ckpt成本(s)，M=MTBF(s)
  C=1s, M=8.5h → T_opt≈247s≈4min

Ckpt 浪费率（L03，fault_tolerance.py::expected_wasted_pct）：
  wasted% = C/T + T/(2M)      ← 在 T=T_opt 时最小化（两项在最优点近似相等）
  C=1s,T=247s(最优),M=8.5h → 0.81%；T=1h(太稀)→5.9%；T=30s(太密)→3.4%

Ray bottom-up scheduler 代价函数（L04，ray_original_minimal.py::choose_node_bottom_up）：
  score(node) = queue_wait_ms(node) + remote_input_mb(task,node) / bandwidth_mb_per_ms
  本地队列不超阈值 → 本地跑；否则选 score 最小的候选节点（兼顾负载与数据 locality）

Ray 任务吞吐估算（论文导读 §15，ray_original_minimal.py::theoretical_task_throughput）：
  tasks_per_second = n_nodes × cores_per_node × (1000 / task_duration_ms)
  200 nodes × 32 cores × (1000/5ms) = 1,280,000 tasks/s      ← 论文用来论证细粒度任务需要百万级调度吞吐

Elastic rendezvous quorum（L05，elastic_training.py::RendezvousState）：
  is_quorum() ⟺ len(current_members) >= min_nodes            ← 达到 quorum 才能训练
  can_admit() ⟺ len(current_members) <  max_nodes            ← 每次 join/leave 都令 generation += 1

Capstone GPU 利用率（L06，capstone_cluster_run.py::simulate_24h）：
  util = Σ(已调度且RUNNING的job.n_gpus) / (n_nodes × gpus_per_node)
  8 job 混合优先级，64节点×8GPU=512GPU → util=0.69（69%，教学结论认为 60-80% 是健康范围）
```

---

## 自测题

1. `fifo_with_backfill` 里为什么"后到但更小"的 job 不会被"先到但太大暂时排不下"的 job 挡住？这和
   `capstone_cluster_run.py` 注释里"A real Slurm loop runs continuously"有什么关系（提示：backfill
   的效果在单次静态调用和持续轮询里分别是怎么体现的）？
2. `fault_tolerance.py` 的 `cluster_mtbf_hours()` 假设失效率（不是 MTBF 本身）可加——为什么 1024 GPU
   加 1 个 fabric 的集群 MTBF 会远小于单独任何一个组件的 MTBF？如果 GPU 数从 1024 涨到 16384，MTBF
   会怎么变（不用重新推导公式，直接说趋势和量级）？
3. Young's Formula `T_opt=√(2CM)` 在 ckpt 成本 C 从 1s 降到 0.01s（async DCP）时，T_opt 会怎么变？
   `lectures/03-fault-tolerance.md`"大集群应对"一节提到这一点，对应公式里的哪个变量？
4. `ray_actors.py` 的 `ParameterServer.push()`/`pull()` 为什么必须用 actor（有状态），而不能用无状态
   task 实现？如果用 task 实现，每次调用需要多传递什么额外信息？
5. `ray_original_minimal.py` 的 `choose_node_bottom_up()` 什么情况下会绕过本地节点、直接交给全局调度
   器？全局调度器的打分公式两项分别对应什么开销来源？
6. `reconstruct_lineage()` 对 actor 方法调用为什么要额外检查 `stateful_dep`（而不是只检查
   `input_refs`）？如果丢失的是一个 actor 方法的输出对象，重建需要 replay 哪些 task？
7. `elastic_training.py` 的 `generation` 计数器在真实 torchrun 里对应什么概念？为什么"节点加入/退出"
   都要让全员 reload state（`lectures/05-elastic.md`"State 必须 elastic-safe"一节）？
8. `gang_scheduling.py` 的 `starvation_check` 只检查"单个 job 需求 > 集群总量"，`lectures/
   02-gang-scheduling.md`"反例"一节描述的场景（40 GPU 分散在 6+2 节点、job-A 需要 8 节点整块空）为什么
   不会被这个函数抓出来？这说明 starvation 有几种不同成因？
9. `capstone_cluster_run.py` 的"故障容忍"数字（MTBF/T_opt）和"调度"数字（util/scheduled）是两条独立
   算出来的线，还是相互影响？如果 8 个 job 里有一个在跑到一半时因故障重启，代码里哪个函数需要改？
10. 本模块两个 `ray_*.py` 都不 `import ray`——`ray_actors.py` 和 `ray_original_minimal.py` 分别对应
    Ray 论文的"编程模型层"还是"系统架构层"？各自缺失了真实 Ray 的哪些能力（提示：真异步 RPC、真分布式
    object store、真 fault injection）？

---

## Git 里程碑

| Tag | 内容 |
|-----|------|
| `training-orch-scheduling` | L01-02：Slurm FIFO/backfill 调度 + Gang scheduling |
| `training-orch-fault-ray` | L03-04：故障容忍 MTBF/Young's formula + Ray actor 模型 |
| `training-orch-elastic` | L05：Elastic training rendezvous |
| `training-orchestration` | L06：Capstone 集群调度快照，模块完结 |

---

## 运行验证（Runbook）

> 本段命令即 [`runbook.yaml`](runbook.yaml) 登记的"文档入口命令"，已在 ERIC-3080Ti（RTX 3080 Ti 16GB）上
> V0+V1 验证通过（8/8，纯 CPU 秒级，无需 GPU；V0 全部 `v0: false` 因为脚本无 argparse，跳过 `--help`
> 探针）。
> 一键复验本模块：
> ```powershell
> python scripts/eric_3080ti_env_audit.py --runbook --modules training-orchestration
> ```

8 个脚本全部**无 argparse**（纯 `_self_test()` 直跑，无可调预算/flag，跑一次就是全部）：

```powershell
python learning/training-orchestration/src/common.py                # [OK] training_orchestration.common
python learning/training-orchestration/src/slurm_scheduler.py       # [OK] slurm_scheduler (2/3 initial + backfill carol)
python learning/training-orchestration/src/gang_scheduling.py       # [OK] gang_scheduling
python learning/training-orchestration/src/fault_tolerance.py       # [OK] fault_tolerance (1024 GPU MTBF 8.5h, T_opt 247s, wasted 0.81%)
python learning/training-orchestration/src/ray_actors.py            # [OK] ray_actors (trainer + PS round-trip)
python learning/training-orchestration/src/ray_original_minimal.py  # [OK] ray_original_minimal (GCS, scheduler, actors, lineage)
python learning/training-orchestration/src/elastic_training.py      # [OK] elastic_training (final world 2, gen 30)
```

**Capstone（L06）：64 节点×8GPU=512GPU 集群单点调度快照**

```powershell
python learning/training-orchestration/src/capstone_cluster_run.py
# [OK] capstone_cluster_run (8/8 jobs, util 0.69, MTBF 16.71h, T_ckpt 347.0s)
```

**关键坑注记**

- **本次验证零代码改动、零文档漂移**：8 个脚本首跑全部一次性 PASS，`lectures/06-capstone.md` 的"结果"
  表格（`8/8 jobs scheduled` / `util 0.69` / `MTBF 16.71h` / `T_ckpt 347s`）和 `lectures/
  03-fault-tolerance.md` 里的全部数字示例（1024GPU MTBF≈8.5h、16384GPU≈0.5h、T_opt≈247s、三档浪费率
  0.81%/5.9%/3.4%）逐一手算+实测核对，**与代码实际输出完全一致，未发现需要订正之处**——这是本次
  runbook 验证系列里少见的"文档从一开始就和代码同步"的模块。
- **6 篇 lecture + 论文导读逐一核查过，没有发现前几个 M8 模块反复出现的"CWD 依赖的 `python -c "import
  sys; sys.path.insert(...)"` 一行流入口"写法**：01-slurm.md 给的是真实 Slurm CLI（`sbatch`/`squeue`/
  `sacct`/`scontrol`，仅作参考不要求在本仓库运行）；04-ray.md 给的是 Ray API 伪代码（`.remote()`，说明
  真实 Ray 长什么样，非本仓库可执行）；05-elastic.md 给的是注释掉的 `torchrun` 命令参考；论文导读 §19
  给出的 `.venv\Scripts\python.exe learning\training-orchestration\src\tests\test_all.py` 是仓库根相对
  路径直跑，已核实可行，且其 §14"最小示例"代码块也已独立跑通核实（`choose_node_bottom_up` 返回 `"B"`）。
- **已独立确认 `ray` 包未装且两个 `ray_*.py` 零 `import ray`**（`.venv` 里 `import ray` 报
  `ModuleNotFoundError`；grep `^import |^from` 全部命中只有 `dataclasses`/`__future__`）——不影响任何
  demo 可跑性，详见上方「环境配置」段"ray 命名澄清"小节的机制拆解。
- 不需要设置 `PYTHONPATH`，也不依赖 CWD：Python 自动把脚本所在目录插进 `sys.path[0]`；本模块 3 个脚本
  （`slurm_scheduler.py`/`gang_scheduling.py`/`capstone_cluster_run.py`）会互相 `from common import
  ...` 等跨文件 import（同 `gpu-architecture`/`cuda-essentials`/`cluster-networking`/`storage-dataops`
  模式），已实测直接 `python learning/training-orchestration/src/X.py`（不设 PYTHONPATH）对全部 8 个
  脚本都可行。
- 全部 8 个脚本零第三方依赖、CPU 秒级 PASS（0.14-0.17s）属正常——这些是纯 dataclass/数值 self-test
  （非训练 demo），逐个手算复核过输出数字（见上方「横向对比」表），不存在"假成功"风险。

**测试（V2）**

```powershell
python learning/training-orchestration/src/tests/test_all.py    # 预期：=== 8/8 passed ===
# 或经审计 harness：python scripts/eric_3080ti_env_audit.py --modules training-orchestration --tests
```

> 注：`test_all.py` 是脚本式聚合器（只有 `def main()`，没有任何 `test_` 前缀函数），pytest 收集会得到
> `no tests ran`（**已实测 rc=5**，非 exit 0）；audit harness 在这种情况下会**自动回退**成 `python
> tests/test_all.py` 直接跑（`_run_test_command`，`scripts/eric_3080ti_env_audit.py` 166-185 行）。
> 已实测：直跑输出 `=== 8/8 passed ===`，是真实 assert 执行，不是空心 PASS。本轮验证**未修改任何
> `src/*.py` 代码**，且该模块已在提交的基线 `docs/local-env/ERIC-3080Ti-test-matrix.md` 里有一行
> `tests-script:test_all.py PASS`——额外用 `--json-out`/`--md-out` 指向系统临时目录复跑一次做双重确认
> （同样 `1/1 pass/skip`），未覆盖任何已提交文件。

---

## 跨专题衔接

| 专题 | 衔接点 |
|---------|-------|
| ← `storage-dataops` | 上一站关心训练数据/checkpoint 怎么在存储和 GPU 之间搬；本站把视角升到"整个训练任务怎么被集群调度、容错、弹性伸缩"——`fault_tolerance.py` 的 ckpt 间隔计算和 `storage-dataops` 的 checkpoint 代价模型是同一枚硬币的两面（一个算"多久存一次"，一个算"存一次多贵"） |
| → `infra-graduation` | M8 毕业模块：把本站的调度器（Slurm/Gang/Ray/Elastic）+ 故障容忍直觉，和其余 6 站（GPU 架构/CUDA/kernel/网络/存储）串成端到端系统设计 |
| 概念呼应 `agent-graduation`(M7) | `ray_actors.py` 的 `TrainerActor`/`ParameterServer` 有状态 actor 模式，和 M7 `multi-agent-orchestration` 里讨论的多 agent 消息总线/状态管理是同一类"有状态并发单元"设计问题在不同 layer 的体现 |

---

## 完成验收（自查）

- [ ] 6 篇 lecture 全过（01 slurm → 06 capstone）
- [ ] `paper/guide_01_ray.md` 通读一遍，能回答文末「闭卷掌握检查」12 条
- [ ] 8 个 `src/*.py` self-test 全部亲自跑过一遍
- [ ] 能不看代码画出 Ray 的四个系统部件（task/actor 统一计算图、object store、GCS、bottom-up scheduler）
- [ ] 能解释为什么 `fifo_with_backfill` 是真 backfill 而不是伪装的 FIFO（提示：队头阻塞 vs 不阻塞）
- [ ] 能默写 Young's Formula `T_opt=√(2CM)` 并说出集群规模变大时 ckpt 频率应该怎么调整
- [ ] 能说清楚 `ray_actors.py` 和 `ray_original_minimal.py` 分别模拟 Ray 论文的哪一层（编程模型 vs 系统架构），以及两者都不实现什么（Placement Group、真异步 RPC）
- [ ] `python scripts/eric_3080ti_env_audit.py --runbook --modules training-orchestration` 全绿（8/8）
- [ ] `python learning/training-orchestration/src/tests/test_all.py` 显示 `8/8 passed`

---

🎓 **Module 8 第 6 专题完成 → 进入 `infra-graduation`：M8 毕业模块，串联 GPU 架构/CUDA/kernel/网络/存储/训练编排六站为端到端系统设计。**
