# TRACE-H 本地工作站与 HiPerGator 分阶段执行方案

- **日期：** 2026-07-11
- **状态：** 2026-07-11 更新；导师组已确认 16 张 B200 GPU 配额，登录后复核 account/QoS/实际并发
- **适用范围：** 72 小时方法杀伤实验、AAAI-27 主实验、公开 baseline 复现
- **方法依据：** [TRACE-H 正式 Proposal](../proposals/trace-h-policy-transport-proposal-zh.md)
- **实验依据：** [72 小时方法杀伤实验](../notes/trace-h-72-hour-method-kill-test-zh.md)
- **集群模板：** [HiPerGator 使用说明](hipergator/README.md)

## 1. 执行结论

TRACE-H 的算法本体不重，昂贵部分是 LLM executor 的大量 end-to-end rollout。资源分工应按“是否需要大规模模型执行”划分，而不是按“代码开发/实验”笼统划分。

| 资源 | 应承担的工作 | 不应承担的工作 |
|---|---|---|
| 本地 CPU/RAM | runner、branch/replay、state hash、schema、OT/LCB、AW、小模型 router、统计、画图、结果审计 | 大规模 32B sealed target、完整 MASA 复现 |
| 本地 RTX 5090 Laptop 24GB | 4B/8B 单卡 smoke、统一量化的 4B/8B/14B pilot、小批 branch bank、吞吐标定 | 把 32B 4-bit 边缘可运行误当成稳定主实验；直接运行 MASA 原始 8-GPU 配置 |
| HiPerGator L4 24GB | B200 排队或依赖不兼容时的 4B/8B BF16 fallback | 作为当前正式实验主资源；14B/32B BF16 |
| HiPerGator B200 180GB | 4B/8B/14B/32B/Gemma 统一 BF16、sealed target、MASA/SkillAdaptor 完整系统 PK、高并发模型服务 | 用 8 张 B200 原样照抄作者配置而不先做单卡资源适配 |

**近期决策：** 先按[本机具体实验计划](../experiments/local-development-experiment-plan-zh.md)完成约 314 条 development episodes，把 runner、branch、transport、seal 和故障恢复验证到可移交状态。正式 750-900-run kill test 与完整论文矩阵使用 16 张 B200 上的统一 BF16 runtime；L4 只作为排队或兼容性 fallback。OT、router 与统计保留在本地或 HiPerGator CPU job，不占 GPU。

## 2. 已核实的本地条件

### 2.1 硬件与系统

2026-07-11 本机审计结果：

| 项 | 当前值 | 对实验的含义 |
|---|---|---|
| 操作系统 | Windows 11 Pro for Workstations；WSL2 Ubuntu 24.04 | Linux-only 推理栈应放 WSL2，不在原生 Windows 强行编译 |
| CPU | Intel Core Ultra 9 275HX，24 cores | 环境 worker、replay、OT、bootstrap 足够 |
| RAM | 191.7 GB | 可缓存环境、轨迹和中等规模 cost matrix |
| GPU | RTX 5090 Laptop，24,463 MiB，compute capability 12.0 | 单卡 24GB；属于 consumer Blackwell `sm_120` |
| NVIDIA driver | 591.74 | 原生 Windows PyTorch 能识别 GPU |
| 本机 PyTorch | nightly `2.13.0.dev20260602+cu130`，CUDA 可用 | 可做 torch/Transformers smoke；不是 MASA 的冻结环境 |
| CUDA 实算 smoke | 4096 x 4096 FP16 matmul 已成功执行 | 证明当前 torch 能执行 `sm_120` kernel；不等于 vLLM/MASA 已兼容 |
| WSL Python | 3.12.3 | 需要新建项目环境；当前无 torch/vLLM/flash-attn/POT |
| WSL 工具 | `uv`、git、gcc/g++、cmake 可用；`nvcc` 缺失 | 纯 Python 环境可建；源码 CUDA 扩展前要补 toolkit |
| Docker | Windows client 已装，daemon/WSL integration 当前未启用 | 暂不能把 Docker 当作已可用路径 |
| C 盘剩余 | 约 374 GB | 不应把全部 BF16 checkpoint、容器和数据重复缓存到本地 |

### 2.2 本地兼容性风险

MASA README 固定 `vllm==0.11.0` 与 `flash-attn==2.7.4.post1`，其脚本依赖 Linux、Ray、verl、vLLM 和 FlashAttention。consumer Blackwell `sm_120` 在这些历史版本附近存在已公开的 kernel/构建兼容问题；当前 FlashAttention 主线已经包含 `sm_120` 编译路径，但这不能反推旧 pin 必然可用。故本地有两条隔离环境：

1. `traceh-core`：当前兼容的 PyTorch/Transformers/POT，用于 TRACE-H runner 和 pilot；
2. `masa-repro`：严格按作者依赖构建，用于验证作者 artifact，失败要记录版本与错误，不能污染 core 环境。

本地 GPU 的使用门为：

```text
torch CUDA smoke
  -> 4B load + 1 deterministic generation
  -> ALFWorld 1 episode
  -> 10-task branch/replay smoke
  -> 才允许批量 pilot
```

任一层失败，都先修运行栈，不用更多任务掩盖系统错误。

## 3. 三个 baseline 的真实资源结构

| 系统 | 代码/训练本体 | 主要成本 | 本地可做程度 |
|---|---|---|---|
| Offline-RL Harness / AW | 18 维 state、小型 64-hidden MLP、20 epochs、3 seeds | rollout buffer 与评估调用 | 控制器训练、单测、统计全部本地；rollout 可迁移集群 |
| SkillAdaptor | Python/OpenAI-compatible client；Localizer/Linker/Reviser/Validator | target failure trajectories、反复 chat/embedding、qualification reruns | dry-run、轨迹解析和 API mock 本地；完整 target feedback frontier 在集群 |
| MASA | verl + Ray + vLLM + flash-attn；作者脚本写死单节点 8 GPU | Qwen3 4B/8B/14B/32B rollout 与官方 runner | JSON/skill 解析本地；原始系统 reproduction 放集群并做最小资源适配审计 |

MASA 的 `trainer.n_gpus_per_node=8` 是作者运行配置，不等于所有 evaluation 科学上都需要 8 GPU。B200 单卡可容纳本项目规划的 32B BF16 权重；应先把 val-only runner 改为实际 GPU 数做小样本等价检查，并将结果标为“作者代码 + 最小资源适配”。只有确有分布式依赖时才申请多卡。

## 4. 模型显存与本地边界

以下是权重下界和工程规划范围，不是特定推理后端的实测峰值。`BF16` 仅按约 2 bytes/parameter 计算；实际还需 KV cache、临时张量、CUDA graph 和 allocator 余量。

| 模型 | BF16 权重下界 | 4-bit 权重下界 | 24GB 单卡规划结论 | B200 180GB |
|---|---:|---:|---|---|
| Qwen3-4B | 8 GB | 2 GB | BF16 可做，仍需实测长上下文 | 轻松；适合高并发 |
| Qwen3-8B | 16 GB | 4 GB | batch=1 BF16 可试；4-bit 更稳 | 轻松；适合高并发 |
| Gemma3-12B | 24 GB | 6 GB | BF16 无安全余量；4-bit 可做 | BF16 可做 |
| Qwen3-14B | 28 GB | 7 GB | BF16 不可作为稳定方案；4-bit 可做 | BF16 可做 |
| Qwen3-32B | 64 GB | 16 GB | 4-bit 仍属边缘，只做 smoke | BF16 单卡可做 |
| Qwen3-Embedding-0.6B | 1.2 GB | 0.3 GB | 本地无压力 | 无需占 B200 |

**科学约束：** quantization 是 executor 定义的一部分。pilot 若采用 4-bit，4B/8B/14B 必须使用同一预声明量化规则；不能 source 用 BF16、pseudo-target 因显存改 4-bit 后把差异全部归因于模型规模。完整主实验优先冻结 BF16 checkpoint family，4B/8B 可在 L4 跑，14B/32B/Gemma target 在 B200 跑。

## 5. 本地现在应完成什么

### 5.1 不需要 GPU 的核心实现

1. ALFWorld 统一 runner 与 task manifest；
2. `NONE/CHECK/RETRY/REPLAN` action contract 和 action mask；
3. snapshot/replay、state hash 与 95% replay 一致性检查；
4. append-only episode record、target information ledger 和 freeze hash；
5. event detector 与结构化 state extractor；
6. Source-AW、Best Fixed、Nearest、kNN、balanced OT、partial OT、LCB/NONE；
7. source LOMO、bootstrap、主表和 error audit；
8. mock executor 与 synthetic branches 的单元测试。

这些工作对 192GB RAM 的本机没有资源压力，而且正是当前最大的不确定性。runner 不稳定时，上集群只会更快地产生不可用日志。

### 5.2 本地 GPU 应完成的证据

按顺序执行：

1. 4B、8B、14B 统一量化后端的 load/generate/显存/吞吐表；
2. 10 个 ALFWorld tasks 的 baseline；
3. 至少 20% branch point 的同 action 重测，检验 effect 符号稳定性；
4. 4B/8B source branch bank 小样本；
5. Qwen3-14B `DEV_TARGET` 的 6 个 baseline calibration，随后执行 development freeze；
6. 8-task、8-method 微型 PK 与 post-seal oracle，完整本机 episodes 上限 314；
7. crash/resume 与 B200 block-runner handoff rehearsal。

本机 micro-pilot 的目标是工程 Go/Stop 和方向信号，不产生 DR-0004。正式 Go/Pivot/Stop 在 B200 上用统一 BF16 运行。Qwen3-32B 和 Gemma sealed targets 在本机不做任何 non-NONE 调试。

### 5.3 明确不在本地硬扛的内容

- Qwen3-32B BF16 或稳定高并发；
- Gemma3-12B BF16；
- 完整两环境、多 target、多 seed 主矩阵；
- MASA 原始 8-GPU/Ray 配置；
- SkillAdaptor 大量 target qualification；
- 需要数日连续运行且不能容忍笔记本休眠、热降频或重启的 sealed jobs。

## 6. HiPerGator 当前可用资源与约束

以下均来自 2026-07-11 查询的 UF Research Computing 官方文档：

| 项 | 当前官方信息 | TRACE-H 用法 |
|---|---|---|
| L4 | `hpg-turin` 节点，每节点 3 张，24GB/GPU；请求 GPU 时可自动选择 | 4B/8B BF16、量化 source arrays |
| B200 | `hpg-b200`，每节点 8 张，180GB/GPU；必须显式指定 partition | 14B/32B/Gemma BF16、sealed target |
| GPU allocation | 导师组已确认 16 张 B200 GPU 配额；仍用 `module load ufrc && slurmInfo -g GROUP` 核验 | 足以让正式 pilot 与主矩阵直接采用 BF16；不再依赖本机量化扩量 |
| CPU 要求 | 每张 GPU 至少请求 1 CPU；L4 资源表为 4 cores/GPU，B200 为 14 cores/GPU | 模板分别用 4/14 cores 起步 |
| 时限 | `hpg-turin`、`hpg-b200` 最大 14 天；默认仅 10 分钟 | 每个 job 必须显式 `--time` |
| GPU QoS | GPU partition 没有 burst QoS | 截止日前不能假设临时 burst 可救场 |
| CUDA | L4 需 CUDA >=12.0；B200 需 CUDA >=12.8.1；公开 module 含 12.8.1/12.9.1 | 建 B200 环境时用 12.8.1+ |
| 容器 | Apptainer GPU 必须 `--nv`；多 GPU B200 加 `--ipc=host` 或 bind `/dev/shm` | reproduction image 放 `/blue` |
| Home | 40GB；不能作为 job 输入输出 | 只放小脚本、SSH 与 shell 配置 |
| Blue | 活跃 workload 路径 | repo、env、models、data、runs |
| Orange | 归档，不能承受活跃并行 I/O | 实验结束后归档 immutable artifacts |
| Node scratch | `$SLURM_TMPDIR`，job 结束自动删除 | stage model/index；结束前复制结果回 `/blue` |
| 传输 | 大文件推荐 Globus `UFRC HiPerGator` collection | checkpoint、数据与结果迁移 |

导师组已确认 16 张 B200 GPU 配额，因此此前按默认 2-NGU trial 做的资源不足判断不再适用。仍需核验 GPU 是否可同时占满 16 张、对应 CPU/RAM、Blue quota、QoS 和排队状态；“有配额”不自动等于“16 张立即同时启动”。

## 7. HiPerGator 上具体做什么

### Phase H0：allocation 与运行栈确认

登录后第一轮只做：

```bash
module load ufrc
slurmInfo -g <group>
showQos
blue_quota
nodeInfo
sinfo -p hpg-b200
```

随后从 `/blue/<group>/<user>/trace-h` 提交 20 分钟 GPU preflight，分别验证 L4 与 B200 的 driver、CUDA capability、PyTorch、vLLM/Transformers 和写回路径。禁止在 login node 加载模型或跑 benchmark。

### Phase H1：模型服务吞吐标定

对每个 `model x precision x GPU type` 运行固定 30-episode block，测：

- load time、峰值显存；
- tokens/s、episode p50/p95；
- 1/4/8/16 个 environment workers 的吞吐；
- parser failure、OOM、timeout；
- 同 seed 的输出稳定性。

只有吞吐表完成后才计算 GPU-hours。B200 应优先提高单卡并发，而不是默认给每个 episode 独占一张卡。

### Phase H2：source branch bank

- Qwen3-4B/8B/14B 全部使用同一 B200 BF16 runtime，减少 backend/precision 混杂；
- 16 张 GPU 按 model-loaded blocks 分配，不为每条 episode 单独加载模型；
- L4 只在 B200 排队或某个依赖明确需要 `sm_89` 时作为 fallback；
- 每个 array element 处理一组 tasks/branches，不是一条 episode；
- 每个 block 30-120 分钟，避免 scheduler 被大量短 job 淹没；
- 输出 append-only JSONL 到 `/blue`，model/data 可 stage 到 `$SLURM_TMPDIR`。

### Phase H3：CPU compile 与 target seal

source bank 收齐后，在本地或 CPU compute node 完成 LOMO、metric、partial OT、LCB 与全部 policy artifacts。随后：

1. B200 上只运行 Qwen3-32B/Gemma 的 `NONE` calibration；
2. 冻结 extractor、cost、coverage、threshold、task order 与 policy hash；
3. 生成 target information ledger；
4. freeze 之前禁止任何 target `CHECK/RETRY/REPLAN`。

### Phase H4：blind target final test

在 B200 上按随机化 block 同时运行全部 same-budget methods。stdout 不输出中间排名，raw results 写入 blind 目录；全部 block 完成后统一解封分析。目标 action outcome 允许在 freeze 后产生，但不得用于调阈值或更换 quantization/parser。

### Phase H5：完整公开系统 PK

- MASA Base/DS/Evolved：B200 上做作者代码 + 最小资源适配 reproduction；
- SkillAdaptor：B200 托管 chat/embedding endpoint，client 与环境可在 CPU/L4；
- target-trained AW：复用 target rollout buffer，控制器训练在 CPU；
- WebShop：先用 small data 验证 replay/index，再构建 full data；
- 第二环境只在 ALFWorld kill test 通过后展开。

## 8. GPU-hours 资源包络

### 8.1 计算方式

先用 intended concurrency 实测每个 block 的 `GPU wall time`，再计算：

```text
model-GPU-hours = sum(block wall-hours * GPUs allocated to block)
```

不要用模型参数量直接猜总成本，也不要把排队时间算成 GPU-hours。多 worker 动态 batching 后，episode 平均时间可能显著下降。

### 8.2 72 小时 pilot

pilot 为 750-900 episode-runs。若单卡等效平均每 run 为：

| 等效平均时长 | 750 runs | 900 runs | 16 B200 理论纯计算时间 |
|---:|---:|---:|---:|
| 2 min | 25 GPU-h | 30 GPU-h | 1.6-1.9 h |
| 5 min | 62.5 GPU-h | 75 GPU-h | 3.9-4.7 h |
| 10 min | 125 GPU-h | 150 GPU-h | 7.8-9.4 h |

正式 pilot 不再在本机启动；本机只执行约 314-episode development protocol。上表用于估算 B200 pilot 的并行 wall-clock，实际值必须由 30-episode throughput block 校准。

### 8.3 完整论文规划量级

在尚未做 power analysis 前，只使用资源包络：

- ALFWorld source bank：约 2,700-3,200 runs；
- 两个 targets、9 个 controlled methods、100 tasks、3 seeds：约 5,400 runs；
- ablation、post-seal oracle 与直接系统 baseline：约 3,000-5,000 runs；
- WebShop 缩放验证：约 3,000-6,000 runs；
- **合计规划量级：约 14,000-20,000 runs。**

| 等效平均时长 | 14k runs | 20k runs |
|---:|---:|---:|
| 2 min | 467 GPU-h | 667 GPU-h |
| 5 min | 1,167 GPU-h | 1,667 GPU-h |
| 10 min | 2,333 GPU-h | 3,333 GPU-h |

这些数字是资源采购/排期上界，不是已冻结实验样本量。正式 task/seeds 由 pilot 方差和 power analysis 决定。

按 16 张 B200 理想并行换算，完整矩阵约为：2 min/run 时 29-42 小时，5 min/run 时 73-104 小时，10 min/run 时 146-208 小时。即使扣除排队和长尾，完整 ALFWorld 主表与两个 sealed targets 已具有现实可行性；WebShop 是否全量仍由 runner 稳定性与 pilot 结果决定，而不是由 GPU 数先行决定。

## 9. 存储规划

建议 `/blue` 至少预留 0.5 TB，1 TB 更稳妥：

```text
/blue/<group>/share/trace-h/
  repo/                 # git checkout
  envs/                 # conda/uv or unpacked runtime
  containers/           # Apptainer SIF
  models/               # immutable checkpoints
  data/                 # ALFWorld/WebShop/index
  manifests/            # frozen task/block manifests
  artifacts/frozen/     # router, hashes, seals
  runs/source_branch/
  runs/target_baseline/
  runs/target_final_blind/
  logs/
```

本地只缓存 development protocol 所需的量化 checkpoint 与 0.6B embedding；全部 BF16 模型、WebShop full data 和大批 raw logs 放 `/blue`。完成后将 immutable release bundle 复制到 `/orange` 和独立外部备份。UF 官方明确说明 Home、Blue、Orange 默认都不等于完整备份。

## 10. 调度原则

1. 一个 GPU job 对应一个 model-loaded block，不对应单 episode；
2. L4 每个 block 初始 4-8 workers，B200 初始 16 workers，再按 p95/OOM 调整；
3. array 用 `%K` 限流，`K` 不超过 `slurmInfo` 显示的 group GPU 余量；
4. 4B/8B 与 14B/32B 分队列，不让小模型等 B200；
5. checkpoint 和 retrieval index 可 stage 到 `$SLURM_TMPDIR`，结果必须原子写回 `/blue`；
6. 失败 block 通过 manifest ID 重提，不覆盖已有 raw records；
7. job 结束记录 git commit、container/env hash、GPU type、driver、CUDA、model hash 和 Slurm job ID。

## 11. Go / Pivot / Stop 门

### 本地到集群的 Go 门

- replay hash 一致率 >=95%；
- branch effect 重测符号稳定率 >=70%；
- 10-task smoke 无 silent parser retry；
- 30-run throughput block 无 OOM/死锁；
- append-only records 与 seal ledger 验证通过。

### 资源 Pivot

- 本地 vLLM/flash-attn 被 `sm_120` 卡住：core runner 改 Transformers/当前兼容 server；MASA reproduction 移 B200/L4；
- B200 排队过长：4B/8B 全部转 L4，B200 只保留 14B/32B/Gemma target；
- 实际并发低于确认配额：按 `slurmInfo` 限流，先完成 ALFWorld strongest-baseline 主表，WebShop 缩放为预声明外部验证；
- 32B target 吞吐太低：提高单 B200 batching，不改变已冻结 checkpoint/precision。

### Stop

- runner/replay 不稳定仍试图用集群扩量；
- 为适配硬件在 freeze 后更换 target precision、prompt 或 parser；
- target action outcome 在 freeze 前泄漏；
- 资源不足时删除 kNN、balanced OT、Source-AW 等 strongest same-budget baseline；
- 只完成 secondary diagnostics，没有 end-to-end utility PK。

## 12. 接下来 24 小时

1. 在 WSL2 建隔离的 `traceh-core` 环境，不先装 MASA 全家桶；
2. 完成本地 4B load/generate 和 ALFWorld 1-episode smoke；
3. 实现 mock executor 下的 branch/replay/state-hash 单测；
4. 把 10-task smoke 变成一个可重入 block runner；
5. 获取 HiPerGator group/account，并用 `slurmInfo` 核验 16 B200、Blue quota 与 QoS；
6. 登录后提交 [GPU preflight](hipergator/slurm/gpu-preflight.sbatch)；
7. 用 30 episodes 实测 p50/p95 后再锁 GPU-hour 预算。

## 13. 官方资料与本地证据

UF 官方资料：

- [GPU Access：L4/B200、partition、GRES、Apptainer GPU flags](https://docs.rc.ufl.edu/scheduler/gpu_access/)
- [Partition Limits：hpg-turin/hpg-b200 时限与 GPU QoS](https://docs.rc.ufl.edu/scheduler/partition_limits/)
- [Computation：禁止 login node 计算、workload 从 Blue 运行](https://docs.rc.ufl.edu/quickstart/computation/)
- [Practical Storage：Home/Blue/Orange/$SLURM_TMPDIR](https://docs.rc.ufl.edu/quickstart/practical_storage/)
- [CUDA Usage：当前 CUDA modules 与 L4/B200 architecture](https://docs.rc.ufl.edu/software/apps/cuda/usage/)
- [Conda Configuration：环境与 cache 放到 Blue](https://docs.rc.ufl.edu/software/conda_configuration/)
- [Apptainer Usage](https://docs.rc.ufl.edu/software/apps/apptainer/usage/)
- [Globus：UFRC HiPerGator collection](https://docs.rc.ufl.edu/data_transfer/globus/)
- [Trial Allocation：默认 16 NCU、2 NGU、1 TB Blue/Orange](https://docs.rc.ufl.edu/resources/trial/)

兼容性依据：

- [vLLM releases](https://github.com/vllm-project/vllm/releases)
- [vLLM RTX 5090 / sm_120 issue](https://github.com/vllm-project/vllm/issues/16901)
- [FlashAttention repository and install requirements](https://github.com/Dao-AILab/flash-attention)
- [FlashAttention current `sm_120` build path](https://github.com/Dao-AILab/flash-attention/blob/main/setup.py)

本地代码证据：

- [MASA README](../foundations/method-wave/code/MASA/README.md)
- [MASA ALFWorld script](../foundations/method-wave/code/MASA/scripts/run_alfworld.sh)
- [MASA WebShop script](../foundations/method-wave/code/MASA/scripts/run_webshop.sh)
- [Offline-RL Harness README](../foundations/method-wave/code/Agentic-RL-harness/README.md)
- [SkillAdaptor README](../foundations/method-wave/code/SkillAdaptor/README.md)
