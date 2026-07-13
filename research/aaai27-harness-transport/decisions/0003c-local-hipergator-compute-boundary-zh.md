# DR-0003C：本地工作站与 HiPerGator 的算力边界

- **日期：** 2026-07-11
- **状态：** accepted for execution；导师组已确认 16 张 B200 配额，登录后复核 account/QoS/并发
- **复审日期：** 首个 HiPerGator 30-episode throughput block 完成后
- **替代：** 无；不占用 pilot 结果对应的 DR-0004

## 问题

TRACE-H 哪些工作应立即在本地完成，哪些工作必须留到 UF HiPerGator，才能在 AAAI-27 截止前最大化方法证据，而不是把时间耗在运行栈和无效扩量上？

## 决策

采用“本地证伪与实现、16 B200 统一 BF16 扩量”的两阶段执行边界：

1. 本地 CPU/RAM 完成 runner、branch/replay、state hash、OT/LCB、baselines、统计和 seal；
2. 本地 RTX 5090 承担统一 4-bit 的 4B/8B/14B engineering micro-pilot，上限约 314 episodes，不承担 DR-0004；
3. HiPerGator 的 16 张 B200 承担统一 BF16 的 4B/8B/14B source、32B/Gemma sealed targets、750-900-run kill test 和完整系统 PK；
4. L4 只作为 B200 排队或特定依赖兼容性的 fallback，不再是主执行资源；
5. 任何批量 GPU job 以一批 episodes 为单位，模型只加载一次；禁止一条 episode 一个 Slurm job；
6. quantization/checkpoint 在 freeze 前预声明，本地因显存使用的量化 pilot 不冒充集群 BF16 主实验。

完整依据和资源包络见[分阶段执行方案](../deployment/local-vs-hipergator-execution-plan-zh.md)。

## 当前证据

### 项目直接观察

- 本机有 RTX 5090 Laptop 24GB、191.7GB RAM 和 24-core CPU，具备强单机开发能力；
- 原生 PyTorch 已实际完成 4096 x 4096 FP16 CUDA matmul，不只是识别到 GPU；
- WSL2 当前尚无 torch/vLLM/flash-attn/POT，Docker daemon 也未启用；
- 原生 Windows nightly PyTorch 可识别 `sm_120` GPU，但不是 MASA 的冻结依赖；
- Offline-RL Harness 的 AW controller 极小，真正成本是 rollout；
- MASA 作者脚本使用 Ray/verl/vLLM/flash-attn 并设置单节点 8 GPU；
- SkillAdaptor 本体轻，成本来自 target trajectory、chat/embedding 和 qualification rerun；
- 本机 development plan 上限约 314 episodes；正式 pilot 750-900 runs，完整矩阵预计进入万级 runs；
- 导师组已确认具有 16 张 B200 GPU 配额。

### 官方集群事实

- HiPerGator 当前提供 24GB L4 与 180GB B200；
- B200 使用 `hpg-b200`，L4 GPU request 可自动落到 `hpg-turin`；
- group 必须有 active NGU，GPU partition 无 burst QoS；
- active workload 必须从 `/blue` 运行，不能在 login node 或 `/home` 跑；
- B200 需要 CUDA >=12.8.1，Apptainer GPU job 需要 `--nv`。

### 尚未知

- 16 张 B200 对应的实际 Slurm account、QoS、同时并发上限、CPU/RAM 与 Blue/Orange quota；
- 每个模型在目标 runner 上的 episode p50/p95 与最佳并发；
- MASA pin 在 HPG B200 当前 software stack 上是否原样兼容；
- WebShop full index 的构建与 replay 稳定性。

## 备选方案

### 全部本地完成

未选择。32B 4-bit 在 24GB 上仅属边缘可运行，长时间 sealed runs 还受 laptop 休眠、热降频、单卡串行和 `sm_120` 旧依赖兼容影响。

### 全部搬到集群后再开发

未选择。runner/replay 是当前最大风险，在计费 GPU 上调试会更快产生不可用数据；login-node 规则也不允许把集群当本地 shell 随意跑。

### 所有模型都用 L4 量化

只作为资源 Pivot，不作为首选主实验。统一量化能扩并行，但会降低与公开 BF16/官方 artifact 的可比性，且 32B 长上下文仍可能紧张。

## 风险与反证

- 若本地 current-stack 无法稳定驱动 ALFWorld 4B，需把 GPU smoke 提前迁往 L4；
- 若 `slurmInfo` 显示实际并发或 QoS 与 16-GPU 配额理解不一致，需要重新计算 wall-clock；
- 若 B200 单卡模型服务不能批量处理环境 workers，GPU-hour 上界会显著上升；
- 若作者 baseline 依赖无法在 L4/B200 复现，需清楚区分 official、minimal fix 和 mechanism reimplementation。

## Go / Pivot / Stop 门

### Go

- 本地 10-task smoke 与 branch/replay 门通过；
- HPG preflight 同时确认至少一种 L4 runtime 和一种 B200 runtime；
- 30-episode block 得到可接受的 p95/OOM；
- actual allocation 足以完成 ALFWorld strongest-baseline 主表。

### Pivot

- `sm_120` 卡住 MASA pin：本地 core 改当前兼容 backend，MASA 移 B200；
- 实际 B200 并发不足：先保 sealed targets 和 ALFWorld 主表，4B/8B 可转 L4；
- scheduler/account 约束高于预期：缩 WebShop 样本与扩展消融，不删主表强 baseline；
- 32B BF16 吞吐不足：提升 B200 batching，不在 freeze 后换 checkpoint。

### Stop

- replay 未过门仍扩量；
- freeze 后因硬件改变 target precision/prompt/parser；
- target action outcome 泄漏；
- 为省算力删除 strongest same-budget baseline。

## 后果

- 新增 [deployment 执行方案](../deployment/local-vs-hipergator-execution-plan-zh.md)；
- 新增 [HiPerGator 提交说明与 Slurm 模板](../deployment/hipergator/README.md)；
- 下一轮首先取得 allocation 与 30-episode throughput，不再以参数量猜 GPU-hours；
- DR-0004 仍专用于 72 小时 pilot 的 Go/Pivot/Stop 结果。

## 变更记录

- 2026-07-11：建立初版边界；待 HPG 实测更新，不静默覆盖原估计。
- 2026-07-11：导师组确认 16 张 B200 GPU 配额；本机从完整 pilot 收缩为约 314-episode engineering micro-pilot，正式 BF16 pilot 全部移到 B200。
- 2026-07-13：本机 4B/8B source pilot 的 17 对 `NONE/REPLAN` terminal scores 全零，未通过 L3 非退化门；根据 [DR-0003D](0003d-local-source-policy-pivot-zh.md) 暂停 B200 扩量，先执行 Source Policy Gate v2。
