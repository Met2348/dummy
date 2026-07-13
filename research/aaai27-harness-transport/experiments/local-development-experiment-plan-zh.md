# TRACE-H 本机具体实验计划：从运行栈到集群交接

> **2026-07-13 执行更新：** L0-L3 的首轮 4B/8B source-policy kill test 已完成，但 17 对 `NONE/REPLAN` terminal scores 全为 0，未通过 response-bank 非退化门。当前暂停 L3 扩展与 L5，先执行 Source Policy Gate v2；原计划保留为 prospective protocol，不把未完成项写成已完成。详见[最终报告](local-none-replan-source-pilot-final-20260713-zh.md)与 [DR-0003D](../decisions/0003d-local-source-policy-pivot-zh.md)。

- **日期：** 2026-07-11
- **执行资源：** RTX 5090 Laptop 24GB、191.7GB RAM、WSL2 Ubuntu 24.04
- **集群条件：** 导师组已确认具有 16 张 NVIDIA B200 GPU 配额；上线后仍需用 `slurmInfo` 核验 account/QoS/并发状态
- **本机定位：** 工程证伪 + 微型方法信号 + B200 交接验收
- **不是：** AAAI 主结果、DR-0004 的正式 72 小时 BF16 kill test
- **上位协议：** [72 小时方法杀伤实验](../notes/trace-h-72-hour-method-kill-test-zh.md)
- **信息隔离：** [Policy Transport Seal](../protocol/policy-transport-seal.md)
- **机器可读矩阵：** [local-development-matrix.tsv](local-development-matrix.tsv)

## 1. 为什么 16 张 B200 之后仍要认真做本机实验

16 张 B200 解决的是扩量与大模型 fidelity，不解决以下风险：

- action 语义没有真正可执行；
- ALFWorld replay 回不到同一 state；
- parser silently retry，造成虚假成功；
- branch bank 的 run ID、prefix hash 或 outcome 对不上；
- partial OT 只在公式上成立，代码对 target-private states 仍强制匹配；
- freeze/ledger/resume 在并行任务中泄漏或覆盖结果；
- Slurm 上把一个坏 runner 放大成几千条坏记录。

因此，本机阶段的唯一职责是把上述风险压到足够低，使集群阶段只做三件事：换成统一 BF16、增加 tasks/seeds、扩大并行度。

## 2. 本机与正式实验的边界

| 项 | 本机 development protocol | B200 正式 protocol |
|---|---|---|
| 模型 | Qwen3-4B/8B source；Qwen3-14B development target | Qwen3-4B/8B/14B source；Qwen3-32B 与 Gemma sealed targets |
| 精度 | 三个模型使用同一个冻结的 4-bit backend | 全模型统一 BF16 runtime |
| 环境 | ALFWorld development tasks | ALFWorld 主实验；通过后 WebShop |
| 任务数 | 10 source + 6 calibration + 8 final | 30/20/30 pilot，再由 power analysis 扩展 |
| 完整 episodes | 上限 314 | pilot 750-900；主矩阵万级 |
| 统计含义 | 工程门与方向信号 | DR-0004 与论文证据 |
| Prospective target | 无；14B 明确标为 `DEV_TARGET` | Qwen3-32B/Gemma 才可称 sealed target |

本机看过 Qwen3-14B 的 action outcomes 后，14B 在正式实验中只能作为 source，不能再冒充 prospective target。Qwen3-32B 和 Gemma checkpoint 在本机阶段禁止运行任何 non-NONE action，最好完全不下载。

## 3. 本机固定配置

### 3.1 文件和环境

代码仍位于共享 workspace；高 I/O 数据放 WSL ext4：

```text
C:\Workspace\dummy\research\aaai27-harness-transport\
  experiments/                 # 小型 manifest、报告、hash、统计表

/home/wsl/traceh-local/
  cache/                       # uv/Hugging Face cache
  models/                      # 4-bit checkpoints
  data/alfworld/               # benchmark data
  raw/                         # 大型 append-only traces
  tmp/                         # 可删除中间文件
```

环境策略：

1. WSL2 Python 3.11；
2. `traceh-core` 使用当前兼容的 PyTorch、Transformers、POT、ALFWorld；
3. 不把 MASA 的 Ray/verl/vLLM/flash-attn pins 装进 core 环境；
4. `uv.lock` 或完整 package freeze 在第一条 L2 episode 前生成；
5. model/data/raw 不放 Git，也不重复缓存到 C 盘多个位置。

### 3.2 模型与采样

| 字段 | 本机冻结值 |
|---|---|
| Source executors | Qwen3-4B、Qwen3-8B |
| Development target | Qwen3-14B |
| Precision | 同一 4-bit backend；不同 backend 不混入同一表 |
| Generation backend | L0 通过后冻结；优先 Transformers-compatible，失败则用当前兼容 OpenAI server |
| Prompt length | 4096 tokens |
| Response cap | 512 tokens |
| Environment max steps | 50 |
| Branch sampling | `temperature=0.4`，seed=0；重复分支使用独立预声明 seed |
| Determinism probe | `temperature=0` |
| Non-NONE cap | 每个 episode 最多一次 |
| Branch points | 每个 source baseline 最多 2 个 |

如果 14B 在 24GB 上不能稳定完成 4096-token smoke，不做 CPU offload 长跑，也不临时换另一种量化。L5 直接移交 B200，L0-L4 继续在本机完成。

### 3.3 Development task split

只从 ALFWorld development pool 选择任务，不碰正式 target final pool：

```text
排序全部 eligible task IDs
  -> 使用 seed 20260711 做确定性 shuffle
  -> 前 10 个：DEV_SOURCE
  -> 接着 6 个：DEV_TARGET_BASELINE
  -> 接着 8 个：DEV_TARGET_FINAL
```

三组 task IDs 与顺序写入 manifest 并计算 SHA-256。任何 source run 后不得换 task；因环境损坏排除 task 时必须写 deviation record。

## 4. Action Contract v0.1

本机首先验证的是以下可执行语义，而不是宽泛的自然语言名称：

| Action | v0.1 执行语义 | 硬预算 | 合法事件 |
|---|---|---|---|
| `NONE` | 不改变 context、调用或环境转移 | 0 extra call/step | 全部 |
| `CHECK` | 只读提取 inventory、最近 tool status、admissible-action/precondition flags，形成结构化 check block | 最多 128 added tokens；0 env step | missing evidence、pre-submit、no-progress |
| `RETRY` | 仅对刚被 parser/environment 拒绝的 action 做确定性规范化并重放一次 | 最多 1 extra env step；不无限重试 | invalid/rejected、no-progress with identical action |
| `REPLAN` | 额外调用同一 executor 生成最多三步短计划，再把计划加入后续 context | 1 extra call；最多 192 output tokens；0 immediate env step | no-progress、missing precondition、budget pressure |

共同限制：

- 不添加 oracle 答案、gold action 或未向 baseline 开放的环境信息；
- action mask 不合法时必须返回 `NONE`，不能自动换成另一个 action；
- budget 超限记为真实失败/成本，不截断后静默重跑；
- 每个 episode 最多一次 non-NONE，避免 intervention-induced distribution compounding。

如果 L1 证明某项语义不可稳定实现，只能在 source-only 阶段修改 contract 并升级版本；开始 L5 calibration 后不得再改。

## 5. 逐项实验

### L0：运行栈与模型兼容矩阵

#### 问题

本机是否能稳定加载三个统一量化模型，并在 512/2048/4096 三种 context 长度下生成？

#### 设计

- 3 models x 3 context bins x 3 repeats = **27 generation probes**；
- 每个模型另做 20 prompts 连续生成 = **60 soak probes**；
- 记录 load time、peak VRAM、tokens/s、首 token latency、OOM、输出是否可解析；
- 先逐模型 load/unload，不同时驻留多个模型。

#### 必过门

- 4B/8B 对应的 18 个 required probes 全部无 OOM；
- 14B 4096 context 若失败，明确转 B200，不使用 CPU offload 掩盖；
- 60 soak probes 无显存持续增长、进程崩溃或 silent fallback to CPU；
- 保存 backend、model hash、driver、torch/CUDA 和量化配置。

#### 产物

```text
experiments/local-dev/reports/L0-runtime-matrix.json
experiments/local-dev/reports/L0-runtime-matrix.md
```

### L1：Action contract 与 schema 单测

#### 问题

四个 actions 是否真的满足跨 executor 一致语义、权限边界和预算？

#### 设计

- 5 event types x 4 actions x 4 fixtures = **80 deterministic cases**；
- fixture 覆盖合法 mask、非法 mask、预算边界和 malformed action；
- 使用 mock executor 和 mock environment，不消耗 GPU；
- 同时检查 episode、branch、ledger、freeze records 的 JSON schema。

#### 必过门

- 80/80 contract cases 通过；
- 非法 action 100% fallback `NONE`；
- `CHECK` 无环境写操作；
- `RETRY` 最多一个额外 env step；
- `REPLAN` 不超过三步和 192 tokens；
- schema validation 100%，未知字段拒绝而非静默丢弃。

#### 产物

```text
experiments/local-dev/contracts/*.json
experiments/local-dev/reports/L1-contract-tests.json
```

### L2：ALFWorld baseline、parser 与 replay

#### 问题

baseline trajectory 能否被准确记录，并从同一 prefix 恢复相同环境 state？

#### 设计

- 2 source models x 10 source tasks = **20 baseline episodes**；
- 每条 episode 选最多 2 个 event prefixes，共最多 **40 replay checks**；
- replay 使用记录下来的环境 actions，不重新采样 LLM；
- hash 至少包括 task state、inventory、location、completed predicates、step index 与最近 tool result。

#### 必过门

- replay hash 一致率 >=95%，且不一致有字段级 diff；
- parser failure <=2%，所有失败保留为 outcome；
- 不允许 silent retry；
- 相同 run ID 第二次写入必须失败；
- 20 episodes 都有完整 token、step、wall-clock、model/config hash。

#### 产物

```text
/home/wsl/traceh-local/raw/source_baseline/*.jsonl
experiments/local-dev/reports/L2-replay-audit.md
```

### L3：Source counterfactual branch bank 微型收集

#### 问题

同 prefix 的四个 actions 是否产生非退化、可重复的 terminal response？

#### 设计

- 复用 L2 的 20 baselines；
- 最多 40 branch points x 4 actions = **最多 160 branch continuations**；
- 固定抽取 20% branch points，即 8 points x 4 actions = **32 repeated continuations**；
- 每次 branch 都先验证 prefix hash，再滚动到 terminal；
- 保存 `G(a)`、`A(a)=G(a)-G(NONE)`、tokens、steps、invalid actions 和成本。

#### 必过门

- 至少获得 20 个合法 branch points；不足说明 event detector/support 有问题；
- branch 起点 hash 100% 与 baseline prefix 一致；
- 同 action advantage 符号稳定率 >=70%；
- 不能出现某 action 因 parser 特权而系统性多一次免费重跑；
- branch bank 可按 `model/task/prefix/action/seed` 唯一索引。

#### 产物

```text
/home/wsl/traceh-local/raw/source_branch/*.jsonl
experiments/local-dev/reports/L3-branch-bank-audit.md
```

### L4：Transport 与 conservative router 的 synthetic kill tests

#### 问题

partial/unbalanced OT 和 LCB/NONE 是否按算法设计工作，而不是依赖真实 LLM 数据碰巧过关？

#### 设计

5 个 scenario x 20 seeds = **100 CPU simulations**：

1. source/target support 完全一致；
2. 只有尺度和频率变化；
3. target 缺少一个 source event class；
4. target 出现 30% private states；
5. 文本相似但 action response 相反。

每个 scenario 比较 kNN、balanced OT、partial OT、partial OT + LCB。

#### 必过门

- support 完全一致时 partial OT 不劣于 balanced OT；
- private-state scenario 中 unmatched mass 随 private 比例单调上升；
- private states 上 conservative router 选择 `NONE` 的比例 >=90%；
- response-conflict scenario 中 semantic-only metric 明显失败，response-aware metric 不沿用错误 action；
- 固定 seed 结果可重现。

这些是软件/机制断言，不是论文 empirical result。

#### 产物

```text
experiments/local-dev/reports/L4-synthetic-transport.json
experiments/local-dev/reports/L4-synthetic-transport.md
```

### L5：Qwen3-14B development-target 微型 PK

#### 问题

整条 `branch -> transport -> compile -> execute` 链能否在 whole-executor holdout 形式下完成，并产生至少可解释的方向信号？

#### 设计

##### Calibration

- Qwen3-14B、6 个 disjoint tasks、`NONE` only = **6 episodes**；
- 冻结 state extractor、metric、OT、coverage、LCB、8 个方法 artifact 和 final order；
- 生成 development seal hash。

##### Final

8 tasks x 8 methods = **64 episodes**：

1. No Harness；
2. Best Fixed；
3. Source-AW；
4. Nearest-AW；
5. MF-Gated AW；
6. kNN-Branch；
7. Balanced-OT；
8. TRACE-H Partial-OT + LCB。

##### Post-seal oracle

- 只在每个 final task 的 TRACE-H 首个 eligible state 上运行四 actions；
- 8 states x 4 actions = **最多 32 branch continuations**；
- 用于检查 negative intervention 与局部 regret，不用于回调 policy。

#### 结果解释

8 tasks 没有统计功效，结果只分三色：

| 结果 | 含义 | 下一步 |
|---|---|---|
| Green | pipeline 全过；TRACE-H mean utility 高于 strongest baseline；partial/LCB 未显示反向信号 | 直接提交 B200 BF16 pilot |
| Yellow | pipeline 全过；方法排序不稳定或差异接近 0 | 保留方法，B200 用正式 30-task pilot 判定 |
| Red | replay/ledger 泄漏、TRACE-H 明显负效用、branch response 退化或 partial/LCB 逻辑错误 | 暂停集群扩量，先修对应模块 |

#### Red 的数值定义

- TRACE-H 比 No Harness utility 低超过 0.10；或
- negative intervention rate >50%；或
- target action outcome 在 development freeze 前被读取；或
- kNN/balanced/TRACE-H 实际加载了错误或相同 artifact；或
- primary table 不能由 raw records 一键重建。

#### 产物

```text
experiments/local-dev/manifests/L5-freeze.json
experiments/local-dev/artifacts/*.json
/home/wsl/traceh-local/raw/dev_target_*/
experiments/local-dev/reports/L5-primary-table.csv
experiments/local-dev/reports/L5-micro-pilot.md
```

### L6：故障恢复与 B200 handoff rehearsal

#### 问题

同一 runner 能否在进程崩溃、block 重提和 Linux/Slurm 路径下不覆盖、不重复、不泄漏？

#### 设计

1. source block 在 50% 处强制退出；
2. 用相同 block ID 重启，只补未完成 run IDs；
3. 重复提交已完成 block，必须拒绝覆盖；
4. target baseline 与 target final 使用不同 output roots；
5. 生成与 HiPerGator `TRACEH_BLOCK_RUNNER` 相同的 CLI；
6. 打包一个 2-task source block、1-task calibration block、1-task blind-final block。

#### 必过门

- 无重复 run ID；
- append-only raw records 在恢复前后 hash 可审计；
- partial file 不会被分析脚本读取；
- runner 不含 `C:\...` 硬编码路径；
- cluster block manifest、env lock、action contract、schema 和 expected output 全部进入 handoff bundle。

#### 产物

```text
experiments/local-dev/handoff/traceh-b200-smoke-bundle/
experiments/local-dev/handoff/SHA256SUMS
experiments/local-dev/reports/L6-resume-audit.md
```

## 6. Run 总量与本机时间

完整 LLM episodes 上限：

| 来源 | Episodes |
|---|---:|
| L2 source baselines | 20 |
| L3 four-action branches | 160 |
| L3 20% repeats | 32 |
| L5 target calibration | 6 |
| L5 8-method final | 64 |
| L5 post-seal oracle | 32 |
| **合计上限** | **314** |

另有：87 条短 generation probes、80 个 action fixtures、40 个 replay checks、100 个 CPU transport simulations 和 3 类 fault-injection tests。

| 本机等效 episode 时长 | 314 episodes 的单 GPU 时间 |
|---:|---:|
| 2 min | 10.5 h |
| 5 min | 26.2 h |
| 10 min | 52.3 h |

这是上限估计。自然 episode 产生不足两个 event 时 branch 数会减少；但不得人为制造虚假错误只为凑满 160 branches。

## 7. 48 小时执行顺序

| 时间 | 工作 | GPU 占用 |
|---|---|---|
| 0-6 h | L0 环境、模型、cache、runtime matrix | 间歇 |
| 6-12 h | L1 contract/schema；确定 task manifest | 无或很少 |
| 12-20 h | L2 baselines、parser、replay | 连续 |
| 20-32 h | L3 source branches；CPU 同时做 L4 synthetic | 连续 |
| 32-36 h | Source LOMO、8 个 artifacts、development calibration | 连续 |
| 36-44 h | L5 blind final 与 post-seal oracle | 连续 |
| 44-48 h | L6 resume、统一分析、B200 handoff bundle | 少量 |

如果平均 episode 超过 10 分钟或 14B runtime 不稳，不延长本机到 72 小时；完成 L0-L4 和 L6，把 L5 移到 B200。

## 8. 本机总体 Go 门

只有以下项目全部通过，才允许用 16 张 B200 扩量：

1. L1 contract/schema 100% 通过；
2. L2 replay hash >=95%；
3. L2 parser failure <=2% 且无 silent retry；
4. L3 至少 20 个合法 branch points，重复符号稳定率 >=70%；
5. L4 五个 synthetic scenario 断言通过；
6. L5 development seal 无 target leakage；
7. L6 crash/resume 无覆盖、重复和 partial-file contamination；
8. B200 handoff bundle 能用单一入口执行 source/calibration/final 三种 block。

TRACE-H 是否在 8-task micro PK 中领先不是硬 Go 门；样本太小。真正的方法 Go/Pivot/Stop 仍由 B200 上 30-task、750-900-run、统一 BF16 pilot 的 DR-0004 决定。

## 9. 禁止事项

- 不在本机下载或试跑 Qwen3-32B/Gemma non-NONE target；
- 不把 4-bit development result 与 BF16 main result 合并；
- 不根据 L5 target action outcomes 修改 event taxonomy、OT metric 或 LCB；
- 不因本机慢而删除 Source-AW、kNN、balanced OT 等强 baseline；
- 不把 infrastructure failure 静默重跑成成功；
- 不在 raw JSONL 中保存 secret、完整 chain-of-thought 或未授权数据；
- 不把微型 8-task 排名写成论文 claim。

## 10. 本机结束时应交付什么

```text
1 个可重入 block runner
4 个可执行 action contracts
1 个冻结的 24-task development manifest
1 个带重复分支的 source response bank
8 个 method artifacts
1 个 development seal + target information ledger
1 张 8-task micro primary table
1 份 replay/branch/parser/error audit
1 个通过故障恢复测试的 B200 handoff bundle
```

完成这些后，16 张 B200 的价值才能真正兑现：并行增加科学样本，而不是并行调试。
