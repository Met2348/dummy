# TRACE-H 本机实验启动状态

- **日期：** 2026-07-12
- **运行环境：** WSL2 Ubuntu 24.04，RTX 5090 Laptop 24 GB
- **结论：** 本机工程实验的首批门槛已通过，可以进入 Qwen-ALFWorld baseline runner 集成；尚未产生任何 TRACE-H 方法有效性证据

## 1. 今日实际完成

| 层级 | 实验 | 实跑结果 | 门槛 |
|---|---|---:|---:|
| L0 | CUDA 13.0 FP16 matmul | 4096x4096 成功 | PASS |
| L0 | Qwen3-4B NF4 推理 | 3 个 context x 3 repeats，9/9 完成 | PASS |
| L1 | contract/schema/state/append-only/transport tests | 97/97 | PASS |
| L2 | ALFWorld 无模型 reset/step/replay | 8 个不同任务，48/48 state hash 一致 | PASS |
| L4 | partial-OT/LCB 合成 kill tests | 5 scenarios x 20 seeds，4/4 assertions | PASS |

机器可运行性、记录契约、环境重放和方法原语都已从“计划”变成了可执行代码。这里的 PASS 只允许我们继续开发 runner，不允许声称 TRACE-H 能提高真实任务成功率。

## 2. 固定环境

| 项目 | 版本/值 |
|---|---|
| Python | 3.11.15 |
| PyTorch | 2.12.1+cu130 |
| CUDA runtime | 13.0 |
| GPU / driver | NVIDIA GeForce RTX 5090 Laptop GPU / 591.74 |
| Transformers | 4.57.6 |
| Accelerate | 1.12.0 |
| bitsandbytes | 0.49.2 |
| POT | 0.9.7 |
| ALFWorld | 0.4.2 |

完整依赖见 [WSL 环境冻结](local-dev/environment/wsl-traceh-core.freeze.txt)。模型使用官方 `Qwen/Qwen3-4B`，本地原始权重 3 个 safetensors 分片、共 8,044,982,000 bytes；关键文件哈希见 [Qwen3-4B SHA-256 清单](local-dev/models/Qwen3-4B.sha256)。

## 3. Qwen3-4B 本机 L0

加载配置为 NF4 4-bit、double quant、BF16 compute。实际检测到 252 个 `Linear4bit` 模块，不是仅在配置中声称量化。

| Input tokens | Repeats | 固定 output tokens | 中位端到端时间 | 中位端到端 output tok/s | Peak allocated |
|---:|---:|---:|---:|---:|---:|
| 512 | 3 | 32 | 1.8524 s | 17.2750 | 2.6612 GiB |
| 2,048 | 3 | 32 | 2.0974 s | 15.2573 | 2.9612 GiB |
| 4,096 | 3 | 32 | 2.4373 s | 13.1295 | 3.3808 GiB |

- 模型载入：8.3416 s；
- 模型 memory footprint：2.4167 GiB；
- 载入后 allocated：2.5154 GiB；
- 9/9 生成非空，无 OOM、crash 或 NaN；
- 512-token 首次生成 2.7815 s，包含首次 kernel 路径开销；
- 4096-token 输出出现过 `</think>` 文本前缀。这是行为格式噪声，不影响 runtime readiness，但说明 L0 不能替代 agent parser 测试。

原始逐次数据见 [L0 JSON](local-dev/reports/L0-qwen3-4b-nf4-probe.json)。这些吞吐是 prefill 与 32-token decode 合并后的端到端值，不应写成纯 decode throughput。

## 4. ALFWorld 重放门

在官方 ALFWorld `eval_in_distribution` 的 140 个候选任务中，固定 seed 后选择 8 个互异任务。每个任务记录 reset state 和最多 5 个确定性 admissible actions，再从新环境实例重放：

- unique tasks：8/8；
- state comparisons：48；
- matching state hashes：48；
- replay hash ratio：1.000；
- unknown task ID：0。

原始轨迹、动作和逐状态 hash/diff 见 [L2 JSON](local-dev/reports/L2-alfworld-no-llm-replay-smoke.json)。该实验验证 TextWorld 环境重放，不验证 LLM action quality，也还不是 same-prefix intervention branch。

## 5. 软件与合成门

`pytest` 全量回归为 **97 passed in 4.10 s**，覆盖：

- 四动作 contract 的 80 个精确案例与 budget/mask；
- canonical state hash 的顺序不变性和非法浮点拒绝；
- append-only record、重复 run ID 和不安全路径拒绝；
- 当前 TRACE-H schemas；
- balanced OT、partial transport、LCB/NONE router。

合成实验共 100 次 simulation。关键 kill 条件全部成立：target-private 30% 场景下 partial-LCB 的 private NONE rate 为 1.0、private coverage 为 0；semantic-conflict 场景下 response-aware accuracy 为 1.0，semantic-only 为 0。详见 [L4 JSON](local-dev/reports/L4-synthetic-transport.json)。这些是定向构造的实现测试，不是对真实 baseline 的胜负证据。

## 6. 当前 Go / No-Go

**GO：** 开始实现并运行 Qwen3-4B + ALFWorld baseline adapter，先做 admissible-action constrained parser、3-task micro run、append-only episode record 与失败保留。

**NO-GO：** 当前不能启动正式 target-final sealed evaluation，不能声称 cross-executor transport 有效，也不能把 4B NF4 本机数字与 B200 BF16 主结果混用。

下一执行块：

1. 将 Qwen chat output 接到 ALFWorld admissible-action parser；
2. 运行 3 个任务的 baseline micro，确认 episode schema、budget、termination 与 raw output 完整；
3. 对可重放 prefix 建立第一批 `CHECK/RETRY/REPLAN/NONE` branch records；
4. 通过 crash/resume 与 duplicate-ID 测试后，再扩展本机 development episodes；
5. Qwen3-8B/14B L0 只在 4B runner 稳定后下载，正式 Qwen3-32B/Gemma 保持 B200 sealed。

