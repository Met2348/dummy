# TRACE-H 本机实验可运行性审计与首轮 Smoke 结果

- **日期：** 2026-07-11
- **目的：** 回答“本机现在能运行哪些实验，是否可以开始”
- **结论：** 可以开始，L0 已实际启动；TRACE-H end-to-end 尚未 ready，必须先完成 runner/schema/ALFWorld 环境
- **计划依据：** [本机具体实验计划](local-development-experiment-plan-zh.md)

## 1. 当前 Ready / Not Ready

| 实验 | 当前状态 | 是否已实跑 | 下一缺口 |
|---|---|---:|---|
| CUDA kernel smoke | Ready | 是 | 无 |
| Transformers 单模型连续生成 | Ready（原生 Windows） | 是 | 换成计划中的 Qwen3 统一 4-bit checkpoints |
| Offline-RL Harness 小型 MLP 前向 | Ready | 是 | 接入 TRACE-H 四动作 state/action schema |
| Offline reward/HMS 单测 | Ready | 是 | 统一 UTF-8 环境变量 |
| L0 Qwen3 4B/8B/14B 矩阵 | 未 ready | 否 | 三个 checkpoints、冻结 4-bit backend、WSL torch runtime |
| L1 action contract/schema | 未 ready | 否 | 当前 schemas 仍是旧 FORECAST-H；尚无 TRACE-H contract implementation |
| L2 ALFWorld baseline/replay | 未 ready | 否 | WSL 无 `alfworld/gymnasium`，无数据，无统一 runner/state hash |
| L3 source branch bank | 未 ready | 否 | 依赖 L1/L2 |
| L4 synthetic transport | 未 ready | 否 | 本机缺 POT，项目尚无 partial-OT/LCB implementation |
| L5 14B development-target PK | 未 ready | 否 | 依赖 L0-L4 与 development seal |
| L6 crash/resume/handoff | 未 ready | 否 | 尚无可重入 block runner |

这意味着“可以开始”不是“现在可以直接跑 314 episodes”。当前应从 L0 runtime 与 L1 软件契约开始；在 L1/L2 过门前不得下载 Qwen3-32B/Gemma 或运行 target actions。

## 2. 本机与软件现状

### GPU 与存储

- GPU：NVIDIA GeForce RTX 5090 Laptop，24,463 MiB；
- 当前空闲显存：约 22.4 GiB；
- driver：591.74；compute capability：`sm_120`；
- C 盘可用：约 402 GB；
- WSL ext4 可用：约 930 GB。

### 原生 Windows Python

| Package | 状态/版本 |
|---|---|
| PyTorch | `2.13.0.dev20260602+cu130`，CUDA 可用 |
| Transformers | `4.57.6` |
| Accelerate | `1.13.0` |
| bitsandbytes | `0.49.2` |
| jsonschema/pandas/numpy | 可导入 |
| POT (`ot`) | 缺失 |
| ALFWorld | 缺失 |
| Gymnasium | 缺失 |

### WSL2 Ubuntu 24.04

- Python 3.12.3；
- `uv 0.11.8` 可用；
- torch、Transformers、POT、ALFWorld 均未安装；
- WSL GPU 能被 `nvidia-smi` 识别，但尚未完成 Python CUDA smoke；
- 未发现 WSL Hugging Face model cache。

### 项目实现

- `experiments/` 目前只有计划、矩阵和三个旧 FORECAST-H schemas；
- 没有 TRACE-H runner、block runner、state hash、branch recorder、partial OT 或 LCB router；
- MASA snapshot 内含 ALFWorld wrapper/code，但模型权重和 `alfworld-download` 数据不在本机；
- 因此不能把 MASA 仓库存在误认为 TRACE-H runner 已存在。

## 3. 已完成的首轮真实实验

### R0：CUDA FP16 matmul

```text
matrix: 4096 x 4096
dtype: FP16
device: RTX 5090 Laptop
result: PASS
```

这证明当前原生 PyTorch 能在 `sm_120` 上实际执行 CUDA kernel，不只是识别设备。

### R1：离线 Transformers 连续生成

使用本机已缓存的 `TinyLlama-1.1B-Chat-v1.0`，完全离线、Transformers SDPA、FP16：

| 指标 | 结果 |
|---|---:|
| Snapshot | `fe8a4ea1ffedaf415f4da2f062534de366a451e6` |
| Model load | 1.728 s |
| 连续 probes | 20/20 完成 |
| 总生成 tokens | 960 |
| 平均单 probe | 1.223 s |
| 范围 | 1.170-1.758 s |
| 吞吐 | 39.24 tokens/s |
| Peak allocated VRAM | 2.085 GiB |
| Crash/OOM | 0 |

结论：本机的 PyTorch + Transformers + SDPA 连续推理链路可用。TinyLlama 不是 TRACE-H executor，这个结果只算 L0 前置 smoke，不进入方法结果。

### R2：Offline-RL Harness MLP

直接加载作者 snapshot 的 `MLPPolicy`，将 action 数调整为 TRACE-H 所需的 4：

```text
input batch: 32 x 18
output logits: 32 x 4
parameters: 1,476
finite outputs: PASS
```

结论：AW controller 的训练/推理资源可以忽略，后续成本确实来自 LLM rollout。

### R3：Offline reward/HMS anchors

- reward aggregator：5/5 hand-computed episodes、rubric、error/early-submit、cost、format checks 全部通过；
- HMS detector：35/35 judgments 通过；
- Windows 默认 GBK 首次无法打印 Unicode `✓`，设置 `PYTHONUTF8=1` 后通过；
- 运行生成的 `anchor_4_results.json`、`anchor_6_results.json` 已删除，作者代码 snapshot 恢复干净。

结论：baseline 的纯 CPU reward/controller 核心可以在本机运行；后续命令统一设置 UTF-8。

## 4. 现在可以立即运行的实验

### A. 不需要新增模型或数据

1. CUDA/PyTorch/Transformers soak；
2. Offline-RL Harness MLP、reward 和 HMS 单测；
3. TRACE-H schemas、action masks、budget、ledger 的 deterministic unit tests；
4. partial OT/LCB synthetic tests，前提是先实现并安装 POT；
5. crash/resume、append-only 和 duplicate run ID tests，前提是先实现 block runner。

### B. 安装本地环境后可运行

1. WSL CUDA smoke；
2. ALFWorld environment reset/step/replay，不需要 Qwen；
3. deterministic task split 与 state-hash audit；
4. mock executor 下的四动作 branch bank。

### C. 下载 Qwen3 4B/8B/14B 后可运行

1. L0 27 个 context probes + 60 个 soak probes；
2. L2 的 20 source baselines；
3. L3 的最多 192 branch/repeat continuations；
4. L5 的 6 calibration + 64 final + 32 oracle。

## 5. 启动决策

### 已启动

- L0 前置 CUDA/model smoke；
- baseline CPU controller/reward/HMS smoke。

### 下一执行顺序

1. 建立 WSL `traceh-core`，先装 PyTorch/POT/jsonschema，不装 MASA 全家桶；
2. 建立当前 TRACE-H schemas 与 action-contract unit tests；
3. 安装 ALFWorld/Gymnasium，完成无 LLM reset/step/replay；
4. 实现 mock executor block runner 与 state hash；
5. 上述全部通过后，再下载 Qwen3-4B，开始正式 L0；
6. 4B 通过后依次下载 8B、14B；
7. Qwen3-32B/Gemma 保持未触碰，留给 B200 sealed protocol。

## 6. 当前硬停止门

以下缺口存在时，不允许声称“本机 TRACE-H 实验已开始收集科学数据”：

- TRACE-H schemas 仍未替换旧 FORECAST-H schemas；
- action contract 没有 80/80 unit tests；
- ALFWorld replay hash 未达到 95%；
- runner 会覆盖相同 run ID；
- target information ledger 尚不存在；
- Qwen3 checkpoint/quantization/backend 未 hash 和冻结。

当前状态是：**工程实验已经开始，科学 rollout 尚未开始。**

