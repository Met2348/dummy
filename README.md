# AI/ML 全栈学习马拉松 — 46 专题知识地图

> 这是一个博士级 AI/ML 研究代码库，围绕 **8 个课程模块（M1、M3-M8）、46 个专题**组织，每个专题都是
> 一个可独立运行、可独立验证的自包含学习包（lectures + 可跑代码 + 论文导读 + 自测）。核心内容在
> [`learning/`](learning/) 目录下；本文件是它的入口地图。
>
> （模块编号从 M1 跳到 M3：早期设计文档里的"Module 2"指 RL/对齐相关内容，`dpo-family`/`reasoning-r1`
> 等专题当时就挂在这个编号下——现在这些专题在下方 M4「改大模型」里，历史编号沿用未回收，不是遗漏。）

---

## 这是什么

`learning/` 下的 46 个专题按 8 个课程模块组织，覆盖从 PEFT 微调、预训练造大模型、RLHF/RL 对齐、
推理部署、评测与安全、Agent 应用层，到 GPU/Infra 系统底层的完整技术栈。每个专题都配有：

- **可运行代码**（`src/`），带 `_self_test()`/`pytest` 断言验证，不是伪代码
- **讲义**（`lectures/`），中文撰写，覆盖原理、公式推导、常见坑
- **论文导读**（`paper/`），精读对应的原始论文/技术报告
- 部分专题还有交互式 **notebook**（`notebooks/`）

全部 **46 个专题的文档命令均已逐条独立验证**（详见下方「怎么运行/验证任意模块」），验证过程和发现的
问题记录在 [`docs/local-env/ERIC-3080Ti-runbook-progress.md`](docs/local-env/ERIC-3080Ti-runbook-progress.md)。

---

## 8 模块 / 46 专题地图

### Module 1 — PEFT（参数高效微调，3 专题）

| # | 专题 | 一句话 |
|---|------|--------|
| 1 | [`prompt-tuning-family`](learning/prompt-tuning-family/README.md) | Soft Prompt / P-Tuning / IPT——不改模型权重的微调方式 |
| 2 | [`lora-family`](learning/lora-family/README.md) | LoRA / QLoRA / DoRA / LoRA+——低秩适配家族，工业界主流 PEFT 方案 |
| 3 | [`adapter-tuning-family`](learning/adapter-tuning-family/README.md) | Pfeiffer / Houlsby / Parallel / Compacter Adapter 家族 |

### Module 3 — 造大模型（预训练全流程，8 专题）

| # | 专题 | 一句话 |
|---|------|--------|
| 4 | [`data-curation`](learning/data-curation/README.md) | 数据准备：CommonCrawl → 1B-token 高质量语料 |
| 5 | [`transformer-deep`](learning/transformer-deep/README.md) | 现代 Transformer 架构骨架（RoPE/RMSNorm/SwiGLU/GQA） |
| 6 | [`moe-architecture`](learning/moe-architecture/README.md) | Mixture of Experts：从 Shazeer 2017 到 DeepSeek-V3 Aux-Free |
| 7 | [`ssm-hybrid`](learning/ssm-hybrid/README.md) | State Space Models：Mamba / RWKV / Hybrid 架构 |
| 8 | [`long-context`](learning/long-context/README.md) | 长上下文：RoPE 外推 / NTK / YaRN / Ring Attention |
| 9 | [`scaling-infra`](learning/scaling-infra/README.md) | 训练与推理基础设施、Chinchilla scaling law |
| 10 | [`pretraining-recipe`](learning/pretraining-recipe/README.md) | 预训练管线：真实从零训练闭环 |
| 11 | [`small-model-graduation`](learning/small-model-graduation/README.md) | **M3 毕业**：Phi-tiny 270M 从零训练 capstone |

### Module 4 — 改大模型（RLHF / 对齐 / 推理 RL，7 专题）

| # | 专题 | 一句话 |
|---|------|--------|
| 12 | [`rl-foundations`](learning/rl-foundations/README.md) | 强化学习基础，PPO + GAE 为核心 |
| 13 | [`rlhf-classic`](learning/rlhf-classic/README.md) | InstructGPT 三段管线（SFT→RM→PPO） |
| 14 | [`dpo-family`](learning/dpo-family/README.md) | DPO 去 RM 革命，RainbowPO 统一视角 + 6 方法 capstone |
| 15 | [`process-reward`](learning/process-reward/README.md) | PRM 训练 + BoN + MCTS + LLM-as-Judge |
| 16 | [`reasoning-r1`](learning/reasoning-r1/README.md) | ⭐ RL 系列高峰：GRPO / R1 / R1-Zero / Kimi k1.5 / TinyZero |
| 17 | [`rl-sota-2026`](learning/rl-sota-2026/README.md) | DAPO / VAPO / PRIME / Dr.GRPO 等 2025-2026 算法升级清单 |
| 18 | [`multimodal-agent`](learning/multimodal-agent/README.md) | **M4 毕业**：VLM-R1 / Vision-R1 / s1 / Safe-RLHF |

### Module 5 — 用大模型（推理部署，7 专题）

| # | 专题 | 一句话 |
|---|------|--------|
| 19 | [`inference-engine-core`](learning/inference-engine-core/README.md) | 手写 mini-vLLM：PagedAttention + continuous batching |
| 20 | [`sglang-radixattention`](learning/sglang-radixattention/README.md) | RadixAttention 前缀复用 + 结构化生成 |
| 21 | [`speculative-decoding`](learning/speculative-decoding/README.md) | 投机解码全谱：classic / Medusa / EAGLE |
| 22 | [`quantization-deploy`](learning/quantization-deploy/README.md) | INT8/GPTQ/AWQ/NF4/FP8/SmoothQuant 量化部署全谱 |
| 23 | [`distributed-inference`](learning/distributed-inference/README.md) | TP/PP/EP/分离式(disaggregated)推理 |
| 24 | [`production-serving`](learning/production-serving/README.md) | 生产部署：成本核算/监控/负载均衡/OpenAI 协议 |
| 25 | [`serving-graduation`](learning/serving-graduation/README.md) | **M5 毕业**：端到端 serving capstone |

### Module 6 — 评测/安全（7 专题）

| # | 专题 | 一句话 |
|---|------|--------|
| 26 | [`eval-foundations`](learning/eval-foundations/README.md) | MMLU/BBH/HELM 等经典 benchmark 评测基础 |
| 27 | [`reasoning-eval`](learning/reasoning-eval/README.md) | GSM8K/MATH/AIME/GPQA 推理 benchmark 深化 |
| 28 | [`agent-code-eval`](learning/agent-code-eval/README.md) | HumanEval/MBPP/SWE-bench 代码评测（真 exec 沙箱） |
| 29 | [`llm-judge-arena`](learning/llm-judge-arena/README.md) | LLM-as-Judge + Chatbot Arena Bradley-Terry/Elo |
| 30 | [`red-team-jailbreak`](learning/red-team-jailbreak/README.md) | GCG/PAIR/AutoDAN/Crescendo 红队攻击复现 |
| 31 | [`safety-defense`](learning/safety-defense/README.md) | Llama Guard/WildGuard/Constitutional Classifier 防御 |
| 32 | [`eval-graduation`](learning/eval-graduation/README.md) | **M6 毕业**：mini-HELM + mini-Arena + 红队/防御综合 |

### Module 7 — Agent 应用层（7 专题）

| # | 专题 | 一句话 |
|---|------|--------|
| 33 | [`agent-foundations`](learning/agent-foundations/README.md) | ReAct/Reflexion/Plan-Execute 等 Agent 基础范式 |
| 34 | [`rag-essential`](learning/rag-essential/README.md) | BM25/混合检索/GraphRAG/HippoRAG 检索增强全谱 |
| 35 | [`tool-use-mcp`](learning/tool-use-mcp/README.md) | Toolformer/Function Calling/MCP 协议化工具调用 |
| 36 | [`multi-agent-orchestration`](learning/multi-agent-orchestration/README.md) | CAMEL/AutoGen/CrewAI/MetaGPT 多 agent 编排 |
| 37 | [`agent-memory-context`](learning/agent-memory-context/README.md) | MemGPT/Mem0/长对话记忆与上下文管理 |
| 38 | [`agent-framework-stack`](learning/agent-framework-stack/README.md) | LangChain/LangGraph/PydanticAI 框架横评选型 |
| 39 | [`agent-graduation`](learning/agent-graduation/README.md) | **M7 毕业**：Deep Research Agent from scratch + Portfolio v2（39 题） |

### Module 8 — 系统与 Infra（GPU / Kernel / 集群，7 专题）

| # | 专题 | 一句话 |
|---|------|--------|
| 40 | [`gpu-architecture`](learning/gpu-architecture/README.md) | GPU 体系结构基础，Roofline 模型为核心 |
| 41 | [`cuda-essentials`](learning/cuda-essentials/README.md) | CUDA 执行模型与内存优化（Python 数值模拟） |
| 42 | [`kernel-engineering`](learning/kernel-engineering/README.md) | Triton/CUTLASS/FlashAttention 三家 kernel 工程实践 |
| 43 | [`cluster-networking`](learning/cluster-networking/README.md) | Fat-Tree/NCCL Collectives/SHARP 集群网络 |
| 44 | [`storage-dataops`](learning/storage-dataops/README.md) | 存储分层/Dataloader/Sharding/Checkpoint |
| 45 | [`training-orchestration`](learning/training-orchestration/README.md) | Slurm/Gang Scheduling/Ray Actor/Elastic Training |
| 46 | [`infra-graduation`](learning/infra-graduation/README.md) | **M8 毕业 + 全系列收官**：Mini-Cluster 模拟器 + TCO + Portfolio v3（46 题） |

> **⚠️ M8 的定位说明**：这 7 个专题名字带 "GPU"/"CUDA"/"kernel"/"cluster" 等词，但 `src/` 下全部是
> 用可断言验证的**纯 Python 数值/机制模拟**去复现这些系统的行为，不是可编译的真实 CUDA/Triton/CUTLASS
> kernel、也不是真实多机分布式训练代码——这是 Windows 工作站上的设计取舍（真实编译/多机需要
> Linux CUDA 工具链/真实多卡集群），每个模块的 README 页首都有对应澄清。`kernel-engineering` 额外带了
> 一个真实的 [Dao-AILab/flash-attention](learning/kernel-engineering/official/repos/flash-attention)
> 只读 git submodule 供源码阅读对照，不参与本仓库的运行验证范围。

---

## 每个专题的"七件套"解剖

46 个专题（数量不完全一致，取决于专题是否配 notebook/官方参考仓库）大体遵循同一套目录结构：

```
learning/<topic>/
├── README.md              # 专题主文档：概览/学习路径/环境/横向对比/cheatsheet/自测题/运行验证/跨专题衔接/验收清单
├── runbook.yaml            # 文档入口命令清单——记录"跑这个专题要执行哪些命令"，供审计 harness 消费
├── lectures/                # 讲义（中文），每篇对应一个知识点/子专题
├── paper/ 或 papers/         # 论文原文 PDF + 中文导读（guide_*.md）
├── notebooks/                # （部分专题有）交互式 notebook，端到端 walkthrough
├── environment/               # （部分专题有）requirements.txt + verify_env.py，专题级额外依赖声明
├── official/                   # （极少数专题有）只读参考：官方开源实现的 git submodule
└── src/
    ├── *.py                     # 可直接运行的实现，每个文件 `if __name__=="__main__"` 触发 `_self_test()`
    └── tests/                    # pytest 用例（部分专题的测试是脚本式聚合器，见下方说明）
```

**没有 `environment/` 目录的专题**（M8 全部 7 个 + 少数 M1-M7 专题）表示该专题 `src/` 下的代码只依赖
Python 标准库，不需要任何额外 `pip install`。

---

## 环境安装（`.venv`）

本仓库使用 repo-local 的 Windows 原生 `.venv`（不需要 WSL2；PyTorch 2.11 原生支持 CUDA）：

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install pytest numpy matplotlib pandas scipy einops transformers==5.10.2 tokenizers accelerate datasets peft sentencepiece tiktoken tqdm fastapi "uvicorn[standard]" pydantic sympy networkx seaborn tensorboard ipykernel jupyterlab
.\.venv\Scripts\python.exe -m pip install warcio trafilatura datasketch simhash gymnasium stable-baselines3 trl sentence-transformers prometheus-eval math-verify sse-starlette bitsandbytes
.\.venv\Scripts\python.exe -m pip install --force-reinstall pyarrow==21.0.0
```

> `pyarrow==21.0.0` 是钉死版本——`pyarrow==24.0.0` 在 `torch → transformers → datasets` 这个 import
> 顺序下会触发原生 access violation（详见 [`docs/local-env/ERIC-3080Ti-final-report.md`](docs/local-env/ERIC-3080Ti-final-report.md)）。

**未随主环境安装、按专题按需 opt-in 的重型栈**（`vllm`/`sglang`/`verl`/`ray`/`adapters` 等）：这些专题
的 `src/` 绝大多数是纯 Python 对这些框架内部机制的手写模拟，本地缺包不影响可跑性；`environment/
verify_env.py` 会诚实标注 `[SKIP]`，不会假装通过。

---

## 怎么运行 / 验证任意一个专题

统一入口是 [`scripts/eric_3080ti_env_audit.py`](scripts/eric_3080ti_env_audit.py)，每个专题在独立子
进程里跑，避免相互污染 import 路径：

```powershell
$env:PYTHONIOENCODING="utf-8"
$env:PYTHONUTF8="1"

# 跑某个专题文档里登记的全部入口命令（对应 runbook.yaml），做 V0(静态)+V1(smoke)验证
python scripts/eric_3080ti_env_audit.py --runbook --modules <topic-name>

# 同时验证多个专题
python scripts/eric_3080ti_env_audit.py --runbook --modules gpu-architecture cuda-essentials

# 跑某专题的 pytest 测试套件（V2）——脚本式聚合测试(无 test_ 前缀函数)会自动回退为直接跑脚本
python scripts/eric_3080ti_env_audit.py --modules <topic-name> --tests

# 跑某专题的 environment/verify_env.py 依赖自检
python scripts/eric_3080ti_env_audit.py --modules <topic-name> --env

# 按预置分组批量跑（A/B/C 是历史上按 GPU 需求/耗时划的批次，M8 是系统与Infra组）
python scripts/eric_3080ti_env_audit.py --tiers A B C M8 --tests --timeout 900
```

也可以完全不用 harness，直接照每个专题 README 的"运行验证（Runbook）"段落里给出的具体命令，逐条
`python learning/<topic>/src/xxx.py` 手动跑——每条命令的预期输出也写在同一段里。

**⚠️ 一个例外**：`infra-graduation` 的 `src/` 是包结构（`sim/`、`eval/` 子包），其中 5 个脚本用包内
绝对导入，不能像其它专题那样直接裸跑（需要 `PYTHONPATH`/`python -m`/harness 三选一），详见该专题
[README 的「环境配置」段](learning/infra-graduation/README.md#环境配置)。

---

## 目录导航

| 目录 | 内容 |
|------|------|
| [`learning/`](learning/) | 本文档地图覆盖的 46 个专题（本仓库主体） |
| [`docs/`](docs/) | 过程文档：`superpowers/`(brainstorming/plan 等设计与执行文档)、`local-env/`(环境搭建报告 + runbook 验证进度账本)、`paper-guides/`(论文导读撰写标准与整体进度追踪) |
| [`for_real_dummy/`](for_real_dummy/) | 一个大二本科生（本仓库的第二使用者）从零学习 `learning/` 这套博士级内容时记录的自学笔记；含 7 条独立"深挖系列"（numpy/python-advanced/torch/huggingface/tensorflow/python-idioms/rhcsa-bash），和 `learning/` 相互独立，互不依赖 |
| [`keynotes/`](keynotes/) | 会议 keynote 笔记归档 |
| [`interviews/`](interviews/) | 业界访谈/技术分享整理 |
| [`live_at_US/`](live_at_US/) | 与技术内容无关的个人生活记录 |
| [`runs/`](runs/) | 部分 RL 专题（如 `rl-foundations`）训练产生的运行日志/checkpoint 快照 |
| [`scripts/`](scripts/) | 仓库级工具脚本，核心是上面用到的 `eric_3080ti_env_audit.py` |

---

## 验证状态

全部 46 个专题的文档命令已完成 V0（静态检查）+ V1（逐命令 smoke 跑通）验证，多数专题额外过了 V2
（`src/tests/` 复核）。验证方法论、标准定义、每个专题的验证记录（含发现的 bug/文档漂移/修复方式）见：

- 验证标准与流程：[`docs/superpowers/specs/2026-06-20-runbook-verification-design.md`](docs/superpowers/specs/2026-06-20-runbook-verification-design.md)
- 活动进度账本（46 个专题逐条记录 + commit sha）：[`docs/local-env/ERIC-3080Ti-runbook-progress.md`](docs/local-env/ERIC-3080Ti-runbook-progress.md)
- 环境搭建与已知兼容性问题：[`docs/local-env/ERIC-3080Ti-final-report.md`](docs/local-env/ERIC-3080Ti-final-report.md)

**未覆盖范围（明确 YAGNI，非遗漏）**：424 个 notebook 的端到端执行（notebook 内容与对应 `src/` 脚本
逻辑重复，留作独立的后续 pass；spec 撰写时是 410 个，此处已用 `find learning/ -name "*.ipynb"` 重新
核实为当前实际数字）；capstone 的完整训练复现到真实收敛指标（太慢，仅按需 opt-in，本仓库验证的是
"脚本能正确跑通、数值符合公式"而非"真的练出 SOTA 模型"）。
