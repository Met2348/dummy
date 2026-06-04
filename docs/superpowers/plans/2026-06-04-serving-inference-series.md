# Module 5 · 用大模型 — 服务/推理深化 7 专题学习系列完整规划

> 设计日期：2026-06-04
> 学习仓库：`c:\Workspace\dummy`
> 系列定位：**造-改-用** 三角形的"用"轴 — Module 3 造大模型 + Module 4 改大模型完成后，本系列让训出来的模型真正**服务起来**
> 模板来源：Module 3 八专题 + Module 4 RL 七专题

---

## Context — 为什么开此系列

### 学习者当前坐标（2026-06-04）

- ✅ Module 1 **PEFT**（28 方法 / ~29h）：prompt-tuning / lora / adapter
- ✅ Module 2 **RL+对齐+推理**（88 方法 / ~101h）：rl-foundations → multimodal-agent
- ✅ Module 3 **造大模型**（118 方法 / ~122h）：data-curation → small-model-graduation
- ⏳ Module 5 **用大模型 · 服务/推理深化**：本次规划

### 为什么这是必修

PEFT + RL 教你"如何调模型"，造大模型教你"如何训模型"。但 2024-2026 LLM 工业落地的最大瓶颈不在算法，**而在推理服务**：

- vLLM (UC Berkeley 2023.06) PagedAttention → 24x 吞吐
- SGLang (LMSys 2024.01) RadixAttention + 编程接口 → agent 场景 5-10x
- EAGLE-2/3 (2024-2025) 投机解码 → 2-4x latency 降
- DeepSeek-V3 (2024.12) FP8 + DualPipe + Disaggregated Prefill/Decode → 推理 100k token/s
- TensorRT-LLM (NVIDIA 2024-2025)：Hopper FP8 + in-flight batching
- Triton + Ollama + LM Studio：从研究到 prod 全栈
- 商业 thinking models (Claude 4 / Gemini 2.5 / o3) 计费按 reasoning token，让推理优化变成钱的事

**不学这条线 → 训出的模型只能跑 demo 不能 serve**。Module 3+4 让你"造-改"，Module 5 让你"用"。

### 系列设计原则

1. **从骨架到生产**：每专题先从 minimal 推理引擎手写起，再对照 vLLM/SGLang/TRT-LLM 源码
2. **三轨代码（继承 Module 3）**：minimal（教学）/ 库（vLLM/SGLang/TRT-LLM）/ kernel（Triton 微 kernel）
3. **承上启下**：第 7 专题的 capstone 把 Module 1+2+3+5 串起来 — 拿一个 R1-tiny ckpt（专题 4 量化）+ FastAPI + 部署上线
4. **环境策略**：全程 WSL2（与 Module 3 后半 / Module 4 后半一致），可保持已有环境

### 输出物

- 本文件：7 专题整体蓝图（执行参考）
- 后续：7 份 `docs/superpowers/specs/2026-MM-DD-<topic>-design.md` + 7 份 `docs/superpowers/plans/2026-MM-DD-<topic>.md`
- 实施：7 个 `learning/<topic>/` 目录（lectures/ + src/ + notebooks/ + environment/ + tests/ + papers/）

---

## 一、7 专题总览

| # | 专题代号 | 一句话定位 | 方法数 | Lec | 时长 | 环境 | git tag |
|---|---------|----------|--------|-----|------|------|---------|
| 1 | `inference-engine-core` | vLLM 骨架 — PagedAttention + Continuous Batching + Chunked Prefill | 14 | 13 | 14h | WSL2 | `infer-engine` |
| 2 | `sglang-radixattention` | SGLang 骨架 — RadixAttention + Constrained Decoding + Frontend | 12 | 11 | 12h | WSL2 | `sglang` |
| 3 | `speculative-decoding` | EAGLE-2/3 + Medusa + Lookahead + Self-speculative | 13 | 12 | 13h | WSL2 | `spec-decode` |
| 4 | `quantization-deploy` | GPTQ + AWQ + FP8 + W4A16/W4A8 + KV cache 量化 | 14 | 13 | 14h | WSL2 | `quant-deploy` |
| 5 | `distributed-inference` | TP + PP + EP for MoE + Disaggregated Prefill/Decode | 13 | 12 | 14h | WSL2/多卡 | `distrib-infer` |
| 6 | `production-serving` | TensorRT-LLM + Triton + Ollama + LM Studio + 商业网关 | 12 | 12 | 12h | WSL2/容器 | `prod-serving` |
| 7 | `serving-graduation` | Agent 推理 + 五线综合毕业 capstone ⭐⭐⭐⭐⭐ | 14 | 14 | 13h | WSL2 | `用-graduation` |
| | **合计** | | **92** | **87** | **92h** | | |

对照 Module 3 (118 方法/122h) 和 Module 4 (88 方法/101h)，本系列体量约 **0.85×**，因服务/推理 lecture 偏短（每个 kernel 多代码少叙）。

### 依赖关系图

```
专题 1: 推理引擎核心  (WSL2, 14h)
        |
        ├──→ 专题 2: SGLang (WSL2, 12h)     \
        ├──→ 专题 3: 投机解码 (WSL2, 13h)    | 并行可能
        ├──→ 专题 4: 量化部署 (WSL2, 14h)    /
                              ↓
        专题 5: 分布式推理 (WSL2/多卡, 14h)
                              ↓
        专题 6: 生产部署 (WSL2/容器, 12h)
                              ↓
        专题 7: 五线综合毕业 (WSL2, 13h, 系列收官)
```

### 实施排期建议

| 月份 | 专题 | 备注 |
|------|------|-----|
| 2026-06 下 | 专题 1 推理引擎核心 | — |
| 2026-07 上 | 专题 2 SGLang | 与专题 3 可并行 |
| 2026-07 下 | 专题 3 投机解码 | — |
| 2026-08 上 | 专题 4 量化部署 | — |
| 2026-08 下 | 专题 5 分布式推理 | 多卡需求高 |
| 2026-09 上 | 专题 6 生产部署 | 容器/CI |
| 2026-09 下 | 专题 7 毕业 | 系列收官 |

共 4 个月。

---

## 二、专题 1：推理引擎核心（`inference-engine-core`）

### 定位
vLLM 是 2023 后 LLM 推理的事实工业标准。本专题完整还原其骨架：PagedAttention（解决 KV cache 碎片）+ Continuous Batching（解决 batch 同步）+ Chunked Prefill（prefill/decode 调度）。

### 章节规划（13 lectures）

| Lec | 主题 | 核心 idea |
|-----|------|---------|
| 01-llm-serving-overview.md | 推理服务全图 | prefill / decode / KV cache / latency 公式 |
| 02-naive-kv-cache.md | naive KV cache | static 分配 / fragmentation / waste 量化 |
| 03-paged-attention.md | **PagedAttention** (vLLM SOSP'23) | 虚拟内存类比 / block table / 物理 vs 逻辑块 |
| 04-paged-attention-kernel.md | PagedAttention CUDA kernel | gather + block load + naive Triton 实现 |
| 05-continuous-batching.md | **Continuous Batching** (Orca OSDI'22) | iter-level scheduling / dynamic admission |
| 06-chunked-prefill.md | Chunked Prefill (DeepSpeed-Inference 2024) | 长 prefill 切块 / decode 抢占 |
| 07-prefix-caching.md | Prefix Caching (vLLM 2024) | 系统 prompt 共享 / hash 命中率 |
| 08-scheduling-policies.md | 调度策略 | FCFS / SJF / priority / starvation 防 |
| 09-cuda-graphs.md | CUDA Graphs 加速 | decode-only graph capture / shape bucketing |
| 10-attention-backends.md | Attention backends | FlashAttention v2/v3 / XFormers / FlashInfer |
| 11-sampling-engines.md | Sampling | top-k / top-p / temperature / guided / Mirostat |
| 12-vllm-source-tour.md | vLLM 源码导读 | engine.py / scheduler.py / executor.py 关键路径 |
| 13-capstone-mini-vllm.md | Capstone：100 行 mini-vLLM | 手写 + 6 case 对照 vLLM throughput |

### src/ 三轨规划

| 文件 | 实现 |
|------|------|
| `common.py` | tokenizer wrap / request abstraction / metrics |
| `naive_kv.py` | static KV cache（基线）+ fragmentation 度量 |
| `paged_kv.py` | PagedAttention block table + alloc/free |
| `paged_attention_triton.py` | Triton 微 kernel for paged attention |
| `continuous_batching.py` | iter-level scheduler + admission control |
| `chunked_prefill.py` | prefill 切块 + decode 抢占 |
| `prefix_cache.py` | hash trie + LRU |
| `mini_vllm.py` | 100-200 行手写 mini-vLLM（capstone）|
| `vllm_compare.py` | 对照 vllm.LLM throughput / latency |
| `tests/test_paged_attention.py` | paged vs naive 数值一致 + 显存峰值降 |
| `tests/test_continuous_batching.py` | iter-level throughput 提升 ≥ 2x |
| `tests/test_mini_vllm.py` | 6 case 与 vLLM 一致 |

### Capstone：mini-vLLM 复刻
- **基座**：Qwen2.5-0.5B
- **任务**：6 个典型场景（短 in / 长 in / 短 out / 长 out / 大 batch / 流式）
- **指标**：吞吐 ≥ vLLM 50% + 显存峰值 ≤ vLLM 1.2×
- **耗时**：5090 24GB ≤ 2h

### 环境
```
torch>=2.5+cu130, transformers>=5.0
vllm>=0.7
flash-attn>=2.7
triton>=3.0
```

### 风险
- PagedAttention CUDA kernel 难写 → 提供 Triton 简化版 + 与 vllm._C 对照
- 0.5B 显存够用，1.5B 起出现 OOM → 教学限 0.5B
- Continuous batching 测时变量多 → 固定 seed + warmup 10 iter

### 退出条件
- [ ] 13 lecture + notebook 全跑通
- [ ] Capstone mini-vLLM 吞吐 ≥ vLLM 50%
- [ ] tag `infer-engine`

---

## 三、专题 2：SGLang RadixAttention（`sglang-radixattention`）

### 定位
SGLang (LMSys 2024.01) 在 agent 场景和 multi-turn 上常常击败 vLLM。核心是 **RadixAttention**（prefix 树共享）+ **Constrained Decoding**（grammar）+ **Frontend Language**（声明式 prompt）。

### 章节规划（11 lectures）

| Lec | 主题 | 核心 idea |
|-----|------|---------|
| 01-sglang-overview.md | SGLang vs vLLM | 设计哲学 / agent 适配性 |
| 02-radix-attention.md | **RadixAttention** | radix tree / share prefix between requests |
| 03-radix-tree-impl.md | radix tree 实现 | insert / match / evict / 引用计数 |
| 04-constrained-decoding.md | Constrained Decoding | regex / json schema / grammar mask |
| 05-grammar-fsm.md | Grammar FSM | finite state machine 转 token mask |
| 06-jump-forward.md | Jump-Forward Decoding | 强制 token 路径跳过 |
| 07-frontend-language.md | Frontend Language | gen / select / fork / parallel |
| 08-agent-patterns.md | Agent 模式 | tree of thought / multi-turn / tool routing |
| 09-zero-overhead-batch.md | Zero-Overhead Batch | CUDA Graph + bucketing |
| 10-sglang-vs-vllm-bench.md | SGLang vs vLLM 横向 | 5 场景 benchmark |
| 11-capstone-agent-server.md | Capstone：Agent 推理服务 | tree-of-thought + tool routing 全跑 |

### src/ 三轨规划

| 文件 | 实现 |
|------|------|
| `radix_tree.py` | radix tree 完整实现（insert/match/evict）|
| `radix_attention.py` | 与 KV cache 集成 |
| `grammar_fsm.py` | regex/json schema → FSM |
| `constrained_sampler.py` | token mask + sampling |
| `jump_forward.py` | 强制路径跳过 |
| `frontend_lang.py` | gen/select/fork mini DSL |
| `agent_patterns.py` | ToT / multi-turn 模板 |
| `sglang_compare.py` | 用 sglang 库跑同样场景对照 |
| `tests/test_radix_tree.py` | 共享率 / hit rate / 正确性 |
| `tests/test_constrained.py` | 100% JSON valid |

### Capstone：Agent 推理服务
- **场景**：ReAct agent + 4 个 tool（search/python/calc/weather）
- **指标**：32 并发，radix hit 率 ≥ 70%
- **耗时**：≤ 1.5h

### 环境
追加 `sglang>=0.4`

### 退出条件
- [ ] 11 lecture + notebook 全跑通
- [ ] Capstone radix hit 率 ≥ 70%
- [ ] tag `sglang`

---

## 四、专题 3：投机解码（`speculative-decoding`）

### 定位
2024-2025 推理加速最热方向。从经典 Speculative Decoding (Leviathan 2023) 到 EAGLE-2/3（2024-2025 SOTA），系统性覆盖所有变体。

### 章节规划（12 lectures）

| Lec | 主方法 | 团队 | 核心 idea |
|-----|--------|------|---------|
| 01-spec-decode-intuition.md | 直觉 | — | "猜测+验证" 范式 |
| 02-classic-spec-decode.md | Speculative Decoding (Leviathan 2023) | DeepMind | rejection sampling 正确性证明 |
| 03-medusa.md | Medusa (Cai 2024) | Princeton | 多 head 自投机 + tree attention |
| 04-eagle.md | EAGLE (Li 2024) | THU | feature-level 投机 |
| 05-eagle-2.md | **EAGLE-2** (Li 2024) | THU | dynamic draft tree |
| 06-eagle-3.md | **EAGLE-3** (Li 2025) | THU | multi-feature + scalable |
| 07-lookahead.md | Lookahead Decoding (2024) | UCSD | n-gram + Jacobi iteration |
| 08-self-spec-decode.md | Self-Speculative | — | skip layer 自投机 |
| 09-tree-attention.md | Tree Attention | — | 并行 verify 多 draft |
| 10-dynamic-tree.md | Dynamic Draft Tree | — | beam → tree 自适应 |
| 11-spec-eval.md | 评测方法 | — | accept rate / wall-clock speedup |
| 12-capstone-eagle3-deploy.md | Capstone：EAGLE-3 部署 | — | Qwen-1.5B + EAGLE-3 全跑 |

### src/ 三轨规划

| 文件 | 实现 |
|------|------|
| `classic_spec_decode.py` | 经典 Spec Decoding |
| `rejection_sampling.py` | 严格 rejection 实现（带数学证明）|
| `medusa_heads.py` | Medusa 多 head |
| `eagle_minimal.py` | EAGLE feature-level minimal |
| `eagle2.py` | EAGLE-2 dynamic tree |
| `lookahead.py` | Lookahead n-gram |
| `self_spec.py` | self-speculative (skip layer) |
| `tree_attention.py` | tree attention kernel (Triton) |
| `spec_eval.py` | accept rate / speedup 量化 |
| `tests/test_rejection_correctness.py` | 投机解码 vs greedy 分布一致 |
| `tests/test_speedup.py` | EAGLE-2 加速 ≥ 1.5x |

### Capstone：EAGLE-3 部署
- **基座**：Qwen2.5-1.5B
- **draft model**：自训小 head
- **指标**：accept rate ≥ 4 token/iter，wall-clock 加速 ≥ 2x
- **耗时**：≤ 4h（含训 head）

### 环境
追加 `eagle-llm` (官方) 备份手写

### 退出条件
- [ ] 12 lecture + notebook 跑通
- [ ] EAGLE-2 加速 ≥ 1.5x
- [ ] tag `spec-decode`

---

## 五、专题 4：量化部署（`quantization-deploy`）

### 定位
权重/激活/KV cache 量化全谱。是推理服务"省显存"的核心 lever — int4 模型可比 fp16 省 4x，FP8 在 Hopper 上吞吐翻倍。

### 章节规划（13 lectures）

| Lec | 主方法 | 团队 | 核心 idea |
|-----|--------|------|---------|
| 01-quant-overview.md | 量化全图 | — | weight/activation/KV/format/算子 |
| 02-int8-basics.md | int8 基础 | — | min-max / per-tensor / per-channel |
| 03-gptq.md | **GPTQ** (Frantar 2023) | IST Austria | 二阶 hessian + greedy column |
| 04-awq.md | **AWQ** (Lin 2024) | MIT | 激活感知 + salient channel 保护 |
| 05-smooth-quant.md | SmoothQuant (Xiao 2022) | — | activation outlier → weight |
| 06-llm-int8.md | LLM.int8() (Dettmers 2022) | — | outlier 分离 + mixed precision |
| 07-fp8-format.md | FP8 (E4M3/E5M2) | NVIDIA | Hopper FP8 硬件支持 |
| 08-fp8-training.md | FP8 训练（回顾 Module 3） | — | DeepSeek-V3 FP8 训推一致 |
| 09-w4a16.md | W4A16 (bitsandbytes) | — | 主流推理量化 |
| 10-w4a8.md | W4A8 | — | 激活也量化 |
| 11-kv-cache-quant.md | KV cache 量化 | — | int8 KV / 显存降 2x |
| 12-quant-eval.md | 量化评测 | — | PPL / MMLU / 实测 speedup |
| 13-capstone-quant-zoo.md | Capstone：量化动物园 | — | Qwen-1.5B × 6 quant 对照 |

### src/ 三轨规划

| 文件 | 实现 |
|------|------|
| `int8_basics.py` | min-max / per-channel demo |
| `gptq_minimal.py` | 手写 GPTQ 主循环 |
| `gptq_lib.py` | auto-gptq 对照 |
| `awq_minimal.py` | 手写 AWQ |
| `awq_lib.py` | autoawq 对照 |
| `smooth_quant.py` | activation 平滑 |
| `fp8_demo.py` | torch FP8 demo（cu130+）|
| `bnb_int4.py` | bitsandbytes 4bit 加载 |
| `kv_quant.py` | KV cache int8 |
| `quant_eval.py` | PPL / MMLU / speedup 量表 |
| `tests/test_gptq_acc.py` | GPTQ 后 PPL 差 < 5% |
| `tests/test_awq_acc.py` | AWQ 后 MMLU 差 < 2pp |

### Capstone：量化动物园
- **基座**：Qwen2.5-1.5B
- **6 个版本**：fp16 / int8 / GPTQ-4bit / AWQ-4bit / FP8 / W4A8
- **指标**：PPL / MMLU / 显存 / 吞吐 6 列表
- **耗时**：≤ 3h

### 环境
追加 `auto-gptq, autoawq, bitsandbytes>=0.43`

### 退出条件
- [ ] 13 lecture + notebook 跑通
- [ ] 量化动物园 6 版本表完成
- [ ] tag `quant-deploy`

---

## 六、专题 5：分布式推理（`distributed-inference`）

### 定位
单卡装不下 7B+ 模型时必须切片。本专题 TP + PP + EP（for MoE）+ Disaggregated（vLLM 0.7 / Mooncake）系统性覆盖。

### 章节规划（12 lectures）

| Lec | 主方法 | 核心 idea |
|-----|--------|---------|
| 01-distrib-overview.md | 分布式推理全图 | TP / PP / EP / DP 何时用 |
| 02-tensor-parallel.md | TP for inference | column/row split / all-reduce |
| 03-tp-megatron-style.md | Megatron-LM 风格 TP | attention/MLP 切法 |
| 04-pipeline-parallel.md | PP for inference | micro-batch / bubble |
| 05-1f1b-schedule.md | 1F1B schedule | bubble 减少 |
| 06-expert-parallel.md | EP for MoE | DeepSeek-V3 EP 256 / cross-node |
| 07-all-to-all.md | All-to-All 通信 | NCCL all-to-all / 带宽瓶颈 |
| 08-disaggregated.md | **Disaggregated Prefill/Decode** | DistServe / Mooncake / vLLM 0.7 |
| 09-kv-transfer.md | KV cache 跨节点传输 | NVLink / RDMA / NVSHMEM |
| 10-prefix-aware-routing.md | Prefix-aware Routing | SGLang/Mooncake routing 策略 |
| 11-multi-node-deploy.md | 多节点部署 | Ray Serve / Kuberay |
| 12-capstone-disagg-serve.md | Capstone：Disaggregated 部署 | 2 GPU 模拟 P/D 分离 |

### src/ 规划

| 文件 | 实现 |
|------|------|
| `tp_demo.py` | column/row split TP |
| `pp_demo.py` | naive PP + 1F1B |
| `ep_demo.py` | MoE expert parallel + all-to-all |
| `disaggregated_mock.py` | P/D 分离 mock（单机模拟）|
| `kv_transfer_mock.py` | KV cache 序列化 / 传输 |
| `routing_policies.py` | round-robin / prefix-aware / load-aware |
| `tests/test_tp_correctness.py` | TP vs single GPU 数值一致 |
| `tests/test_disagg.py` | P/D 分离吞吐 ≥ 单实例 1.3x |

### Capstone：Disaggregated 部署
- **场景**：单机 2 GPU 模拟（1 prefill + 1 decode）
- **基座**：Qwen2.5-1.5B
- **指标**：长 prompt 场景吞吐提升 ≥ 1.3x
- **耗时**：≤ 3h

### 环境
追加 `ray[serve]`, `nvidia-nccl-cu12`

### 退出条件
- [ ] 12 lecture + notebook 跑通
- [ ] Disaggregated 演示 ≥ 1.3x 提升
- [ ] tag `distrib-infer`

---

## 七、专题 6：生产部署（`production-serving`）

### 定位
从研究 demo 到工业服务。覆盖 TensorRT-LLM（NVIDIA 官方栈）+ Triton Server + Ollama + LM Studio + 商业 API 网关接入。

### 章节规划（12 lectures）

| Lec | 主题 | 核心 idea |
|-----|------|---------|
| 01-prod-overview.md | 生产部署全图 | latency SLO / availability / 监控 |
| 02-tensorrt-llm.md | **TensorRT-LLM** | NVIDIA 官方栈 / Hopper FP8 |
| 03-trtllm-build.md | TRT-LLM build | engine 编译 / in-flight batching |
| 04-triton-server.md | **Triton Inference Server** | model repo / ensemble / Python backend |
| 05-ollama.md | Ollama | gguf / modelfile / 端侧部署 |
| 06-llama-cpp.md | llama.cpp + gguf | CPU/Metal/CUDA backend |
| 07-lm-studio.md | LM Studio | Mac/Win 端侧 GUI |
| 08-openai-api-spec.md | OpenAI-compatible API | chat/completions/embeddings 规范 |
| 09-fastapi-wrap.md | FastAPI wrap | uvicorn + 流式 + SSE |
| 10-monitoring.md | 生产监控 | Prometheus / Grafana / latency p50/p99 |
| 11-cost-engineering.md | 成本工程 | $/M token / cache 命中 / batch 收益 |
| 12-capstone-prod-stack.md | Capstone：生产栈 | TRT-LLM + Triton + 监控 全跑通 |

### src/ 规划

| 文件 | 实现 |
|------|------|
| `trtllm_build.py` | TRT-LLM engine 构建脚本模板 |
| `triton_model_repo/` | Triton model_repository 完整示例 |
| `ollama_modelfile/` | Modelfile + ollama serve |
| `openai_api_server.py` | FastAPI 实现 OpenAI 兼容 API |
| `streaming_sse.py` | SSE 流式输出 |
| `metrics_prometheus.py` | latency/throughput Prometheus exporter |
| `cost_calc.py` | $/M token 计算器 |
| `tests/test_openai_compat.py` | 与 openai-python client 兼容 |

### Capstone：生产栈完整部署
- **栈**：TRT-LLM build → Triton serve → FastAPI proxy → Prometheus 监控
- **指标**：p50 < 500ms，p99 < 2s，100 QPS 稳定
- **耗时**：≤ 4h

### 环境
追加 `tensorrt-llm` (NVIDIA pip), `tritonclient`, `fastapi`, `uvicorn[standard]`, `prometheus-client`

### 退出条件
- [ ] 12 lecture + notebook 跑通
- [ ] Capstone 生产栈 100 QPS 稳定
- [ ] tag `prod-serving`

---

## 八、专题 7：用大模型毕业 capstone（`serving-graduation`）⭐⭐⭐⭐⭐

### 定位
**系列毕业专题**。覆盖 Agent 推理特化场景 + **五线综合 capstone**（造-改-用闭环）：拿 Module 3 训出的 Phi-tiny → Module 4 R1-Zero ckpt → Module 5 量化部署 → FastAPI 上线。

### 章节规划（14 lectures，主线 11 + capstone 3）

| Lec | 主题 | 核心 idea |
|-----|------|---------|
| 01-agent-inference.md | Agent 推理特化 | prefix share / tool routing / multi-step |
| 02-thinking-budget.md | Thinking budget | Claude/Gemini 2.5 thinking_budget 设计 |
| 03-reasoning-cache.md | Reasoning cache | reasoning trace 复用 |
| 04-long-ctx-serving.md | 长上下文服务 | 100k+ KV / sliding / RoPE-scaling 服务化 |
| 05-multi-model-routing.md | 多模型路由 | small fast / large smart / router model |
| 06-batch-vs-online.md | Batch vs Online | 离线批 / 在线低延迟 / 不同 SLO |
| 07-vlm-serving.md | VLM 推理 | image encoder offload / vision-text 异步 |
| 08-embedding-serving.md | Embedding 服务 | batch encode / FAISS 集成 |
| 09-cold-start.md | 冷启动 | model load / warmup / canary |
| 10-fault-tolerance.md | 容错 | retry / fallback / circuit breaker |
| 11-serving-takeaway.md | 服务工程 takeaway | 5 个原则 |
| 12-capstone-r1-tiny-deploy.md | Capstone-1：R1-tiny 部署 | 拿 Module 4 ckpt 量化部署 |
| 13-five-line-graduation.md | ⭐⭐⭐ **五线综合 lecture** | 造-改-用 闭环 |
| 14-capstone-graduation.md | Capstone-2：毕业作品 | 18 专题串联，端到端 demo |

### src/ 规划

| 文件 | 实现 |
|------|------|
| `agent_inference_demo.py` | 多 step agent + radix cache |
| `thinking_budget.py` | budget forcing / early stop |
| `multi_model_router.py` | small/large router |
| `vlm_serve.py` | VLM 推理服务模板 |
| `embedding_serve.py` | embedding batch encode |
| `r1_tiny_deploy/` | Module 4 R1-Zero ckpt → AWQ-4bit → FastAPI |
| `graduation_e2e/` | 18 专题端到端串联 demo |
| `tests/test_e2e.py` | 端到端验证 |

### Capstone-1：R1-tiny 部署
- **基座**：Module 4 reasoning-r1 capstone-A 训出的 GPT-2-M R1-Zero ckpt
- **流程**：AWQ 4bit 量化 → vLLM serve → FastAPI 包装 → 流式输出
- **指标**：p50 < 1s，可流式输出推理过程
- **耗时**：≤ 3h

### Capstone-2：五线综合毕业作品 ⭐⭐⭐⭐⭐⭐
- **同一道 GSM8K 题**
- **5 个推理路径**：
  1. **Vanilla** GPT-2 base 直接 vLLM 服务
  2. **LoRA** 微调版（Module 1 ckpt）vLLM 服务
  3. **DPO** 对齐版（Module 4 专题 3 ckpt）vLLM 服务
  4. **R1-Zero** 推理版（Module 4 专题 5 ckpt）vLLM 服务
  5. **Phi-tiny 270M**（Module 3 专题 7 训出）vLLM 服务
- **可视化**：5 个服务并存的 grafana 看板 + 同问 5 答对照
- **输出**：notebook 完整展示「我从 PEFT 到部署的全程作品」

### 五线综合 Lecture (L13) 详细大纲

90 min，30 slides：

**Part I：18 专题地图回顾 (20 min, 6 slides)**
- 5 大 module / 18 专题 / 200+ 方法
- 「造-改-用」三角形构图

**Part II：服务视角的统一公式 (30 min, 10 slides)**
- 任意 LLM 服务 = `Engine(Model, Quant, Schedule, KV, Spec) → API`
- 5 个 lever 各贡献多少 throughput / latency / cost
- 真实案例：DeepSeek-V3 100k tok/s / Claude Sonnet 价格分析

**Part III：选型决策树 (25 min, 8 slides)**
- 4 个真实场景：单用户 chatbot / 大规模批 / agent / VLM
- 给出选型表（vLLM / SGLang / TRT-LLM / Ollama）

**Part IV：5 大 module 闭环 + 下一程 (15 min, 6 slides)**
- 「造」要服务长 ctx → 「改」要训出可量化模型 → 「用」要服务多 ckpt
- 下一程：Eval/Safety/Agent 应用层 / LLMOps

### 环境
继承前 6 专题环境

### 退出条件
- [ ] 14 lecture + notebook 跑通
- [ ] Capstone-1 R1-tiny p50 < 1s
- [ ] Capstone-2 五线综合 demo 完成
- [ ] tag `用-graduation` ⭐ 系列收官

---

## 九、跨专题工程策略

### 三轨代码策略（汇总）

| 专题 | minimal | 库 | kernel |
|------|---------|-----|--------|
| 1 推理引擎 | 100 行 mini-vLLM | **vLLM** | Triton paged attn |
| 2 SGLang | radix tree + grammar FSM | **sglang** | tree attn |
| 3 投机解码 | 手写 EAGLE / Medusa | sglang/vllm spec | tree attn kernel |
| 4 量化部署 | 手写 GPTQ / AWQ | auto-gptq / autoawq / bnb | FP8 demo |
| 5 分布式 | TP/PP/EP mock | vllm TP / ray | NCCL |
| 6 生产 | FastAPI 包装 | **TRT-LLM / Triton** | — |
| 7 毕业 | 端到端 e2e | 5 service grafana | — |

### 环境策略

全程 **WSL2**（沿用 Module 3+4 后半 / Module 4）。原因：
- vLLM/SGLang/TRT-LLM Windows native 全装不上
- 量化库 auto-gptq/autoawq Windows 兼容差
- Triton kernel 需要 cuda toolkit + Linux 环境
- 与 Module 4 后半完美衔接，可保持 conda env

### 一致性测试新定义（区别于 Module 3 的 numerical）

| 测试类型 | 标准 | 示例 |
|---------|------|------|
| **正确性** | 分布一致 (KL < 1e-3) | Spec decode greedy vs no-spec |
| **数值** | 量化后 PPL diff < 5% | GPTQ / AWQ |
| **性能** | 吞吐/延迟达标 | mini-vLLM ≥ vLLM 50% |
| **稳定性** | 100 QPS 30min 无 OOM | 生产栈 |

### Git 里程碑

| Tag | 时机 | 内容 |
|-----|------|------|
| `infer-engine` | 专题 1 末 | mini-vLLM + vLLM 对照 |
| `sglang` | 专题 2 末 | RadixAttention + Constrained |
| `spec-decode` | 专题 3 末 | EAGLE-2/3 + Medusa |
| `quant-deploy` | 专题 4 末 | 量化动物园 |
| `distrib-infer` | 专题 5 末 | TP/PP/EP/Disaggregated |
| `prod-serving` | 专题 6 末 | TRT-LLM + Triton + 监控 |
| `用-graduation` | 专题 7 末 | ⭐⭐⭐ 五线综合毕业作品 |

---

## 十、2025-2026 高影响力方法补充清单

### 推理引擎（2024-2026）
| 方法 | 团队/时间 | 影响 | 一句话 |
|------|----------|------|--------|
| **PagedAttention / vLLM** | UCB SOSP'23 | ⭐⭐⭐⭐⭐ | 行业标准 |
| **Continuous Batching / Orca** | OSDI'22 | ⭐⭐⭐⭐⭐ | 迭代级调度 |
| **Chunked Prefill** | DeepSpeed 2024 | ⭐⭐⭐⭐ | prefill/decode 互不干扰 |
| **Prefix Caching** | vLLM 2024 | ⭐⭐⭐⭐ | 系统 prompt 共享 |
| **RadixAttention / SGLang** | LMSys 2024.01 | ⭐⭐⭐⭐⭐ | agent 场景王 |
| **FlashAttention v3** | Tri Dao 2024.07 | ⭐⭐⭐⭐⭐ | Hopper 优化 |
| **FlashInfer** | 2024 | ⭐⭐⭐⭐ | inference-specific attn 库 |

### 投机解码（2024-2025）
| 方法 | 团队/时间 | 影响 | 一句话 |
|------|----------|------|--------|
| **EAGLE-2** | THU 2024 | ⭐⭐⭐⭐⭐ | dynamic draft tree SOTA |
| **EAGLE-3** | THU 2025 | ⭐⭐⭐⭐⭐ | multi-feature scalable |
| **Medusa** | Princeton 2024 | ⭐⭐⭐⭐ | 多 head self-speculative |
| **Lookahead** | UCSD 2024 | ⭐⭐⭐ | Jacobi iteration |
| **SpecDec++** / SpecExec | 2024 | ⭐⭐⭐ | tree expansion 策略 |

### 量化（2024-2025）
| 方法 | 团队/时间 | 影响 | 一句话 |
|------|----------|------|--------|
| **GPTQ** | IST Austria 2023 | ⭐⭐⭐⭐⭐ | 二阶 hessian |
| **AWQ** | MIT 2024 | ⭐⭐⭐⭐⭐ | 激活感知 |
| **SmoothQuant** | Han et al 2022 | ⭐⭐⭐⭐ | activation→weight 平滑 |
| **FP8** (E4M3/E5M2) | NVIDIA 2023+ | ⭐⭐⭐⭐⭐ | Hopper 硬件 |
| **W4A8** | 2024 | ⭐⭐⭐⭐ | activation 也量化 |
| **KV int8/int4** | 2024 | ⭐⭐⭐⭐ | 显存 2-4x 降 |

### 分布式推理（2024-2025）
| 方法 | 团队/时间 | 影响 | 一句话 |
|------|----------|------|--------|
| **DistServe** | PKU OSDI'24 | ⭐⭐⭐⭐⭐ | P/D 分离开山 |
| **Mooncake** | Moonshot 2024 | ⭐⭐⭐⭐⭐ | 商业 P/D 分离 |
| **vLLM 0.7 Disagg** | vLLM 2025 | ⭐⭐⭐⭐ | 开源 P/D |
| **DeepSeek-V3 EP** | DeepSeek 2024.12 | ⭐⭐⭐⭐⭐ | EP-256 cross-node |
| **DualPipe** | DeepSeek 2024.12 | ⭐⭐⭐⭐⭐ | 训推一体调度 |

### 生产部署（2024-2025）
| 方法 | 团队/时间 | 影响 | 一句话 |
|------|----------|------|--------|
| **TensorRT-LLM** | NVIDIA 2023-2025 | ⭐⭐⭐⭐⭐ | 官方栈 |
| **Triton Inference Server** | NVIDIA | ⭐⭐⭐⭐ | 多模型 serving |
| **Ollama** | 2024 | ⭐⭐⭐⭐⭐ | 端侧部署王 |
| **llama.cpp / gguf** | 2023+ | ⭐⭐⭐⭐⭐ | CPU/Metal 必备 |
| **vLLM OpenAI API** | 2024 | ⭐⭐⭐⭐⭐ | de facto 标准 |

### Agent/Long-ctx 推理（2025）
| 方法 | 团队/时间 | 影响 | 一句话 |
|------|----------|------|--------|
| **Claude Extended Thinking** | Anthropic 2025 | ⭐⭐⭐⭐⭐ | thinking budget API |
| **Gemini 2.5 thinking_budget** | Google 2025 | ⭐⭐⭐⭐⭐ | 显式预算 |
| **OpenAI Realtime API** | OpenAI 2024 | ⭐⭐⭐⭐ | 流式低延迟 |
| **CRAG / Self-RAG serving** | 2024-2025 | ⭐⭐⭐ | RAG-aware 推理 |

---

## 十一、风险总览（系列级）

| 风险 | 缓解 |
|------|------|
| WSL2 NCCL 多节点装不上 | 单机多卡 mock + Docker 备份 |
| TRT-LLM 安装 ≥ 30GB | 提供 ngc 预编译 image |
| 投机解码 draft model 训费时 | 提供预训好 ckpt |
| 量化精度损失评估难 | 用统一 PPL/MMLU 子集 |
| 24 GB 5090 装不下 7B | 全程限 0.5B-1.5B + 4bit |
| 监控栈复杂 | docker-compose 一键起 |

---

## 十二、复用模板说明

继承 Module 3 / Module 4 同款目录结构（参考 [scaling-infra README](learning/scaling-infra/README.md)）：

```
learning/<topic>/
├── README.md
├── environment/
│   ├── requirements.txt
│   └── verify_env.py
├── papers/
├── lectures/
│   └── NN-topic-name.md
├── src/
│   ├── common.py
│   ├── <method>_minimal.py
│   ├── <method>_lib.py
│   └── tests/test_<method>.py
└── notebooks/
    └── NN-topic-name.ipynb
```

文件命名规范同 Module 3 / 4。

---

## 十三、验证方法（end-to-end）

```powershell
# 每专题完成后
python learning/<topic>/environment/verify_env.py
python -m pytest learning/<topic>/src/tests/ -v

# 系列收官
python learning/serving-graduation/src/graduation_e2e/run.py
# 预期：5 个 service grafana 看板 + 5 答对照可视化
```

---

## 十四、关键文件路径速查

### 现有模板
- Module 3 全图：`docs/superpowers/plans/2026-06-04-pretraining-architecture-series.md`
- Module 4 全图：`C:\Users\ericp\.claude\plans\partitioned-squishing-stream.md`
- 最近毕业 capstone 参考：`learning/small-model-graduation/`

### 即将新建
- `c:\Workspace\dummy\learning\inference-engine-core\`
- `c:\Workspace\dummy\learning\sglang-radixattention\`
- `c:\Workspace\dummy\learning\speculative-decoding\`
- `c:\Workspace\dummy\learning\quantization-deploy\`
- `c:\Workspace\dummy\learning\distributed-inference\`
- `c:\Workspace\dummy\learning\production-serving\`
- `c:\Workspace\dummy\learning\serving-graduation\`

---

## 十五、本次任务的"下一步"

按 Module 3 / 4 同款节奏：

1. **commit 本蓝图** → 进入专题 1 inference-engine-core
2. 每专题：scaffold → lectures 分 phase → src 三轨 → tests → notebooks → README → tag
3. 全 7 专题完成后 → `用-graduation` 系列收官 → 报告整体进度

按用户「一口气分步一个个地执行完 不需要反复批准」精神 — 本规划写完后直接推进专题 1，不再单独 ask。
