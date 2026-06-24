# Topic 2: SGLang RadixAttention（agent 推理王）

> Module 5 「用大模型」第 2 专题 · 11 lectures · 11 notebooks · ~12h

## 总览

| Lecture | 主题 | 代码 |
|---------|------|------|
| L01 | SGLang 全图 vs vLLM | — |
| L02 | RadixAttention 概念 | — |
| L03 | Radix tree 实现 | `radix_tree.py` |
| L04 | Constrained decoding | `constrained_sampler.py` |
| L05 | Grammar FSM | `grammar_fsm.py` |
| L06 | Jump-Forward Decoding | `jump_forward.py` |
| L07 | Frontend Language (DSL) | `frontend_lang.py` |
| L08 | Agent patterns (ReAct/ToT/SC) | `agent_patterns.py` |
| L09 | Zero-Overhead Batch | — |
| L10 | SGLang vs vLLM 5 场景 | `sglang_compare.py` |
| L11 | **Capstone: 32-agent server** ⭐ | `agent_server.py` |

## Tags

- `sg-radix-tree` — L01-L03 radix tree 基础
- `sg-constrained` — L04-L06 constrained + grammar + jump-fwd
- `sg-frontend` — L07-L10 DSL + agents + bench
- `sglang` — 最终（含 Capstone + README）

## Capstone 实测

32 并发 ReAct agent，共享 SYSTEM_PROMPT (~2000 char) → radix hit_rate **91.7%**:

```json
{
  "n_agents": 32,
  "n_forwards": 160,
  "radix_hit_rate": 0.917,
  "forwards_per_agent": 5.0,
  "tool_calls": {"search": 32, "calc": 32}
}
```

## 与 Topic 1 (vLLM) 的关系

| 场景 | Topic 1 vLLM | Topic 2 SGLang |
|------|-------------|----------------|
| 单 prompt 大 batch | ★★★★★ | ★★★★ |
| ToT 8 路 | ★★ | **★★★★★** |
| JSON 结构化 | ★★★ | **★★★★★** |
| ReAct 5 步 | ★★ | **★★★★★** |
| 长 prompt 单请求 | ★★★★★ | ★★★★ |

## 环境

```powershell
python environment/verify_env.py
```

## 运行验证（Runbook）

文档入口命令集中在 [`runbook.yaml`](runbook.yaml)，一键验证（在 3080 Ti repo-local `.venv`）：

```bash
python scripts/eric_3080ti_env_audit.py --runbook --modules sglang-radixattention \
  --json-out docs/local-env/ERIC-3080Ti-runbook-results.json \
  --md-out  docs/local-env/ERIC-3080Ti-runbook-matrix.md
```

可跑入口（均 repo-root 相对；全部纯 CPU、秒级、无重型依赖；无 argparse → 直跑无参）：

| 入口 | 命令 | 说明 |
|------|------|------|
| radix tree | `python learning/sglang-radixattention/src/radix_tree.py` | L03 RadixAttention radix 树：前缀匹配 + segment 分裂 + refcount/LRU 淘汰 |
| grammar FSM | `python learning/sglang-radixattention/src/grammar_fsm.py` | L05 mini-FSM 编译 + token 表 mask |
| jump-forward | `python learning/sglang-radixattention/src/jump_forward.py` | L06 沿 FSM 唯一合法字符确定边跳过前向 |
| constrained sampler | `python learning/sglang-radixattention/src/constrained_sampler.py` | L04 对 logits 施加 FSM mask（非法 token→-inf） |
| frontend DSL | `python learning/sglang-radixattention/src/frontend_lang.py` | L07 Stream + gen/select/fork 原语 |
| agent patterns | `python learning/sglang-radixattention/src/agent_patterns.py` | L08 ReAct / Tree-of-Thought / Self-Consistency |
| paper-shaped minimal | `python learning/sglang-radixattention/src/sglang_original_minimal.py` | L02 radix 复用省 prefill + jump-forward（含 self-test） |
| SGLang vs vLLM | `python learning/sglang-radixattention/src/sglang_compare.py` | L10 5 场景**合成对照 mock**（不需 GPU；非真 benchmark） |
| **agent server (capstone)** | `python learning/sglang-radixattention/src/agent_server.py` | L11 32-agent 进程内并发模拟 + radix 命中统计 |

> 这些 demo 是纯 CPU / 纯数值演示，<1s 即出真实数字（radix matched/hit_rate / FSM mask / jump-forward 强制步数 / agent radix_hit_rate≈0.917），秒级 PASS 非 no-op。

**关键坑注记：**
- 本模块 **src 零 `import sglang`**（仅 docstring / 讲义里有 `sgl.` 字符串示例）。9 个脚本全部无依赖、离线可跑。
- `agent_server.py`（capstone）**不是真网络服务**，是**进程内 32-agent 并发模拟**：`run_server()` 返回 `AgentMetrics` 后即 `exit`，无挂起风险。生产替换：把 mock generator 换成 `sglang.RuntimeEndpoint` 客户端。
- `sglang_compare.py`：**诚实标注的合成对照 mock**（docstring 明言「No real engine called」，讲义 L10 标「mock benchmark（不需 GPU）」），不 import 任何重型库。**已修一处自相矛盾 bug（2026-06）**：原 `cost_vllm`/`cost_sglang` 公式完全相同 → 两列 cost 恒相等（fork 场景也不分裂），却另配一个硬编码 `estimated_sglang_gain` 显示 +83.3%——两列相等却报 83% 收益，自相矛盾。修正：`cost_vllm` 的 fork 分支改为 `fork_k·(prefix+suffix)`（vLLM 各 fork 独立重 prefill，与 `gain_pct` 分母本来就假设的一致），`sglang` 保持共享前缀只 prefill 一次；显示的 `sglang_prefill_gain` 现**纯由两列 cost 推出**（`1−s/v`，绝不再矛盾）；本 cost 模型未建模的其它优势（xgrammar/jump-forward/跨 step KV 复用）改成单列**定性说明**而非假百分比。现 `tot_8way` 真实显示 vllm 8400 vs sglang 1400 = +83.3%，fork_k==1 场景显示 +0.0%(前缀两引擎都缓存)。装真 `sglang`（Linux+CUDA）才能做真实 tok/s 横评。
- `jump_forward.py` 忠实于论文/讲义 L06 算法（「沿唯一合法字符确定边跳跃」），非占位：在 `{"name":"\d{4}}` 上正确强制前缀 9 字符后停在 `\d{4}` 分支。

**测试（V2）：**

```bash
python -m pytest learning/sglang-radixattention/src/tests -q   # 32 passed
```

## 关键文献

- SGLang (LMSys NeurIPS 2024)
- RadixAttention 论文章节
- xgrammar (NVIDIA 2024)
- Outlines (2023)
- Jump-forward decoding paper

## 一句话总结

> **vLLM 是 block-hash + paged。SGLang 是 trie + grammar fast path**。
> 任意 agent / 多轮 / 结构化场景，SGLang 是 first call。
