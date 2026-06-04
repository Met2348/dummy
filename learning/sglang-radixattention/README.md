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

## 运行

```powershell
# 测试（27/27 全绿）
python -c "import sys; sys.path.insert(0,'src'); sys.path.insert(0,'src/tests'); import test_radix_tree, test_constrained, test_frontend, test_agent_server"

# Capstone agent server
python src/agent_server.py
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
