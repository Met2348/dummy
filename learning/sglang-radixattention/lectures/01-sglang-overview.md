# L01 · SGLang 全图 — 与 vLLM 的设计哲学差异

## 1 · 起源
- 2024.01 LMSys 团队（vLLM 同源），针对 vLLM 在 agent 场景的不足
- 论文：*Efficiently Programming Large Language Models using SGLang* (NeurIPS 2024)

## 2 · 三大创新
1. **RadixAttention**: KV cache 按 radix tree 共享（前缀树）
2. **Constrained Decoding**: regex / JSON schema / grammar 的 fast path
3. **Frontend Language**: `gen()` / `select()` / `fork()` DSL

## 3 · 与 vLLM 对比
| 维度 | vLLM | SGLang |
|------|------|--------|
| KV cache | block 哈希 | radix tree |
| 共享粒度 | block (16 token) | token (1) |
| 共享类型 | 完全相同前缀 | 任意公共前缀 |
| Constrained Decoding | outlines (慢) | xgrammar (快 10x) |
| Frontend | OpenAI API | DSL + OpenAI API |
| Agent | 良好 | **极佳** |
| 通用 chat | 极佳 | 良好 |

## 4 · 适用场景
| 场景 | 谁更好 |
|------|-------|
| 单 system prompt 大量请求 | 平手 |
| 多 system prompt | SGLang |
| tree-of-thought | SGLang |
| multi-turn agent | SGLang |
| JSON 结构化输出 | SGLang |
| 长 prompt 单请求 | vLLM 略胜 |

## 5 · 本专题路线图
- L02-L03 RadixAttention 数据结构 + 实现
- L04-L06 Constrained Decoding (FSM/jump-forward)
- L07-L08 Frontend + Agent patterns
- L09 zero-overhead batch
- L10 vs vLLM bench
- L11 Capstone：agent 推理服务

## 6 · 一句话
> SGLang = **PagedAttention 的 token 级版本 + grammar 加速 + 程序员友好的 DSL**。
