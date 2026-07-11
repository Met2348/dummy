# Topic 5: Agent Memory & Context（记忆 + 上下文）

> Module 7 第 5 专题 · 12 lectures · ~12h
>
> Letta (MemGPT) / Mem0 / 4 memory 类 / 上下文管理 / prompt cache

## 总览

| Lecture | 主题 | 代码 |
|---------|------|------|
| L01 | Memory 分类 (working / episodic / semantic / procedural) | (intro) |
| L02 | **Letta** (前 MemGPT) — OS-inspired paging | `letta_mock.py` + `memgpt_virtual_context.py`（MemGPT 论文 virtual context 补充实现：warning/flush/recursive-summary/nested-KV chaining，见 `paper/guide_01_memgpt.md`） |
| L03 | **Mem0** — extract / update / retrieve | `mem0_mock.py` |
| L04 | Vector DB 选型 (Pinecone/Weaviate/Chroma/Qdrant/Milvus/pgvector) | `vector_store.py` |
| L05 | **Episodic memory** | `episodic_memory.py` |
| L06 | **Semantic memory** + KG | `semantic_memory.py` |
| L07 | Context mgmt (summary / pruning / compression) | `context_mgmt.py` |
| L08 | **Prompt caching** (Anthropic / OpenAI) | `prompt_cache.py` |
| L09 | KV-cache mgmt + PagedAttention | `kv_cache_mock.py` |
| L10 | 长对话策略 | `long_conv.py` |
| L11 | 个性化 memory | `personalization.py` |
| L12 | **Capstone**: 3-layer memory chatbot | `capstone_memory_chat.py` |

## Tags

- `agent-memory` — Module 7 第 5 专题

## 运行验证（Runbook）

> 本段命令即 [`runbook.yaml`](runbook.yaml) 登记的"文档入口命令"，已在 ERIC-3080Ti（RTX 3080 Ti 16GB）上 V0+V1 验证通过（13/13，纯 CPU 秒级）。
> 一键复验本模块：
> ```powershell
> python scripts/eric_3080ti_env_audit.py --runbook --modules agent-memory-context
> ```

13 个脚本全是**手写 memory 数据结构 mock**（blake2b 哈希 embedding / KG triples / PagedAttention 风格 KV block /
prompt cache TTL 等，零外部依赖、纯 stdlib、纯 CPU）。每个直跑都会执行内置 `_self_test()`（真断言，非 print-only）。
直接 `python <脚本>` 即可（脚本无 argparse；harness 会自动把 `src/` 加进 `PYTHONPATH`，且 Python 本身也会把脚本
所在目录插入 `sys.path[0]`，故脱离 harness 单独跑也不依赖 CWD）：

```powershell
# 共享后端（blake2b 稳定哈希 embedding / cosine / mock_now / tokenize）
python learning/agent-memory-context/src/common.py
# L02 Letta(前 MemGPT)：main context(core+recent) + archival memory 溢出转存
python learning/agent-memory-context/src/letta_mock.py
# L02 补充：MemGPT 论文 virtual context toy（warning/flush/recursive-summary + nested-KV chaining + pagination）
python learning/agent-memory-context/src/memgpt_virtual_context.py
# L03 Mem0：extract_facts 正则抽取 + decide_action ADD/UPDATE/NONE
python learning/agent-memory-context/src/mem0_mock.py
# L04 向量库：upsert/search/delete + metadata filter_fn
python learning/agent-memory-context/src/vector_store.py
# L05 Episodic memory：带时间戳事件存储 + user/time-range 过滤检索
python learning/agent-memory-context/src/episodic_memory.py
# L06 Semantic memory：KG triples add/update + multi-hop 查询
python learning/agent-memory-context/src/semantic_memory.py
# L07 Context 管理：sliding-window / importance-prune / rolling-summary / rag-history
python learning/agent-memory-context/src/context_mgmt.py
# L08 Prompt caching：TTL 缓存命中/未命中 + Anthropic 定价 cost_estimate
python learning/agent-memory-context/src/prompt_cache.py
# L09 KV-cache：PagedAttention 风格 block 分配 + share_prefix 前缀共享 + free_seq 回收
python learning/agent-memory-context/src/kv_cache_mock.py
# L10 长对话：4 层架构（recent + rolling-summary + core-facts）
python learning/agent-memory-context/src/long_conv.py
# L11 个性化 memory：偏好/知识水平/风格档案 + 点赞点踩反馈更新
python learning/agent-memory-context/src/personalization.py
```

**Capstone（L12）：3-layer memory chatbot（10-turn 对话，Turn 10 从 semantic profile 召回 Turn 1 偏好）**

```powershell
python learning/agent-memory-context/src/capstone_memory_chat.py
```

> 直跑先打印 `_self_test` 断言（Turn 10 从 turn 1 的 semantic profile 召回 "Anthropic"），再打印完整 markdown 报告。
> （历史版本用过 CWD 依赖的 `python -c "import sys; sys.path.insert(0,'learning/agent-memory-context/src'); ..."`
> 一行流；已改为直接脚本调用，效果等价但不再依赖"当前目录=repo-root"这个隐藏前提。）

**关键坑注记**

- 全模块 embedding 都是 `common.hash_embed`：用 `hashlib.blake2b` 逐 token 稳定哈希分桶再归一化，**跨进程/跨
  运行确定性一致**（不像 `rag-essential` 的 `common.hash_embed` 用内置 `hash()`，受 `PYTHONHASHSEED` 逐进程加盐
  影响而有 ±0.05~0.1 浮动）——这是本模块的设计亮点，非巧合。
- capstone 的 Turn 10 recall **真的**从 `semantic_memory`/`UserProfile.preferences` 里查出来（`_mock_answer`
  命中 `"preferred" in text and "llm" in text` 分支才返回 `profile.preferences["llm"]`），不是硬编码答案；
  `_self_test` 显式断言 `recall_source == "semantic_profile"`，而非只看 verdict 字符串。
- `kv_cache_mock.share_prefix` 和 `prompt_cache.cost_estimate` 都是**真计算**：前者把 `seq_a` 已分配的 block id
  原样并入 `seq_b`（不重新分配，验证 PagedAttention 前缀共享不重复占用显存）；后者的 88.9% 省成本数字由
  `write_price`/`cached_read_price`/`regular_price` 三个参数实算得出（自检 `assert savings_pct > 80`），
  不是写死的百分比。
- `memgpt_virtual_context.py` 是为 `paper/guide_01_memgpt.md` 补的 toy 实现，专门覆盖 `letta_mock.py` 未展开的
  MemGPT 论文机制（70% warning / 100% flush / recursive summary / heartbeat 式 nested-KV chaining / 分页检索）。

**测试（V2）**

```powershell
python learning/agent-memory-context/src/tests/test_memory.py    # 预期：=== 13/13 modules passed ===
# 或经审计 harness：python scripts/eric_3080ti_env_audit.py --modules agent-memory-context --tests
```

> 注：`test_memory.py` 是脚本式聚合器（汇总 13 个模块的 `_self_test`），无 `test_` 函数；
> 经 harness 时 pytest 收集为空会**自动回退**按脚本直跑。

## 4 memory 类对照

| 类 | 期限 | 例 |
|----|------|---|
| Working | 当下对话 | 当前 turn 上下文 |
| Episodic | 事件 | "上周三聊了 RAG" |
| Semantic | 概念 | "用户偏爱 Anthropic 模型" |
| Procedural | 技能 | "如何调 search tool" |

## 一句话

> 4 memory 类 + Letta paging + Mem0 extract-update — 让 agent 记住用户跨 session 的偏好。
