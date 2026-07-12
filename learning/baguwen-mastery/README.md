# Baguwen Mastery —— 大厂"八股文"面试知识库（AI/算法岗 + 后端通用，两条赛道，306 题）

> 定位：`leetcode-mastery` 练的是"没见过的题现场推导"，`interview-prep/mlcoding` 练的是"真 PyTorch 手写"，本 track 补的是第三块拼图——**标准化、高频重复、面试官按套路追问的知识问答**。这不是知识差，是"知道 -> 30 秒内脱口而出关键词 + 扛住追问"之间的表现差。

## 先读这篇

[`lectures/00-how-to-answer.md`](lectures/00-how-to-answer.md) —— 通用作答四步法（是什么->为什么->怎么做->追问链）、`grade()` 关键词自查怎么用、两条赛道的时间分配建议。

## 两条赛道

| 赛道 | 类别数 | 题量 | 定位 |
|---|:--:|:--:|---|
| AI/算法岗八股（`src/ai_qa/`） | 15 | 178 | ML基础/Transformer/分布式训练/微调对齐/2026新增的Agent-RAG-MCP-MLOps趋势。和你 PhD/NLP 背景高度重合，本质是把已学知识练成临场表达。 |
| 后端通用八股（`src/backend_qa/`） | 8 | 128 | 操作系统/计算机网络/数据库/JVM/分布式系统基础/缓存/设计模式/Linux。国内大厂通用后端面试环节的地基。 |

**合计 306 题 / 23 类**，全部通过 `_self_test()`（每题标准答案自洽性 >= 60% 采分关键词命中，见 `src/qa_common.py` 的 `grade()`）。

## 23 类总览

| # | AI 赛道类别 | 题量 | 讲义 | # | 后端赛道类别 | 题量 | 讲义 |
|---|---|:--:|---|---|---|:--:|---|
| 01 | 优化与训练动力学 | 12 | [ai-01](lectures/ai-01-optimization.md) | 01 | 操作系统 | 16 | [be-01](lectures/be-01-os.md) |
| 02 | 正则化与泛化 | 12 | [ai-02](lectures/ai-02-regularization.md) | 02 | 计算机网络 | 16 | [be-02](lectures/be-02-network.md) |
| 03 | 归一化 | 10 | [ai-03](lectures/ai-03-normalization.md) | 03 | 数据库原理 | 16 | [be-03](lectures/be-03-database.md) |
| 04 | 评估指标 | 11 | [ai-04](lectures/ai-04-metrics.md) | 04 | JVM与并发 | 16 | [be-04](lectures/be-04-jvm-concurrency.md) |
| 05 | Transformer核心架构 | 14 | [ai-05](lectures/ai-05-transformer.md) | 05 | 分布式系统基础 | 16 | [be-05](lectures/be-05-distributed-systems.md) |
| 06 | Tokenizer与数据 | 11 | [ai-06](lectures/ai-06-tokenizer-data.md) | 06 | 缓存与存储 | 17 | [be-06](lectures/be-06-cache-storage.md) |
| 07 | 微调与参数高效方法(PEFT) | 12 | [ai-07](lectures/ai-07-peft.md) | 07 | 设计模式与工程实践 | 18 | [be-07](lectures/be-07-design-patterns.md) |
| 08 | RLHF与对齐 | 14 | [ai-08](lectures/ai-08-rlhf.md) | 08 | Linux与运维基础 | 13 | [be-08](lectures/be-08-linux-ops.md) |
| 09 | MoE架构 | 10 | [ai-09](lectures/ai-09-moe.md) | | | | |
| 10 | 分布式训练 | 14 | [ai-10](lectures/ai-10-distributed-training.md) | | | | |
| 11 | 推理部署与服务化 | 12 | [ai-11](lectures/ai-11-inference-serving.md) | | | | |
| 12 | Agent与RAG与工具调用 | 13 | [ai-12](lectures/ai-12-agent-rag.md) | | | | |
| 13 | 可解释性 | 10 | [ai-13](lectures/ai-13-interpretability.md) | | | | |
| 14 | 经典机器学习基础 | 13 | [ai-14](lectures/ai-14-classic-ml.md) | | | | |
| 15 | 系统设计(AI向) | 10 | [ai-15](lectures/ai-15-system-design.md) | | | | |

## 代码验证模块（11 个，纯 stdlib）

| 赛道 | 模块 | 验证了什么结构性质 |
|---|---|---|
| AI | `ai_coding/moe_routing.py` | 偏斜路由的负载均衡损失显著高于均匀路由，且均匀情况精确等于理论最小值 |
| AI | `ai_coding/dist_memory_calc.py` | Adam+fp16 训练显存里优化器状态占大头，具体数值与 ZeRO 论文的"16字节/参数"估算吻合 |
| AI | `ai_coding/tool_calling_schema.py` | 合法/非法 function-calling 参数的 schema 校验 + 错误反馈 + 修正重试 |
| AI | `ai_coding/rag_retrieval.py` | BM25 检索在确定性小语料上 top-1 命中 + 关键词覆盖二次重排序 |
| AI | `ai_coding/quantization_infer.py` | INT8 对称量化/反量化往返误差落在理论上界 `scale/2` 内 |
| 后端 | `backend_coding/sync_primitives.py` | 手写 Lock+Condition 生产者消费者，`join()` 后断言生产=消费、无丢失无重复 |
| 后端 | `backend_coding/lru_lfu_cache.py` | LRU/LFU 淘汰顺序对拍手工构造的操作序列 |
| 后端 | `backend_coding/consistent_hashing.py` | 加节点后重分布比例 ≈ 理论值 1/N，且迁移只流向新节点 |
| 后端 | `backend_coding/rate_limiter.py` | 令牌桶限流行为，整数 tick 驱动（非真实时钟） |
| 后端 | `backend_coding/bplus_tree_sim.py` | 简化 B+树 insert/search/range_query 对拍暴力排序列表 |
| 后端 | `backend_coding/design_patterns_demo.py` | 单例/工厂/观察者/装饰器/策略五个模式的行为契约 |

## 与 `interview-prep/src/mlqa/qbank.py` 的边界

`interview-prep` 里已有 48 题/8类的 AI 八股 rapid-fire，定位是"研究岗面试前的精简地板速查"，**保持不动**。本 track 的 `ai_qa/`（178题/15类）是同一形式的完整扩展版，覆盖面广得多（含 2026 新趋势 MoE/分布式训练/Agent-RAG-MCP），两者共用 `QA` 数据结构和 `grade()` 打分逻辑（本 track 额外加了 `follow_ups` 追问链字段），但各自独立成册。

## 关于"要不要背后端八股"的诚实说明

如果你的目标是海外前沿 Lab 的纯研究岗，`backend_qa/` 大概率用不上，优先级应明显低于 `ai_qa/` 和 `interview-prep/` 的真 torch 手撕、研究项目深挖。只有当你也在准备国内大厂算法岗/后端岗，或想两手准备时，这条赛道才值得投入——而即便如此，也建议先把 AI 赛道过一遍，因为那是你的核心竞争力所在，后端八股只是"不被完全卡在门槛外"的保底项。

## 环境与测试

```powershell
$env:PYTHONIOENCODING="utf-8"
python learning/baguwen-mastery/environment/verify_env.py
python learning/baguwen-mastery/src/tests/test_all.py     # 预期 37/37 modules passed
```

单模块直接跑也可（每个都有 `__main__`）：
```powershell
python learning/baguwen-mastery/src/ai_qa/qbank_transformer.py
python learning/baguwen-mastery/src/backend_qa/qbank_os.py
python learning/baguwen-mastery/src/ai_coding/moe_routing.py
python learning/baguwen-mastery/src/backend_coding/consistent_hashing.py
```

## 间隔复习

```python
from tracker import ReviewTracker, seed_from_qa
from ai_qa import ALL_QA as AI_QA
from backend_qa import ALL_QA as BE_QA

t = ReviewTracker()
seed_from_qa(t, AI_QA + BE_QA)   # 306 张卡片全部种下
```

详见 [`lectures/99-review-strategy.md`](lectures/99-review-strategy.md)，复用 `leetcode-mastery` 同款 SM-2 简化算法。

## 一句话

> 八股文不是让你重新学一遍知识，是把你已经会的东西，磨成"面试官问什么都能条件反射答出采分点、还接得住三层追问"的临场表现。AI 赛道是你的主战场，后端赛道按你实际要投的岗位量力而行。
