# L05 · 多模型路由

## 1 · 思路
- 简单 query → small model（便宜 / 快）
- 复杂 query → large model（贵 / 慢）
- 节省 50-70% 总成本

## 2 · Router 实现
| 方式 | 准确度 |
|------|-------|
| 关键词 (heuristic) | 中 |
| BERT classifier | 高 |
| LLM (small) routing | 极高 |
| user explicit (mode toggle) | N/A |

## 3 · 典型层级
- L1: small 1B model
- L2: middle 7B
- L3: large 70B
- L4: thinking 70B + reasoning

router 决定走哪层。

## 4 · 商业例
- Cursor: small for simple completion / large for refactor
- Cline: cheap base model + expensive for plans
- OpenAI: 4o-mini → 4o → o3 escalation

## 5 · 实现：[multi_model_router.py](../src/multi_model_router.py)
- complexity scoring (heuristic)
- 模拟 3 层 routing
- cost 对比
