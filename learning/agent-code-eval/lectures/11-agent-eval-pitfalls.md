# L11 · Agent 评测陷阱

## 1. Sandbox 安全

```python
exec(model_code)  # 危险
```

需：
- whitelist builtins
- subprocess + timeout
- 容器隔离
- 文件 system 只读

## 2. 复现性

```
跑 1: GPT-4o WebArena 35%
跑 2 (1 周后): GPT-4o WebArena 28%
```

原因：
- 网站 DOM 改了
- API quota 限速
- 模型版本静默更新
- 随机种子

**对策**：snapshot 网站 / 固定 model version / multi-seed avg

## 3. Cost 控制

```
WebArena 一题 ~5-10 USD（GPT-4o）
全 812 题 ~5000 USD
```

需要：
- subset 评测（200 题）
- 并行 + cache
- 廉价模型预筛

## 4. Partial credit 缺失

Agent bench 多是 0/1：
- 写了 90% 正确代码 → 0
- 5 步对 1 步错 → 0

→ progress reward 难定义 → RL 难训。
对策：sub-task scoring (SWE-Bench 部分支持)。

## 5. Tool 失败 ≠ Model 失败

```
模型说: search_web("XYZ")
工具回: API rate limited
```

是模型错还是 infra 错？
对策：区分 `model_error` vs `infra_error`，infra fail 重试。

## 6. Hallucinated 完成

```
模型说: "I clicked checkout, order placed."
真实状态: 没 click 任何东西。
```

LLM 输出"叙述"而非"动作" → end-state check 是黄金标准（不看 transcript）。

## 7. 模型自己写测试

危险模式：
```
模型: 我写一个简单的 add 函数 + 一个永真断言
def add(): return 0
assert True
```

→ test 过但代码错。
对策：测试 hidden 给 model，evaluator 跑。

## 8. Token 上限

```
LLM ctx 128k, 但 OSWorld 一 episode 可能 200k token
```

需 summarization / memory / context compression。

## 9. 数据集 leak

```
SWE-Bench 在 2024 收录 → GPT-4o 2024-10 训练 cutoff
→ 可能见过
```

对策：用 SWE-Bench Live（rolling）。

## 10. Anchor 偏差

第一个 tool call 错了，后续不会 backtrack。
对策：reflection prompts / multi-rollout vote。

## 一句话

> Agent 评测不是"算 accuracy"，是工程问题——sandbox、cost、复现都要做对。
