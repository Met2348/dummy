# L02 · Thinking Budget

## 1 · 2025 商业 thinking model
- Claude 3.7 Sonnet: extended_thinking
- Gemini 2.5: thinking_budget
- OpenAI o1/o3: reasoning_effort

## 2 · API 设计
```json
{
    "model": "claude-3.7-sonnet",
    "thinking": {"type": "enabled", "budget_tokens": 16000}
}
```
- budget = 推理时的 reasoning token 上限
- 超过预算 → 强制 close </thinking> 出 answer

## 3 · 为什么需要 budget
- 推理 model 倾向于长 thinking → 高 cost + 高 latency
- 实际 50% 任务不需要 16k thinking
- 显式预算：用户选 fast vs smart

## 4 · 实现：budget forcing
```python
def generate_with_budget(prompt, budget):
    tokens = []
    in_thinking = False
    while len(tokens) < budget:
        tok = model.next_token(prompt + tokens)
        if tok == "<thinking>":
            in_thinking = True
        elif tok == "</thinking>":
            in_thinking = False
        tokens.append(tok)
        if in_thinking and len(tokens) >= budget * 0.9:
            # 强制 close
            tokens.append("</thinking>")
            in_thinking = False
    return tokens
```

## 5 · s1 budget forcing (Stanford 2025)
- 训了一个小模型，把 "Wait" 注入到 thinking 流
- 强制延长思考
- 小模型 + budget forcing 击败大模型

## 6 · 实现：[thinking_budget.py](../src/thinking_budget.py)
- early_stop / Wait 注入
- 测试不同 budget 对答案质量的影响
