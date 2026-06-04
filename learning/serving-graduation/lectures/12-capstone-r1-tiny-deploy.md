# L12 · Capstone-1 — R1-tiny 部署

## 1 · 目标
拿 Module 4 reasoning-r1 capstone-A 训出的 GPT-2-M R1-Zero ckpt → 完整部署 pipeline。

## 2 · 步骤
```
GPT-2-M R1-Zero ckpt (Module 4)
    ↓ AWQ 4bit 量化 (Module 5 Topic 4)
qwen-r1-tiny-awq.bin
    ↓ vLLM 加载 + 流式 API (Topic 6)
http://localhost:8000/v1/chat/completions
    ↓ 测试 5 道数学题
看 reasoning trace 流式输出 + 答案准确率
```

## 3 · 关键代码（[r1_tiny_deploy/](../src/r1_tiny_deploy/)）
```python
# 1. 模拟 ckpt 加载
class MockR1Model:
    def stream(self, prompt):
        yield "<think>Let me solve this step by step. "
        for token in self.thinking(prompt):
            yield token
        yield "</think><answer>"
        yield self.answer(prompt)
        yield "</answer>"

# 2. FastAPI 包装（继承 Topic 6 的 openai_api_server）
# 3. 测试 5 道题
```

## 4 · 退出条件
- p50 latency < 1s（thinking 流式开始）
- 5 道数学题 ≥ 3 道正确
- thinking trace 可流式输出
