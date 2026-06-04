# L12 · Capstone — EAGLE-3 部署（教学版）

## 1 · 目标
- 跑通 classic + Medusa + EAGLE-1/2 三种 spec decoding minimal 版
- 测 accept rate / MAU / 模拟加速
- 5 任务对照表

## 2 · 5 任务
| 任务 | prompt | greedy expected acc |
|------|--------|---------------------|
| code  | def fib(n):\n    return | 高 (boilerplate) |
| math  | 2+2=  | 高 (确定性) |
| qa    | Capital of France is | 中 |
| story | Once upon a time | 低 (创意) |
| json  | {"name": | 高 (结构) |

## 3 · 输出
```
method    task   accept_rate  MAU    sim_speedup
classic   code   0.78        4.1    2.8x
medusa    code   0.80        4.5    3.2x
eagle1    code   0.84        4.8    3.5x
eagle2    code   0.87        5.4    4.0x
classic   story  0.45        2.4    1.5x
...
```

## 4 · 数据流
draft → verify → accept_loop → metrics 表

## 5 · 实现：[capstone_eagle3.py](../src/capstone_eagle3.py)
- 用 mock target / mock draft（不依赖真模型）
- 严格 rejection sampling
- 5 任务 batch run
- 输出 markdown table

## 6 · 真训扩展
真跑 Qwen-1.5B target + 200M draft：
```bash
# 用 vLLM 内置 spec
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-1.5B \
    --speculative-model Qwen/Qwen2.5-0.5B \
    --num-speculative-tokens 5
```
预期 accept rate 0.7-0.8 / speedup 1.8x（Qwen-1.5B 已经够小，gain 有限；70B 时 gain 大）

## 7 · 退出条件
- 5 任务全跑通
- EAGLE-2 MAU > Medusa > classic
- (可选) 真模型 vLLM 跑通
