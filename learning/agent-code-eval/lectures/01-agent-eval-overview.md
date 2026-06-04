# L01 · Agent / Code 评测全景

## 为什么这是 2024-2026 新主轴

LLM 输出从"文本"扩展到"代码 + 工具调用 + 浏览器动作"：
- 知识 bench → 测**知道**
- 推理 bench → 测**算**
- **Agent bench → 测做**

## 4 类

| 类 | 代表 | 测什么 |
|----|------|------|
| **Code** | HumanEval / LiveCodeBench / BigCodeBench | 写代码通过单元测试 |
| **SWE** | SWE-Bench / SWE-Bench Verified | 改 repo 修 issue |
| **Web** | WebArena / VisualWebArena / Mind2Web | 浏览器操作 |
| **Tool/OS** | BFCL / GAIA / OSWorld / AndroidWorld | 函数调用 / OS / 移动 |

## 评测协议本质 = exec + outcome check

```
agent_output (code / actions / API call)
    ↓
sandbox executes
    ↓
hidden tests / goal state check
    ↓
pass / fail (binary)
```

## 关键挑战

1. **沙箱安全**：exec()、Docker、firejail、subprocess
2. **复现性**：网页 DOM 变化、API rate limit、模型 sample 抖
3. **成本**：每 episode 几十秒-几分钟，跑全套 100-1000+ episode
4. **失败模式**：模型给"看起来对"的答案，实际没 work

## 本 Topic 覆盖

L02-L03: Code 基础（HumanEval / MBPP）
L04-L05: 新 code bench (BigCodeBench / LiveCodeBench)
L06: SWE-Bench
L07-L08: Web 类 (WebArena / GAIA)
L09: OSWorld
L10: BFCL (function calling)
L11: VLM (MMMU / MathVista)
L12: Capstone mini-agent

## 一句话

> Agent bench = LLM 的"实习考评"，测的是"能不能交付"。
