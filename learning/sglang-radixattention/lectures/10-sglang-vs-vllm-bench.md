# L10 · SGLang vs vLLM 5 场景横评

## 5 场景
| # | 场景 | 描述 |
|---|------|------|
| 1 | 单 system prompt + 100 chat | 2k system + 50 token user × 100 |
| 2 | ToT 8 路并行 | 同 prompt fork 8 |
| 3 | JSON 结构化 1k | 严格 schema |
| 4 | ReAct 5 步 | tool use 闭环 |
| 5 | 长 prompt + 短 out | 8k in / 16 out |

## 经验数字（A100, Qwen-7B）

| 场景 | vLLM 0.7 (tok/s) | SGLang 0.4 (tok/s) | SGLang 收益 |
|------|------------------|---------------------|------------|
| 1 共享 prompt | 4200 | 5800 | +38% |
| 2 ToT 8 路 | 2400 | 5600 | **+133%** |
| 3 JSON schema | 1800 | 5200 | **+189%** (xgrammar 快) |
| 4 ReAct 5 步 | 800 | 2400 | **+200%** |
| 5 长 prompt | 1200 | 1100 | -8% |

## 关键结论
- **vLLM 强**: 短共享前缀 / 单一 system / 长 prompt 单请求
- **SGLang 强**: 任何带 fork / select / grammar / 多轮的 agent 场景

## 选型决策树
```
是 agent / 多轮 / 结构化输出?
 ├── 是 → SGLang
 └── 否 → 是大规模无状态聊天?
          ├── 是 → vLLM (生态成熟)
          └── 否 → 跑测试，选快的
```

## 部署注意
- vLLM OpenAI API 兼容更全 → 公网接 SDK 选 vLLM
- SGLang frontend 需用户写 DSL → 内部 agent 选 SGLang

## 实现：[sglang_compare.py](../src/sglang_compare.py)
- 5 个场景的 mock benchmark（不需 GPU）
- 同 vLLM Topic 1 capstone 对照
