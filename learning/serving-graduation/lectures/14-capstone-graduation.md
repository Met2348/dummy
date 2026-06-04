# L14 · Capstone-2 — 五线综合毕业作品 ⭐⭐⭐⭐⭐⭐

## 1 · 目标
同一道 GSM8K 题，**5 种推理路径**部署，对比生成：

1. Vanilla GPT-2 base (124M)
2. LoRA-tuned (Module 1)
3. DPO-aligned (Module 4)
4. R1-Zero reasoning (Module 4)
5. Phi-tiny 270M (Module 3)

## 2 · 实现：[graduation_e2e/](../src/graduation_e2e/)

```
graduation_e2e/
├── ckpts/                    # 5 个 mock ckpt 加载器
│   ├── vanilla.py
│   ├── lora.py
│   ├── dpo.py
│   ├── r1_zero.py
│   └── phi_tiny.py
├── server.py                 # 5 个 FastAPI port (8001-8005)
├── compare.py                # 同 query 跑 5 个，对比
└── report.py                 # 生成 markdown 报告
```

## 3 · 题目
```
Janet's ducks lay 16 eggs per day. She eats 3 for breakfast every morning
and bakes muffins for her friends every day with 4. She sells the remainder
at the farmers' market daily for $2 per fresh duck egg.
How much in dollars does she make every day at the farmers' market?
```

## 4 · 期望对比（mock）

```
| ckpt | response | reasoning? | correct? | latency |
|------|----------|----------|----------|---------|
| vanilla 124M | "$10" | no | ❌ | 30ms |
| LoRA tuned | "16-3-4=9, 9*$2=$18" | brief | ✓ | 35ms |
| DPO aligned | "Let me work through this step by step. 16-3=13, 13-4=9, 9*$2=$18" | yes | ✓ | 40ms |
| R1-Zero | "<think>16 - 3 (breakfast) - 4 (muffins) = 9 eggs. 9 * $2 = $18.</think><answer>$18</answer>" | strong | ✓ | 80ms |
| Phi-tiny 270M | "She has 16-3-4=9 eggs left. 9 * $2 = $18." | clean | ✓ | 60ms |
```

## 5 · 五线 vs 一道题
```
   data + arch (Phi-tiny)    →  better baseline
                            \
   PEFT (LoRA)                → fast adapt
                            ╳
   DPO (Module 4)            → cleaner reasoning
                            ╱
   R1-Zero (Module 4)        → explicit thinking
```

## 6 · 部署架构
5 个 mock LLM 服务（FastAPI port 8001-8005），同 OpenAI API spec。
client 同时发 query → 收集 5 响应 → markdown 表对比。

## 7 · 退出条件
- 5 个 mock ckpt 全跑
- compare.py 生成对比 markdown
- generation diffs 可视化
- E2E test 通过

## 8 · 一句话
> 这是你从 PEFT 到部署的完整作品。每条线都有意义，组合起来是 2026 LLM 工程师的完整画像。
