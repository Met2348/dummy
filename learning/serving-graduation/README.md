# Topic 7: Serving Graduation（用大模型毕业 capstone）⭐⭐⭐⭐⭐⭐

> Module 5 「用大模型」第 7 专题 — **系列毕业** · 14 lectures · 14 notebooks · ~13h

## 总览

| Lecture | 主题 | 代码 |
|---------|------|------|
| L01 | Agent 推理特化 | `agent_inference_demo.py` |
| L02 | Thinking budget | `thinking_budget.py` |
| L03 | Reasoning cache | — |
| L04 | 长 ctx 服务 | — |
| L05 | 多模型路由 | `multi_model_router.py` |
| L06 | Batch vs Online | — |
| L07 | VLM 推理 | `vlm_serve.py` |
| L08 | Embedding 服务 | `embedding_serve.py` |
| L09 | 冷启动 | — |
| L10 | 容错 | — |
| L11 | 服务工程 5 原则 | — |
| L12 | **Capstone-1: R1-tiny 部署** | `r1_tiny_deploy/serve.py` |
| L13 | ⭐⭐⭐ **五线综合 lecture** | — |
| L14 | **Capstone-2: 五线综合毕业** ⭐⭐⭐⭐⭐⭐ | `graduation_e2e/` |

## Tags

- `用-graduation` — 系列收官（含双 Capstone + README）⭐⭐⭐⭐⭐⭐
- `module5-complete` — Module 5 整体完成

## Capstone-2 五线综合对比

同一道 Janet 鸡蛋题，5 个 ckpt 的对比：

| ckpt | reasoning | correct | latency | response 摘要 |
|------|-----------|---------|---------|---------------|
| vanilla 124M | none | NO | 30ms | "$10" (错) |
| LoRA tuned | brief | OK | 35ms | "16-3-4=9, 9*$2=$18" |
| DPO aligned | yes | OK | 40ms | "step by step. 16-3=13, 13-4=9, 9*$2=$18" |
| **R1-Zero** | **strong** | OK | 80ms | `<think>...</think><answer>$18</answer>` |
| **Phi-tiny 270M** | **clean** | OK | 60ms | "16-3-4=9. 9 * $2 = $18." |

## 与全系列其它 module 的关系

```
Module 3 造大模型  -> Phi-tiny 270M ckpt
Module 4 改大模型  -> R1-Zero ckpt / DPO ckpt
Module 1 PEFT     -> LoRA ckpt
                       ↓
Module 5 用大模型  -> 量化 + serve + 路由
                       ↓
                  Capstone-2 五线综合对比
```

## 环境

```powershell
python environment/verify_env.py
```

## 运行

```powershell
# 测试 (20/20 全绿)
python -c "import sys; sys.path.insert(0,'src'); sys.path.insert(0,'src/tests'); import test_graduation"

# Capstone-1: R1-tiny demo
python -c "import sys; sys.path.insert(0,'src'); from r1_tiny_deploy.serve import run_demo; print(run_demo())"

# Capstone-2: 五线综合报告
python -c "import sys; sys.path.insert(0,'src'); from graduation_e2e.run import main; import sys as s; s.argv=['x','--out','./report']; main()"
```

## 毕业 checklist

- [x] 14 lecture + 14 notebook
- [x] 20 tests pass
- [x] Capstone-1 R1-tiny 部署 demo
- [x] Capstone-2 五线综合对比 report (md + json)
- [x] git tag `用-graduation` ⭐⭐⭐⭐⭐⭐
- [x] git tag `module5-complete`

## 关键文献

- DeepSeek-V3 (推理服务参考)
- Claude 3.7 / Gemini 2.5 thinking_budget
- s1 (Stanford budget forcing 2025)
- vLLM / SGLang / TRT-LLM 文档

## 一句话总结

> Module 5 闭环 = **训出的 ckpt 量化 + 调度 + serve + 五线综合**。
> 25 个 topic 学完，你已具备 2026 LLM 全栈工程师画像。
