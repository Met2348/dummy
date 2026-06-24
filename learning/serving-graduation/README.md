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

## 运行验证（Runbook）

> 本模块的"可运行入口"即 [`runbook.yaml`](runbook.yaml) 登记的 8 个直跑入口（6 支持 demo + 2 毕业 capstone），已在 ERIC-3080Ti（RTX 3080 Ti 16GB）V1 验证通过。全部纯 CPU 秒级。
> 一键复验：
> ```powershell
> python scripts/eric_3080ti_env_audit.py --runbook --modules serving-graduation
> ```

**6 个支持 demo**（serving 全家：缓存/嵌入/路由/评分/预算/VLM）：

```powershell
python learning/serving-graduation/src/agent_inference_demo.py   # 多轮 naive vs radix-cached prefill
python learning/serving-graduation/src/embedding_serve.py        # mock 嵌入 + 余弦
python learning/serving-graduation/src/multi_model_router.py     # 按复杂度路由 tier + 成本
python learning/serving-graduation/src/serving_scorecard.py      # correctness+SLO+goodput 评分卡
python learning/serving-graduation/src/thinking_budget.py        # 思考预算强制 + Wait 注入
python learning/serving-graduation/src/vlm_serve.py              # mock VLM 服务
```

**2 个毕业 capstone**（子包）：

```powershell
# Capstone-1：R1-tiny 部署 demo
python learning/serving-graduation/src/r1_tiny_deploy/serve.py

# Capstone-2：五线综合 e2e 报告（默认写 tempdir；--out 指定目录）
python learning/serving-graduation/src/graduation_e2e/run.py
# 或包形式（从 src 目录）：python -m graduation_e2e.run --out report/
```

> 注：这 8 个入口原先**全都直跑无输出**（6 支持脚本 + capstone1 缺 `__main__`，capstone2 用相对 import 只能 `-m`）。本轮补齐 `demo()`/`__main__`，并让 capstone2 `run.py` 兼容"直接当脚本"和"`-m` 包"两种跑法。所有 serving demo 都是单进程模拟（mock 模型/嵌入/图像编码），无需 GPU。

**测试（V2）**：23 个测试覆盖缓存/路由/评分卡/预算/两 capstone：

```powershell
python -m pytest learning/serving-graduation/src/tests/ -v
# 或经审计 harness：python scripts/eric_3080ti_env_audit.py --modules serving-graduation --tests
```

## 毕业 checklist

- [x] 14 lecture + 14 notebook
- [x] 23 tests pass
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
