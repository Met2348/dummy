# Topic 7: Eval Graduation（评测毕业 ⭐⭐⭐⭐⭐⭐）

> Module 6「评」第 7 专题 — **系列毕业** · 14 lectures · ~14h
>
> 全部 25 prior + 7 Module 6 = **32 专题学习马拉松** 收官 capstone

## 总览

| Lecture | 主题 | 代码 |
|---------|------|------|
| L01 | 评测毕业总览 | (intro) |
| L02 | 25 专题 ckpt zoo | `ckpt_zoo/` |
| L03 | bench × ckpt 矩阵设计 | (lecture) |
| L04 | mini-HELM | `mini_helm.py` |
| L05 | mini-Arena | `mini_arena.py` |
| L06 | mini 红队 | `mini_red_team.py` |
| L07 | mini 防御 | `mini_defense.py` |
| L08 | 成本工程 | (lecture) |
| L09 | Portfolio 设计 | `portfolio.py` |
| L10 | blog 风格 README | (lecture) |
| L11 | 选型决策树 | (lecture) |
| L12 | **Capstone-1: mini-HELM** | `mini_helm.py` |
| L13 | **Capstone-2: Arena + 红队 + 防御** | 3 个 src |
| L14 | **Capstone-3: Portfolio ⭐⭐⭐⭐⭐⭐** | `portfolio.py` |

## Tags

- `评-graduation` — 系列收官（3 Capstone + Portfolio）⭐⭐⭐⭐⭐⭐
- `module6-complete` — Module 6 整体完成

## 运行验证（Runbook）

> 本段命令即 [`runbook.yaml`](runbook.yaml) 登记的"文档入口命令"，已在 ERIC-3080Ti（RTX 3080 Ti 16GB）上 V0+V1 验证通过（6/6 PASS）。
> 一键复验本模块：
> ```powershell
> python scripts/eric_3080ti_env_audit.py --runbook --modules eval-graduation
> ```

6 个脚本（1 个共享 `ckpt_zoo` 库 + 5 个 capstone）全部无 argparse、纯 stdlib（无 torch/transformers/网络依赖）→ 直跑到完成即验证（runbook 里 `v0: false`，跳过 `--help` 探针）：

```powershell
python learning/eval-graduation/src/ckpt_zoo/interface.py   # L02 共享 ckpt_zoo：5 mock ckpt 接口 self-test
python learning/eval-graduation/src/mini_helm.py            # Capstone-1: mini-HELM 5×4 打分表 + ASCII 雷达
python learning/eval-graduation/src/mini_arena.py           # Capstone-2A: mini-Arena round-robin + BT-Elo
python learning/eval-graduation/src/mini_red_team.py        # Capstone-2B: 3 攻击 × 5 ckpt 红队 ASR 矩阵
python learning/eval-graduation/src/mini_defense.py         # Capstone-2C: 输入+输出分类器防御前后对照
python learning/eval-graduation/src/portfolio.py            # Capstone-3⭐⭐⭐⭐⭐⭐: 32 专题 Portfolio（落盘到系统临时目录）
```

> ℹ️ **5 个 mock ckpt 是手写字面量回复**（`vanilla`/`lora`/`dpo`/`r1_tiny`/`phi_tiny`，定义在 `ckpt_zoo/interface.py`），不加载真实权重；4 维评分 / battle 判定 / 攻击合规判定 / 分类器防御都是**真计算**（非硬编码分数——例如 vanilla 因回答含"23"被 `score_reasoning` 判 0 分、`is_compliant`/`input_classifier`/`output_classifier` 靠关键词真扫描响应文本），已在各脚本 `_self_test()` 用非平凡断言锁住。红队 3 种攻击（`direct`/`persona_wrap`/`multi_turn`）对同一 mock ckpt 输出 ASR 相同是**诚实局限**（docstring 已标注：mock ckpt 无状态，不区分攻击手法），非 bug。
>
> ℹ️ `portfolio.py` 直跑时会额外把完整 Portfolio 写到系统临时目录（`%TEMP%/eval_graduation_portfolio.md`，**不写入 repo**），对应 L14 讲义"落盘 portfolio.md"的文档承诺；若要落盘到指定路径：`from portfolio import write_portfolio; write_portfolio("your/path.md")`（`write_portfolio()` 默认参数是相对路径 `"portfolio.md"`，从 repo 根直接调用会落在 repo 根——请显式传路径，避免污染 repo）。

**测试（V2）**：

```powershell
python learning/eval-graduation/src/tests/test_graduation.py
# 或经审计 harness：python scripts/eric_3080ti_env_audit.py --modules eval-graduation --tests
```

预期：`6/6 modules passed`（逐模块跑 `_self_test()`）。

> 注：`test_graduation.py` 是脚本式 runner（无 pytest `test_*` 函数），`pytest` 会 collect 到 0 个用例（exit 5，"no tests ran"）→ 审计 harness 自动回退直接跑脚本执行真断言。

## 5 个 ckpt 对应

| ckpt | 出处 |
|------|------|
| `vanilla` | Module 3 baseline GPT-2 base |
| `lora` | Module 1 PEFT 后 |
| `dpo` | Module 4 改大模型 (RLHF) |
| `r1_tiny` | Module 4 reasoning-r1 |
| `phi_tiny` | Module 3 pretraining-recipe |

## 跨 Module 闭环图

```
Module 1 PEFT (3)        ┐
Module 3 造大模型 (8)     ├─→ Module 6 评测/安全 (本 capstone)
Module 4 改大模型 (7)     │       ↓
Module 5 用大模型 (7)     ┘  32-topic Portfolio ⭐⭐⭐⭐⭐⭐
```

## 关键文献

- HELM (Stanford 2022)
- Chatbot Arena Elo (LMSYS 2024)
- HarmBench (CMU 2024)
- Constitutional Classifiers (Anthropic 2025)
- Llama Guard 3 (Meta 2024)
- 整个 Module 6 引用文献

## 一句话

> Module 6 收官 = Module 1+3+4+5 的 25 专题学完后，加 1 份 portfolio.md ——
> 你的"LLM 全栈工程师 ID 卡"。
