# L10 · 评测 harness 本身

> Part III · 40-min lecture · 配套代码 `src/harness_eval.py` · notebook `N3` · 目标: 把「harness 是性能的自变量」变成可测量的实验。

---

## 0. 关键事实：同模型，换 harness，差几十分

> 固定同一个底层模型, **只换 harness, SWE-bench 的解决率能差几十分**。Claude Code、Codex CLI、Aider、OpenHands 跑同一个模型, 分数显著不同——差的就是 harness。

推论很重要: **「这个 agent 强不强」很大程度是「这个 harness 强不强」。** 所以 harness 必须像模型一样**被评测**——否则你在盲目调一个看不见好坏的东西。

---

## 1. 把 harness 当自变量做对照实验

评测 harness 的方法论, 本质是控制变量:

```
固定:  模型 (provider)  +  任务 (task)
变化:  harness 配置 (compaction? hook? 预算? subagent?)
测量:  成功率 (success rate)  +  成本 (token/$/步数)  +  延迟
```

`src/harness_eval.py` 把这个做成可跑的对照。三个教学配置 (`default_configs`):

```python
HarnessConfig("A_naive",           compaction=False, hook=False)   # 玩具
HarnessConfig("B_hook_only",       compaction=False, hook=True)    # 救回长任务, 但贵
HarnessConfig("C_hook_compaction", compaction=True,  hook=True)    # 生产级: 成且省

rows = evaluate(default_configs(), work_dir, total_steps=6, early_stop_at=2)
```

跑出来的对照 (N3 会画成表 + 图):

```
harness            success  windows  steps  context_tokens   解读
A_naive            ✗        1        2      低              没 hook → early-stop 就废了
B_hook_only        ✓        2        6      高              hook 救回, 但不压缩 → 烧 token
C_hook_compaction  ✓        2        6      低 (↓)          生产级: 又成又省
```

这张表把两件抽象的事钉成了数字:
- **hook 决定成功率**: A vs B——没 hook 的长任务直接失败 (`aborted_early=True`)。
- **compaction 决定成本**: B vs C——同样成功, compaction 把 `context_tokens` 显著压低。

> 这正是 `src/tests/test_all.py::test_configs_differ` 断言的: `A.success == False`, `B/C.success == True`, 且 `C.context_tokens < B.context_tokens`。**「换 harness 改变结果」不是口号, 是被单测钉死的事实。**

---

## 2. 一个诚实的方法论坑：compaction 在小任务上看不出差别

构建 `harness_eval` 时踩到一个真坑, 值得记 (这本身是 `critical-reading-gap` L2 攻击清单 Q3「有没有统计显著性」的镜像):

> 最初演示任务太小, 上下文从没逼近预算, compaction 无从触发 → 开/关 compaction 的成本**完全相同**。必须把工具输出调大、让窗口内上下文真的累积, 差别才显现。

教训: **harness 优化 (compaction/subagent 等) 的收益, 只在「上下文压力够大」时才显现。** 在玩具任务上 benchmark harness, 会得出「优化没用」的假阴性结论。**评测 harness 必须用足够长、足够重的任务**——这也是为什么 SWE-bench(-Pro) 这类真实软件工程任务是好的 harness 试金石。

---

## 3. 真实世界的 harness 评测基准

```
SWE-bench / SWE-Bench-Pro   真实 GitHub issue 修复 (长程、多文件) ← harness 差异最显著
τ-bench                     工具使用 + 多轮对话 (你 Module 7 跑过)
WebArena / OSWorld / GAIA   网页/操作系统/通用 agent 任务
```

- 这些 benchmark **名义上测 agent, 实质上同时测 harness**——因为同模型下分差来自 harness。
- 一个 2026 里程碑: **Confucius Code Agent (CCA)**, 靠统一 orchestrator + 高级 context 管理 + 跨会话持久笔记 + 自动 build-test-improve 的 meta-agent, 在 **SWE-Bench-Pro 上拿到 59% Resolve@1**——这个数字里, harness 工程的贡献是主要的。

> 衔接你的研究技能: 当你读到「我们的 agent 在 SWE-bench 上 SOTA」, 用 `critical-reading-gap` 的攻击清单先问——**提升来自模型还是 harness? 他们固定 harness 只换模型、或固定模型只换 harness 了吗?** 很多论文在这里是糊的, 这本身就是 gap (L13)。

---

## 4. 本讲小结 + 通往 L11

- 同模型换 harness 差几十分 → harness 必须被当自变量评测。
- 方法: 固定模型+任务, 变 harness 配置, 测成功率 + 成本 + 延迟 (`harness_eval`)。
- 诚实坑: 小任务上 harness 优化看不出差别, 评测要用足够重的任务 (SWE-Bench-Pro)。
- 读 agent 论文时追问: 提升来自模型还是 harness?

> **下一讲 L11**: 我们造的是「Agent Loop」一种形态。但 70 个开源 agent 系统的实证研究发现了**五大架构模式**。L11 给你这张地图 + 统一调度框架, 让你按任务选对形态。

**动手**: 打开 `notebooks/N3-harness-eval.ipynb`, 跑三配置对照, 亲眼看 hook 决定成败、compaction 决定成本。
