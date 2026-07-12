# 04 · harness-engineering Capstone 深挖

> 素材来源：`learning/harness-engineering/README.md` + `src/provider.py`/`compaction.py`/`long_horizon.py`/`otel_trace.py`/`harness_eval.py` + `src/tests/test_all.py` + `lectures/L01`/`L04`/`L05`/`L09`/`L10`/`L13`/`L14` + `notebooks/N3-harness-eval.ipynb`/`N4-capstone.ipynb`（实际执行取数）+ `.gitignore` + `templates/README.md`。
> 本文档是脚手架，不是台词稿——练法见 [`00-how-to-defend.md`](00-how-to-defend.md)。

---

## 1. 背景与目标

2026 年前沿模型在数周内密集商品化（GPT-5.5/Claude Sonnet 4.6/Gemini 3.1 Pro/DeepSeek-V4），"选哪个模型"不再是竞争焦点，护城河转移到模型外面那层运行时——loop、工具路由、上下文管理、可观测、评测。README 引用的两个数字："**88%** 的企业 agent 项目到不了生产，其中 **65%** 的失败根因是 harness 缺陷（context drift / schema 错配 / state 退化），而非模型推理"（`L01-moat-and-commoditization.md`）。

本专题是 Module 7 `agent-harness-design`（理解层：从零搭一个 mini-harness，懂零件）的进阶续作，目标是把玩具 harness 推到"工程层 + 研究前沿"：
- **工程层**：接真模型的 provider 抽象、5 阶段渐进式 compaction、跨窗口 long-horizon 自治、OpenTelemetry 式可观测、真正评测 harness 本身（同模型换 harness 的分差可测量）。
- **研究前沿层**（Part IV）：用 Module 9 `critical-reading-gap` 的 6 类 gap 雷达扫出 harness 领域的开放问题，产出候选 PhD 研究题目的 idea 卡。

Capstone（L14 + `N4-capstone.ipynb`）把 `provider` + `compaction` + `long_horizon`/hook + `otel_trace` + `harness_eval` 五个组件串成一个整体，跑通一个会 early-stop、被 hook 救回的跨窗口长任务，全程可观测、可评测，最后产出研究 idea 卡。

---

## 2. 个人贡献

自学项目，独立完成的部分：
- `src/provider.py`：`Provider` 抽象基类 + 确定性 `MockProvider`（脚本回放或默认策略两种模式）+ `AnthropicProvider` 教学骨架。
- `src/compaction.py`：5 阶段渐进式压缩器 `Compactor`（budget_reduction → snip → microcompact → context_collapse → auto_compact），带不变量（system/pinned 永不丢、最近 N 条受保护、单调减 token）。
- `src/long_horizon.py`：`FileStateStore`（落盘状态）+ `ToolRegistry` + `run_long_horizon()`（loop-with-hook 外循环，拦截模型的 early-stop）。
- `src/otel_trace.py`：`Tracer`/`Span`，OpenTelemetry 风格的嵌套 span 树，用逻辑时钟保证可复现。
- `src/harness_eval.py`：`HarnessConfig` + `evaluate()`，把 harness 配置当自变量做对照实验（同任务、同模型，只变 compaction/hook 开关）。
- `src/tests/test_all.py`：11 个 stdlib `unittest` 单测，覆盖以上全部组件。
- 14 篇 lecture（`L01`-`L14`）+ 4 个 notebook（`N1`-`N4`），把每个组件的设计意图、不变量、踩过的方法论坑写成教案。

---

## 3. 关键技术决策与理由

### 3.1 5 阶段渐进式 compaction：为什么不是一刀切 summarize

`compaction.py` 的设计原则（`L04` §0-1）：上下文窗口逼近预算时，**能用轻手段就不用重手段，逐级升级**：

| Stage | 名字 | 做什么 | 代码位置 |
|---|---|---|---|
| 1 | budget_reduction | 截断单条超过 `SINGLE_MSG_CAP=400` token 的消息 | `_stage1_budget` |
| 2 | snip | 丢最老的低价值消息（保护 system + 最近 `RECENT_PROTECT=3` 条） | `_stage2_snip` |
| 3 | microcompact | 把较老一段压成一条摘要（约原 token 的 25%） | `_stage3_microcompact` |
| 4 | context_collapse | 只留最近 K 条，其余全塌缩成一条摘要（约 12%） | `_stage4_collapse` |
| 5 | auto_compact | 整窗重置：`[system, 全量摘要(~8%), 最近 2 条]` | `_stage5_autocompact` |

`compact()` 按 1→5 顺序，每个阶段反复施用直到本阶段无效才升级，直到落进预算或手段用尽（`while total_tokens(msgs) > self.max_tokens and guard < 200`）。理由：一刀切 summarize 会一次性丢掉大量信息；渐进式让"最该丢的先丢"，system/pinned 消息永不参与压缩。这个设计是"逆向 Claude Code 揭示的生产级做法"（`L04` 标题下引用）。

**诚实的设计声明**：`_summarize()` 是**确定性占位**，不调用真实 LLM，只是把若干消息替换成一条按 token 目标长度填充的标记字符串（`while approx_tokens(body) < target: body += " …(保留要点)"`）。`L04` §3 明确写"真实 harness 会在 Stage 3/4/5 调一次模型做语义摘要……机制对了，换真模型只是替换一个函数"——这是刻意的简化，用来演示"信息被有损压缩"这件事本身和 token 曲线，不是真摘要质量的演示。

### 3.2 loop-with-hook：为什么"判定权在 harness，不在模型"

`long_horizon.py` 针对模型在长任务上的三个病（early stopping / 分解差 / 跨窗口失忆，`L05` §0）：
- **文件系统是真相之源**：`FileStateStore` 把 progress/notes/todo 落盘成 `state.json`；每个新窗口从干净上下文起步，但开头通过 `_seed_messages()` 读回 `store.summary()`。
- **hook 拦截 early-stop**：`run_long_horizon()` 里，模型发出 `stop` 时，`goal_met(store.load())` 才是唯一的完成判据——不是"模型说完了就完了"。若未达成且 `hook=True`，标记 `stop_intercepted=True`，换新窗口继续；若 `hook=False`（对照组），直接放行，`res.aborted_early=True`。

关键代码：
```python
if stop:
    if goal_met(store.load()):
        res.success = True
    elif hook:
        rec.stop_intercepted = True   # 拦截 early-stop, 触发换窗续跑
    else:
        res.aborted_early = True       # 对照组: 不拦截, 整个 run 中断
```
理由：模型的"我觉得做完了"不可信（这是 2026 公认的三病之一），完成判据必须是 harness 侧可验证的客观状态（这里是 `progress >= total_steps`），而不是模型自我报告。

### 3.3 harness_eval：把 harness 当自变量做对照实验

`harness_eval.py` 的方法论（`L10` §1）：固定模型（`provider`）+ 固定任务（`demo_setup`），只变 harness 配置（`compaction`/`hook` 开关），测量成功率 + 成本（`context_tokens_total`）。三个教学配置：
```python
HarnessConfig("A_naive",           compaction=False, hook=False)
HarnessConfig("B_hook_only",       compaction=False, hook=True)
HarnessConfig("C_hook_compaction", compaction=True,  hook=True)
```
理由：README 和 `L10` 反复强调的关键事实——"固定同一个模型，只换 harness，SWE-bench 解决率能差几十分"。这个模块的意义是把这句话从口号变成本地可跑、可测量的三行对照（实测数字见第 5 节）。

### 3.4 OTel 式 trace：为什么用逻辑时钟而非墙钟

`otel_trace.py` 的 `Tracer._tick()` 用单调计数器代替 `time.time()`。理由（`L09` §1 明确写出）：为了让 notebook/测试的 trace 输出**可复现**——墙钟每次运行都不同，没法做确定性断言；生产环境换成真实时间戳即可，"机制对了，换个时钟源就上生产"。span 层级设计（`window` → `reasoning` → `tool`/`subagent` 子 span）对齐 2026 的 OTel + LLM span 标准：每个 reasoning step 是一个 span，每个 tool call 是它的 child span。

### 3.5 demo_setup 里刻意放大工具输出，为什么

`long_horizon.py` 的 `demo_setup()` 里每次 `do_step` 会返回一个 `"x"*1200`（约 300 token）的"大 tool 输出"。代码注释直接写明原因：**"让窗口内上下文真的累积，这样开/关 compaction 才会在成本上拉开差距（小任务上看不出差别）"**。这个决策背后是 `L10` §2 记录的一个真实方法论坑（见第 4 节）。

---

## 4. 踩过的坑与解决

**`L10-evaluating-the-harness.md` §2 明确记载的方法论坑**（原文小标题"一个诚实的方法论坑：compaction 在小任务上看不出差别"）：

> 最初演示任务太小，上下文从没逼近预算，compaction 无从触发 → 开/关 compaction 的成本**完全相同**。必须把工具输出调大、让窗口内上下文真的累积，差别才显现。

这直接对应 `demo_setup()` 里 `bulky = "x" * 1200` 这一行代码——不是随手写的常量，是为了让 `harness_eval` 的对照实验能测出差异而刻意加大的。教训被明确写下："harness 优化的收益，只在上下文压力够大时才显现，在玩具任务上 benchmark harness 会得出'优化没用'的假阴性结论"——这也是为什么 SWE-bench(-Pro) 这类真实、长程任务是好的 harness 试金石的原因。

**（推测，非文档明确记载）** `AnthropicProvider.stream()` 的实现是：
```python
def stream(self, messages, tools=None):  # pragma: no cover - 需真实 key
    try:
        import anthropic
    except Exception as e:
        raise RuntimeError(...) from e
    raise NotImplementedError("教学骨架: 在此把 Anthropic 流式事件映射为 Chunk。默认用 MockProvider。")
```
`# pragma: no cover` 注释表明作者清楚这段代码从未被测试覆盖、也从未真正执行过。这暗示"provider 可替换"这条核心原则（`L01`："Model proposes, harness disposes"）在本项目里**只在 MockProvider 上验证过**，接真模型这一步是有意留白的骨架，不是踩坑后放弃，而是从一开始就没打算在本专题内做（超出"无需 API key"的教学约束）。

**（推测，非文档明确记载）** `compaction.py` 的 `_summarize()` 用 `body = body[: target * 4]` 按字符数截断到目标 token 量级（`approx_tokens` 的换算是 `len(text)//4`）——这个"4 字符≈1 token"的估计对中文不准确（中文通常 1-2 字符/token），`approx_tokens` 的 docstring 也承认"中文按字符。够教学/对照用"，即作者知道这是不准的估计，只是明确声明了这个近似的边界，而非在真实中文场景里踩坑后才发现。

---

## 5. 结果与诚实局限

### 结果（实际运行取得的数字）

**测试套件**：`python -m unittest discover -s src/tests` 实测 **11/11 测试通过**（0.028s），覆盖 `TestProvider`（2）/`TestCompaction`（4）/`TestTracer`（1）/`TestLongHorizon`（3）/`TestHarnessEval`（1）。关键断言：
- `test_hook_rescues_early_stop`：`hook=True` 时任务成功、至少换了一次窗口、且确实发生过 `stop_intercepted`。
- `test_no_hook_fails_long_task`：`hook=False` 时任务失败、`aborted_early=True`。
- `test_configs_differ`：`A_naive` 失败，`B_hook_only`/`C_hook_compaction` 成功，且 `C.context_tokens < B.context_tokens`。

**N3 (`N3-harness-eval.ipynb`) 实际执行输出**（`total_steps=6, early_stop_at=2`）：

| harness | success | windows | steps | context_tokens |
|---|---|---|---|---|
| A_naive | False | 1 | 2 | 346 |
| B_hook_only | True | 2 | 6 | 2274 |
| C_hook_compaction | True | 2 | 6 | 1382 |

C 相比 B 省 **892 tokens**（notebook 里直接打印的数字），约合 39.2%（892/2274）。

**N4 (`N4-capstone.ipynb`) 实际执行输出**（capstone 装配后的跑法，任务规模与 N3 不同——8 步、2 窗口）：
```
成功: True | 窗口: 2 | 步: 8 | 累计上下文 tokens: 2072
```
span 树实际打印：window-0 内 3 个 reasoning span（各带一个 tool:do_step 子 span），window-1 内 5 个（同样各带 tool 子 span）——与"6 步任务、第 2 步 early-stop、hook 救回后到第 8 步真正完成"的场景吻合。同一 notebook 里的 harness_eval 对照：

| harness | success | context_tokens |
|---|---|---|
| A_naive | False | 978 |
| B_hook_only | True | 4168 |
| C_hook_compaction | True | 2072 |

（N3 和 N4 的具体数字不同是因为任务规模不同——N3 用 6 步默认设置，N4 capstone 用了不同的 `total_steps`/`early_stop_at` 组合；两次跑法都稳定复现"A 失败 / B 成功但贵 / C 成功且省"的定性结论。）

### 诚实局限

1. **这是 MockProvider-only 的教学级 mini 实现，没有接过真实模型**。`AnthropicProvider.stream()` 从未被真正调用过（`# pragma: no cover`），README 明确写"核心运行只用 stdlib + MockProvider，无需 GPU、无需 API key"——**本专题从未在真实 LLM 上验证过 5 阶段 compaction、loop-with-hook 或 harness_eval 的任何一个结论**，全部结论都是在确定性 mock 场景下得出的。
2. **14 讲里只有 5 个 lecture 有对应的可运行 src 代码**：`L03`（provider.py）、`L04`（compaction.py）、`L05`（long_horizon.py）、`L09`（otel_trace.py）、`L10`（harness_eval.py）。其余 8 讲（`L01`/`L02`/`L06`/`L07`/`L08`/`L11`/`L12`/`L13`）在 README 的表格里被明确标注为 `(concept)` 或"复用已有 subagents"——**没有本专题独立新增的实现文件**。L07（工具/MCP 控制平面）甚至没有独立文件，共享 `provider.py` 里的 tool dispatch 逻辑。也就是说"14 讲"是课件规模，不是"14 个系统组件都被工程实现"。
3. **compaction 的摘要是确定性占位符，不是真实语义压缩**——`_summarize()` 从不调用 LLM，只是字符串填充。如果面试官问"你的 compaction 效果怎么样"，诚实的回答是"验证了机制（token 曲线单调下降、system/pinned 不丢、阶段升级顺序对），没有验证真实语义压缩后下游任务成功率是否保持"——这一层验证需要接真模型才能做。
4. **可观测性只到 Level 2（span 树），没有接真实 OTel 后端**。`L09` §3 自己给出的成熟度阶梯里，本专题明确停在"Level 2 span 树（生产入门）"，Level 3（接 Jaeger/Honeycomb + 成本/质量指标）被称为"工程接线问题"，未实现。
5. **Part IV 的"idea 卡"研究产出是运行时生成、且被 gitignore 排除的临时文件，不是持久化的研究交付物**。`N4-capstone.ipynb` 最后一个 cell 会把 `critical-reading-gap/templates/idea-card.md` 复制两份成 `IDEA-harness-G9-repro-variance.md`/`IDEA-harness-G10-model-vs-harness.md`，但写入路径是 `notebooks/_capstone_output/`——这个目录在 `learning/harness-engineering/.gitignore` 里被显式排除（"该目录已 gitignore，是你的个人草稿区"）。也就是说，**这两张 idea 卡从未作为仓库里的实际研究产出被提交过**，每次重跑 notebook 才会重新生成、内容也只是模板 + 一行占位注释（`<!-- 预填: 来自 harness-engineering L13, gap=... -->`），远未填满 idea-card 模板要求的"可证伪假设 + 最小验证实验 + 三筛"内容。如果面试官要求"给我看你写的 idea 卡"，目前仓库里**拿不出一份已经填好内容的实际文件**。
6. **N3 和 N4 的数字不能直接跨 notebook 比较**——因为 `demo_setup` 的 `total_steps`/`early_stop_at` 参数在两个 notebook 里配置不同（N3 是 6 步/第 2 步收工，N4 capstone 是不同的步数组合），只能各自内部横向比较三个 harness 配置，不能说"N4 的 2072 tokens 比 N3 的 1382 tokens 更省"——这是任务规模不同导致的，不是 harness 配置本身的差异。
7. **规模差距（相对生产级系统）**：这是一个单文件、单进程、无并发、无真实网络调用、无鉴权、无预算硬限制的教学实现；`L08`（安全与控制：权限门/hooks/预算/企业 RBAC）整讲是"(concept)"，没有任何一行代码实现权限校验或预算硬拦截——虽然 README 的定位表格里写"权限门 + destructive hooks + 预算护栏 + 企业 RBAC"是本专题相对 `agent-harness-design` 的升级点之一，但目前源码里**没有一个对应的实现文件**，这一条目前是课件层面的规划，未兑现为代码。

---

## 6. 追问树（5 条链，每条 3-4 层）

### 追问 1 · 与 PhD 方向的衔接

**trigger**："这套基建具体和你说的 PhD 研究方向（model-vs-harness 因果归因）怎么接上，能不能现场画一下数据怎么流转？"

1. **Q**: 从工程组件到候选研究题目，具体是怎么接的？
   **A**: `L13-research-gaps.md` 用 Module 9 `critical-reading-gap` 的 6 类 gap 雷达（方法/评测/假设/泛化/复现/理论）逐类扫这门课，扫出 12 个候选 gap（G1-G12），每个都标了来源讲次和"起手友好度"；优先级最高的是 ⑤复现类的 G9（严肃复现某篇 agent 论文的真实方差）和 G10（把 SOTA agent 论文的提升拆成"模型贡献 vs harness 贡献"）——这两个直接建在 `long_horizon.py` + `harness_eval.py` 已有的代码基础上。
2. **Q**: 具体数据怎么流转，从跑一个任务到产出一张 idea 卡？
   **A**: `run_long_horizon()` 驱动一个任务 → `Tracer` 记录 span 树（成本/时长归因）→ `harness_eval.evaluate()` 把"改一个 harness 旋钮（compaction/hook）"跑成对照组，输出 `success`/`context_tokens` 这些结构化指标 → 这套"控制变量测 harness"的方法论，正是 G10（model vs harness 因果归因）第一篇论文需要的实验框架雏形——只是目前只有 2 个旋钮（compaction/hook），真正的研究要扩展到"model"这个自变量（比如换 provider）和更多 harness 旋钮的交互项。
3. **Q**: 这套 idea 卡实际产出了吗，能给我看看具体内容吗？
   **A**: 诚实说，`N4-capstone.ipynb` 的确会生成两个文件名（`IDEA-harness-G9-repro-variance.md`、`IDEA-harness-G10-model-vs-harness.md`），但写入路径 `notebooks/_capstone_output/` 被 `.gitignore` 显式排除，仓库里**没有一份持久化的、填满内容的 idea 卡**——生成的文件只是模板原文 + 一行"预填"注释，还没有真正填写"可证伪假设/最小验证实验/三筛"这些内容。目前"接上 PhD 方向"更多是**课件层面画好了路径图**，不是已经产出可展示的研究文档。
4. **Q**: 如果现在要你把 G10 做成真正的第一篇论文实验，最小可行版本是什么?
   **A**: 固定一个"任务"（比如 `demo_setup` 这类合成长任务，或者迁移到一个真实小规模 SWE 风格任务），把"模型"和"harness"都做成可切换的自变量（模型至少两个：MockProvider 的两种确定性策略变体，或者真的接一个真实小模型；harness 至少复用现有 `compaction`/`hook` 两个旋钮），跑一个 2×2（或更多）的析因设计，测 `success`/`context_tokens`，用统计方法（比如双因素方差分析）拆解"模型主效应 vs harness 主效应 vs 交互项"——`harness_eval.py` 现在的三配置对照已经是这个设计的简化雏形，缺的是"模型"这一维的变化和统计显著性检验。

**pitfall**：容易把"L13 扫出了 gap 列表""L14 生成了两个文件名"直接等同于"已经有研究产出"——一旦被追问"给我看内容"，如果不老实说清楚这两个文件是 gitignore 掉的运行时占位产物，会显得夸大了完成度。

---

### 追问 2 · 测试到底验证了什么性质

**trigger**："11 个测试具体验证了什么，有没有测试是'能跑不报错'但没验证真正的性质？"

1. **Q**: 11 个测试具体分布在哪些组件，各自验证什么核心性质？
   **A**: `TestProvider`(2)：`MockProvider` 默认策略确定性可复现 + 脚本模式按顺序推进；`TestCompaction`(4)：压缩后确实降到预算内、system 消息幸存、预算内时不动、阶段会升级（`max(stages) >= 2`）；`TestTracer`(1)：span 正确嵌套 + 聚合统计正确；`TestLongHorizon`(3)：hook 救回 early-stop、无 hook 任务失败、状态确实落盘；`TestHarnessEval`(1)：三配置的成功率和成本差异符合预期方向。
2. **Q**: 这些测试里，哪个是纯"能跑不报错"、没有验证真正性质的？
   **A**: 最接近这个描述的是 `TestProvider.test_mock_default_deterministic`——它验证的是"两次跑同样输入，输出逐字节相同"和"输出里包含至少一个 tool_call 和一个 done"，这确认了确定性和基本结构，但**没有验证内容语义的正确性**（比如"echo 工具真的被正确调用、参数对不对"这种更强的断言）；不过这本来就是 mock 组件，"确定性可复现"就是它唯一需要满足的性质，所以这条测试和它的目的是匹配的，不算"测试写得偷懒"。
3. **Q**: 那有没有测试是断言写得偏弱、可能漏掉真正该测的性质？
   **A**: `test_stages_escalate` 只断言 `max(stages) >= 2`（至少升级到了 stage 2），没有断言"最终确实落进了预算内"（虽然 `compact()` 内部逻辑会保证这点，但这条测试本身没有重复断言这一点，是靠其他测试如 `test_reduces_under_budget` 间接覆盖）；另外，**没有任何测试验证 compaction 的"正确性"维度**——即"被压缩/丢弃的内容是否恰好是低价值信息、有没有可能把关键信息当冗余丢了"，这本身就是 `L04` §4 和 `L13` G1 明确列为未解决的研究问题，测试套件目前只能测"机制运行正确"（token 单调下降、阶段顺序对），测不了"压缩决策本身是否语义正确"。
4. **Q**: RainbowPO... 啊不对，这是 harness 专题——那 otel_trace 的测试呢，有没有验证"生产可用性"？
   **A**: `test_nesting_and_stats` 只验证了嵌套结构对（window→reasoning→tool 三层）和 `stats()` 聚合数字对（`stats()["tool"]["count"] == 1`），完全没有测试真实 OTel 后端对接（比如 `to_dict()` 导出的 JSON 是否符合 OTel 协议 schema）——这条完全没测，因为 `L09` §3 自己承认本专题只做到 Level 2（span 树），Level 3（接后端）根本没实现，也就没有对应测试。

**pitfall**：容易笼统地说"11 个测试全绿，覆盖很全面"，一旦被追问"具体哪条测试验证了 compaction 的压缩决策是不是语义正确"，答不出来就会暴露"测试覆盖的是机制而非语义正确性"这个真实边界。

---

### 追问 3 · loop-with-hook 的完成判据

**trigger**："loop-with-hook 靠 goal_met 判断任务是否真的完成，这个函数在真实开放任务里怎么定义？"

1. **Q**: 当前代码里 `goal_met` 是怎么实现的？
   **A**: `demo_setup()` 里定义为 `state.get("progress", 0) >= total_steps`——一个非常简单的计数器阈值判断，因为这是一个"可验证任务"（progress 有明确的标量目标）。
2. **Q**: 这在真实的、开放式的任务（比如"帮我重构这个代码库"）里还适用吗？
   **A**: 不直接适用。`L05` §5 和 `L13` G2 都明确指出这是研究 gap："completion-goal 怎么定义才既不 early-stop 也不 over-run——可验证任务（有标准答案）好办，开放任务很难"。目前项目里**只实现并测试了可验证任务**（progress 计数器），完全没有涉及"如何为开放任务学一个『任务是否真完成』的判别器"这个更难的问题。
3. **Q**: 如果 goal_met 判断错了（比如任务其实没做完，但计数器被工具函数错误地提前加满了），harness 会怎样？
   **A**: 会提前结束——`run_long_horizon()` 完全信任 `goal_met(store.load())` 的判断，没有第二层校验机制；如果 `do_step` 这类工具函数本身逻辑有 bug（比如重复计数），`goal_met` 就会被錯误触发，harness 没有内建的"完成状态二次确认"或"回滚"能力。这是当前设计里"判定权在 harness"这条原则的一个隐藏假设——**默认工具函数本身是可信的**，harness 不做进一步验证。
4. **Q**: 这算不算是 L13 里提到的 G5（"文件系统是真相之源"的假设边界）？
   **A**: 相关但不完全一样——G5 讲的是"哪些任务的 state 无法良好序列化"（比如隐性上下文、连续控制），而这里的问题更接近"落盘的 state 本身可能是错的，harness 没有验证 state 正确性的机制"，这是一个 G5 之外、文档里没有明确列出的子问题（**推测，非文档明确记载**）：`FileStateStore` 只负责读写，不做 schema 校验或一致性检查，这本身也是一个潜在的假设 gap。

**pitfall**：容易把"文件系统是真相之源"讲成"这样就绝对可靠了"，而不去承认"harness 完全信任 state 和工具函数的正确性，没有做二次验证"这个隐藏假设——这正是考官想戳的"你是不是把工程解法当成了理论上完备的解"。

---

### 追问 4 · compaction 和正确性的权衡

**trigger**："5 阶段 compaction 听起来都是为了省 token，会不会压丢关键信息导致任务失败？你怎么知道压缩是安全的？"

1. **Q**: 代码里有没有防止"压丢关键信息"的机制？
   **A**: 有不变量层面的保护——`system`/`pinned` 消息（`_is_pinned()` 判断 `role=="system"` 或 `pinned=True`）在所有 5 个阶段都不会被丢弃或压缩；`RECENT_PROTECT=3` 保证最近 3 条消息不参与早期阶段的压缩。但这只保护"标记为重要"和"最近"的消息，**不保护"中间某条看似普通但其实关键的历史消息"**。
2. **Q**: 如果一条关键信息既不是 system/pinned，又不在最近几条里，会被 stage 2/3/4 丢掉/压缩吗？
   **A**: 会。`_stage2_snip` 只按"是不是 tool_result 或 user/assistant 角色 + 不在保护范围内"这个粗粒度规则丢弃，不做任何语义重要性判断；`_stage3/4` 的摘要是确定性占位符（字符串填充），**不是真实的语义摘要**，所以真实场景下如果关键信息只出现在一条"看起来普通"的历史消息里，当前实现完全没有能力识别并保护它。
3. **Q**: 那 harness engineering 领域现在怎么解决"什么该 pin、什么该压"这个问题？
   **A**: 目前是工程直觉，没有理论。`L04` §4 明确写"什么该 pin、什么该压、压多狠，是 harness engineering 的核心手艺，也是 L13 的研究 gap 之一（context folding 的理论目前并不成熟）"；`L13` G11 把这个问题正式列为理论 gap："能否给『在预算 B 下保留信息以最大化任务成功』一个形式刻画"——目前业界（包括本专题）都还没有答案，只有"pin 系统提示和目标、保护最近对话"这种经验规则。
4. **Q**: 那你怎么验证你的 5 阶段 compaction 在真实场景里"够安全"?
   **A**: 诚实回答——没有验证。测试套件（`TestCompaction`）只验证了"token 数确实降到预算内""system 消息幸存""不到预算时不动作""阶段会升级"这几条**机制正确性**，完全没有测试"压缩后下游任务成功率是否受影响"。`harness_eval.py` 里 compaction=on/off 两组对照测的是**成本**（token 数）而不是**正确性**（任务是否因为压缩而失败）——事实上 N3/N4 的结果里，`hook=True` 时无论 compaction 开关都成功（`B_hook_only`/`C_hook_compaction` 都 `success=True`），说明这个特定的合成场景里压缩没有造成信息丢失导致失败，但这只是因为这个 mock 任务的"关键信息"就是那几条 `pinned` 的 system 消息，非 pinned 的 tool_result 全是可丢的冗余——**这是一个被设计出来的、compaction 一定安全的场景，不构成"compaction 在真实场景里安全"的证据**。

**pitfall**：容易拿"11 个测试全绿"当作"compaction 是安全的"的证据，而没有意识到测试验证的是"机制运行正确"而不是"压缩决策在真实语义上不丢关键信息"——这两者是完全不同的主张，混淆会被追问出来。

---

### 追问 5 · harness_eval 的实验设计有效性

**trigger**："你说同模型换 harness 能测出分差，这个'评测 harness'的实验设计本身靠谱吗？会不会有混淆变量？"

1. **Q**: `harness_eval.py` 的实验设计具体是怎么控制变量的？
   **A**: 固定 `provider`（同一个 `MockProvider` 脚本，`demo_setup()` 生成同一套确定性场景）和固定任务（`total_steps`/`early_stop_at` 相同），只变 `HarnessConfig` 里的 `compaction`/`hook` 两个布尔开关；每个配置用 `run_dir = Path(work_dir) / cfg.name` 隔离状态文件，避免跨配置的状态污染。
2. **Q**: 这个设计有没有混淆变量、或者你自己在 lecture 里提过的坑？
   **A**: 有——`L10` §2 明确记录的坑是"任务规模"这个变量最初没控制好："最初演示任务太小，上下文从没逼近预算，compaction 无从触发",导致 compaction 开/关看不出成本差异，是一次**假阴性**。修复方式是把 `demo_setup` 里工具输出调大（`bulky = "x"*1200`），让上下文压力足够大——这说明"评测 harness"这件事本身对"任务是否足够重"高度敏感，如果复现别人的 harness 评测结论，第一件要检查的事就是任务规模是否足够触发被测的那个机制。
3. **Q**: 现在这个三配置对照（A/B/C），有没有统计显著性检验？
   **A**: 没有。`evaluate()` 对每个配置只跑一次确定性场景（`MockProvider` 的脚本回放是完全确定性的，没有随机性），所以"跑多次取方差"这件事在当前实现里没有意义——但这也意味着这个实验设计**没有测试对真实非确定性模型（会有采样随机性）的鲁棒性**；`L13` G9 把"很多 agent 论文 harness 散落、单配置、不报方差"列为复现类 gap，本专题自己的 `harness_eval` 目前也是"单配置、不报方差"，同样的批评可以直接套用在自己身上。
4. **Q**: 如果要把这个评测协议做严谨，最少要加什么？
   **A**: 至少三件事：(1) 接入有真实随机性的 provider（哪怕是加了采样噪声的 mock），每个配置跑多次取均值和方差；(2) 扩大任务集合而不是单一 `demo_setup` 场景，避免"只在一个人为设计的任务上成立"的结论；(3) 像 `L10` §3 提到的把评测迁移到真实 benchmark（SWE-bench-Pro 这类），因为"这些 benchmark 名义上测 agent，实质上同时测 harness"——目前完全没有做到任何一条，这正是 `L13` 里 G3（harness 专用 benchmark）和 G9（复现+方差）两个 gap 分别指向的缺口。

**pitfall**：容易把"三配置对照跑出了预期方向的数字"直接等同于"这是一个严谨的评测协议"，而回避"单次确定性跑法、无统计检验、单一任务场景"这些和自己在 L13 里批评别人论文时用的同一套标准——诚实的态度是承认这套 harness_eval 目前也只是"方法论雏形"，不是可以直接发表的评测协议。

---

## 附：核心数字来源速查

| 数字/断言 | 来源 |
|---|---|
| 88% 落地失败 / 65% 源于 harness 缺陷 | `lectures/L01-moat-and-commoditization.md` §2 |
| 5 阶段 compaction 具体常量（400/3） | `src/compaction.py` `SINGLE_MSG_CAP`/`RECENT_PROTECT` |
| N3 三配置对照（346/2274/1382 tokens，省 892） | 实际执行 `notebooks/N3-harness-eval.ipynb` 输出 |
| N4 capstone 跑法（2 窗口/8 步/2072 tokens + span 树） | 实际执行 `notebooks/N4-capstone.ipynb` 输出 |
| 11/11 测试通过，0.028s | 本次复核 `python -m unittest discover -s src/tests` 实测 |
| compaction 小任务测不出差异的坑 | `lectures/L10-evaluating-the-harness.md` §2 |
| idea 卡写入 gitignore 目录、未持久化 | `learning/harness-engineering/.gitignore` + `N4-capstone.ipynb` 最后一个 code cell 源码 |
| 14 讲里仅 5 讲有独立 src 文件 | `learning/harness-engineering/README.md` "14 讲总览" 表格逐行核对 |
| AnthropicProvider 从未真正调用 | `src/provider.py` `# pragma: no cover` 注释 + README "无需 GPU/API key" |
