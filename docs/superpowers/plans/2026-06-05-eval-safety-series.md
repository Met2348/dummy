# Module 6 「评」 — 评测 / 安全 / 对齐审计 7 专题完整规划

> 设计日期：2026-06-05
> 学习仓库：`c:\Workspace\dummy`
> 上游：Module 1 PEFT (3) + Module 3 造大模型 (8) + Module 4 改大模型 (7) + Module 5 用大模型 (7) = 25 专题
> 模板来源：Module 5 「用大模型」7 专题（2026-06-04-serving-inference-series.md）
> 配套：Stanford HELM / OpenAI Evals / Anthropic Red-team / Apollo Research

---

## 一、Context — 为什么需要"评"这一程

### 学习全图位置
```
造 (Module 3) → 改 (Module 1+4) → 用 (Module 5) → 评 (Module 6) → 应用 (Module 7?)
                                                   ⬆ 现在
```

**「评」不是"造-改-用"的可选附录**，而是 LLM 工程的第四主轴：
- **没有评测就没有迭代**：训练全靠 eval 闭环
- **没有 jailbreak 红队就没有安全 ship**：生产模型必经
- **2025-2026 现状**：评测 hack 化、benchmark 污染、jailbreak 升级、Constitutional Classifiers 新防御 — 全是新方法

### 用户决策（2026-06-05 拍板）
- ✅ **跨 module 决策**：Module 5 已收官，明确选 Module 6 评测/安全/对齐审计
- ✅ **7 专题方案**（评测 4 + 安全 2 + 综合毕业 1）
- ✅ **保持节奏**：复用 Module 5 三轨代码 + 中文 lecture + capstone 模板
- ✅ **承接 25 专题**：Topic 7 毕业作品在所有 ckpt 上跑全套评测

### 输出物
- 本文件：Module 6 整体蓝图
- 7 个 `learning/<topic>/` 目录
- 终态 tags：`eval-foundations` / `reasoning-eval` / `agent-code-eval` / `llm-judge-arena` / `red-team-jailbreak` / `safety-defense` / `评-graduation` / `module6-complete`

---

## 二、7 专题总览

| # | 专题代号 | 一句话定位 | 方法/Bench 数 | Lecture | 时长 | git tag |
|---|---------|----------|--------|---------|------|---------|
| 1 | `eval-foundations` | 通用 LLM benchmark + lm-eval-harness 上手 | 13 | 12 | 12h | `eval-foundations` |
| 2 | `reasoning-eval` | 推理 benchmark 深化 (数学/科学/编程推理) | 12 | 12 | 12h | `reasoning-eval` |
| 3 | `agent-code-eval` | Agent / Code / OS / Web benchmark | 13 | 12 | 13h | `agent-code-eval` |
| 4 | `llm-judge-arena` | LLM-as-Judge + Arena 风格评测 | 12 | 12 | 12h | `llm-judge-arena` |
| 5 | `red-team-jailbreak` | 红队 + 攻击方法 (GCG/PAIR/Crescendo) | 13 | 12 | 13h | `red-team-jailbreak` |
| 6 | `safety-defense` | 防御 / Guardrails / Constitutional Classifiers | 12 | 12 | 12h | `safety-defense` |
| 7 | `eval-graduation` | 25 专题全 ckpt 跑全套评测 + 安全红队 + portfolio | 14 | 14 | 14h | **`评-graduation`** + **`module6-complete`** ⭐⭐⭐⭐⭐⭐ |
| | | **合计** | **89** | **86** | **88h** | — |

### 依赖关系图
```
Topic 1: eval-foundations (基础, 12h)
        ├─→ Topic 2: reasoning-eval (12h)
        ├─→ Topic 3: agent-code-eval (13h)
        ├─→ Topic 4: llm-judge-arena (12h)
                                ↓
        Topic 5: red-team-jailbreak (13h)
                ↓
        Topic 6: safety-defense (12h)
                ↓
        Topic 7: eval-graduation 毕业 (14h)
```

---

## 三、专题 1：eval-foundations（评测基础 + 通用 benchmark）

### 定位
教会"评测从零跑通"，覆盖 MMLU / MMLU-Pro / HELM / Open LLM Leaderboard / lm-eval-harness。

### 章节规划（12 lectures）

| Lecture | 主题 | 核心 idea |
|---------|------|----------|
| 01-eval-paradigms.md | 评测 4 大范式 | log-prob / generative / preference / programmatic |
| 02-mmlu.md | MMLU (57 学科) | 5-shot multiple choice，2020 至今的"金标准" |
| 03-mmlu-pro.md | MMLU-Pro (2024) | 10-option，难度 ↑，去污染 |
| 04-helm.md | HELM (Stanford 2022→) | 16 场景 × 7 维度 holistic |
| 05-open-llm-leaderboard.md | Open LLM Leaderboard v2 (HF 2024) | IFEval/BBH/MATH/GPQA/MuSR/MMLU-Pro |
| 06-bbh.md | Big-Bench Hard (23 任务) | 难题精选 |
| 07-truthfulqa.md | TruthfulQA | 反 imitation 倾向 |
| 08-hellaswag-arc-winogrande.md | 经典 commonsense bench | 历史背景 + 已饱和 |
| 09-lm-eval-harness.md | lm-evaluation-harness 上手 | EleutherAI 标准工具 |
| 10-contamination.md | benchmark 污染 + LiveBench (2024) | n-gram + canary string |
| 11-eval-pitfalls.md | 评测陷阱合集 | prompt format / option order / chain-of-thought 影响 |
| 12-capstone-eval-pipeline.md | Capstone：GPT-2-M 跑全套 | mmlu/bbh/truthfulqa/hellaswag 4 bench 联跑 |

### src/ 三轨
- `mmlu_runner.py` (手写 MMLU 推理 + scoring，无依赖)
- `lm_eval_adapter.py` (调 lm-evaluation-harness API)
- `helm_local.py` (HELM 简化本地复现)
- `contamination_check.py` (n-gram 污染检测)
- `eval_pipeline.py` (capstone：4-bench 联跑)
- `tests/test_*_eval.py` ×8

### 退出条件
- [ ] 12 lecture + notebook + capstone
- [ ] tag `eval-foundations`

---

## 四、专题 2：reasoning-eval（推理 benchmark 深化）

### 定位
推理是 2024-2026 LLM 卷的核心战场。深入 GSM8K / MATH / AIME / GPQA / Humanity's Last Exam。

### 章节规划（12 lectures）

| Lecture | 主题 | 核心 idea |
|---------|------|----------|
| 01-reasoning-bench-overview.md | 推理 bench 全景 | 数学/科学/逻辑/常识 4 大类 |
| 02-gsm8k.md | GSM8K (OpenAI 2021) | grade school math，`####` parse |
| 03-math.md | MATH (Hendrycks 2021) | competition 难度，LaTeX |
| 04-aime.md | AIME 2024/2025 | 30 题真竞赛，o1/R1 战场 |
| 05-math-shepherd-verifier.md | 自动 verifier (math-verify/sympy) | 程序判分 |
| 06-gpqa.md | GPQA Diamond (2023) | google-proof 物理/生物/化学 |
| 07-humanity-last-exam.md | Humanity's Last Exam (Scale 2025) | 2025 最难，3000 expert-author 题 |
| 08-arc-agi.md | ARC-AGI (Chollet) | 真 reasoning vs pattern matching |
| 09-bigcodebench-zebra.md | ZebraLogic / Big-Bench Zebra | 逻辑推理 |
| 10-mt-math.md | 多轮数学推理 + tool-aug | 用 calculator 调用 |
| 11-reasoning-eval-pitfalls.md | 评测陷阱 | answer extraction 错 / pass@1 vs pass@k |
| 12-capstone-reasoning-bench.md | Capstone：R1-tiny vs GPT-2 推理对照 | 5 bench × 2 model = 10 cell |

### src/ 三轨
- `gsm8k_runner.py` (handles `####` parse)
- `math_runner.py` (LaTeX answer extract)
- `gpqa_runner.py` (5-shot CoT)
- `math_verify_demo.py` (sympy + math-verify)
- `humanity_last_exam_mock.py` (小子集 mock)
- `capstone_reasoning_compare.py`
- `tests/test_*.py` ×8

### 退出条件
- [ ] 12 lecture + notebook + capstone
- [ ] tag `reasoning-eval`

---

## 五、专题 3：agent-code-eval（Agent / Code / OS / Web benchmark）

### 定位
2024-2026 evaluation 新主轴 — agent benchmark。

### 章节规划（12 lectures）

| Lecture | 主题 | 核心 idea |
|---------|------|----------|
| 01-agent-eval-overview.md | Agent benchmark 全景 | code / web / OS / multi-modal 4 类 |
| 02-humaneval-mbpp.md | HumanEval (164) + MBPP | code 评测起点 |
| 03-bigcodebench.md | BigCodeBench (2024) | 1140 真实任务 |
| 04-livecodebench.md | LiveCodeBench (2024) | 月度滚动，去污染 |
| 05-swe-bench.md | SWE-Bench / SWE-Bench Verified | 真实 GitHub issue |
| 06-webarena.md | WebArena (CMU 2024) | shopping/gitlab/social 4 站 |
| 07-gaia.md | GAIA (Meta 2024) | 真实工具调用 multi-step |
| 08-osworld.md | OSWorld (清华 2024) | OS-level 操作 |
| 09-bfcl.md | Berkeley Function Calling Leaderboard | tool use 标准 |
| 10-mmlu-cs.md | MMMU / MathVista / OCRBench (VLM) | 多模态 |
| 11-agent-eval-pitfalls.md | sandboxing / 复现性 / cost 控制 | 工程陷阱 |
| 12-capstone-agent-eval.md | Capstone：mini-agent 跑 HumanEval 子集 + 假 SWE issue | — |

### src/ 三轨
- `humaneval_runner.py` (exec + pass@1)
- `swebench_mock.py` (1 个假 issue 完整流程)
- `webarena_mock.py` (本地静态 HTML 模拟)
- `bfcl_runner.py` (function calling 评测)
- `mini_agent.py` (capstone 基础)
- `tests/test_*.py` ×9

### 退出条件
- [ ] 12 lecture + notebook + capstone
- [ ] tag `agent-code-eval`

---

## 六、专题 4：llm-judge-arena（LLM-as-Judge + Arena 风格评测）

### 定位
开放式生成不能 exact-match → 必须用 LLM judge / Arena。

### 章节规划（12 lectures）

| Lecture | 主题 | 核心 idea |
|---------|------|----------|
| 01-judge-overview.md | LLM-as-Judge 4 类 | pointwise / pairwise / listwise / panel |
| 02-mt-bench.md | MT-Bench (Zheng 2023) | 80 题 multi-turn |
| 03-arena-hard.md | Arena-Hard (LMSYS 2024) | 500 题，model-vs-model |
| 04-chatbot-arena.md | Chatbot Arena Elo | 真人盲对 → Bradley-Terry |
| 05-alpaca-eval-2.md | AlpacaEval 2 LC (length-controlled) | 自动 judge + length 校正 |
| 06-prometheus2.md | Prometheus 2 (2024) | 开源 judge 模型 |
| 07-g-eval.md | G-Eval (NLG eval) | CoT + form-filling |
| 08-judge-bias.md | judge 4 大 bias | position / verbosity / self-preference / style |
| 09-judgebench.md | JudgeBench (评 judge) | 元评测 |
| 10-pairwise-vs-pointwise.md | pairwise vs pointwise 对照 | win rate + tie 处理 |
| 11-eval-cost-engineering.md | judge 成本工程 | GPT-4 → Prometheus → 小模型 ladder |
| 12-capstone-arena-mini.md | Capstone：5 个 Module 4 ckpt mini-arena | DPO/R1/Vanilla/LoRA/SFT 5 对 5 |

### src/ 三轨
- `mt_bench_runner.py` (multi-turn + GPT-4 judge mock)
- `arena_hard_runner.py` (pairwise judge)
- `bradley_terry.py` (Elo 推 BT)
- `prometheus2_judge.py` (开源 judge mock)
- `judge_bias_demo.py` (position bias 重现)
- `mini_arena.py` (capstone)
- `tests/test_*.py` ×8

### 退出条件
- [ ] 12 lecture + notebook + capstone
- [ ] tag `llm-judge-arena`

---

## 七、专题 5：red-team-jailbreak（红队 + 攻击方法）

### 定位
**Module 6 进入安全维度**。覆盖 2023-2026 最重要的 jailbreak 攻击。

### 章节规划（12 lectures）

| Lecture | 主题 | 核心 idea |
|---------|------|----------|
| 01-red-team-overview.md | 红队历史 + Anthropic 框架 | manual → automated red team |
| 02-jailbreak-taxonomy.md | jailbreak 分类 | prompt / encoding / persona / multi-turn |
| 03-gcg.md | GCG (Zou 2023) | gradient-based suffix |
| 04-pair.md | PAIR (Princeton 2024) | LLM 攻 LLM 单轮 |
| 05-autodan.md | AutoDAN (UMD 2024) | genetic + handcrafted seed |
| 06-crescendo.md | Crescendo (Microsoft 2024) | multi-turn 渐进升级 |
| 07-many-shot.md | Many-shot Jailbreak (Anthropic 2024) | long context 攻击 |
| 08-prompt-injection.md | Prompt Injection / IPI | direct + indirect (tool) |
| 09-prefilling-attack.md | Prefilling attack | 强制 assistant 前缀 |
| 10-multi-modal-jailbreak.md | 多模态 jailbreak (image/audio) | typographic / steganography |
| 11-jailbench.md | JailbreakBench / HarmBench (eval 红队效果) | 标准化 |
| 12-capstone-red-team.md | Capstone：3 种攻击 × 3 模型 ckpt | 自家 ckpt 红队矩阵 |

### src/ 三轨
- `gcg_minimal.py` (gradient suffix，玩具版)
- `pair_minimal.py` (attacker LLM 循环)
- `autodan_minimal.py` (genetic algorithm)
- `crescendo_demo.py` (multi-turn 升级)
- `prompt_injection_demo.py` (direct + indirect)
- `jailbench_runner.py` (标准化评测)
- `red_team_matrix.py` (capstone)
- `tests/test_*.py` ×8

### 安全声明
所有红队代码仅用于**自家 ckpt 攻防研究**，不针对外部模型，不发布有效 jailbreak prompt。

### 退出条件
- [ ] 12 lecture + notebook + capstone
- [ ] tag `red-team-jailbreak`

---

## 八、专题 6：safety-defense（防御 / Guardrails / Constitutional Classifiers）

### 定位
Module 6 的"防"——攻防对立面，2025 最新防御方法。

### 章节规划（12 lectures）

| Lecture | 主题 | 核心 idea |
|---------|------|----------|
| 01-defense-overview.md | 防御 4 层 | system prompt / fine-tune / classifier / monitor |
| 02-llama-guard-3.md | Llama Guard 3 (Meta 2024) | input/output classifier |
| 03-shield-gemma.md | ShieldGemma (Google 2024) | open-source guard family |
| 04-wildguard.md | WildGuard (AI2 2024) | adversarial-trained |
| 05-nemo-guardrails.md | NeMo Guardrails (NVIDIA) | rule-based + LLM hybrid |
| 06-constitutional-classifiers.md | Constitutional Classifiers (Anthropic 2025) | 大规模合成 + jailbreak 防御 |
| 07-prompt-injection-defense.md | IPI 防御 | input parse / privilege / sandboxing |
| 08-content-moderation.md | 内容审核传统方法 | OpenAI moderation API spec |
| 09-pii-redaction.md | PII 检测 + 脱敏 | Presidio / regex / NER |
| 10-monitoring-incident.md | 在线监控 + 事件响应 | prompt logging / audit / kill-switch |
| 11-safety-eval.md | 安全评测 bench | HHH / WildBench / TrustLLM / SALAD |
| 12-capstone-defense-pipeline.md | Capstone：4 层防御部署到 R1-tiny | 攻防对照表 |

### src/ 三轨
- `llama_guard_mock.py` (input/output classifier 简化)
- `wildguard_mock.py` (adversarial guard)
- `nemo_guardrails_mock.py` (rule-based)
- `constitutional_classifier.py` (synthetic 训 classifier)
- `pii_redaction.py` (regex + mock NER)
- `prompt_injection_defense.py` (privilege + sandboxing)
- `safety_eval_runner.py`
- `defense_pipeline.py` (capstone)
- `tests/test_*.py` ×9

### 退出条件
- [ ] 12 lecture + notebook + capstone
- [ ] tag `safety-defense`

---

## 九、专题 7：eval-graduation（25 专题全跑分 + 红队 + portfolio）⭐⭐⭐⭐⭐⭐

### 定位
**Module 6 收官** = 整个学习系列的"考试日"。所有自家 ckpt 全跑评测、红队、防御，输出 portfolio。

### 章节规划（14 lectures）

| Lecture | 主题 | 核心 idea |
|---------|------|----------|
| 01-grad-overview.md | 评测毕业 = 系列大考 | 4 主轴回顾 |
| 02-ckpt-zoo.md | 25 专题 ckpt 清单 | vanilla / LoRA / Adapter / DPO / R1-Zero / Phi-tiny / 量化版... |
| 03-bench-matrix-design.md | bench × ckpt 矩阵设计 | 5 ckpt × 5 bench = 25 cell |
| 04-mini-helm.md | 自建 mini-HELM | knowledge / reasoning / safety / efficiency 4 维 |
| 05-mini-arena.md | mini-Arena Elo + BT | 自家 ckpt 互打 |
| 06-mini-red-team.md | mini 红队 | GCG/PAIR/Crescendo 各 1 |
| 07-mini-defense.md | mini 防御部署 | Constitutional + Llama Guard mock |
| 08-cost-engineering.md | 成本工程 (token/latency/$) | 跨 ckpt 对照 |
| 09-portfolio-design.md | Portfolio 设计 | README + 雷达图 + 决策树 |
| 10-blog-style-readme.md | blog 风格 README 写法 | 面试展示 |
| 11-decision-tree.md | "选型决策树" | 给场景选 ckpt + benchmark |
| 12-Capstone-1-mini-helm.md | **Capstone-1: mini-HELM 全跑** | 5 ckpt × 5 bench 全表 |
| 13-Capstone-2-arena-redteam.md | **Capstone-2: mini-Arena + 红队矩阵** | Elo + 防御表 |
| 14-Capstone-3-portfolio.md | **Capstone-3: 25 专题 Portfolio README** ⭐⭐⭐⭐⭐⭐ | 系列毕业的"出门作品集" |

### src/ 三轨

```
src/
├── ckpt_zoo/
│   ├── load_all.py        # 加载 25 个 mock ckpt 元数据
│   └── interface.py        # 统一 generate(prompt) 接口
├── mini_helm/
│   ├── knowledge.py        # MMLU 子集
│   ├── reasoning.py        # GSM8K 子集
│   ├── safety.py           # WildGuard 子集
│   ├── efficiency.py       # latency/tok/s
│   └── runner.py           # 5 ckpt × 4 维度
├── mini_arena/
│   ├── pairwise.py
│   ├── bt_elo.py
│   └── runner.py
├── mini_red_team/
│   ├── gcg_demo.py
│   ├── pair_demo.py
│   └── matrix.py
├── mini_defense/
│   └── pipeline.py
├── portfolio/
│   ├── radar_chart.py     # ASCII 雷达图
│   ├── decision_tree.py   # 场景→ckpt 决策树
│   ├── readme_gen.py      # blog 风格 README 生成
│   └── run.py             # 一键生成 portfolio/*.md
└── tests/test_*.py ×12
```

### 三 Capstone 设计

**Capstone-1: mini-HELM 跑 5 ckpt × 4 维**
- 5 ckpt: vanilla / LoRA / DPO / R1-Zero / Phi-tiny
- 4 维: knowledge (MMLU 10 题) / reasoning (GSM8K 10 题) / safety (HarmBench 5 题) / efficiency (tok/s)
- 输出 markdown 表 + ASCII 雷达图

**Capstone-2: mini-Arena Elo + 红队矩阵**
- mini-Arena: 5 ckpt 互打 100 局 → BT-Elo
- 红队矩阵: 3 攻击 × 5 ckpt → 防御成功率表
- 防御加 Constitutional Classifier 后对照

**Capstone-3: Portfolio README ⭐ 出门作品集**
- 25 专题一句话总结
- 4 主轴时间线（造-改-用-评）
- 雷达图 / 决策树 / 选型表
- blog 风格 markdown
- "我能做什么"画像

### 退出条件
- [ ] 14 lecture + 14 notebook
- [ ] 3 Capstone 全部 PASS
- [ ] tag `评-graduation` ⭐⭐⭐⭐⭐⭐
- [ ] tag `module6-complete`
- [ ] Portfolio README.md 生成到 `learning/eval-graduation/portfolio/`

---

## 十、跨专题工程策略

### 三轨代码（汇总）

| 专题 | minimal | 库 1 | 库 2 |
|------|---------|------|------|
| 1 eval-foundations | 手写 mmlu_runner | lm-evaluation-harness | HELM 简化 |
| 2 reasoning-eval | 手写 gsm8k/math runner | math-verify | sympy verify |
| 3 agent-code-eval | 手写 humaneval exec | bfcl mock | swe-bench mock |
| 4 llm-judge | 手写 BT-Elo | Prometheus 2 mock | mt-bench runner |
| 5 red-team | 手写 GCG/PAIR | jailbench mock | harmbench mock |
| 6 safety-defense | 手写 4 层 pipeline | llama-guard mock | constitutional-classifier mock |
| 7 graduation | 三大 capstone 合一 | — | — |

### Mock vs Real
所有专题保持 "**code 能在 CPU/无 GPU/无外部 API key 跑通**" 的设计：
- judge 用 mock 函数（接收 prompt 输出固定 score）
- 真 LLM 调用做接口预留，但默认 mock 后端
- 真 benchmark 用 10-100 题精选子集

### Git 里程碑
| Tag | 时机 | 内容 |
|-----|------|------|
| `eval-foundations` | Topic 1 末 | MMLU/HELM/lm-eval-harness 全跑 |
| `reasoning-eval` | Topic 2 末 | GSM8K/MATH/AIME/GPQA |
| `agent-code-eval` | Topic 3 末 | HumanEval/SWE-Bench/WebArena |
| `llm-judge-arena` | Topic 4 末 | MT-Bench + Arena Elo |
| `red-team-jailbreak` | Topic 5 末 | GCG/PAIR/AutoDAN/Crescendo |
| `safety-defense` | Topic 6 末 | Constitutional Classifiers + 4 层防御 |
| `评-graduation` | Topic 7 末 | ⭐⭐⭐ 三大 Capstone + Portfolio |
| `module6-complete` | Topic 7 末 | Module 6 整体收官 |

---

## 十一、2025-2026 高影响力方法（用户特别要求）

### 新 benchmark
- **LiveCodeBench / LiveBench**：滚动 / 去污染
- **Humanity's Last Exam** (Scale 2025)：3000 expert 题
- **Arena-Hard v2** (2025)
- **MMLU-Pro** (2024)
- **GPQA Diamond** (2023)
- **ZebraLogic** (2024)
- **AIME 2024/2025**
- **SWE-Bench Verified** (2024)

### 新攻击
- **Crescendo** (Microsoft 2024)
- **Many-shot Jailbreak** (Anthropic 2024)
- **Multi-modal jailbreak (image)** (2024-2025)
- **Prefilling attack**
- **Encoding (Base64 / Cipher)**

### 新防御
- **Constitutional Classifiers** (Anthropic 2025) — 重点
- **Llama Guard 3** (Meta 2024)
- **ShieldGemma** (Google 2024)
- **WildGuard** (AI2 2024)
- **NeMo Guardrails v2** (NVIDIA)

### Judge
- **Prometheus 2** (2024)
- **Skywork-Critic** (2024)
- **JudgeBench** (2024)
- **PandaLM**

### Eval engineering
- **lm-evaluation-harness** (EleutherAI)
- **OpenCompass** (Shanghai AI Lab)
- **HELM v2** (Stanford)
- **LiveBench monthly rotation**
- **Contamination detection (Min-K%++ 2024)**

---

## 十二、风险总览

| 风险 | 缓解 |
|------|------|
| 真 benchmark 数据集太大 | 10-100 题精选子集 + 提供 jsonl |
| GPT-4 judge 需 API key | mock judge 函数，输出可解释分数 |
| 红队代码安全敏感 | 仅自家 ckpt + 不发布有效 prompt |
| Windows GBK 编码 | 沿用 Module 5 教训，避免 ✓✗→ unicode |
| 评测复现性差 | 固定 seed + 5-shot prompt 模板 |
| sandbox 执行 (humaneval) | `exec()` 加 timeout + 受限 namespace |
| 24h+ 训练 | 全用预存 mock ckpt 元数据 |

---

## 十三、实施排期

| 时段 | Topic | tag |
|------|-------|-----|
| 2026-06-05 PM | Topic 1 eval-foundations | `eval-foundations` |
| 2026-06-05 PM | Topic 2 reasoning-eval | `reasoning-eval` |
| 2026-06-05 PM | Topic 3 agent-code-eval | `agent-code-eval` |
| 2026-06-05 PM | Topic 4 llm-judge-arena | `llm-judge-arena` |
| 2026-06-05 PM | Topic 5 red-team-jailbreak | `red-team-jailbreak` |
| 2026-06-05 PM | Topic 6 safety-defense | `safety-defense` |
| 2026-06-05 PM | Topic 7 eval-graduation | `评-graduation` + `module6-complete` ⭐ |

模式：参照 Module 5 节奏，按 Topic 顺序串行推进，每 Topic 结束 commit + tag。

---

## 十四、跨 Module 关系图

```
Module 1 PEFT (3)         ┐
Module 3 造大模型 (8)      ├─→ Module 6 评测/安全 (本)
Module 4 改大模型 (7)      │       ↓
Module 5 用大模型 (7)      ┘  25 专题 Portfolio ⭐
```

完工 = 32 专题（25 + 7）的 LLM 全栈工程师画像 + 出门作品集。

---

## 十五、下一步

本 plan 完成后立即进入 Topic 1 eval-foundations 实施。
