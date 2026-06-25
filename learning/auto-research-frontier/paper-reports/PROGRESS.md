# 论文组会汇报文档 · 建设账本（source of truth，抗 compaction）

> 任务：为 `papers/` 里**全部 40 篇 PDF**各写一份 50 分钟组会级、PPT 风格、约 20 页的 md
> （见 [`_STYLE-GUIDE.md`](_STYLE-GUIDE.md)）。intention/why > how；数学细节全；符号公式前先解释。
> 用户指令：「深度调研每个 paper…20 页…自主全部执行 提交」（2026-06-26）。多次 compaction，子代理并行。

## 执行方式
- 每篇真读对应 PDF（`papers/<id>-<slug>.pdf`），严格按 `_STYLE-GUIDE.md` 写到
  `paper-reports/<id>-<slug>.md`。忠于原文、标章节/公式编号、不编造。
- 我先做 1 篇标杆（2408.06292），其余派子代理并行；抽检 + 分批提交。
- 标杆做完后，子代理 prompt 指向标杆 + 风格规范。

## 进度（40 篇）
状态：⬜ 未做 / 🟡 进行中 / ✅ 已写并抽检

### A. 综述（6）
- ⬜ 2505.13259 survey-automation-to-autonomy ⭐(第一篇/坐标系)
- ⬜ 2503.24047 survey-scientific-intelligence-agents
- ⬜ 2506.18096 survey-deep-research-agents-roadmap
- ⬜ 2507.01903 survey-ai4research
- ⬜ 2508.12752 survey-deep-research-autonomous
- ⬜ 2512.02038 survey-deep-research-systematic

### B. 端到端 AI Scientist（8）
- ✅ 2408.06292 ai-scientist-v1 ⭐⭐(标杆/已做，23.8k字)
- ✅ 2504.08066 ai-scientist-v2-tree-search ⭐
- ✅ 2502.18864 google-ai-co-scientist ⭐
- ⬜ 2501.04227 agent-laboratory
- ⬜ 2505.18705 ai-researcher-hkuds
- ⬜ 2505.16938 novelseek-internagent
- ⬜ 2503.18102 agentrxiv-collaborative
- ⬜ 2411.11910 aigs-automated-falsification

### C. 创意/假设生成（5）
- ✅ 2409.04109 can-llms-generate-novel-ideas ⭐
- ✅ 2506.20803 ideation-execution-gap ⭐
- ⬜ 2404.07738 researchagent-iterative-ideation
- ⬜ 2412.14141 llm-combinatorial-creativity
- ⬜ 2511.02238 deep-ideation-concept-network

### D. Deep Research/综述合成（4）
- ✅ 2402.14207 storm-wikipedia-from-scratch ⭐
- ⬜ 2408.15232 co-storm-unknown-unknowns
- ⬜ 2411.14199 openscholar-ai2 ⭐
- ⬜ 2508.20033 deepscholar-bench

### E. 评测/Benchmark（10）
- ⬜ 2504.01848 paperbench-openai ⭐
- ⬜ 2410.07095 mle-bench-openai ⭐
- ⬜ 2411.15114 re-bench-metr ⭐
- ⬜ 2410.05080 scienceagentbench
- ⬜ 2502.14499 mlgym-meta
- ⬜ 2505.19955 mlr-bench
- ⬜ 2310.03302 mlagentbench
- ⬜ 2510.21652 astabench-ai2
- ⬜ 2409.11363 core-bench-reproducibility
- ⬜ 2407.13168 scicode-benchmark

### F. 自我改进/进化（2）
- ⬜ 2505.22954 darwin-godel-machine ⭐
- ⬜ 2408.08435 adas-agentic-system-design ⭐

### G. 批判/陷阱（4）
- ⬜ 2502.14297 critique-wishful-thinking-ari
- ✅ 2509.08713 critique-hidden-pitfalls ⭐
- ⬜ 2506.01372 critique-fail-without-implementation
- ⬜ 2601.03315 critique-why-not-scientists-yet ⭐

### H. 前沿（1）
- ⬜ 2511.16931 omniscientist-coevolving

## 已提交批次
（每批填：commit + 含哪几篇）
- f6288f4 标杆：2408.06292(v1)
- 批次1(核心6篇)：2504.08066 v2 / 2502.18864 co-scientist / 2409.04109 novel-ideas /
  2506.20803 ideation-gap / 2402.14207 STORM / 2509.08713 hidden-pitfalls

## 最近进度
- 2026-06-26：标杆 + 批次1 共 7/40 完成。子代理+模板+标杆这套质量过关（各 42–53k 字符、
  公式前给直觉+符号、指标有定义式、数字标出处、诚实区分宣称vs实测）。继续批次2…
