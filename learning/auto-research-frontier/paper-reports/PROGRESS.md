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
- ✅ 2505.13259 survey-automation-to-autonomy ⭐(第一篇/坐标系)
- ✅ 2503.24047 survey-scientific-intelligence-agents
- ✅ 2506.18096 survey-deep-research-agents-roadmap
- ✅ 2507.01903 survey-ai4research
- ✅ 2508.12752 survey-deep-research-autonomous
- ✅ 2512.02038 survey-deep-research-systematic

### B. 端到端 AI Scientist（8）
- ✅ 2408.06292 ai-scientist-v1 ⭐⭐(标杆/已做，23.8k字)
- ✅ 2504.08066 ai-scientist-v2-tree-search ⭐
- ✅ 2502.18864 google-ai-co-scientist ⭐
- ✅ 2501.04227 agent-laboratory
- ✅ 2505.18705 ai-researcher-hkuds
- ✅ 2505.16938 novelseek-internagent
- ✅ 2503.18102 agentrxiv-collaborative
- ✅ 2411.11910 aigs-automated-falsification

### C. 创意/假设生成（5）
- ✅ 2409.04109 can-llms-generate-novel-ideas ⭐
- ✅ 2506.20803 ideation-execution-gap ⭐
- ✅ 2404.07738 researchagent-iterative-ideation
- ✅ 2412.14141 llm-combinatorial-creativity
- ✅ 2511.02238 deep-ideation-concept-network

### D. Deep Research/综述合成（4）
- ✅ 2402.14207 storm-wikipedia-from-scratch ⭐
- ✅ 2408.15232 co-storm-unknown-unknowns
- ✅ 2411.14199 openscholar-ai2 ⭐
- ✅ 2508.20033 deepscholar-bench

### E. 评测/Benchmark（10）
- ✅ 2504.01848 paperbench-openai ⭐
- ✅ 2410.07095 mle-bench-openai ⭐
- ✅ 2411.15114 re-bench-metr ⭐
- ✅ 2410.05080 scienceagentbench
- ✅ 2502.14499 mlgym-meta
- ✅ 2505.19955 mlr-bench
- ✅ 2310.03302 mlagentbench
- ✅ 2510.21652 astabench-ai2
- ✅ 2409.11363 core-bench-reproducibility
- ✅ 2407.13168 scicode-benchmark

### F. 自我改进/进化（2）
- ✅ 2505.22954 darwin-godel-machine ⭐
- ✅ 2408.08435 adas-agentic-system-design ⭐

### G. 批判/陷阱（4）
- ✅ 2502.14297 critique-wishful-thinking-ari
- ✅ 2509.08713 critique-hidden-pitfalls ⭐
- ✅ 2506.01372 critique-fail-without-implementation
- ✅ 2601.03315 critique-why-not-scientists-yet ⭐

### H. 前沿（1）
- ✅ 2511.16931 omniscientist-coevolving

## 已提交批次
（每批填：commit + 含哪几篇）
- f6288f4 标杆：2408.06292(v1)
- 批次1(核心6篇)：2504.08066 v2 / 2502.18864 co-scientist / 2409.04109 novel-ideas /
  2506.20803 ideation-gap / 2402.14207 STORM / 2509.08713 hidden-pitfalls

## 最近进度
- 2026-06-26：标杆 + 批次1 + 批次2 共 **13/40** 完成。批次2(B组5+ResearchAgent)：
  agent-lab/ai-researcher/novelseek/agentrxiv/aigs/researchagent，各 37–50k 字符。
  注：批次2 中 4 个子代理触发 API 5h 限额(429)，但文件**已全部落盘且结构完整**(429 只杀了返回消息)。
- 批次3(综述5)→eba71bc(18/40)；批次4(A末+C+D 6篇)→f0af716(24/40)；
  批次5(评测6篇 paperbench/mle/re-bench/scienceagent/mlgym/mlagent)→**30/40**，
  benchmark 指标定义式齐(复现分/pass@k/归一化分/AUP/SR/CBS)。
  剩 10 篇：E×4(mlr-bench/astabench/core-bench/scicode) + F×2(DGM/ADAS) + G×3批判 + H×1。
- 批次6(评测4+自改进2)→251d9cf(36/40)；批次7(批判3+前沿1)→**40/40 全部完成** 🎉。
  全 40 篇各 20–53k 字符、20页骨架、公式前给直觉+先定义符号、指标有定义式、数字标 §/Table/Eq 出处、
  诚实区分「宣称 vs 实测/局限」。子代理并行 + 模板 + 标杆 + 产物落盘账本，扛过一次 API 5h 限额中断。
  收尾：加 paper-reports/README.md 索引。

---

# 第二批扩充（2026-06-26 老师批示：量 ≥70、更新更权威、解读更详尽，**新增 why++ 与 inspires-us**）

> 现有 40 篇**不动**；新增 **34 篇**（全部新、权威、不重复）→ 合计 **74 篇 ≥70**。
> 新增遵循 [`_STYLE-GUIDE-v2-why-and-inspiration.md`](_STYLE-GUIDE-v2-why-and-inspiration.md)：v1 全部要求 + Why 三连 + 强制
> `## ★ 对我们的启发（Inspires Us）` 一节。下载脚本已扩到 74 篇（Virtual Lab bioRxiv 反爬，未纳入）。
> 标杆：v1=2408.06292；**v2 新标杆=2506.13131 AlphaEvolve（我亲自写，演示 why三连+inspires-us）**。

## 第二批清单（34 篇，状态 ⬜未做 / 🟡进行 / ✅已写抽检）
### A 综述(2)
- ⬜ 2508.21148 survey-scientific-llms-data-to-agent
- ⬜ 2501.04306 survey-llm4sr
### B 端到端/多智能体(5)
- ⬜ 2411.00816 cycleresearcher · ⬜ 2505.13400 robin · ⬜ 2506.22653 ursa
- ⬜ 2509.26603 deepscientist · ⬜ 2506.15692 mle-star
### C 创意/假设(5)
- ⬜ 2409.05556 sciagents · ⬜ 2412.17767 researchtown · ⬜ 2404.04326 hypogenic
- ⬜ 2410.09403 many-heads · ⬜ 2506.08140 autosdt
### D Deep Research(4)
- ⬜ 2409.13740 paperqa2 · ⬜ 2504.03160 deepresearcher · ⬜ 2504.21776 webthinker · ⬜ 2507.02592 websailor
### E 评测(7)
- ⬜ 2407.01725 discoverybench · ⬜ 2409.07440 super · ⬜ 2505.24785 exp-bench
- ⬜ 2506.02314 researchcodebench · ⬜ 2503.00096 bixbench · ⬜ 2510.27598 innovatorbench · ⬜ 2504.11524 hypobench
### F 自我改进/算法发现(6)
- 🟡 2506.13131 alphaevolve（v2 标杆，我写） · ⬜ 2506.10943 seal · ⬜ 2410.04444 godel-agent
- ⬜ 2404.18400 llm-sr · ⬜ 2505.22451 ai-mathematician · ⬜ 2510.14150 codeevolve
### G 批判(1)
- ⬜ 2508.16613 critique-biomedical-acceleration-limits
### H 前沿生态(1)
- ⬜ 2508.15126 aixiv-ai-scientist-ecosystem
### I 域内落地发现(3，新增组)
- ⬜ 2304.05376 chemcrow · ⬜ 2508.02956 sparksmatter · ⬜ 2509.06917 paper2agent

## 第二批已提交批次
（每批：commit + 含哪几篇）
- （待填）
