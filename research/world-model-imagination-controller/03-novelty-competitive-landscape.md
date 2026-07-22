# Novelty 风险 / 竞争格局详细核验记录

> 这是 [`02-deep-gap-analysis.md`](02-deep-gap-analysis.md) 结论背后的完整调研过程和逐篇核验记录,
> 供需要查证某个具体论断出处时使用。核验方法:能查到的一律用 WebFetch 对照 arXiv 原文(不满足于
> 摘要转述,优先看正文/HTML全文),每条给出"确认/有出入/agent二手转述未再核实"三档状态,不含糊。

---

## 1. 调研方法与过程(含限额中断的完整记录)

按照"不能偷懒"的要求,原计划六路独立子agent并行调研:①idea1/7抢发风险、②近8-10周最新文献扫描、
③相邻领域空白挖掘、④理论框架锋利度复核、⑤自有pilot发现文献先例核查、⑥头部实验室战略动向扫描。

**六路首轮全部在中途撞上API 5小时限额被强制中断**(`api key 5小时限额已用完`)。处理方式:没有
立即重新发起全部六路(会再次触发同一限流),而是先用 SendMessage 续跑进展最好的三路(①③⑥,
均已产出可用的中间发现),让它们完成收尾;②④证据在①③⑥完成后被充分覆盖,判断为无需单独补跑;
⑤(自有发现文献先例)因为是独立于其他五路的独特问题,零进度重新发起了一次全新调用,顺利完成。

七路(含⑤重新发起)全部完成后,我又亲自用 WebFetch 直接核验了约17篇最关键的新发现论文——这是
必要的一步,不是走流程:子agent报告里已经出现至少一次"两次检索给出的arXiv编号互相矛盾"的情况
(见下文"Inference-Time Scaling Laws for Embodied Agents"条目),说明agent二手转述不能直接采信,
必须按本项目一贯的纪律逐条核实才能写进正式文档。

---

## 2. 核验状态总表(F组20篇 + 部分F组外论文)

| 论文 | arXiv ID | 核验方式 | 状态 |
|---|---|---|---|
| Astra (Thinking with Imagination) | 2606.06476 | 我方WebFetch摘要页 | 确认标题/作者/机制,RL课程学习非理论驱动 |
| Imagine-then-Plan (ITP) | 2601.08955 | 我方WebFetch **HTML全文**(非摘要) | **逐条确认**POIMDP/K-head predictor/代价函数/4个benchmark全部属实 |
| ELASTIC | 2606.31132 | 我方WebFetch摘要页 | 确认标题/作者/机制,非world model |
| Finding the Time to Think | 2606.26463 | 我方WebFetch摘要页(x2轮,agent+我) | 确认标题/作者/日期/摘要机制;SMDP/MCTS/预算结转等细节摘要未明确写出,无法证伪也无法证实 |
| ROI-Reasoning | 2601.03822 | 我方WebFetch摘要页(x2轮) | 确认标题/作者/knapsack形式化;"不可逆限制后续预算"是我方推断措辞,原文未逐字使用 |
| World-in-World | 2510.18135 | 我方WebFetch摘要页(x2轮)+2026-07-23 OpenReview/ICLR官方日程交叉核实 | 确认标题/作者/提交日期(2025-10-20)/代码仓库;**ICLR接收状态已确认为ICLR 2026 Oral**(详见`07-baseline-reproducibility-audit.md`) |
| AdaNav | 2509.24387 | 我方WebFetch摘要页(x2轮) | 确认标题/作者/机制 |
| RARRL (When Should a Robot Think) | 2603.16673 | 我方WebFetch摘要页 | 确认标题(15位作者)/机制/ALFRED基准 |
| Current Agents Fail to Leverage World Model | 2601.03905 | 我方WebFetch摘要页 | 确认标题/具体数字(<1%触发率/~15%误用/最多5%性能下降) |
| Active Inference Test-Time Scaling Law | 2606.22813 | 我方WebFetch摘要页 | 确认标题/6位作者/机制(软贝叶斯推断,非二元门控) |
| Cognitive Friction | 2603.30031 | 我方WebFetch摘要页 | 确认标题/单作者/HJB停止边界机制;"明确引用Russell&Wefald"和"optimal-stopping envelope"逐字措辞**未在摘要中确认**(原文用"HJB-inspired stopping boundary") |
| Sezener & Dayan (VOC in MCTS) | 2002.04335 | 我方WebFetch摘要页 | 确认标题/作者/机制/UAI 2020 |
| Value Equivalence Principle | 2011.03506 | 我方WebFetch摘要页 | 确认标题/作者(Grimm/Barreto/Singh/Silver)/NeurIPS 2020/核心论点 |
| Beyond the One-Step Greedy Approach | 1802.03654 | 我方WebFetch摘要页 | 确认标题/作者/ICML 2018;"证明无单调改进保证"这个具体措辞摘要未直接支持,原文摘要只说"首次系统分析+证明收敛性" |
| Multiple-Step Greedy Policies | 1805.07956 | **未核验,仅agent转述** | PDF已下载确认存在,机制描述待正式写作前核实 |
| Hamrick et al. role of planning | 2011.04021 | **未核验,仅agent转述** | 同上 |
| Hanna & Corrado | 2506.17124 | **未核验,仅agent转述** | 同上 |
| Metacontrol | 1705.02670 | **未核验,仅agent转述** | 同上(2017 DeepMind,历史脉络类引用,风险相对低) |
| Thinker | 2307.14993 | **未核验,仅agent转述** | 同上(2023 NeurIPS,风险相对低) |
| LLMs Cannot Self-Correct | 2310.01798 | **未核验,仅agent转述** | 同上(2024 ICLR,较知名,风险相对低) |
| Hay et al. Selecting Computations | 1207.5879 | **已在原68篇库里,本次未重新核验** | 早在初版文献调研阶段已核验 |
| "Inference-Time Scaling Laws for Embodied Agents" | **未确认** | 我方WebSearch | **红旗**:只在第三方镜像站(attractorstate.com)找到,该站URL字符串含"2405.14005"(2024年格式ID),与"2026"论文的宣称矛盾。**不应引用此论文**,除非能在arxiv.org本站直接确认其真实ID。搜索过程中意外找到一篇独立确认真实存在的相关论文(Active Inference Test-Time Scaling Law,2606.22813,已收录F组),可以替代使用 |

---

## 3. 十个idea逐一风险重估(不止1/7/10,补全其余7个)

以下是这次调研对全部10个idea的完整过一遍,`02-deep-gap-analysis.md` §6 只列了受冲击最大的4个,
这里补全其余部分供完整留档:

- **idea 2(投机式想象draft-then-verify)**:本次调研没有发现新的直接竞争论文。Video-T1仍是最贴近
  的同期工作(已在第一轮审计里核验过),风险判断不变。
- **idea 4(conformal校准停止规则)**:本次调研的Cognitive Friction用的是HJB最优停止而非conformal
  prediction,C3(2512.05927)仍是训练时校准而非事后conformal方法,没有发现直接撞车的新论文,
  风险判断不变,但可以补充引用ROI-Reasoning/Finding-the-Time-to-Think作为"budget-aware stopping"
  这个大类下的邻近工作。
- **idea 5(免集成分歧估计)**:定位是"组件级,不建议独立成篇",本次调研没有专门排查这条,维持原判断。
- **idea 6(观测驱动反应式重规划,扩展FFDC)**:本次调研没有发现新的多候选版本FFDC扩展论文,风险
  判断不变(仍然是"需要真实/高保真机器人环境,时间风险高于idea本身好不好"这个瓶颈)。
- **idea 8(预算受限想象树搜索,adaptive submodularity理论保证版)**:和idea 3一样属于"理论驱动"
  类,本次调研的核心发现(VOC/最优停止理论有明确的方法论空当)同样支撑idea 8,但adaptive
  submodularity本身更适合"树结构展开选择"这个更具体的子问题,而不是"该不该想/想多深"这个更基础
  的门控问题——如果理论路线被选中,idea 3(VOC)和idea 8(adaptive submodularity)的关系需要
  在会议上明确,大概率仍然是"二选一,不建议同时做"这个原来的判断成立。
- **idea 9(oracle蒸馏为轻量门控)**:本次调研没有发现直接竞争论文,风险判断不变,天花板相对最低
  这个原判断也不变。

---

## 4. 特别记录:两篇论文的具体技术细节全文核验(供写作时直接引用)

### Imagine-then-Plan (arXiv 2601.08955) —— 全部技术claim均逐字核对HTML全文原句

- POIMDP:"We move beyond the Partially Observable Markov Decision Process (POMDP)... toward a
  Partially Observable and Imaginable MDP (POIMDP)"
- K-head predictor:"we augment the agent with a lightweight K-head predictor Pθ(Kt|st), which is
  a linear layer built on top of a backbone LLM",用A2C联合优化
- 代价函数:"K̃t = argmax₀≤k≤Kmax [log pθ0(at\*|st,τ̂t(k)) − λK·k]"
- 世界模型性质:"we learn a LLM-based world model Mφ that approximates the environment dynamics
  pφ(s'|s,a)",token级自回归文本生成,不是隐空间动力学模型
- 决策机制:先一次性选定Kt,再完整执行Kt步rollout,三阶段流程(horizon selection→world-model
  imagination→reflective policy generation)里没有中途止损设计

### Cognitive Friction (arXiv 2603.30031) —— 摘要层面核验,机制细节需正式写作前读全文

- 标题:"Cognitive Friction: A Decision-Theoretic Framework for Bounded Deliberation in
  Tool-Using Agents",单作者Davide Di Gioia
- 核心机制:三元认知架构(Triadic Cognitive Architecture),"combining nonlinear filtering,
  congestion-dependent cost dynamics, and HJB optimal"停止,基于rollout近似计算信息价值
- 摘要用词是"value-of-information"和"HJB-inspired stopping boundary",**不是**逐字的
  "Value of Computation"或"optimal-stopping envelope"(这是此前agent转述时用词更接近我们
  自己框架的措辞,实际论文原文措辞略有不同,引用时要用论文自己的原始表述)

---

## 5. 本次调研没有解决的问题(明确留白,不要假装查清楚了)

1. ~~World-in-World是否真的是ICLR 2026 Oral~~ ——**已于2026-07-23解决**:OpenReview页面
   ([openreview.net/forum?id=yDmb7xAfeb](https://openreview.net/forum?id=yDmb7xAfeb))与ICLR
   官方日程交叉核实,确认是ICLR 2026 Oral,详见`07-baseline-reproducibility-audit.md`。
2. F2组后6篇论文的具体机制描述(尤其"是否证明了无单调改进保证"这类精确的理论claim)只做了
   PDF存在性验证,没有逐字核对论文正文,写作前必须补做。
3. "Inference-Time Scaling Laws for Embodied Agents"这篇论文的真实身份完全没有确认清楚,不应
   在任何后续文档里引用,除非能找到arxiv.org本站的直接确认。
4. 竞争格局扫描依赖的"招聘信息""大厂博客"类证据本身就偏间接推测性质,子agent自己也标注了这点,
   这里如实转达,不因为是关键结论就掩盖证据强度的局限。
