# 09 · 收尾:模拟审稿意见 + Rebuttal 攻防 Capstone

> 总览见 [00-roadmap.md](00-roadmap.md)。这不是第 9 个"分类",是全系列的应用出口——把 01-08 类讲过的
> 判断力工具串成一次完整的攻防:一段摘要 → 四条真实典型的审稿意见(每条对应
> [07 类](07-reviewer-perspective-and-rejection-patterns.md)识别出的一种红旗模式)→ 用 07/08 类
> 已经验证过的代码工具做投稿后的内部分诊 → 逐条写出真正的 rebuttal 正文。

**性质声明**:下面的论文摘要、四条审稿意见、内部分诊数据,全部是**围绕 `research/world-model-
imagination-controller/` 项目真实问题设定(测试时想象预算分配)重新设计的教学场景**,不是这个真实
项目实际收到的审稿意见——这个真实项目目前还在会前准备阶段,尚未投稿,不存在真实的审稿记录。摘要里
出现的具体数字(如 82.0%/63.7%、35.6%→10.0%)取自该项目 `01-meeting-briefing.md` 里真实的 pilot
结果,按设计文档约定的边界直接引用;四条审稿意见是根据 07 类调研到的**真实红旗模式**设计出来的,
不是凭空编造的"稻草人"审稿人。

---

## 一、模拟论文:摘要与文献库片段

**标题**:*Budget-Aware Imagination: Deciding When a World-Model Agent Should Look Ahead*
(按 [01 类](01-narrative-structure-and-elevator-pitch.md)知识点 5 的判断力,标题没有用"novel
architecture"这类容易被追问的词,只描述做了什么判断)

**摘要**(按 01 类知识点 3 的 Context-Gap-Contribution-Result 公式撰写):

> World models let an agent simulate future outcomes before acting, but existing systems spend
> a fixed computation budget on this simulation regardless of whether the current decision
> benefits from it. We show that when an imagined rollout and the agent's baseline policy
> share exactly the same model, spending more computation at decision time does not change the
> expected outcome and only adds variance: across five seeds, the decision-change rate falls
> from 35.6% to 10.0% as the number of sampled candidates grows from 1 to 10, and most changed
> decisions make things worse. We then show that giving the rollout a genuine information
> advantage over the baseline reverses this result: task-conditioned imagination reaches an
> 82.0% hit rate versus 63.7% for unconditioned imagination across three target settings.
> Building on this, we propose a lightweight controller that gates imagination based on an
> estimated information advantage rather than a fixed schedule.

**论文自己的参考文献库(节选,和 07 类知识点 4 的真实案例结构一致)**:

```
[12] Selecting Computations: Theory and Applications
[31] When and How Much to Imagine: Adaptive Test-Time Scaling with World Models
[47] Mastering Diverse Domains through World Models
```

**Method 一节的真实描述(4.1 小节)**:"Our gating network is a 2-layer MLP that takes the current
state and a summary of the candidate rollout as input and outputs a scalar imagine/skip
decision."——按知识点二的六步模板设计,这句描述本身没有问题,但和摘要/Introduction 用"a novel
architecture"这个措辞放在一起看,会成为 07 类知识点 2 的经典红旗。

---

## 二、四条模拟审稿意见

每条意见对应 [07 类](07-reviewer-perspective-and-rejection-patterns.md)已经用真实调研支撑过的
一种红旗模式,不是凭空设计的刁难。

**Reviewer 1(评分 4/10,对应 07 类知识点 4:reinventing the wheel 红旗)**

> The central claim in Section 3 — that additional computation does not change the decision
> when the rollout shares exactly the same model as the baseline policy — appears to be a
> specific instance of the classical Value of Computation framework (Russell & Wefald, 1991),
> later formalized for Bayesian selection problems (already cited in this paper as [12]). The
> paper does not discuss this connection or explain what is genuinely new beyond re-deriving
> a known result in a specific setting. Even setting the novelty question aside, the empirical
> evidence for both findings comes entirely from a small synthetic gridworld. Without
> clarifying the relationship to [12] and what Section 4 adds beyond it, I lean toward
> rejection.

**Reviewer 2(评分 5/10,对应 07 类知识点 3:消融/基线/种子方差三件套缺失)**

> The headline numbers (82.0% vs. 63.7%) are reported as point estimates in the abstract and
> main text without seed counts or variance in most places. I could not find a direct
> comparison against the most relevant recent baseline, [31] (AVIC-R). An ablation over which
> components of the gating signal actually drive the improvement is also missing. Without
> these, it is hard to assess whether the reported gains are robust or specific to this exact
> setup.

**Reviewer 3(评分 5/10,对应 07 类知识点 2:novel architecture 与标准骨干的落差)**

> The abstract and introduction repeatedly describe the controller as "a novel architecture,"
> but Section 4.1 describes it as a standard 2-layer MLP taking [state, candidate summary] as
> input. I don't see what is architecturally novel here. If the real contribution is the
> training signal or the way the gating decision is calibrated rather than the architecture
> itself, the paper should say so explicitly instead of calling it a novel architecture.

**Reviewer 4(评分 6/10,对应 05 类"诚实局限性自曝"的镜像——审稿人主动替作者指出了论文自己没有
充分讨论的局限):**

> The experiments are conducted entirely in a small tabular gridworld (32 states). I
> appreciate the controlled setting for isolating the causal mechanism, but I'm not convinced
> the conclusions transfer to high-dimensional latent world models (e.g., [47]), where the
> rollout and the baseline critic are unlikely to share exactly the same model in the clean
> way this paper assumes. Some explicit discussion of this gap, or a more carefully scoped
> claim, would strengthen the paper.

---

## 三、投稿后第一步:用 07/08 类已验证的工具做内部分诊

收到评审的当天,不是立刻动笔写 rebuttal,而是先做一轮结构化分诊——这正是 07/08 类工具真正的用途:
不是用来"应付审稿人",是用来**在动笔之前把散落在四条评审里的信息整理清楚**。下面这段代码把
[07 类](07-reviewer-perspective-and-rejection-patterns.md)知识点 3/4 和
[08 类](08-rebuttal-writing-techniques.md)知识点 2/4/5 的工具在这个具体场景里完整地跑了一遍:

```python
import re
from collections import Counter

# ---------- 复用07类知识点4:自己文献库红旗扫描 ----------
STOP = {"the", "a", "an", "of", "to", "and", "in", "for", "with", "on", "via", "using",
        "that", "which", "has", "its", "this"}

def stem(word):
    if word.endswith("s") and len(word) > 4:
        return word[:-1]
    return word

def keywords(text):
    words = re.findall(r"[A-Za-z]+", text.lower())
    return {stem(w) for w in words if w not in STOP and len(w) > 3}

def check_reinventing_wheel(discovery_description, own_bibliography_titles, min_overlap=3):
    disc_kw = keywords(discovery_description)
    suspects = []
    for title in own_bibliography_titles:
        title_kw = keywords(title)
        overlap = disc_kw & title_kw
        if len(overlap) >= min_overlap:
            suspects.append({"title": title, "overlap_keywords": sorted(overlap)})
    return suspects

paper_central_claim = (
    "We show that additional computation has no effect on the decision once the rollout and "
    "the baseline policy share exactly the same model. This connects to a classical theory of "
    "selecting when to compute, and has direct applications to imagination gating in world "
    "models."
)
paper_own_bibliography = [
    "Selecting Computations: Theory and Applications",
    "When and How Much to Imagine: Adaptive Test-Time Scaling with World Models",
    "Mastering Diverse Domains through World Models",
]
suspects = check_reinventing_wheel(paper_central_claim, paper_own_bibliography)
assert len(suspects) == 2  # 两条都被命中,含义不同,正文会分别解读
print("=== the self-check that should have happened pre-submission: bibliography red-flag scan (run after the fact, to verify R1's concern holds up) ===")
for s in suspects:
    print(" ", s)

# ---------- 复用07类知识点3:严谨性记分卡 ----------
CHECKS = {
    "has_seed_count": r"\b\d+\s*seeds?\b",
    "has_std_or_ci": r"±|standard deviation|confidence interval|\bstd\b",
    "mentions_baseline": r"\bbaseline\b",
    "mentions_ablation": r"\bablat\w*",
}
def rigor_scorecard(results_text):
    hits = {name: bool(re.search(pat, results_text, re.IGNORECASE)) for name, pat in CHECKS.items()}
    return {"scorecard": hits, "score": sum(hits.values()), "max_score": len(CHECKS)}

submitted_results_text = (
    "Our controller reaches an 82.0% hit rate compared to 63.7% for the unconditioned "
    "baseline across three target settings."
)
revised_results_text = submitted_results_text + (
    " All numbers are mean ± std over 5 seeds. We additionally report a comparison against "
    "AVIC-R and a full ablation over gating signal components in Table 4."
)
print("\n=== submitted version vs. the planned revised version: rigor scorecard (verifying R2's concern holds up) ===")
print("submitted:", rigor_scorecard(submitted_results_text))
print("revised (version promised in rebuttal):", rigor_scorecard(revised_results_text))
assert rigor_scorecard(submitted_results_text)["score"] <= 1
assert rigor_scorecard(revised_results_text)["score"] == 4

# ---------- 复用08类知识点2:话题去重与共性识别 ----------
reviewer_issues = {
    "R1": ["reinventing_wheel_voc", "synthetic_env_scope"],
    "R2": ["missing_baseline_avic", "no_seed_variance", "missing_ablation"],
    "R3": ["novelty_architecture_mismatch"],
    "R4": ["synthetic_env_scope", "generalization_to_latent_models"],
}
all_issues = [iss for issues in reviewer_issues.values() for iss in issues]
counts = Counter(all_issues)
shared = {k: v for k, v in counts.items() if v > 1}
print("\n=== real topic count after deduping the four reviewers' comments ===")
print(f"distinct topics: {len(counts)}, shared (merge into one reply): {shared}")
assert shared == {"synthetic_env_scope": 2}
assert len(counts) == 7

# ---------- 复用08类知识点5:按严重程度加权分配750词预算 ----------
severity = {
    "reinventing_wheel_voc": 5, "missing_baseline_avic": 4, "no_seed_variance": 3,
    "missing_ablation": 3, "novelty_architecture_mismatch": 3,
    "synthetic_env_scope": 2, "generalization_to_latent_models": 2,
}
concerns = [{"topic": t, "severity": s} for t, s in severity.items()]

def weighted_budget(concerns, total_words=750, min_words_per_topic=40):
    total_severity = sum(c["severity"] for c in concerns)
    remaining = total_words - min_words_per_topic * len(concerns)
    assert remaining >= 0
    return [{"topic": c["topic"],
              "words": round(min_words_per_topic + remaining * (c["severity"] / total_severity))}
             for c in concerns]

budget = weighted_budget(concerns)
print("\n=== 750-word budget allocation (severity-weighted) ===")
for b in sorted(budget, key=lambda x: -x["words"]):
    print(f"  {b['topic']}: {b['words']} words")
assert max(budget, key=lambda b: b["words"])["topic"] == "reinventing_wheel_voc"

# ---------- 复用08类知识点4:每条意见的回应策略 ----------
def choose_response_strategy(request, feasible, changes_core_claim):
    if feasible:
        return "commit_with_concrete_plan"
    if not changes_core_claim:
        return "argue_not_decisive"
    return "concede_scope_limit"

strategies = {
    "missing_ablation": choose_response_strategy("gating-signal ablation", True, False),
    "no_seed_variance": choose_response_strategy("5-seed variance", True, False),
    "missing_baseline_avic": choose_response_strategy("AVIC-R comparison", True, False),
    "generalization_to_latent_models": choose_response_strategy(
        "DreamerV3-scale validation", False, True),
    "reinventing_wheel_voc": choose_response_strategy(
        "reframe contribution around task-conditioning result", True, False),
}
print("\n=== response strategy for each real topic ===")
for k, v in strategies.items():
    print(f"  {k}: {v}")
assert strategies["generalization_to_latent_models"] == "concede_scope_limit"
assert strategies["missing_ablation"] == "commit_with_concrete_plan"
print("\nOK: the triage pipeline (red-flag scan -> rigor scorecard -> topic dedup -> weighted budget -> response strategy) runs end to end")
```

本机实测输出摘录:
- 文献库红旗扫描命中**两条**,不是预想中的一条——除了真正的"reinventing the wheel"来源
  `[12] Selecting Computations`,`[31]`(AVIC)也被列为高关键词重叠条目。**这不是工具的误报,
  是一个有价值的提醒**:关键词重叠工具分不清"这是同一个定理"和"这是一个应该被拿来比较的强相关
  竞品工作"——但两者都值得回头精读确认,而 `[31]` 恰好正是 Reviewer 2 独立要求对比的同一篇工作,
  说明这个"廉价初筛"工具即使给出的判断理由是模糊的,指向的方向仍然是对的。
- `submitted_results_text` 的严谨性记分卡只有 1/4(仅命中"mentions_baseline",因为句子里出现了
  "baseline"这个词,但没有种子数/方差/消融字样),印证 Reviewer 2 的质疑并非无的放矢。
- 四条意见共提出 7 条真实不同的话题,其中 `synthetic_env_scope` 被 R1 和 R4 独立提出,应合并
  回复;分配 750 词预算时,`reinventing_wheel_voc`(R1 的核心质疑)拿到最多篇幅(147 词),
  与它对论文可接受性的潜在杀伤力相称。
- 五条可行性判断中,只有"扩展到 DreamerV3 规模的验证"被判定为`concede_scope_limit`(做不到,
  且确实是核心结论的边界),其余全部可以在 rebuttal 窗口内给出具体承诺。

---

## 四、Rebuttal 正文(按 08 类技巧组织)

按 [08 类](08-rebuttal-writing-techniques.md)知识点 2 的结构建议:开头总结改动,正文按去重后的
话题(不是按审稿人)组织,`synthetic_env_scope` 合并回复,每位审稿人在正文中都被点名提及至少一次。

> **Summary of changes.** We thank all four reviewers for their careful reading. In response,
> we (1) reframe our central contribution around the task-conditioning result rather than the
> computation-invariance result, and explicitly relate the latter to prior Value-of-Computation
> theory [12]; (2) add a direct comparison to AVIC-R [31] and a component-wise ablation of the
> gating signal, both with variance over 5 seeds; (3) revise the abstract and Section 4.1 to
> describe our contribution as a calibrated gating *signal*, not a novel architecture; and (4)
> add an explicit scoping paragraph discussing generalization to high-dimensional world models.
>
> **On the relationship to prior Value-of-Computation theory (Reviewer 1).** We agree, and this
> is the most important change in this revision. Our computation-invariance result (old Section
> 3) is indeed a specific instance of the classical VOC stopping rule [12]; we do not claim it
> as a new finding, and we have rewritten Section 3 to present it explicitly as a controlled,
> reproducible instantiation of that theory rather than a discovery. Our actual novel claim is
> the task-conditioning result (old Section 4): giving the rollout a genuine, decision-relevant
> information advantage reverses the sign of the comparison, and this reversal's magnitude
> scales continuously with how much of the advantage is exposed, which — to the best of our
> searching — is not addressed by [12] or by the more recent gating methods that assume a
> binary "imagine or not" decision. We have moved this result to the paper's title and abstract
> as the primary contribution.
>
> **On missing comparisons and variance (Reviewer 2, related to Reviewer 1's scope point).**
> We agree this was a real gap. We have re-run our main experiments (5 seeds; numbers now
> reported as mean ± std throughout) and add a direct comparison to AVIC-R [31]: our controller
> reaches 82.0% ± 1.4% versus AVIC-R's <specific number to be measured>% on the same synthetic
> benchmark. We also add a component ablation (new Table 4) isolating the contribution of each
> input to the gating signal.
>
> **On "novel architecture" (Reviewer 3).** This is a fair criticism of our phrasing, not of
> the underlying claim. Our gating network is intentionally a plain 2-layer MLP — the
> contribution is the training signal and calibration procedure that decides *when* to trust
> imagination, not the network topology. We have removed "novel architecture" from the abstract
> and introduction and replaced it with language specific to the gating signal and its
> calibration target.
>
> **On generalization beyond the synthetic gridworld (Reviewers 1 and 4).** We acknowledge this
> is a genuine boundary of the current work, not something we can resolve within the rebuttal
> window. The synthetic environment was a deliberate choice — it lets us compute the ground-truth
> optimal policy and isolate the causal mechanism from measurement noise, which would not be
> possible in a high-dimensional environment like [47] without a separate, expensive evaluation
> protocol. We have added an explicit paragraph in the revised Limitations section stating that
> our claims are scoped to settings where the rollout and the baseline can be shown to share (or
> knowably not share) the same model, and that extending the diagnostic protocol to latent world
> models such as [47] is future work, not a claim this paper makes.

**这份 rebuttal 里能对照到的具体技巧(呼应 08 类):**
- 开头"Summary of changes"直接对应 08 类知识点 2 的结构建议。
- 回应 Reviewer 1 时用了"we agree""we do not claim it as a new finding"——诚实承认,同时说明
  贡献的价值被重新定位,不是被推翻,对应 [05 类](05-limitations-and-honest-disclosure.md)知识点 5
  和 08 类知识点 4 的 `commit_with_concrete_plan`/坦然承认策略。
- 全文没有出现"flaw"/"mistake"/"final version",用的是"a fair criticism"/"a genuine boundary"/
  "revised Limitations section"这类留有余地但不失诚恳的措辞,对应 08 类知识点 3。
- 回应 Reviewer 4(和 Reviewer 1 的 `synthetic_env_scope` 部分)时,没有硬辩"我们的合成环境已经
  足够",而是诚实解释了这个选择的方法论理由**并且承认了范围边界**——对应 08 类知识点 4 的
  `concede_scope_limit` 策略,也呼应 05 类"局限性连接到对结果解读的影响"这条纪律。
- 回应 Reviewer 2 时给出的是可以核实的具体行动(重新跑 5 个种子、加 AVIC-R 对比、加消融表),
  不是"我们会考虑"这种空泛承诺,对应 08 类知识点 4 的 `commit_with_concrete_plan`。

---

## 五、如果这是真实场景,接下来大概率会发生什么

按 [08 类](08-rebuttal-writing-techniques.md)知识点 1 引用的真实统计(*Insights from the ICLR
Peer Review and Rebuttal Process*,19000+ 篇论文/74000+ 条评审):分数在 rebuttal 后提升的论文,
最终录用率是 55.7%-57.6%;分数不变的论文只有 7.8%-12.4%。这份模拟 rebuttal 如果要真正提升分数,
最关键的一步不是"语言技巧用得多漂亮",而是**Reviewer 1 提出的核心质疑是否被真正诚实地处理**——
如果论文只是嘴上说"我们同意",却没有真的把 Introduction/摘要的核心贡献叙事改掉,Reviewer 1 大概率
会在后续讨论阶段追问"贡献列表第一条现在具体是什么",这正是
[01 类](01-narrative-structure-and-elevator-pitch.md)知识点 4 讲的"贡献要收敛到一个能扛住追问
的核心",在 rebuttal 阶段依然适用,不是只有投稿前才用得上。

**诚实的收尾**:这个模拟场景设计成"四条意见都能被合理回应、且论文的核心贡献本身依然站得住"这种
相对理想的情况——现实中相当一部分论文收到的意见是没有办法在一轮 rebuttal 里妥善处理的(比如
Reviewer 1 如果认定"task-conditioning 这个结果本身也已经被某篇文献做过",而这个判断是正确的,
那这篇论文可能就是真的需要大改甚至改投的情况,rebuttal 技巧再好也无法把一个真实站不住的贡献
说服成立)。**Rebuttal 写作技巧能做的是确保一篇本身站得住的论文,不会因为呈现方式的问题被误伤**;
它不能把一篇真正有硬伤的论文伪装成没有硬伤——这条边界呼应 [07 类](07-reviewer-perspective-and-rejection-patterns.md)
的核心立场:识别红旗的目的是尽早自己发现问题并解决,不是学会怎么在红旗被发现后蒙混过关。

---

## 六、这个 capstone 串起来的方法论回顾

| 环节 | 用到的知识点 |
|---|---|
| 摘要撰写 | [01 类](01-narrative-structure-and-elevator-pitch.md)知识点 3(C-C-C 摘要公式)、
  知识点 5(标题判断力) |
| Method 措辞和摘要的落差 | [07 类](07-reviewer-perspective-and-rejection-patterns.md)知识点 2 |
| 严谨性缺口(种子/方差/基线/消融) | [07 类](07-reviewer-perspective-and-rejection-patterns.md)
  知识点 3、[03 类](03-method-and-results-presentation.md)知识点 3 |
| 核心贡献是否是已知理论的推论 | [07 类](07-reviewer-perspective-and-rejection-patterns.md)
  知识点 4、[01 类](01-narrative-structure-and-elevator-pitch.md)知识点 4、
  [05 类](05-limitations-and-honest-disclosure.md)知识点 5 |
| Rebuttal 结构与话题去重 | [08 类](08-rebuttal-writing-techniques.md)知识点 2 |
| 承认 vs 辩护的语言技巧 | [08 类](08-rebuttal-writing-techniques.md)知识点 3 |
| "做不到的实验"怎么回应 | [08 类](08-rebuttal-writing-techniques.md)知识点 4 |
| 有限篇幅的优先级分配 | [08 类](08-rebuttal-writing-techniques.md)知识点 5 |

全系列 01-08 类讲的判断力,最终都在"一篇论文从投稿到被评审"这条真实链路上派上用场——这也是这篇
capstone 选择"模拟审稿+rebuttal 攻防"而不是另一种形式的原因:它不是一次新的知识点讲解,是对前面
八篇内容的一次真实场景压力测试。

---

*上一篇:[08-rebuttal-writing-techniques.md](08-rebuttal-writing-techniques.md)。返回
[00-roadmap.md](00-roadmap.md)。*
