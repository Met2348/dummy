# guide_Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena

<!-- manual-deep-guide -->

> 原论文: [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685)
>
> 本地原文 PDF: `learning/llm-judge-arena/paper/01_mt_bench_chatbot_arena.pdf`
>
> 作者: Zheng et al., LMSYS
>
> 年份: 2023
>
> 类型: paper

## 0. 这篇论文到底在改写什么

这篇论文解决的是一个很现实的问题: 传统 NLP benchmark 很难评价 chat assistant 的真实使用体验。一个模型在 MMLU、TruthfulQA、HumanEval 这类闭集或短答案任务上分数不错，不代表它在多轮对话、指令遵循、解释质量、风格偏好上更受用户喜欢。

作者提出两套互补评测:

1. MT-Bench: 80 个高质量多轮开放问题，用来测试 chat model 的多轮对话和指令遵循能力。
2. Chatbot Arena: 匿名双模型对战平台，让真实用户和两个匿名模型同时对话，然后投票更喜欢哪一个。

但人类评测慢且贵，所以论文系统研究了 LLM-as-a-judge: 用强 LLM, 主要是 GPT-4, 代替人类裁判，对开放回答做 pairwise comparison 或 single answer grading。论文的核心结论是: 经过一定偏差缓解后，GPT-4 judge 和人类偏好的一致率可以超过 80%, 达到人类之间一致率的水平。

这篇论文真正重要的地方不是“GPT-4 可以打分”这句口号，而是它把自动开放式评测拆成了一个可审计流程:

- 评测题如何构造。
- 两个回答如何匿名比较。
- 裁判提示词如何写。
- position bias、verbosity bias、self-enhancement bias、reasoning failure 如何测。
- LLM judge 和专家/众包人类的一致性如何计算。
- 自动评测和传统标准化 benchmark 如何互补。

这对你利用 AI agent 学习尤其重要。Agent 可以当老师、考官、reviewer，但它不是天生可靠的真理机器。你必须学会给 agent 设计评测协议，也必须学会审计它的偏差。

## 1. 论文结构地图

原文可以按下面顺序读:

1. Abstract: 给出主要发现，GPT-4 judge 可达到超过 80% 的人类一致率，并公开 MT-Bench、3K expert votes、30K arena conversations。
2. Introduction: 解释为什么 MMLU/HELM 等传统 benchmark 不能充分反映人类对 chat assistants 的偏好。
3. Section 2 MT-Bench and Chatbot Arena: 定义 80 个多轮问题和匿名双模型对战平台。
4. Section 3 LLM as a Judge: 定义三种 judge 形式，并系统分析 position、verbosity、self-enhancement、math/reasoning 失败。
5. Section 3.4 Addressing limitations: 讲 swap positions、few-shot judge、CoT judge、reference-guided judge。
6. Section 3.5 Multi-turn judge: 解释为什么多轮评测要把完整 conversation 放进一个 prompt。
7. Section 4 Agreement Evaluation: 主证据，比较 GPT-4/GPT-3.5/Claude 与人类在 MT-Bench 和 Chatbot Arena 上的一致性。
8. Section 5 Human Preference Benchmark and Standardized Benchmark: 说明 MT-Bench/Arena 与 MMLU/TruthfulQA 互补。
9. Section 6 Discussion: 局限性，尤其是论文主要看 helpfulness，较少覆盖 safety/honesty/harmlessness。
10. Appendix: 很重要，包含 judge prompts、人类数据收集、position bias 分解、agreement 计算细节、Vicuna judge fine-tuning。

建议优先精读 Figure 1、Table 2、Table 3、Table 4、Table 5、Table 6、Table 8、Figure 5 到 Figure 10 的 prompt templates。它们分别对应: 为什么多轮偏好评测必要、judge 偏差、math 误判、与人类一致性、标准化 benchmark 的互补、实际 prompt 格式。

## 2. 背景: 为什么传统 benchmark 不够

传统 benchmark 主要评价模型的 core capability。例如:

- MMLU 测多学科选择题。
- GSM8K 测数学 word problem。
- HumanEval 测代码生成。
- HellaSwag、ARC、WinoGrande 测常识和推理。

这些任务通常有标准答案，便于自动判分。它们很重要，但不等价于用户偏好。一个 chat assistant 的真实价值还包括:

- 能不能遵循复杂指令。
- 能不能在第二轮理解上下文。
- 能不能解释清楚。
- 能不能把答案写成用户想要的风格。
- 能不能在开放问题里兼顾准确、相关、深入、创造性和细节。

论文 Figure 1 用 LLaMA-13B 和 Vicuna-13B 的例子说明这个差异。LLaMA base 在传统 benchmark 上可能有竞争力，但面对开放 follow-up 问题时，回答会重复、缺少例子、不贴合用户需求。Vicuna 的开放对话体验更好，这种差异不是 MMLU 选择题能完整衡量的。

所以论文的问题不是“传统 benchmark 错了”，而是“传统 benchmark 测的是另一种东西”。正确评价 chat assistant 需要同时看 core capability 和 human preference。

## 3. MT-Bench: 80 个多轮开放问题

MT-Bench 是一个小而精的 controlled benchmark:

- 80 个 high-quality multi-turn questions。
- 8 个类别，每类 10 个问题。
- 每个问题有两轮，第一轮提问，第二轮 follow-up。

8 个类别是:

- writing。
- roleplay。
- extraction。
- reasoning。
- math。
- coding。
- knowledge I, 也就是 STEM。
- knowledge II, 也就是 humanities/social science。

它的设计目的不是覆盖所有任务，而是用比较少的高质量开放问题区分 chatbots。多轮结构是关键。很多模型第一轮能答，但第二轮必须引用前文、遵循新约束、改写前一答复、或者在原推理上继续推进。

论文里一个典型问题是:

```text
Turn 1:
Compose an engaging travel blog post about a recent trip to Hawaii.

Turn 2:
Rewrite your previous response. Start every sentence with the letter A.
```

第二轮不是新问题，而是对第一轮输出的再约束。这正是 chat assistant 和普通 QA benchmark 的区别。

## 4. Chatbot Arena: 匿名对战收集偏好

Chatbot Arena 是一个 crowdsourced battle platform。用户同时和两个匿名模型对话，两个模型收到同样的问题。用户投票:

```text
Model A wins
Model B wins
Tie
Both bad
```

投票后才揭示模型身份。这种匿名设计减少品牌偏见和先验印象。论文当时运行一个月后收集了约 30K votes。论文主实验中又随机采样了 3K single-turn votes，覆盖 GPT-4、GPT-3.5、Claude、Vicuna、Koala、Alpaca、LLaMA、Dolly 等模型，并来自 2114 个 unique IPs。

MT-Bench 和 Arena 的差别是:

- MT-Bench 是 controlled: 固定 80 个问题，方便复现和横向比较。
- Arena 是 in the wild: 用户自由提问，覆盖真实使用分布，但噪声更大。

这两个东西互补。只做 MT-Bench 容易题集过窄；只做 Arena 又不够可控，用户问题分布也会变。论文同时用两者来验证 LLM judge 是否接近人类偏好。

## 5. LLM-as-a-Judge 的三种形式

论文定义了三类 LLM judge:

第一，pairwise comparison。给 judge 一个问题和两个回答，让它输出 A 更好、B 更好或 tie。这种方式适合 Arena，因为人类偏好本来就是相对比较。

```text
question
answer_a
answer_b
judge -> A / B / tie
```

第二，single answer grading。给 judge 一个问题和一个回答，让它给 1 到 10 分。论文后面用 GPT-4 single-answer grading 给 MT-Bench 打分，每个 turn 一次，最终平均 160 个分数位置，也就是 80 questions x 2 turns。

```text
question
answer
judge -> score 1..10
```

第三，reference-guided grading。对数学、推理等有参考答案的任务，把 reference answer 也放进 judge prompt，让 judge 对照参考答案判断。

```text
question
reference_answer
answer_a
answer_b
judge -> A / B / tie
```

三种方式各有取舍。Pairwise 更适合细微偏好，但模型数量增加时 pair 数量是二次增长。Single grading 更可扩展，但绝对分可能随 judge model 和 prompt 改变而漂移。Reference-guided 对可验证题更可靠，但不是所有开放问题都有 reference。

## 6. Prompt 设计不是小事

论文 Appendix A 给了实际 judge prompt。默认 pairwise prompt 要求 judge 考虑:

- helpfulness。
- relevance。
- accuracy。
- depth。
- creativity。
- level of detail。

同时明确要求:

- 避免 position bias。
- 不要让回答长度影响判断。
- 不要偏好 assistant 名字。
- 输出必须严格是 `[[A]]`, `[[B]]`, `[[C]]`。

这些指令不是装饰。LLM judge 如果输出格式不稳定，就无法批量解析；如果不明确提醒 position/length/name，它的偏差会更明显。

多轮 judge 的 prompt 更 tricky。论文发现，把两轮拆成两个 prompt 会让 GPT-4 搞错“第二个例子”到底是 Assistant A 上一轮的第二个例子，还是 Assistant B 上一轮的第二个例子。Figure 16 就是这种失败。最后作者倾向于把完整 conversation 放进一个 prompt，让 judge 看到每个 assistant 自己的上下文。

这对 agent 学习很有启发: 你让 agent 评价你的答案时，必须把题目、你的回答、参考标准、对话历史都给齐。上下文组织错误，judge 很可能评价错对象。

## 7. 偏差 1: Position Bias

Position bias 是 judge 偏好某个位置的回答，例如总喜欢第一个答案。论文用一个 swap 实验测这个偏差:

```text
prompt: same question
order 1: answer_a = model X, answer_b = model Y
order 2: answer_a = model Y, answer_b = model X

consistent if winner swaps accordingly
```

Table 2 的结果很醒目:

- Claude-v1 default prompt consistency 23.8%, biased toward first 75.0%。
- GPT-3.5 default prompt consistency 46.2%, biased toward first 50.0%。
- GPT-4 default prompt consistency 65.0%, biased toward first 30.0%。

GPT-4 最好，但仍然不完美。论文也测试了 rename prompt，发现 Claude 还有 name bias，会偏好 Assistant A 这个名字。

缓解方法有两种:

1. Conservative swap: 调两次 judge，交换 A/B 顺序。只有两次都支持同一实际模型时才判 win，否则 tie。
2. Random position: 大规模时随机放置位置，用期望抵消偏差。

论文主实验使用 conservative swap。这会增加成本，但能明显降低 position bias 风险。

## 8. 偏差 2: Verbosity Bias

Verbosity bias 是 judge 偏好更长、更啰嗦的回答。论文设计了 repetitive list attack:

1. 从 MT-Bench 选出 23 个含编号列表的回答。
2. 用 GPT-4 把原列表里的每一项换个说法，不添加新信息。
3. 把重复改写后的列表插到原列表前面。
4. 如果 judge 认为这个更长但无新增信息的回答更好，就算攻击成功。

Table 3 的失败率:

- Claude-v1: 91.3%。
- GPT-3.5: 91.3%。
- GPT-4: 8.7%。

这个结果有两层含义。第一，GPT-4 明显更抗 verbosity bias。第二，即使 GPT-4 也不是零失败。后来的 AlpacaEval 2 LC、Arena-Hard 等工作会进一步处理 length bias，本仓库的 `alpaca_eval.py` 就实现了一个 length-controlled win rate 的 toy 版本。

## 9. 偏差 3: Self-Enhancement Bias

Self-enhancement bias 指 judge 偏好自己生成的回答。论文做了统计观察，而不是强结论:

- GPT-4 相比人类对 GPT-4 自己的 win rate 高约 10%。
- Claude-v1 对自己高约 25%。
- GPT-3.5 并不明显偏好自己。

作者谨慎地说，数据量和差异都有限，不能确定这就是严格的 self-enhancement bias。因为要控制风格而不改变质量很难。这个态度很值得学: 看到现象不等于证明机制。

## 10. 偏差 4: Math 和 Reasoning 误判

LLM judge 的一个危险点是: 它可能能单独解出题，但在评价别人答案时被错误答案带偏。论文给了数学例子，GPT-4 单独解题能得到正确答案，但 judge 时复制了错误助手的思路，最后判错。

Table 4 测了 10 道 math questions，包含 LLaMA-13B vs Vicuna-13B，并交换位置，共 20 次判断。失败定义是 GPT-4 说错误答案正确。结果:

- Default prompt: 14/20 failures。
- CoT prompt: 6/20 failures。
- Reference-guided prompt: 3/20 failures。

结论非常重要: 让 judge “先独立思考”有帮助，但仍可能被上下文污染；提供 reference answer 更稳。对于数学、代码、安全、事实核查，能用 reference、unit tests、tools 或 verifiable reward 时，不要只靠裸 LLM judge。

这也解释了为什么你学习时不能让 agent 只凭感觉判你的数学推导。更好的方式是让它先生成标准解、或者给它参考答案、或者让它运行代码/单元测试。

## 11. Agreement: 怎么衡量 judge 是否像人

论文的 agreement metric 是:

```text
agreement(J1, J2) =
probability that a random judge of type J1
and a random judge of type J2
give the same vote on a random question
```

如果包含 A/B/tie 三种标签，随机 baseline 是 33%。如果只看 non-tie 的 A/B，随机 baseline 是 50%。

Table 5 在 MT-Bench 上比较 GPT-4、GPT-4 single grading 和 expert humans。关键结果:

- First turn, S2 non-tie: GPT-4 pairwise vs Human 是 85%。
- First turn, S2 non-tie: Human vs Human 是 81%。
- Second turn, S2 non-tie: GPT-4 pairwise vs Human 是 85%。
- Second turn, S2 non-tie: Human vs Human 是 82%。

也就是说，在这个设置下，GPT-4 和人类的非平局一致率达到甚至略高于随机两个人类之间的一致率。

论文还报告了一个有意思的人类复核实验: 当人类选择和 GPT-4 不同时，研究者展示 GPT-4 的判断并询问是否合理。人类在 75% 的分歧中认为 GPT-4 判断合理，并在 34% 的分歧中愿意改选。这不是说 GPT-4 永远对，而是说明它的解释常常足够有说服力。

## 12. Chatbot Arena 上的一致性

Table 6 在 Chatbot Arena 的 3K crowd votes 上做 agreement。关键结果:

- GPT-4 pairwise vs Human, S2 non-tie: 87%。
- GPT-4 single grading vs Human, S2 non-tie: 85%。
- GPT-3.5 pairwise vs Human, S2 non-tie: 83%。
- Claude pairwise vs Human, S2 non-tie: 84%。

论文还指出，GPT-4 给出 non-tie 的数量更多，也就是说它更 affirmative，受 position bias 影响更小。其他 judge 在给出明确答案时也不错，但更容易 tie 或格式/偏差问题。

Figure 2 还展示了一个直觉: 模型差距越大，GPT-4 和人类越容易一致。当两个模型水平接近时，agreement 可能从约 70% 起；当 win rate difference 大时，agreement 接近 100%。这说明 judge 不确定性和模型间差距强相关。

## 13. Win Rate、BT 和 Elo: 论文原文与后续 Arena

原论文在主实验里大量使用 average win rate 和 agreement。Chatbot Arena 后续 leaderboard 更常见的是 Elo 或 Bradley-Terry 风格排名。本仓库把这个后续统计抽象也放进来了。

Pairwise battle 数据形状是:

```text
record = {
  qid,
  model_a,
  model_b,
  winner: A / B / tie
}
```

Bradley-Terry 模型假设每个模型有一个潜在 strength `s_i`:

```text
P(i beats j) = exp(s_i) / (exp(s_i) + exp(s_j))
```

给一堆 pairwise outcomes 后，用最大似然估计每个 `s_i`。再把 log-strength 转成 Elo-like rating:

```text
elo_i = 1500 + scale * s_i
```

本仓库 `bradley_terry.py` 用 MM algorithm 做 toy 拟合，并把 tie 当作双方各 0.5 win。真实 leaderboard 还需要 bootstrap confidence interval、采样权重、时间更新、反作弊和数据清洗。导读里要分清: BT/Elo 是 Arena 排名工程化的重要工具，但不是论文中验证 LLM-as-a-judge 的唯一证据。

## 14. MT-Bench 与传统 benchmark 互补

Table 8 很有价值，因为它说明不同 benchmark 测的东西不同。

一些关键数值:

- LLaMA-13B: MMLU 47.0, MT-Bench 2.61。
- Alpaca-13B: MMLU 48.1, MT-Bench 4.53。
- Vicuna-7B selected: 只用约 4.8M tokens 高质量对话训练，MMLU 37.3, MT-Bench 5.95。
- Vicuna-13B all: MMLU 52.1, MT-Bench 6.39。
- GPT-3.5: MMLU 70.0, MT-Bench 7.94。
- GPT-4: MMLU 86.4, MT-Bench 8.99。

Vicuna-7B selected 的例子尤其说明问题: 少量高质量对话数据能很快教会模型一种 GPT-4/人类偏好的聊天风格，使 MT-Bench 分数上升，但不一定显著提升 MMLU 这类核心知识能力。

所以论文主张 hybrid evaluation:

```text
capability benchmarks
  MMLU, GSM8K, HumanEval, HELM ...

plus

preference benchmarks
  MT-Bench, Chatbot Arena, LLM-as-a-judge ...
```

只看前者会漏掉对话体验；只看后者会把风格、长度、judge 偏差误当作能力。

## 15. 和本仓库的连接

本专题文件可以按下面顺序读:

1. `learning/llm-judge-arena/src/common.py`: 定义 PairBattle、Sample、toy judges。
2. `learning/llm-judge-arena/src/mt_bench_runner.py`: toy MT-Bench pointwise scoring。
3. `learning/llm-judge-arena/src/arena_hard_runner.py`: pairwise battles 和 win rate。
4. `learning/llm-judge-arena/src/judge_bias_demo.py`: position、verbosity、swap consistency 的 toy 复现。
5. `learning/llm-judge-arena/src/bradley_terry.py`: BT strength 和 Elo 转换。
6. `learning/llm-judge-arena/src/alpaca_eval.py`: length-controlled win rate。
7. `learning/llm-judge-arena/src/prometheus2_judge.py`: open judge model/rubric 的 toy 版。
8. `learning/llm-judge-arena/src/mini_arena.py`: round-robin battles 到 leaderboard。

最小运行命令:

```powershell
python learning\llm-judge-arena\src\tests\test_judge.py
python learning\llm-judge-arena\src\mini_arena.py
python learning\llm-judge-arena\src\judge_bias_demo.py
```

`mini_arena.py` 的 Elo 数值可能很极端，因为 toy 数据接近完全可分，真实系统需要更多噪声、更多 votes 和置信区间。这正好是一个学习点: 排名数字看起来精确，不代表不确定性小。

## 16. 张量/数据结构视角

评测论文没有神经网络 tensor，但有清晰的数据结构。

MT-Bench item:

```text
sample = {
  qid,
  category,
  turn_1_question,
  turn_2_question,
  model_answer_turn_1,
  model_answer_turn_2,
  judge_score_turn_1,
  judge_score_turn_2
}
```

Pairwise battle:

```text
battle = {
  qid,
  prompt,
  model_a,
  answer_a,
  model_b,
  answer_b,
  winner: A / B / tie,
  judge_type: human / gpt4 / gpt35 / claude
}
```

Agreement calculation input:

```text
votes_by_question = {
  qid_1: [human votes, gpt4 vote, claude vote, ...],
  qid_2: [...]
}
```

BT/Elo input:

```text
battles: list[PairBattle]
models: set of model names
win_counts[i, j] = number of times i beats j
tie contributes 0.5 win to both sides
```

这比“让 GPT-4 打个分”严谨得多。真正的评测系统首先是数据协议，其次才是 judge model。

## 17. 代码样例: swap 检查 position bias

下面是论文 position bias 思路的最小版:

```python
def judge_once(judge, question, answer_a, answer_b):
    return judge(question, answer_a, answer_b)

def swap_check(judge, question, x, y):
    v1 = judge_once(judge, question, x, y)
    v2 = judge_once(judge, question, y, x)

    if v1 == "A" and v2 == "B":
        return "x_wins_consistently"
    if v1 == "B" and v2 == "A":
        return "y_wins_consistently"
    if v1 == "tie" and v2 == "tie":
        return "tie_consistently"
    return "inconsistent_position_sensitive"
```

这个函数的价值很大。你以后让 agent 比较两个方案、两段代码、两篇笔记时，都可以让它先 A/B 判一次，再 B/A 判一次。如果结论翻转不一致，就不要把它当强证据。

## 18. 30-60 分钟本地实验

实验 A: verbosity bias

```text
1. 打开 common.py 的 make_length_judge。
2. 用 judge_bias_demo.py 计算 verbosity_bias_score。
3. 观察 prefer_longer=True 时 long_pick_rate 是 1.0。
4. 把 judge 改成 make_keyword_judge，看长度偏差是否下降。
5. 写一句解释: 为什么“更长”不等于“更好”。
```

实验 B: pairwise 到 Elo

```text
1. 运行 mini_arena.py。
2. 看 5 个 mock model 的 leaderboard。
3. 修改 make_arena_models，让某个短回答模型加入关键词。
4. 把 judge 从 length_judge 换成 keyword_judge。
5. 观察 leaderboard 如何变化。
```

实验 C: agreement

```text
1. 手工构造 10 个 qid。
2. 给 human_vote 和 gpt4_vote 两列 A/B/tie。
3. 分别计算包含 tie 的 agreement 和只看 non-tie 的 agreement。
4. 解释为什么 random baseline 从 33% 变成 50%。
```

实验 D: reference-guided judge

```text
1. 选一个简单数学题。
2. 构造两个回答: 一个最终错但解释流畅，一个最终对但简短。
3. 让 judge 在没有 reference 时判一次。
4. 给 reference answer 后再判一次。
5. 记录是否更稳定。
```

## 19. AI agent 应该怎样辅助你学这篇

学这篇论文时，agent 最适合当“评测协议审计员”，而不是直接当裁判。

推荐提示词:

```text
我正在学习 MT-Bench 和 Chatbot Arena。
请你不要只总结论文。
请按 examiner 模式一次问我一个问题。
每个问题都必须让我说明:
1. 这个评测协议输入是什么。
2. judge 输出是什么。
3. 可能有什么偏差。
4. 论文用什么实验测这个偏差。
5. 本仓库哪个函数能做 toy 复现。
如果我把 LLM judge 当成真理，请你指出风险并要求我设计 swap 或 reference 检查。
```

用 agent 给你学习评分时，可以使用这个安全版流程:

```text
请评价我的回答。
请先写出评分 rubric。
请分别从准确性、完整性、清晰度、代码对应四个维度打分。
请指出一个最严重错误。
请给出一个可以验证的参考依据。
然后请把我的回答和另一个候选回答交换顺序再评一次，检查是否有位置偏差。
```

这种用法比“你觉得我答得怎么样”强很多。它把 judge 变成可审计工具，而不是黑箱权威。

## 20. 常见误读

误读一: 有 GPT-4 judge，就不需要人类评测。

更准确: 论文证明 GPT-4 在这些设置下和人类高度一致，但人类偏好仍然是 gold standard。LLM judge 是 scalable approximation。

误读二: GPT-4 judge 超过 80% agreement，所以它总是正确。

更准确: agreement 是概率指标。论文也展示了 position bias、verbosity bias、math/reasoning failure。数学类 default prompt 甚至有 14/20 failure。

误读三: MT-Bench 可以替代 MMLU。

更准确: MT-Bench 测 human preference 和对话体验，MMLU 测知识/能力。论文主张 hybrid evaluation。

误读四: Arena 排名数字越高，模型绝对更强。

更准确: Arena 是基于用户分布、对战样本、judge/human votes 和统计模型的相对偏好排名。样本量、置信区间、用户群体和题目分布都重要。

误读五: 更长的回答一般更好。

更准确: 论文专门测 verbosity bias。长回答可能只是重复，judge 需要被审计和校正。

误读六: CoT judge 能解决数学误判。

更准确: CoT prompt 把 failure 从 14/20 降到 6/20，但 reference-guided 才进一步降到 3/20。CoT 仍可能被候选错误答案污染。

## 21. 现代意义

这篇论文是现代 LLM 体验评测的奠基点之一。它影响了后来的:

- Chatbot Arena leaderboard。
- MT-Bench 风格开放问题评测。
- Arena-Hard。
- AlpacaEval 2 的 length-controlled win rate。
- Prometheus、JudgeLM、PandaLM 等 open judge model。
- G-Eval、LLM-as-a-judge rubric prompting。
- Agent benchmark 中的 pairwise preference evaluation。

对今天的学习者来说，它最重要的启发是: 评测不是一道脚本命令，而是一套社会技术系统。题目、用户、模型、judge、prompt、bias、统计模型、置信区间共同决定了最后的排行榜。

如果你要用 AI agent 加速学习，这篇论文给你的底层习惯是:

- 让 agent 出题，但要保留参考答案或可验证标准。
- 让 agent 评分，但要给 rubric。
- 对重要判断做 swap。
- 对数学/代码题用 reference 或工具。
- 不迷信单一分数，要看错误类型和不确定性。

## 22. 闭卷掌握检查

读完后你应该能闭卷回答:

1. 为什么 MMLU/HELM 等传统 benchmark 不能充分评价 chat assistants。
2. MT-Bench 有多少问题、几轮、几个类别，为什么设计成多轮。
3. Chatbot Arena 为什么要匿名，为什么问题不预设。
4. Pairwise comparison、single answer grading、reference-guided grading 分别适合什么场景。
5. Position bias 怎么测，Table 2 中 GPT-4 的 consistency 大约是多少。
6. Verbosity bias 的 repetitive list attack 是怎么构造的，GPT-4 failure rate 大约是多少。
7. 为什么 GPT-4 judge 在 math/reasoning 上会被错误答案带偏。
8. Default、CoT、reference-guided math judge 的 failure 数分别是多少。
9. Agreement metric 怎么定义，为什么包含 tie 和不包含 tie 的 random baseline 不同。
10. MT-Bench 上 GPT-4 pairwise 与 human 的 non-tie agreement 大约是多少。
11. Chatbot Arena 上 GPT-4 与 human 的 non-tie agreement 大约是多少。
12. Bradley-Terry 模型如何把 pairwise battle 转成 latent strength。
13. 为什么 MT-Bench 和 MMLU/TruthfulQA 是互补关系。
14. 在本仓库里，哪个文件模拟 judge bias，哪个文件拟合 BT/Elo，哪个文件跑 mini Arena。
15. 如果你用 agent 给自己的学习笔记打分，应该怎样设计 swap、rubric 和 reference 检查。

真正掌握的标志是: 你能自己设计一个小型 Arena，构造 5 个模型、20 个问题、一个有偏 judge、一个校正后的 judge，并解释两个 leaderboard 为什么不同。
