# guide_TinyStories: How Small Can Language Models Be and Still Speak Coherent English?

> manual-deep-guide: true
>
> 原论文: TinyStories: How Small Can Language Models Be and Still Speak Coherent English?
>
> 本地原文 PDF: `learning/small-model-graduation/paper/01_tinystories.pdf`
>
> 作者: Ronen Eldan and Yuanzhi Li
>
> 版本: arXiv v2, 2023-05-24
>
> 本地机制代码:
> `learning/small-model-graduation/src/tinystories_original_minimal.py`

## 0. 这篇导读怎么读

TinyStories 是整个学习仓库的一个非常好的收官论文。它问的问题很朴素:
小模型到底是因为模型太小而不会说连贯英语, 还是因为训练数据太宽、太杂、
太重知识负担, 让小模型被压垮了?

论文的答案不是换一个神奇架构, 而是控制数据分布。作者构造了一个合成短故事
数据集, 故事只使用 3-4 岁儿童通常能理解的简单词汇, 但仍然包含语法、事实、
因果、人物一致性和简单推理。这样一来, 小模型不需要记住 Wikipedia 级别的
世界知识, 可以把容量用于学习语言组织和故事连贯性。

一句话概括:

TinyStories 通过受控合成数据把自然语言任务的宽度降下来, 让百万级到千万级
参数的小语言模型也能生成多段、语法基本正确、情节相对连贯的英文故事, 并用
GPT-4 teacher-style rubric 和 Rouge overlap 检查能力与记忆。

读这篇论文要抓住四根线:

- 数据线: TinyStories 和 TinyStories-Instruct 怎样生成。
- 模型线: 1M 到 80M 左右的小 GPT-Neo 风格模型怎样表现。
- 评测线: GPT-Eval 怎样像老师批作文一样给多维分数。
- 解释线: 小模型的 attention head 和 MLP neuron 为什么更容易看懂。

## 1. 历史语境: 小模型为什么显得笨

在 TinyStories 之前, 社区已经看到 scaling law: 模型越大、数据越多、训练越久,
能力通常越强。GPT-2 small 或 GPT-Neo small 这种 125M 级别模型, 即使在
大语料上训练, 生成长文本时也常常重复、跳题、语法崩坏、人物关系混乱。

这很容易让人得出一个结论:

连贯英文生成必须依赖几亿、几十亿、甚至更大模型。

TinyStories 提醒我们另一个可能性:

小模型不是完全没有语言能力, 而是被数据分布的复杂度淹没了。Wikipedia、Common
Crawl、Pile 这类语料不仅教模型说英语, 还要求它编码海量事实、专业领域、
文体、实体、日期、长尾词汇和世界知识。小模型容量有限, 它可能连基础语言
模式还没学稳, 就被太多主题和知识负担分散了。

论文要隔离的问题是:

如果把数据分布限制在儿童故事这个小世界里, 小模型能不能学到自然语言的核心
机制, 例如语法、指代、因果、情节延续、简单常识和简单推理?

这就是 TinyStories 的核心价值。

## 2. 论文主张

论文的主要主张可以拆成三层。

第一层是数据主张:

- 自然语言能力不一定只能在非常宽的 web 语料上研究。
- 可以构造一个小而精的 synthetic dataset, 保留语言的核心要素。
- TinyStories 的约束是短故事、简单词汇、儿童可理解事实、简单情节。

第二层是模型主张:

- 低于 10M 参数的小模型也能生成多段连贯故事。
- 只有一个 transformer block 的模型也能产生有意义文本。
- 语法可能较早掌握, 一致性、创造性、指令跟随需要更大宽度或更多层。

第三层是评测主张:

- 传统 benchmark 常要求单词或短答案, 不适合评价自由生成故事。
- GPT-4 可以像老师批作文一样给 grammar, creativity, consistency 等维度。
- 这种 GPT-Eval 能让我们观察小模型能力如何随 loss、宽度、深度变化。

你可以把论文的技术路线画成:

```text
controlled synthetic story generation
        |
        v
TinyStories dataset
        |
        v
train small autoregressive LMs
        |
        v
generate story completions
        |
        v
GPT-4 teacher-style evaluation
        |
        v
capacity, data, width-depth and interpretability analysis
```

## 3. TinyStories 数据集怎样生成

TinyStories 的目标是保留自然语言的质感, 但压低主题和词汇复杂度。论文使用
GPT-3.5 和 GPT-4 生成短故事, 并要求故事只使用典型 3 岁儿童大概率能理解的
简单词。

关键设计不是一句请写儿童故事, 而是多样性控制。作者发现, 直接让大模型写很多
儿童故事会变得重复。为了解决这个问题, 他们构造了一个约 1500 个 basic words
的词表, 按 noun, verb, adjective 等类别组织。每次生成故事时随机选 3 个词:

- 一个 verb。
- 一个 noun。
- 一个 adjective。

然后 prompt 要求故事必须把这 3 个词自然地放进去。

此外, 作者还准备了一组 story features, 每个故事随机抽取一部分, 例如:

- dialogue。
- plot twist。
- bad ending。
- moral value。
- foreshadowing。
- conflict。

这样做有两个作用:

- 随机词保证故事覆盖儿童词汇空间, 不是反复讲同几种故事。
- 随机特征保证情节结构变化, 不是只有温和 happy ending。

论文给出的 prompt 形式大意是:

```text
Write a short story using only very simple words that a 3 year old
child would likely understand. The story should use a given verb,
noun and adjective. The story should have selected features.
Remember to only use simple words.
```

这就是 synthetic data 的关键: 不是让 teacher 随机发挥, 而是用约束驱动数据
覆盖。

本仓库对应函数:

`build_story_prompt` in `tinystories_original_minimal.py`

## 4. TinyStories-Instruct

TinyStories-Instruct 是论文的指令跟随变体。它不是只给故事, 而是在每个故事
前面加一组 instructions, 然后让模型学习根据指令生成故事。

指令有四类:

- Words: 故事必须包含的词。
- Sentence: 故事中必须出现的一句话。
- Features: 例如 dialogue, bad ending, moral value, plot twist。
- Summary: 1-2 行故事概要。

每个样本随机选择这些 instruction 的一个子集, 然后接 story 本文。这样模型在
训练后可以根据任意组合的指令生成故事。

论文还做了一个很有意思的 OOD 测试:

- 构造 TinyStories-Instruct-OOD。
- 训练集中故意不出现某种 instruction combination。
- 例如不让 Words 和 Summary 同时出现。
- 测试时要求模型同时满足这两类指令。

结果显示 33M 参数模型可以在一定程度上组合这两种从未一起见过的指令类型。
这不是强泛化的最终证明, 但说明模型不只是死记一种模板。

## 5. 为什么这是小模型研究的好数据集

TinyStories 把语言任务的复杂度分成两部分:

```text
core language mechanics:
  grammar
  simple vocabulary
  subject-object relations
  causal events
  character consistency
  simple reasoning

broad world complexity:
  many domains
  rare words
  encyclopedic facts
  professional styles
  long-tail entities
  noisy web text
```

大 web 语料把两部分混在一起。TinyStories 刻意拿掉第二部分的大部分负担,
让小模型专注第一部分。

这就是论文的核心假设:

如果数据分布的熵更低, 模型容量和任务复杂度匹配, 小模型也能学到可观察的
语言能力。

这不是说儿童故事比真实语言更重要, 而是说它是一个实验台。你可以在很少 GPU
资源下研究 width, depth, heads, training steps, loss, generation quality,
interpretability 之间的关系。

## 6. 模型设置和张量级别图

论文使用 GPT-Neo 风格 autoregressive LM。脚注写到:

- 模型公开名包括 TinyStories-1M, 3M, 9M, 28M, 33M, 1Layer, 2Layer 等。
- 使用 GPT-Neo architecture。
- window size 是 256。
- context length 是 512。
- 使用 GPT-Neo tokenizer, 但只保留 top 10K most common tokens。

训练目标仍然是标准 next-token prediction, 没有神秘新 loss。

张量级别如下:

```text
input_ids:      [B, T]
target_ids:     [B, T]
token_embed:    [B, T, H]
logits:         [B, T, V]

loss uses:
  logits[:, 0:T-1, :] to predict input_ids[:, 1:T]
```

用公式写:

```text
loss = average cross_entropy(
    logits at position t,
    token id at position t+1
)
```

所以 TinyStories 的方法重点不在 optimizer, 而在数据分布:

```text
same next-token loss
different data distribution
smaller vocabulary burden
lower topic entropy
clearer evaluation target
```

本仓库对应函数:

`next_token_cross_entropy` in `tinystories_original_minimal.py`

## 7. GPT-Eval: 像老师批作文一样评测

传统 NLP benchmark 常常要求模型输出一个结构化答案。可是故事生成没有单一
正确答案。TinyStories 引入 GPT-Eval, 用 GPT-4 对生成故事做多维评分。

基本流程:

```text
manual beginning prompt
        |
        v
small LM generates 10 completions, temperature 1
        |
        v
GPT-4 receives beginning + completion
        |
        v
GPT-4 gives verbal assessment and scores
```

普通 TinyStories completion 的评分维度包括:

- grammar。
- creativity。
- consistency with story beginning。
- guessed age group。

TinyStories-Instruct 的评分维度增加:

- consistency with instructions。
- plot coherence。

论文的一个具体例子里, GPT-4 对 28M 模型生成的故事给出:

- Grammar: 8/10。
- Creativity: 7/10。
- Consistency: 7/10。
- Age group: E, 即 10-12。

这套评测的优点:

- 可以评价自由文本, 不要求单一答案。
- 能拆开语法、创意、一致性等不同能力。
- 能观察 loss 下降时哪些能力先提升。

这套评测的风险:

- GPT-4 judge 不是绝对真理。
- prompt 设计会影响评分。
- 评分维度之间可能相关。
- 不能完全替代人工评估。

但在 2023 年的语境下, 它很适合做小模型生成能力研究。

本仓库对应函数:

- `gpt_eval_prompt`
- `toy_teacher_scores`

`toy_teacher_scores` 不调用 GPT-4, 只是把 rubric contract 做成可测试玩具版。

## 8. Fig.3 和 Fig.4: loss 与能力不是同一个标尺

Fig.3 显示 GPT-Neo 模型训练过程中:

- evaluation loss 下降。
- GPT-Eval 的 grammar, creativity, consistency 分数上升。

这支持一个常见直觉:

较低 next-token loss 通常对应更好生成质量。

但论文的更细观察是:

- grammar 分数更早 plateau。
- consistency 和 creativity 需要更大模型或更多训练。
- shallower models 在 grammar 上可能还不错, 但在内容一致性上弱。
- depth 对保持上下文更重要。
- hidden size 从 64 增加到 128 时, story beginning consistency 开始明显出现。

这对你学习小模型很重要。不要只盯 val_loss。loss 是平均 token 预测指标,
但 story quality 是多维能力:

```text
val_loss
  -> token-level predictive fit

grammar
  -> local syntax and common phrase structure

consistency
  -> character, event, and plot tracking

creativity
  -> diverse but plausible continuation

instruction-following
  -> obey external constraints over the whole generation
```

## 9. 主结果: 小模型真的会写故事吗

论文开头用一个 28M TinyStories 模型对比 GPT-2XL。GPT-2XL 有约 1.5B 参数,
大约比 28M 模型大两个数量级。给定 Tom 和 Jane 喝汤的 prompt:

- GPT-2XL 生成的内容出现怪异冲突和重复。
- 28M TinyStories 模型能续写 bitter soup, apology, bread and cheese,
  friend relationship 等连贯情节。

这不是说 28M 模型总体比 GPT-2XL 强。更准确的说法是:

在 TinyStories 分布上, 小模型的训练数据和任务高度匹配, 因此在这个受控域内
可以胜过更大但训练分布不匹配的模型。

论文还展示了若干 prompt:

- factual knowledge。
- reasoning。
- context tracking。

Figure 2 说明:

- 2.5M 模型已经能处理一些事实性补全。
- 33M 4-layer 模型能做更强的 reasoning 和 contextual continuation。
- 21M 1-layer 模型也能生成有意义文本, 但上下文跟踪受限。
- GPT-2XL 在这些儿童故事 prompt 上并不稳定。

这就是 TinyStories 最强的教学价值:

它把能力涌现从巨大模型缩小到了可以在单 GPU 上研究的小模型。

## 10. Fig.6-Fig.12: 宽度、深度和能力

论文在 1M 到 35M 左右模型上展示了多个生成例子。模型层数从 1 到 8 层不等。
它们都可以在单个 V100 GPU 上最多约 30 小时内训练。

一个典型例子是 Lucy 爬梯子卡住的故事。GPT-4 评分大致显示:

- 1M 8-layer: Grammar 6, Creativity 3, Consistency 2。
- 2.5M 8-layer: Grammar 5, Creativity 6, Consistency 3。
- 8.3M 8-layer: Grammar 7, Creativity 5, Consistency 5。
- 28M 8-layer: Grammar 9, Creativity 6, Consistency 9。
- 21M 1-layer: Grammar 8, Creativity 3, Consistency 7。
- 33M 2-layer: Grammar 7, Creativity 6, Consistency 4。

这类例子不是严格统计主表, 但它们帮助你看见能力差异:

- 小模型常重复。
- 更大 hidden size 往往带来更好事实知识和表达。
- 更多层数帮助长程依赖和上下文跟踪。
- 1-layer 模型能写顺句子, 但容易在人物和事件上漂移。

论文 Sec.4.2 总结了一个重要发现:

- factual knowledge 更依赖 embedding dimension。
- context tracking 更依赖 number of layers。

这很符合直觉:

- width 给模型更多并行表征槽位, 更容易存词义和事实。
- depth 让信息经过多次组合, 更容易维护跨句结构和事件链。

本仓库对应函数:

- `rough_gpt_params`
- `capability_profile`

这些只是 toy profile, 用来帮助你把 width-depth 直觉变成可测试代码。

## 11. TinyStories-Instruct 和 OOD 指令组合

Figure 12 展示 instruction-following prompt, 例如要求故事包含 words:
`dive`, `job`, `sorry`。模型越大, 越能满足词约束并保持 plot coherent。

论文中的一个例子:

- 1M 8-layer: Grammar 4, Creativity 3, Plot 4, Consistency 6。
- 2.5M 8-layer: Grammar 6, Creativity 4, Plot 5, Consistency 7。
- 8.3M 8-layer: Grammar 7, Creativity 6, Plot 6, Consistency 8。
- 28M 8-layer: Grammar 7, Creativity 6, Plot 7, Consistency 9。
- 33M 4-layer: Grammar 8, Creativity 7, Plot 8, Consistency 9。
- 21M 1-layer: Grammar 7, Creativity 5, Plot 6, Consistency 4。
- 33M 2-layer: Grammar 7, Creativity 6, Plot 7, Consistency 8。

这里可以看到 depth 对指令跟随的帮助。1-layer 模型即便语法尚可, 也更难把
多个约束一直带到故事结尾。

OOD 指令组合实验的意义:

- 训练时没有见过 Words plus Summary 的组合。
- 测试时要求模型同时满足 Words 和 Summary。
- 33M 模型仍能生成大致符合要求的故事。

这说明模型在一定程度上学到了 instruction types 的组合规则, 不只是记住固定
模板。

## 12. 记忆检查: 为什么不是简单背故事

TinyStories 论文很认真地处理了一个质疑:

小模型会不会只是背下训练集中少量故事或模板?

论文区分三种 memorization:

- exact memorization: 直接复制整篇或大段故事。
- simple template matching: 换名字或实体, 情节基本照搬。
- complex template matching: 抽象情节模式相似, 但细节变化大。

作者承认 complex template matching 很难量化, 但他们用多种方法证明模型不是
简单复制:

- 人工检查生成故事。
- 截断训练故事前 40%, 让模型生成 alternative completion。
- 对比生成结尾和原始结尾。
- 检查 instruction combinations 是否能泛化到未见组合。
- 用 Rouge-k overlap 衡量 n-gram 相似度。

Rouge-k precision 的直觉:

```text
source k-grams = all k-word chunks in generated story
target k-grams = all k-word chunks in reference or training story

precision = how many source k-grams appear in target
```

论文关注:

- 生成结尾和原结尾的 Rouge2 precision。
- 100 个生成故事之间的最大 Rouge2 fmeasure。
- 生成 k-gram 在整个训练集里出现的频率。
- 每个生成故事到最近训练故事的最高 Rouge precision。

Figure 16 说明大多数 4-gram 和 5-gram 在训练数据中一次都没有出现。Figure 17
说明生成故事没有在复制某个具体训练故事。

本仓库对应函数:

- `rouge_k_precision`
- `rouge_k_fmeasure`
- `nearest_training_overlap`

## 13. 可解释性: 为什么小模型更容易看

TinyStories 的另一个贡献是 interpretability。作者认为小模型参数少、层数少,
组件更可能呈现可解释角色。

他们看两个对象:

- attention heads。
- MLP neurons。

Attention heads:

- 使用 1-layer, hidden dimension 1024, 16 heads 的模型。
- 因为只有一层, head 对输出 token 的贡献更直接。
- 作者观察到 distance-based heads 和 semantic heads 的分离。

Distance-based heads:

- 某些 head 主要关注固定相对距离的 token。
- 这些 head 更像局部语法工具。

Semantic heads:

- 某些 head 会让 article 或相关 token 关注 banana, park, Tom, Lucy 等实体。
- 这些 head 帮助模型保持主题和实体一致性。

MLP neurons:

- 作者在约 20 个故事, 约 8000 tokens 上记录 neuron activation。
- 对每个 neuron 找激活最高的 tokens。
- 在 1M TinyStories 模型里, 有 neuron 对 subject pronouns, action words,
  adjectives, protagonist introduction 等有清晰偏好。
- 对比 GPT-2XL 的某些 neuron, 角色没有那么显然。

这部分不是严格完整的 mechanistic interpretability 证明, 但它支持一个研究方向:

用受控数据和小模型做可解释性实验, 可能比直接看巨型 web 模型更清楚。

## 14. Scaling law 和超参探索

论文最后把 TinyStories 当作小型 NLP testbed, 研究架构和超参。

一个实验是 fixed FLOPs 下的 model size versus training steps trade-off:

- 改 layers: 2, 4, 8, 12。
- 改 hidden dimension: 64, 128, 256, 512, 768, 1024, 2048。
- 对每个 compute budget, 选 validation loss 最低的组合。
- Figure 23 显示近似 polynomial scaling law。

作者也很谨慎地说点数不多, 不能过度下结论。但这个结果说明:

TinyStories 虽小, 仍能复现某些 LLM scaling-like 现象。

另一个实验是 attention heads 数量。Figure 24 给出 hidden size 768 的例子:

- 2 layers, 2 heads: eval loss 1.38, grammar 7.77, creativity 6.5,
  consistency 7.78。
- 2 layers, 4 heads: eval loss 1.34, grammar 8.05, creativity 6.57,
  consistency 8.16。
- 2 layers, 8 heads: eval loss 1.33, grammar 8.25, creativity 6.53,
  consistency 8.16。
- 1 layer, 2 heads: eval loss 1.58, grammar 7.13, creativity 5.83,
  consistency 6.38。
- 1 layer, 4 heads: eval loss 1.56, grammar 7.43, creativity 5.90,
  consistency 6.75。
- 1 layer, 8 heads: eval loss 1.54, grammar 7.45, creativity 6.28,
  consistency 7.02。

结论:

- head 数太少时, 增加 heads 有帮助。
- 2-layer 比 1-layer 更强。
- grammar 和 consistency 都随配置改善, 但 improvement pattern 不完全相同。

## 15. 本仓库代码怎么对应论文

新增论文机制文件:

`learning/small-model-graduation/src/tinystories_original_minimal.py`

对应关系:

- `StoryConstraints`: 随机 verb, noun, adjective 和 features。
- `build_story_prompt`: TinyStories synthetic prompt。
- `child_vocab_fraction`: 儿童词汇分布约束的 toy 检查。
- `lexical_diversity`: 多样性粗指标。
- `rouge_k_precision`: 论文 Sec.4.4 的 Rouge-k precision。
- `nearest_training_overlap`: 最近训练故事 overlap 检查。
- `gpt_eval_prompt`: GPT-Eval teacher prompt。
- `toy_teacher_scores`: 本地可测试的 rubric mock。
- `ModelSpec` and `rough_gpt_params`: 小 GPT 风格参数量直觉。
- `capability_profile`: width-depth 能力差异的 toy profile。
- `next_token_cross_entropy`: 标准 autoregressive loss。

你可以运行:

```powershell
.venv\Scripts\python.exe learning\small-model-graduation\src\tinystories_original_minimal.py
.venv\Scripts\python.exe -m pytest learning\small-model-graduation\src\tests -q
```

测试覆盖:

- prompt 是否包含随机词和 features。
- 简单儿童词汇比例是否高于复杂专业词。
- Rouge 是否能发现复制。
- GPT-Eval prompt 是否包含 rubric 维度。
- width 和 depth toy profile 是否体现论文直觉。
- next-token loss 是否按 `[B,T,V]` 和 `[B,T]` 正常工作。

## 16. 代码样例: 数据约束如何改变任务

```python
from tinystories_original_minimal import (
    StoryConstraints,
    build_story_prompt,
    child_vocab_fraction,
)

constraints = StoryConstraints(
    verb="decorate",
    noun="thunder",
    adjective="ancient",
    features=("dialogue", "bad ending"),
)

prompt = build_story_prompt(constraints)
print(prompt)

simple = "the happy dog went to the tree"
hard = "quantum governance optimized semiconductor portfolios"

print(child_vocab_fraction(simple))
print(child_vocab_fraction(hard))
```

这段代码对应论文 Sec.2:

- 随机词提高覆盖。
- features 提高结构多样性。
- simple vocabulary 降低小模型学习负担。

你要观察的是 distribution design, 不是单条故事文笔。

## 17. 代码样例: Rouge 记忆检查

```python
from tinystories_original_minimal import (
    rouge_k_precision,
    rouge_k_fmeasure,
    nearest_training_overlap,
)

story = "Lily saw a red ball and helped a sad dog"
copied = "Lily saw a red ball and helped a sad dog"
changed = "Tom baked bread with mom in the house"

print(rouge_k_precision(copied, story, k=2))
print(rouge_k_fmeasure(copied, story, k=2))
print(rouge_k_precision(changed, story, k=2))
print(nearest_training_overlap(copied, [story, changed], k=2))
```

如果生成故事和训练故事高度重合, Rouge 会高。TinyStories 论文用类似思路
说明生成故事通常不是训练故事的简单复制。

## 18. 代码样例: 宽度和深度的直觉

```python
from tinystories_original_minimal import (
    ModelSpec,
    rough_gpt_params,
    capability_profile,
)

shallow = ModelSpec(hidden=256, layers=1, heads=4)
deep = ModelSpec(hidden=256, layers=8, heads=4)
wide = ModelSpec(hidden=1024, layers=1, heads=8)

print(rough_gpt_params(shallow))
print(rough_gpt_params(deep))
print(rough_gpt_params(wide))
print(capability_profile(deep))
print(capability_profile(wide))
```

这不是论文真实训练结果, 而是把论文的 width-depth 观察变成可运行 toy:

- wide 更偏 factual knowledge。
- deep 更偏 context tracking。
- 参数量随 hidden 和 layers 增长。

## 19. 新手容易误读的地方

误读 1: TinyStories 证明小模型普遍强于大模型。

更准确: 它证明在受控儿童故事分布上, 数据匹配的小模型可以生成比不匹配大模型
更连贯的文本。

误读 2: Synthetic data 越多越好。

更准确: 论文强调 diversity controls, 包括随机词和 features。没有控制的合成数据
会重复。

误读 3: GPT-Eval 就是完美评测。

更准确: GPT-Eval 是适合自由生成故事的实用评测范式, 但 judge bias 和 prompt
敏感性仍然存在。

误读 4: 小模型不需要知识。

更准确: TinyStories 保留儿童级别的事实和推理, 只是避免百科级和专业级复杂度。

误读 5: Rouge 低就完全没有记忆。

更准确: Rouge 能排除 exact memorization 和 simple template matching 的一部分,
但 complex template matching 很难量化。

## 20. 局限性

论文自己也有边界:

- 领域非常窄, 主要是简单英文儿童故事。
- 数据由 GPT-3.5 和 GPT-4 生成, 继承 teacher 风格和偏差。
- GPT-4 评分不是独立人类真值。
- 小模型的能力在该分布内成立, 不代表真实 web 任务也成立。
- Creativity 的定义依赖 GPT-Eval prompt。
- Complex template matching 很难用 Rouge 完全排除。
- 解释性部分是 preliminary evidence, 不是完整机制证明。

这篇论文最好的读法不是把它当成 small model 可以取代 LLM 的证明, 而是把它
当成数据分布控制和小模型实验设计的范例。

## 21. 对本学习仓库的意义

这个仓库前面大量专题都在讲更大的模型、更复杂的训练、更强的系统:

- Transformer。
- LoRA/Adapter。
- RLHF/DPO/GRPO。
- serving。
- evaluation。
- safety。
- multimodal agent。

TinyStories 给你一个反方向的提醒:

很多时候学习和研究不需要一上来就追最大模型。你可以通过缩小任务分布,
建立一个可运行、可评测、可解释的小世界, 用它来验证想法。

对 AI agent 学习尤其重要:

- 不要让 agent 只给你堆资料。
- 让 agent 帮你构造小而闭合的 toy domain。
- 让 agent 帮你写 evaluation rubric。
- 让 agent 帮你检查是否只是 memorization。
- 让 agent 逼你解释 data distribution 和 model capacity 的匹配。

这就是 TinyStories 对学习方法的启发。

## 22. 怎样用 AI agent 学这篇论文

推荐流程:

1. 自己先读 Abstract, Sec.2, Sec.3, Sec.4.4。
2. 闭卷写一句话:
   TinyStories 到底改变了模型、数据、评测还是 optimizer。
3. 让 agent 只问你一个问题, 例如:
   为什么随机 verb/noun/adjective 能增加 synthetic data diversity。
4. 打开 `tinystories_original_minimal.py`, 把每个函数映射到论文 section。
5. 修改一个故事约束, 预测 prompt 如何变化。
6. 构造一个 copied story 和 changed story, 预测 Rouge 如何变化。
7. 让 agent 指出你是否把域内能力误读成通用能力。

可直接使用的提示词:

```text
我正在学习 TinyStories 论文。
请按 dataset construction, TinyStories-Instruct, GPT-Eval,
small model evidence, memorization check, interpretability,
scaling-law testbed 的顺序考我。
一次只问一个问题。
每次我回答后, 请指出:
1. 我的说法对应论文哪一节或哪张图。
2. 我是否混淆了数据分布、模型容量和评测方法。
3. 我应该打开本仓库哪个函数做 toy 验证。
不要直接替我总结整篇论文。
```

## 23. 30-60 分钟本地练习

练习 A: prompt constraint。

- 改 `StoryConstraints` 的 verb, noun, adjective。
- 加入 `plot twist` 或 `bad ending`。
- 观察 `build_story_prompt` 输出。
- 解释为什么这比直接写儿童故事更多样。

练习 B: vocabulary complexity。

- 写一个儿童故事句子。
- 写一个专业领域句子。
- 用 `child_vocab_fraction` 比较。
- 解释为什么小模型更容易在前者上学会连贯生成。

练习 C: memorization。

- 构造一个 copied story。
- 构造一个 changed story。
- 用 Rouge2 precision 比较。
- 解释 Rouge 能抓住什么, 抓不住什么。

练习 D: width-depth。

- 比较 `ModelSpec(hidden=256, layers=1)` 和 `ModelSpec(hidden=256, layers=8)`。
- 比较 `ModelSpec(hidden=1024, layers=1)`。
- 用一句话解释 width 和 depth 的不同角色。

## 24. 闭卷掌握检查

读完后你应该能回答:

- TinyStories 要隔离的科学问题是什么。
- 为什么小模型在 web-scale 语料上容易生成不连贯文本。
- 数据集为什么限制到 3-4 岁儿童词汇。
- 随机 verb/noun/adjective 和 story features 解决什么问题。
- TinyStories-Instruct 的四类 instruction 是什么。
- GPT-Eval 具体怎样评分, 它比传统 benchmark 解决了什么。
- grammar, consistency, creativity 的增长规律有什么不同。
- width 和 depth 在论文观察中分别更偏向什么能力。
- 论文怎样检查不是简单 memorization。
- interpretability 部分看了哪些 attention head 和 neuron 现象。
- 本仓库哪个文件实现了 TinyStories 的原机制玩具版。

## 25. 一句话总结

TinyStories 的核心不是小模型突然有魔法, 而是数据分布被设计到足够窄、
足够干净、足够多样, 让小模型容量和任务复杂度匹配起来。它用合成儿童故事
数据、GPT-4 teacher-style evaluation、Rouge 记忆检查和小模型可解释性实验,
展示了连贯语言能力可以在远小于传统大模型的尺度上被观察、研究和教学。
