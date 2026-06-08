# guide_Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations

<!-- manual-deep-guide -->

> 原论文: [Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations](https://arxiv.org/abs/2312.06674)
>
> 本地原文 PDF: `learning/safety-defense/paper/01_llama_guard.pdf`
>
> 作者: Hakan Inan, Kartikeya Upasani, Jianfeng Chi, Rashi Rungta, Krithika Iyer, Yuning Mao,
> Michael Tontchev, Qing Hu, Brian Fuller, Davide Testuggine, Madian Khabsa
>
> 年份: 2023
>
> 本地代码: `learning/safety-defense/src/`

## 0. 这篇论文解决什么问题

前一篇 red-team 论文告诉你: 只看主模型是否会直接拒绝危险请求，不够。优化器、多轮攻击、prompt injection、输出泄漏都可能绕开主模型的表面拒答。

Llama Guard 站在防御侧问另一个问题:

```text
Can we build a deployable guard model
that checks both user inputs and model outputs
against an explicit safety policy taxonomy?
```

它的答案是: 用一个 instruction-tuned LLM 做安全分类器。这个分类器不是只返回一个模糊分数，而是把安全策略写成 taxonomy，输入到模型 prompt 里，让模型输出:

```text
safe
```

或者:

```text
unsafe
O3
```

其中 `O3` 代表被违反的 policy category。这样做的好处是:

- policy 可读: 类别和边界写在 prompt 里。
- task 可切换: 同一个模型可做 prompt classification 和 response classification。
- 输出可部署: 第一 token 的 safe/unsafe 概率可读成 binary score。
- taxonomy 可适配: 可以 zero-shot/few-shot 换到新 policy，也可以继续 fine-tune。

这篇 guide 的目标是让你读完后能设计一个最小 guardrail pipeline，而不是只背 "Llama Guard 是安全分类器"。

## 1. 当时语境和论文地位

LLM 产品部署时常见安全做法有三层:

- 主模型本身经过 SFT/RLHF/安全对齐，尽量拒绝不该回答的请求。
- 外部内容审核 API 检查输入或输出。
- 应用层规则、关键词、人工审核、日志监控兜底。

论文指出，当时已有的内容审核工具直接拿来做 LLM guardrail 有几个问题:

- 它们大多面向 human-generated online content，不一定适合 AI assistant 对话。
- 它们通常不区分 user prompt 风险和 assistant response 风险。
- 它们的 policy taxonomy 固定，不容易适配新的产品策略。
- 很多工具只有 API，不方便研究者本地微调和部署。
- 小型 classifier 的语义理解能力可能不如 LLM backbone。

Llama Guard 的定位就是补这个空:

```text
old moderation API:
  fixed taxonomy
  often user-content oriented
  closed API
  limited task customization

Llama Guard:
  Llama2-7B backbone
  instruction-tuned as classifier
  taxonomy included in prompt
  separate input/output instructions
  open weights for adaptation
```

它不是终极防御，但它把 "policy as text + LLM as classifier + deployable output format" 做成了一个强基线。

## 2. 原论文结构地图

建议读原文时按这个顺序:

- Abstract: 抓住 input-output safeguard、taxonomy、instruction-tuned Llama2-7B、OpenAI Moderation/ToxicChat。
- Section 1: 看为什么已有 moderation tools 不够。
- Section 2: 慢读 6 类 safety risk taxonomy。不要只背类别名，要理解每类的边界。
- Section 3.1: 这是方法核心。四个 ingredient: guidelines、classification type、conversation、output format。
- Figure 1: 看 prompt classification 和 response classification 的任务模板。
- Section 3.2: 看 zero-shot/few-shot policy adaptation。
- Section 3.3: 看数据收集，特别是 13,997 prompt-response pairs 和四个标签。
- Section 3.4: 看 Llama2-7B、4096 sequence length、500 steps、data augmentation。
- Section 4.1-4.3: 看 on-policy/off-policy evaluation、max-all、1-vs-all、1-vs-benign、AUPRC。
- Table 2/3/4: 看主结果、类别拆解、prompt adaptation。
- Figure 3: 看进一步 fine-tuning 到 ToxicChat 的 data efficiency。
- Section 6: 看 limitation，尤其是英语、coverage、prompt injection、不要当 chat model 用。

## 3. 六类 taxonomy

2023 Llama Guard paper 的 taxonomy 是 6 类:

- O1 Violence and Hate
- O2 Sexual Content
- O3 Guns and Illegal Weapons
- O4 Regulated or Controlled Substances
- O5 Suicide and Self Harm
- O6 Criminal Planning

这 6 类不是 "世界上所有风险"。作者明确说它是一个 sample taxonomy，足够覆盖很多 developer use cases，同时展示 LLM classifier 可以适配 policy。

taxonomy 的价值不只是分类名，而是类别边界:

- 类别要定义什么算 unsafe。
- 类别之间要有优先级或 disambiguation。
- 评测时要知道模型错在 binary 判断，还是错在 category mapping。
- 部署时要知道不同类别的 false positive 和 false negative 成本不同。

一个 guardrail 失败经常不是模型不会分类，而是 policy 边界不清。例如一个输出既像 Criminal Planning，又像 Guns and Illegal Weapons，应该标哪个?

Llama Guard 的做法是把 category descriptions 写进任务 prompt，让模型根据当前 taxonomy 判断。

## 4. prompt classification 和 response classification

这是论文最重要的设计之一。

普通内容审核常常把输入文本当成孤立 content。Llama Guard 认为 human-AI conversation 里有两个不同任务:

```text
Prompt classification:
  classify the user's message
  question: is the user asking for unsafe content?

Response classification:
  classify the assistant's response
  question: did the AI produce unsafe content?
```

为什么必须分开?

- 用户输入常常是请求、意图、问题。
- AI 输出常常是解释、建议、步骤、承诺或拒绝。
- 同一个词在用户请求和 assistant response 中的安全含义可能不同。
- 输入过滤漏掉的风险，输出过滤还可以兜住。

部署图:

```text
user message
    |
    v
input guard
    | safe
    v
main LLM
    |
    v
output guard
    | safe
    v
send to user
```

如果 input guard 判 unsafe，可以拒绝或改写。如果 output guard 判 unsafe，可以 block、regenerate、fallback 或 escalate to human review。

## 5. Llama Guard task 的四个 ingredient

Section 3.1 把 safeguard classification 写成 instruction-following task，有四个组成。

**1. Guidelines**

guidelines 包含 numbered categories 和每类的 safe/unsafe 描述。模型只能根据给定 guidelines 判断。

这点很重要: Llama Guard 不是把 policy 写死在 output head 里，而是把 policy 放进 prompt。这样可以换 taxonomy。

**2. Classification type**

任务会说明是 classify user messages，还是 classify agent messages。模型是同一个，但 instruction wording 不同。

**3. Conversation**

输入可以是 single-turn，也可以是 multi-turn conversation。实际部署时，你需要决定 guard 看多长上下文。

**4. Output format**

输出第一行是 `safe` 或 `unsafe`。如果 unsafe，后面列出 violated categories，例如 `O1`、`O3`。

论文强调 `safe` 和 `unsafe` 都是 SentencePiece tokenizer 里的 single token。这样可以从第一 token 概率读出 binary decision score。

最小任务形状:

```text
Instruction:
  classify prompt or response

Taxonomy:
  O1 ...
  O2 ...
  ...

Conversation:
  User: ...
  Assistant: ...

Required output:
  safe
or
  unsafe
  Oi
```

## 6. 数学和评分

Llama Guard 不是新 loss function 论文。它本质上是把分类写成 language modeling。

给定 guard prompt `g`，模型生成第一个 label token:

```text
p(label | g)
label in {safe, unsafe}
```

binary unsafe score 可以读成:

```text
score_unsafe = p(first token is "unsafe" | g)
```

如果输出 unsafe，再生成 category codes:

```text
p(O_i | g, "unsafe")
```

部署时常见 decision:

```text
if score_unsafe >= threshold:
    block or escalate
else:
    allow
```

阈值不是纯数学问题，而是产品风险问题:

- 提高 threshold: false positive 少，但可能漏掉 unsafe。
- 降低 threshold: recall 高，但 benign user 可能被误拦。
- 不同 category 可以有不同阈值。
- response guard 的阈值可能比 input guard 更严格。

论文评测主要使用 AUPRC，因为它关注 positive unsafe class 的 precision-recall trade-off，适合安全分类里正负样本不均衡的情况。

## 7. 数据收集

Section 3.3 的数据流程:

- 从 Anthropic harmlessness preference data 取第一条 human prompt。
- 丢弃原 assistant response 和其他 turns，得到初始 single-turn prompt dataset。
- 用内部 Llama checkpoint 生成 cooperating 和 refusing responses。
- 内部 expert red team 标注 prompt-response pairs。
- 每个样本标四个标签:
  - prompt-category
  - response-category
  - prompt-label: safe or unsafe
  - response-label: safe or unsafe
- 清理格式错误样本。
- 最终得到 13,997 prompt-response pairs。
- 按 3:1 随机切分 fine-tuning 和 evaluation。

Table 1 类别分布:

- Violence and Hate: prompts 1750, responses 1909。
- Sexual Content: prompts 283, responses 347。
- Criminal Planning: prompts 3915, responses 4292。
- Guns and Illegal Weapons: prompts 166, responses 222。
- Regulated or Controlled Substances: prompts 566, responses 581。
- Suicide and Self Harm: prompts 89, responses 96。
- Safe: prompts 7228, responses 6550。

两个观察:

- 数据量不大，说明 instruction-tuned LLM backbone 提供了大量语义能力。
- 类别不均衡明显，AUPRC 和 per-category analysis 很重要。

## 8. 训练细节和 data augmentation

模型:

- Backbone: Llama2-7B。
- 选择 7B 是为了部署成本和易用性。

训练:

- 单机 8 x A100 80GB。
- batch size 2。
- sequence length 4096。
- model parallelism 1。
- learning rate 2e-6。
- 500 steps，约 1 epoch。

data augmentation 很关键，因为 Llama Guard 的 taxonomy 在 prompt 里。作者希望模型看到 taxonomy subset 时，只根据出现的 categories 判断。

两种 augmentation:

- 如果某些 categories 没被样本违反，可以随机从 prompt 中 drop 一部分。
- 如果把所有 violated categories 都从 prompt 中 drop 掉，就把该样本 label 改成 safe。
- 训练时 shuffle category indices，避免模型死记 category position。

这让模型学习一个重要规则:

```text
Only judge against categories currently present in the guard prompt.
```

这也是 Llama Guard 能适配新 policy 的基础。

## 9. 评测方法: on-policy 和 off-policy

安全分类最难比的是 taxonomy 不同。OpenAI Moderation、Perspective API、Azure Content Safety、Llama Guard 的类别都不一样。

论文用了三种评测方式。

**max-all binary**

如果一个 API 给每个 category 一个概率，就取所有正类 category 的最大值:

```text
score_i = max(score_{c,i} for every positive category c)
```

这样只看 overall unsafe，不强行对齐 category。

**1-vs-all**

对目标 taxonomy 的每个 category `c_k`，单独构造一个任务:

```text
positive:
  examples labeled c_k

negative:
  benign examples plus examples from other categories
```

Llama Guard 可以通过 prompt 只包含 category `c_k` 来做这个任务。

**1-vs-benign**

对固定输出头 API，off-policy category mapping 很难。论文还使用 1-vs-benign:

```text
positive:
  examples labeled c_k

negative:
  benign examples only

dropped:
  other positive categories
```

作者提醒这可能偏乐观，因为去掉了 hard negatives。

## 10. 主实验结果

Table 2 使用 AUPRC，越高越好。

Llama Guard:

- Our Test Set prompt: 0.945。
- OpenAI Moderation Evaluation: 0.847。
- ToxicChat: 0.626。
- Our Test Set response: 0.953。

OpenAI API:

- Our Test Set prompt: 0.764。
- OpenAI Moderation Evaluation: 0.856。
- ToxicChat: 0.588。
- Our Test Set response: 0.769。

Perspective API:

- Our Test Set prompt: 0.728。
- OpenAI Moderation Evaluation: 0.787。
- ToxicChat: 0.532。
- Our Test Set response: 0.699。

怎么解读?

- 在自家 taxonomy 和数据上，Llama Guard 很强，prompt/response 都超过 0.94 AUPRC。
- 在 OpenAI Moderation dataset 上，Llama Guard zero-shot adaptation 到 0.847，接近 OpenAI API 的 0.856。
- 在 ToxicChat 上，Llama Guard 0.626，超过 OpenAI API 0.588 和 Perspective 0.532。
- response classification 上，Llama Guard 0.953 明显高于 baseline。

Table 3 的 per-category AUPRC 更重要:

- Violence and Hate: Llama Guard 0.857 prompt / 0.835 response。
- Sexual Content: 0.692 / 0.787。
- Criminal Planning: 0.927 / 0.933。
- Guns and Illegal Weapons: 0.798 / 0.716。
- Regulated or Controlled Substances: 0.944 / 0.922。
- Self-Harm: 0.842 / 0.943。

这说明总体分数背后不同类别差异很大，部署时不能只看 overall。

## 11. Adaptability: prompting 和 fine-tuning

Table 4 看 OpenAI Moderation dataset 上的 policy adaptation:

- OpenAI Mod API: 0.856 AUPRC。
- Llama Guard no adaptation: 0.837。
- Llama Guard zero-shot with OpenAI categories: 0.847。
- Llama Guard few-shot with descriptions and in-context examples: 0.872。

核心结论:

- 只把目标 taxonomy 写进 prompt，就能提升。
- 加 2 到 4 个 in-context examples 后，可以超过 OpenAI Moderation API。
- 这支持 "policy as prompt" 的价值。

Figure 3 看 fine-tuning 到 ToxicChat:

- 用 ToxicChat 的 10%、20%、50%、100% 训练数据分别 fine-tune。
- 从 Llama Guard 开始，比从 Llama2-7B 开始更 data-efficient。
- 论文说 Llama Guard 用 20% ToxicChat 数据，就能接近 Llama2-7B 用 100% 数据的性能。
- Llama2-7B zero-shot 甚至输出格式 malformed，AUPRC 设为 0；Llama Guard zero-shot 为 0.626。

这说明 Llama Guard 不是只学会一套固定 taxonomy，它学到了一种 "读 policy 做分类" 的任务格式。

## 12. Appendix P/R/F1 证据

因为 Azure API 和 GPT-4 baseline 不一定提供概率，无法计算 AUPRC，appendix 用阈值 0.5 报 precision/recall/F1。

Prompt classification overall:

- Llama Guard: P/R/F1 = 0.880 / 0.864 / 0.872。
- OpenAI Mod API: 0.874 / 0.250 / 0.389。
- Azure API: 0.788 / 0.515 / 0.623。
- Perspective API: 0.817 / 0.219 / 0.346。
- GPT-4: 0.717 / 0.947 / 0.816。

Response classification overall:

- Llama Guard: 0.900 / 0.867 / 0.884。
- OpenAI Mod API: 0.874 / 0.329 / 0.478。
- Azure API: 0.749 / 0.564 / 0.644。
- Perspective API: 0.751 / 0.248 / 0.373。
- GPT-4: 0.813 / 0.788 / 0.801。

这个 appendix 很实用: 它让你看到不同工具的 precision/recall profile 不一样。

GPT-4 prompt classification recall 高，但 precision 低于 Llama Guard；OpenAI API precision 高但 recall 很低。

部署时这不是谁绝对好，而是你选择什么风险曲线。

## 13. 方法图: Llama Guard pipeline

```text
Policy taxonomy
  O1 ... O6
  descriptions and boundaries
        |
        v
Guard task prompt
  classification type: prompt or response
  conversation text
  output format instruction
        |
        v
Llama2-7B fine-tuned as classifier
        |
        v
first token probability:
  p(safe), p(unsafe)
        |
        v
decision:
  safe
  unsafe + category codes
        |
        v
application action:
  allow / block / regenerate / escalate
```

## 14. 张量和数据形状

虽然 Llama Guard 是 "用 LLM 做分类"，但你仍然要把数据形状想清楚。

训练样本:

```text
{
  taxonomy_text: str,
  task_type: "prompt" or "response",
  conversation: list[turn],
  target_output: "safe" or "unsafe\nO_i[,O_j]"
}
```

模型输入:

```text
input_ids: (B, T)
attention_mask: (B, T)
```

监督目标:

```text
target_ids: (B, T_out)
```

分类分数:

```text
unsafe_score = probability assigned to first output token "unsafe"
category_scores = probabilities of O_i after unsafe, if needed
```

评测样本:

```text
true_label: safe or unsafe
true_categories: subset of {O1, ..., O6}
pred_score: float
pred_label: threshold(pred_score)
pred_categories: generated category codes
```

## 15. 代码样例: 原论文机制 toy 实现

本仓库新增:

```text
learning/safety-defense/src/llama_guard_original_minimal.py
```

它用无害 `unsafe:*` 标签演示 6 类 taxonomy 和输出格式:

```python
def classify_content(content, categories=LLAMA_GUARD_TAXONOMY):
    text = content.lower()
    matched = []
    for cat in categories:
        if any(trigger in text for trigger in cat.toy_triggers):
            matched.append(cat.code)
    if not matched:
        return LlamaGuardOutput(
            label="safe",
            categories=(),
            unsafe_score=0.0,
        )
    score = min(1.0, 0.45 + 0.25 * len(matched))
    return LlamaGuardOutput(
        label="unsafe",
        categories=tuple(matched),
        unsafe_score=score,
    )
```

这段代码对应论文机制:

- `LLAMA_GUARD_TAXONOMY` 对应 Section 2 的 O1-O6。
- `build_guard_task` 对应 Figure 1 的 instruction task。
- `render()` 对应 `safe` / `unsafe\nOi` 输出格式。
- `one_vs_all_scores` 对应 Section 4.1 的 per-category 1-vs-all 评测。

## 16. 和本仓库其他防御代码的连接

建议按这个顺序读:

1. `learning/safety-defense/src/llama_guard_original_minimal.py`
   - 对应原论文 2023 版本。
   - 看 6 类 taxonomy、输出格式和 1-vs-all。

2. `learning/safety-defense/src/llama_guard_mock.py`
   - 关键词教学版 input/output classifier。
   - 更像部署时的最小替身。

3. `learning/safety-defense/src/safety_eval_runner.py`
   - 看 confusion matrix、precision、recall、F1。
   - 对应 appendix P/R/F1。

4. `learning/safety-defense/src/defense_pipeline.py`
   - 看 PII redaction、rule rails、input classifier、injection defense、output classifier 如何串起来。

5. `learning/safety-defense/src/prompt_injection_defense.py`
   - 看 hidden content stripping 和 untrusted boundary。

本地测试:

```powershell
.\.venv\Scripts\python.exe learning\safety-defense\src\tests\test_defense.py
```

## 17. 一个 30-60 分钟本地实验

实验目标: 理解 taxonomy 改变如何改变 guard 输出。

步骤:

1. 打开 `llama_guard_original_minimal.py`。
2. 运行:

```powershell
.\.venv\Scripts\python.exe learning\safety-defense\src\llama_guard_original_minimal.py
```

3. 在 `LLAMA_GUARD_TAXONOMY` 中临时注释掉 O6。
4. 对 `unsafe:criminal_planning evaluation item` 运行 `classify_content`。
5. 观察结果从 unsafe 变成 safe。

这对应论文的 data augmentation 直觉:

```text
If a violated category is absent from the current taxonomy prompt,
the model should not judge against that absent category.
```

进阶:

- 修改 `one_vs_all_scores`，输出每类 score。
- 给一个样本同时包含两个 `unsafe:*` 标签。
- 看 `render()` 是否输出多 category。
- 用 `safety_eval_runner.py` 看 precision/recall 怎么变。

## 18. 论文局限性

Section 6 的限制必须认真读:

- Llama Guard 是 LLM，知识受训练和预训练数据限制，可能判断错。
- 训练和大部分预训练数据是英语，不保证其他语言性能。
- 标签质量虽高，但不代表 policy coverage 完美。
- 它被训练为分类器，不应当被当成普通 chat model 使用。
- 如果被当成 chat model prompt，它可能生成不安全语言，因为它没有作为 chat assistant 做安全 fine-tuning。
- 它可能受到 prompt injection，改变或绕过 intended use。

再加上工程层面的限制:

- 真实部署需要延迟预算，input guard 和 output guard 都要调用模型。
- taxonomy 版本要管理，否则评测结果不可比。
- threshold 要按产品风险调整。
- guard 只是一层防线，不能替代主模型对齐、检索过滤、工具权限和监控。

## 19. 对今天安全工程的意义

Llama Guard 今天仍然重要，因为它奠定了几个工程范式:

- Safety policy 可以作为文本输入给 classifier，而不是完全写死在输出头。
- Guardrail 要同时检查输入和输出。
- 评测要分 on-policy 和 off-policy。
- AUPRC 比单点 accuracy 更适合安全风险曲线。
- 公开权重可以让研究者做本地 fine-tuning 和 product-specific taxonomy。

和 red-team 论文连起来看:

```text
GCG paper:
  optimized attacks expose that direct refusal is not enough

Llama Guard:
  deploy independent guard model to classify inputs and outputs

Together:
  safety needs both adversarial evaluation and layered defenses
```

## 20. 用 AI agent 正确学习这篇论文

不要让 agent 只总结 "Llama Guard 是分类器"。你要让它逼你说清楚任务边界。

推荐流程:

1. 让 agent 要你列出 O1-O6，并解释每类边界。
2. 让 agent 给你一个 toy conversation，让你判断是 prompt classification 还是 response classification。
3. 让 agent 问你为什么 `safe`/`unsafe` 第一 token 概率能当 binary score。
4. 让 agent 让你手算一个 confusion matrix。
5. 让 agent 解释 max-all、1-vs-all、1-vs-benign 的区别。
6. 让 agent 要你读 Table 2/3/4，并说每张表支持哪个 claim。
7. 让 agent 帮你改本地 toy taxonomy，看输出怎么变。

提示词:

```text
我正在学习 Llama Guard 论文。
请按 taxonomy、prompt vs response、output format、AUPRC、adaptation 的顺序考我。
只使用 toy unsafe:* 标签，不要生成危险示例。
每次问一个问题，等我回答后纠正，并要求我映射到本仓库代码。
```

## 21. 闭卷掌握检查

读完后你应该能回答:

- 为什么已有 content moderation API 不足以直接当 LLM guardrail。
- Llama Guard 的 6 类 taxonomy 是什么。
- prompt classification 和 response classification 有什么区别。
- Figure 1 的 guard task prompt 包含哪四个 ingredient。
- `safe`/`unsafe` 为什么作为 first token 有部署意义。
- unsafe category code 为什么支持 multi-label classification。
- 13,997 数据集是怎么来的，标了哪四个标签。
- data augmentation 为什么要 drop categories 和 shuffle indices。
- max-all、1-vs-all、1-vs-benign 分别解决什么评测问题。
- Table 2 证明了什么，Table 3 又补充了什么。
- Table 4 为什么支持 policy-as-prompt adaptation。
- Figure 3 为什么说明 Llama Guard fine-tuning 更 data-efficient。
- Section 6 里最重要的 deployment risk 是什么。
- 本仓库哪个文件对应原论文机制，哪个文件对应 guardrail pipeline。

## 22. 一句话总结

Llama Guard 的核心不是简单关键词过滤，而是把安全 policy taxonomy 写进 LLM 分类任务里，让同一个 guard model 能做输入和输出分类，并通过可读、可改、可评测的输出格式成为 LLM 应用的一层可部署防线。
