# L10 · 指令数据合成 — Alpaca → Self-Instruct → Magpie

> 24 slides | 70 min | Data Curation 第 10 讲 ⭐⭐⭐⭐⭐

---

## 学习目标

1. 理解 SFT 指令数据演化
2. 掌握 Self-Instruct / Magpie 完整流程
3. 知道质量评估与多样性度量
4. 写一个 Magpie 玩具版

---

## Slide 1 · 指令数据从哪来

```
2022 手工 (Anthropic-HH / FLAN)
2023 GPT 蒸馏 (Alpaca / WizardLM)
2024 自合成 (Self-Instruct / Magpie / Tülu 3)
2025 私有为主（OpenAI / Anthropic / Google）
```

成本：手工 $30/条 → GPT 蒸馏 $0.10 → Magpie 自合成 $0.01。

---

## Slide 2 · FLAN（2021）

Google T5 → FLAN：
- 1.8k 任务模板
- 10M+ examples
- 人工写 prompt 模板，自动套数据集
- 涌现：zero-shot 大幅提升

```
"Translate to French: {text}"   → 多语言能力
"Classify sentiment: {text}"    → 分类能力
"Solve: {math_problem}"         → 数学能力
```

---

## Slide 3 · Alpaca 流程（2023）

Stanford：

```
1. 175 个 seed instruction (人手写)
2. GPT-3.5 prompt → 生成 52k 新 instruction + answer
3. 过滤近似的（rouge < 0.7）
4. fine-tune Llama-7B → Alpaca
```

成本：$500 调 API。让"小模型对齐"变得可行。

---

## Slide 4 · WizardLM（2023）

"Evol-Instruct"：

```
seed instruction
   ↓ + complexity
变深、变难、变长
   ↓
WizardLM data 70k
```

策略：广度（多 topic）+ 深度（一题多变体）。

---

## Slide 5 · Self-Instruct（2022 → 2023 主流）

UW + AllenAI：

```
Round 0: 175 seed
Round 1:
  - 模型生成 8 个新 instruction
  - 区分 class / open-ended
  - 模型自答
  - 过滤重复
  - 加回 seed
Round N: 收 50k+
```

→ "模型给自己出题 + 自己答题"，递归扩展。

---

## Slide 6 · Magpie（2024）

CMU + AllenAI："无 seed 全合成"：

```
对 Llama-3-Instruct：
1. 给空 system prompt + 开头 token <|begin_user|>
2. 模型自然 continue → 写出一个 instruction
3. 用同模型答这个 instruction
4. 收集 instruction + response
```

**核心**：模型已会的能力，可被它自己 elicit。无 seed 必要。

---

## Slide 7 · Magpie 流程图

```
empty prompt → tokenizer(<|im_start|>user) → model.generate
              ↓
        user instruction
              ↓
   completed prompt → model.generate
              ↓
        assistant response
              ↓
     (instruction, response) 对入库
```

收集 1M 对仅需 ~500 GPU 小时（Llama-3-70B）。

---

## Slide 8 · Magpie 质量

Tülu 3 / Open-Hermes 实测：
- 多样性优于 Alpaca
- 数学 / code 能力得分高
- 安全性需补（合成可能 leak unsafe pattern）

→ 配合 RLHF 仍需补漏。

---

## Slide 9 · 评估"多样性"

```
1. embedding clustering 聚 1000 cluster，看分布均匀
2. n-gram diversity (distinct-3 / distinct-4)
3. instruction length 分布
4. category distribution（reasoning/coding/creative/factual/...）
```

Magpie 通常 distinct-3 ≈ 0.7（较高），但 reasoning 类偏少。

---

## Slide 10 · Tülu 3（2024.11）

AllenAI 高质量混合 SFT：

```
3M instruction-response (混合)
  - 30% Magpie
  - 20% WildChat (real)
  - 15% MetaMath / OpenMathInstruct
  - 10% Code-Instruct (合成)
  - 10% Roleplay (Persona)
  - 15% Misc
```

→ 平均开源 SFT 数据中 Tülu 3 最强（截至 2024.12）。

---

## Slide 11 · MetaMath / GSM8K-rewrite

数学专项合成：

```
原 GSM8K 题  ─► GPT-4 rewrite ─► 8 个变体 (改数字、改语境、扩展)
                                 ↓
                            MetaMath 800k pairs
```

效果：Llama-2-7B 在 GSM8K +20pt。

---

## Slide 12 · Self-Rewarding 数据

Meta 2024：
- 模型自己生成 instruction
- 模型自己评分（self-reward）
- 选高分对训 DPO
- 反复

→ "自合成 + 自打分 + 自对齐"闭环。

---

## Slide 13 · WildChat

真实 ChatGPT API 用户对话：
- ~500k 对话
- 真实分布、真实 jailbreak 尝试
- 含敏感 / PII（需脱敏）

合成数据 + 真实数据混合是 2024 SOTA 配方。

---

## Slide 14 · 长上下文 SFT 数据

```
LongAlpaca (2023)    32k token instruction
LongAlign            8-128k
LongLoRA             64k
合成 long-doc reading comprehension
```

合成方式：把书 / 长 wiki 摘要后让 LLM 答关于细节的问题。

---

## Slide 15 · 多语言 SFT

合成多语言 → 翻译合成：

```
1. 高质量 en SFT → GPT-4 translate to {zh, ja, fr, es}
2. 译后 sanity check (BLEU vs back-translate)
3. 与英文混合训
```

Qwen / Aya 都用此。

---

## Slide 16 · 多模态 instruction

```
LLaVA-Instruct      → image + (Q, A)
ShareGPT-4V         → image + 多轮对话
M3IT                → 多模态多任务
```

合成方式：CLIP encoder + GPT-4V → 自动生成图片描述与问题。

---

## Slide 17 · 安全 SFT

```
HH-RLHF (Anthropic)   每对话两 response，rank
Constitutional AI      用 16 条原则让 LLM 自批改
Safe-RLHF             helpful + harmless 双 RM
```

合成数据后必须叠加安全数据，否则模型不知 refusal。

---

## Slide 18 · "code instruction"

```
WizardCoder           Evol-Instruct on Code
Magicoder OSS-Instruct 从开源 snippet 反向 generate
CodeFeedback          含运行错误 + LLM 修复
```

Code 合成已是 2024 主流，HumanEval 提升明显。

---

## Slide 19 · 数据"含金量"指标

```
平均长度       ≥ 200 token (response)
多轮率         ≥ 30%
"refusal" 率   ≤ 5%（避免学会 refuse 常规问题）
benchmark contamination ≈ 0
```

Tülu 3 公开了完整 metric，自查任何 SFT 数据集。

---

## Slide 20 · 数据规模与模型

```
small (1B)   30k-100k        SFT
medium (7B)  300k-1M
large (70B)  1M-10M
```

数据 ≠ 越多越好；超过后 SFT 数据"稀释"模型基座。

---

## Slide 21 · 合成数据"陷阱"

| 陷阱 | 后果 |
|------|------|
| teacher 模型偏见 | 学生继承偏见 |
| 同质性 | 多样性低 |
| 长度膨胀 | response 过长 |
| 数学错误传播 | teacher 错 → 学生错 |
| 安全短板 | 学不到 refusal |

防御：混真实数据 + 独立 verifier check。

---

## Slide 22 · 评估 SFT 数据

```
1. MT-Bench (single-turn 8 类，GPT-4 评)
2. AlpacaEval 2 (LC win-rate)
3. AGIEval / IFEval
4. WildBench (in-the-wild)
```

数据集本身评不了 → 必须训小模型在评测看。

---

## Slide 23 · Magpie 玩具版（伪）

```python
def magpie_one(model, tokenizer):
    # 让模型自由生成 instruction（无 system prompt）
    prompt_inst = "<|im_start|>user\n"
    out_inst = model.generate(prompt_inst, max_new_tokens=200)
    inst = extract_user_text(out_inst)

    # 让模型答这个 instruction
    full = f"<|im_start|>user\n{inst}<|im_end|>\n<|im_start|>assistant\n"
    out_resp = model.generate(full, max_new_tokens=400)
    resp = extract_assistant_text(out_resp)
    return inst, resp
```

详见 `src/magpie_synthesis.py`。

---

## Slide 24 · 课后思考

1. Magpie 用 base 模型还是 instruct 模型？为什么？
2. 合成数据中 hallucinated 答案如何检测？
3. 多轮 vs 单轮：哪种 SFT 数据更重要？
4. 数据"含金量"和"多样性"的关系？

---

## 参考

- Alpaca: Taori 2023
- Self-Instruct: Wang 2022
- WizardLM Evol-Instruct: Xu 2023
- Magpie: Xu 2024
- Tülu 3: AllenAI 2024
