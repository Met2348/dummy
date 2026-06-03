# Lecture 7: K-Adapter + MAD-X — 应用层 PEFT

> 论文 1: Wang et al. 2020, "K-Adapter: Infusing Knowledge into Pre-Trained Models with Adapters" (ACL Findings)
> 论文 2: Pfeiffer et al. 2020, "MAD-X: An Adapter-Based Framework for Multi-Task Cross-Lingual Transfer" (EMNLP)
> 配套代码: `src/k_adapter_minimal.py` + `src/madx_minimal.py`
> 配套 notebook: `notebooks/07-k-adapter-mad-x.ipynb`

---

## Slide 1: 本节路线

```
应用层 PEFT → K-Adapter (知识注入) → MAD-X (跨语言) → 实验
```

**学完本节，你应该能**:
1. 解释"知识注入"为什么用 adapter
2. 推算 MAD-X 在 100 语言上的参数量
3. 理解 invertible adapter 的设计动机

---

## Slide 2: 痛点 — PLM 缺什么？

预训练 PLM (BERT/GPT-2) 虽然强，但**缺两类东西**:

1. **结构化知识**: Wikidata 三元组、医学/法律领域术语
2. **稀有语言/多语言对齐**: 训练数据语言分布不均

**Naive 方案**: 用领域数据 continued pre-training → 训练昂贵 + 灾难遗忘。

**Adapter 方案**: 把"知识"封装到 adapter（任意叠加，base 不变）。

---

## Slide 3: K-Adapter 架构

```
                ┌──────────────────────┐
       input ──→│   冻结 PLM (BERT/GPT)│
                └──────────────────────┘
                       ↓ (hidden)
                ┌──────┴──────┐
                ↓              ↓
        ┌─────────────┐  ┌─────────────┐
        │ Factual K-A │  │ Linguistic  │
        │  (Wikidata) │  │   K-A (dep) │
        └─────────────┘  └─────────────┘
                ↓              ↓
                └──→ Σ (sum) ──┘
                       ↓
                    output
```

每个 K-Adapter 独立预训于特定知识源，**训完冻结**，可任意组合。

---

## Slide 4: K-Adapter 预训练数据

| 类型 | 数据源 | 任务 |
|------|-------|------|
| **Factual** | Wikidata | (entity, relation) prediction |
| **Linguistic** | Stanford dependency | Dependency parsing |
| **Commonsense** | ConceptNet | Concept relation |
| **Medical** | UMLS | Medical entity linking |

**用法**: 用户根据下游任务选 adapter 组合
- 医疗问答 → factual + medical
- 阅读理解 → linguistic + commonsense

---

## Slide 5: K-Adapter 代码（简化教学版）

```python
class MultiKnowledgeGPT2(nn.Module):
    def __init__(self, knowledge_types=("factual", "linguistic")):
        self.lm = GPT2LMHeadModel.from_pretrained("gpt2")
        freeze_base_model(self.lm)

        for block in self.lm.transformer.h:
            adapters = nn.ModuleList([
                KAdapter(d, r, kt) for kt in knowledge_types
            ])
            block.mlp = _KMlpWrapper(block.mlp, adapters)

    def freeze_adapter(self, kt):
        """训完某类知识后冻结。"""
        for block in self.lm.transformer.h:
            for ka in block.mlp.adapters:
                if ka.knowledge_type == kt:
                    for p in ka.parameters():
                        p.requires_grad = False
```

---

## Slide 6: K-Adapter 实验（论文）

Wang et al. 在 LAMA factual probing:

| 方法 | LAMA | T-REx | ConceptNet |
|------|------|-------|-----------|
| BERT-base | 31.4 | 39.4 | 14.6 |
| BERT + factual K-A | **34.1** | **42.5** | 14.7 |
| BERT + linguistic K-A | 31.5 | 39.5 | 14.7 |
| BERT + factual + linguistic | **34.4** | **42.6** | **15.1** |

**关键 takeaway**: 多 adapter 叠加 > 单 adapter > base。

---

## Slide 7: 转到 MAD-X — 跨语言痛点

**问题**: 想做"German NER" 但只有 English NER 训练数据。

**朴素方案**:
- 翻译数据 → 损失语言特性
- mBERT 端到端 → 语言间干扰

**MAD-X 思路**: 把"语言"和"任务"**解耦**到不同 adapter:
- **Language Adapter (LA)**: 学每种语言特性
- **Task Adapter (TA)**: 学每个任务（与语言无关）

→ 用法: 选合适的 LA + TA 组合就行。

---

## Slide 8: MAD-X 架构

```
                  ┌──────────────────────┐
       input ───→ │   Invertible Adapter │ ← 处理 embedding 层
       (lang_de)  │   (per language)     │
                  └──────────────────────┘
                          ↓
                  ┌──────────────────────┐
                  │   冻结 PLM           │
                  └──────────────────────┘
                          ↓
                  ┌──────────────────────┐
                  │  Language Adapter    │ ← 学语言特性
                  │  (lang_de in this    │
                  │   batch)             │
                  └──────────────────────┘
                          ↓
                  ┌──────────────────────┐
                  │   Task Adapter       │ ← 学任务（无关语言）
                  │   (task_ner)         │
                  └──────────────────────┘
                          ↓
                  task output
```

---

## Slide 9: MAD-X 用法 — Cross-lingual Transfer

**Stage 1**: 用单语数据预训 LA
- 每种语言用 MLM 训自己的 LA
- 训完冻结

**Stage 2**: 用英语任务数据训 TA
- 用 LA_en + TA_ner 组合
- 只训 TA_ner

**Stage 3**: Inference on German
- 用 LA_de + TA_ner（替换 LA！）
- → zero-shot 德语 NER

**论文 claim**: zero-shot 德语 NER 接近 supervised 80%。

---

## Slide 10: Invertible Adapter

**为什么需要**:
- Transformer 的 token embedding 层 (50k 词表 × d) 占大量参数
- 每语言一个完整 embedding 太重
- 用 invertible adapter 在 embedding 上做 token 级"变换"

**简化公式**:
$$y = \sigma \odot x + \mu$$
$$x = (y - \mu) / \sigma \quad \text{(invertible)}$$

参数: $2d$（极少），但能让 embedding 适配语言。

---

## Slide 11: MAD-X 参数量

GPT-2 d=768, r=16, 3 lang + 1 task + 3 IA:

| 组件 | 数量 | per layer | 12 layer | 总计 |
|------|------|-----------|----------|------|
| Language Adapters | 3 | 25,360 | 304,320 | 912,960 |
| Task Adapters | 1 | 25,360 | 304,320 | 304,320 |
| Invertible Adapters | 3 | (embedding 层) | — | 4,608 |
| **总计** | | | | **1,221,888** |

**扩展到 100 语言 + 5 任务**: ~104M（仍小于 base 124M）

---

## Slide 12: 代码核心（MAD-X）

```python
class _MADXMlpWrapper(nn.Module):
    def __init__(self, base_mlp, d, r, languages, tasks):
        super().__init__()
        self.base_mlp = base_mlp
        self.language_adapters = nn.ModuleDict({
            lang: HoulsbyAdapter(d, r) for lang in languages
        })
        self.task_adapters = nn.ModuleDict({
            task: HoulsbyAdapter(d, r) for task in tasks
        })
        self.active_language = languages[0]
        self.active_task = tasks[0]

    def forward(self, x):
        h = self.base_mlp(x)
        h = self.language_adapters[self.active_language](h)  # 先 lang
        h = self.task_adapters[self.active_task](h)          # 再 task
        return h

# 切换语言
model.set_active("de", "ner")
```

---

## Slide 13: 与 adapters 库对照

adapters 库原生支持 MAD-X 风格的 **Stack composition**:

```python
from adapters.composition import Stack
model.active_adapters = Stack("lang_en", "task_ner")  # MAD-X 风格
```

**参数对比**:
- minimal: 1,221,888（含 invertible adapter）
- adapters lib: 1,217,280（lib 没用 invertible adapter）
- 差异: 4,608 = 3 × 2 × 768（正好是 3 个 IA 的参数）

---

## Slide 14: K-Adapter vs MAD-X 对比

| 维度 | K-Adapter | MAD-X |
|------|-----------|-------|
| **设计目的** | 注入领域知识 | 跨语言 transfer |
| **adapter 类型** | 同构（都是 Houlsby）| 异构（LA + TA + IA） |
| **组合方式** | 求和 (sum) | 串联 (Stack) |
| **切换粒度** | 加 / 减 知识类别 | 切换 active 组合 |
| **典型场景** | 医疗问答、KG QA | 100 语言 NER |

**共同点**: 都是 "adapter 即插即用 + 组合即赢" 思想。

---

## Slide 15: 思考题

**公式题**:
1. 算 K-Adapter 在 mBERT (d=768) 上叠 5 类知识的参数量。
2. 推导 invertible adapter 的 forward 和 inverse 是否真"互逆"。
3. 证明：MAD-X 的"Stack lang→task"是非交换的（顺序重要）。

**设计题**:
4. 如果想加新语言到 MAD-X，需要训什么？训多久？
5. 用 K-Adapter 加多模态（vision adapter），可行吗？
6. 把 K-Adapter 的"和" 改成 attention 融合（类 AdapterFusion）会怎样？

**对比题**:
7. K-Adapter vs AdapterFusion 的本质差异？
8. MAD-X 与"多 LoRA 切换"有什么相似/差异？

---

## Slide 16: 工程选型

| 场景 | 推荐 |
|------|------|
| 多领域知识库 + 单任务 | K-Adapter ⭐ |
| 多语言 + 多任务 | MAD-X ⭐⭐⭐ |
| 100+ 语言 NLP 平台 | MAD-X |
| 单任务 + 单语言 | LoRA（这俩太重）|

**今天还在用吗**: ✅ MAD-X 在跨语言研究和 multilingual benchmark 仍活跃。AdapterHub 提供数百个预训 LA + TA。

---

## Slide 17: 与下节衔接 — AdaMix

**到此为止**: 学了"显式组合" (K-Adapter sum, MAD-X stack)。

**下节 AdaMix**: 用 **MoE 路由**自动选 adapter
- 训练时随机选 1 个 expert
- 推理时 average
- 比手动组合更智能

下节细讲。

---

## Slide 18: 本节小结

```
┌─────────────────────────────────────────────┐
│ K-Adapter (2020)                            │
│   多个领域 K-A 叠加，知识可插拔              │
│   factual + linguistic 在 LAMA 上 +3.0      │
├─────────────────────────────────────────────┤
│ MAD-X (2020)                                │
│   LA + TA + IA 三类 adapter 解耦            │
│   Stack(lang, task) 实现 zero-shot transfer │
│   100 语言 NLP 仍主流                       │
└─────────────────────────────────────────────┘
              ↓
       下节: AdaMix (MoE 路由)
```
