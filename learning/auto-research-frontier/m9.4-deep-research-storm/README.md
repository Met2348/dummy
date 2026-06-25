# 9.4 · Deep Research / 文献综述 Agent（mini-STORM）

> 把"自动读完一个领域、写出带引用的综述"做成可跑的缩小版，并守住这个范式最致命的一关：
> **引用存在 ≠ 引用忠实**。这是 8 个专题里**当天就能用在你自己科研上**的一个。

## 一句话结论（本模块跑出来的）

```
naive 检查（只问 id 在不在库）：5/5 句通过 —— 看起来全合规。
忠实度检查（问被引文献真支持吗）：揪出 2 句不忠实：
  · AI Scientist v2 的结果已经过湿实验验证 [2504.08066]
    → 引的 [2504.08066] 真实存在，但它并不支持 'wet-lab-validated' 这个论断。
```

两句话都引了**真实存在、且确实被检索到**的论文，却主张了那篇论文根本没说的事
（"湿实验验证"是 co-scientist 干的，不是 v2）。只核对"引用是否存在"的人会被完全骗过。

## grounding 三连里的最深一环

| 关卡 | 问题 | 谁守 |
|------|------|------|
| 存在性 | 引的 id 在不在？（防幻觉一个假 id） | 9.2 的 grounding 检查 |
| **忠实度** | **被引文献到底支不支持这句话？** | **9.4（本模块）** |
| 可信验证 | 谁来独立复算这些判断？ | 9.6 评测 / 9.8 红队 |

`faithfulness.py` 把这层差别写成两个函数：`existence_only`（只看 id 在库）vs
`check_sentence`（看 `claim ⊆ 被引文献的事实账本`）。一个引用可以**存在却不忠实**。

## 跑起来

```powershell
python src/run.py                                  # 默认主题，逐句忠实度 + 两种检查对比
python src/run.py --topic "autonomous research agents"

python scripts/eric_3080ti_env_audit.py --runbook --tests `
  --modules auto-research-frontier/m9.4-deep-research-storm `
  --json-out $env:TEMP/m9.json --md-out $env:TEMP/m9.md
```

## 目录

```
m9.4-deep-research-storm/
├── runbook.yaml
├── lectures/
│   ├── 01-storm-paradigm.md            多视角 + outline-driven RAG
│   ├── 02-exists-vs-faithful.md        核心：引用存在 ≠ 引用忠实
│   └── 03-verifying-faithfulness.md    怎么真核查 + DeepScholar-Bench + 守卫
└── src/
    ├── run.py
    ├── mini_storm/
    │   ├── corpus.py        带"事实账本"(supports)的文献库 + 检索
    │   ├── storm.py         多视角提问→检索→带引用合成(注入2句不忠实)
    │   └── faithfulness.py  existence_only vs check_sentence
    └── tests/test_faithfulness.py  6 测试：忠实句过、不忠实句被抓、存在性被骗
```

## Hands-on（轮到你）

`check_sentence` 现在用"claim 集合 ⊆ 文献 supports 集合"判忠实——这是 NLI/蕴含的极简代理。

1. **加一种"半忠实"**：让一句话的 claim 有一半 token 被支持、一半不被支持。
   现在的"全包含才算 faithful"会判它什么？你觉得综述里这种"部分支持"该算忠实吗？
   （这正是真实忠实度评测最难的灰区。）
2. **接真检索**：把 `corpus` 换成对 `papers/INDEX.md` 的真实条目，
   `supports` 用每篇摘要的关键句——看真实摘要下忠实度还好不好判。
3. **自查你自己的综述**：把你最近写的一段带引用的文字，按"每句 claim + 引的文献"喂进来，
   亲手核一遍——有多少句其实是"引了真论文但它没这么说"？

## 桥接

- 复用 **rag-essential**（检索）· **agent-code-eval**（"核查"的评测思维）。
- 呼应 **9.2**（存在性 grounding）→ 本模块（忠实度）→ **9.6/9.8**（独立验证 / 红队幻觉引用）。
