# 12.2 probing-and-activations — 论文清单

> 探针 + logit lens 论文。读法接 L4: 探针读出 ≠ 模型在用 (相关非因果)。

## 必读 (核心)
- **Designing and Interpreting Probes** (Hewitt & Liang, 2019) — 探针的方法论 + 控制任务 (探针太强会作弊)。
- **A Structural Probe for Syntax** (Hewitt & Manning, 2019) — 经典: BERT 线性编码句法树。
- **logit lens** (nostalgebraist, 2020) — 中间层过 unembed 看预测成形。
- **Tuned Lens** (Belrose et al., 2023) — logit lens 的严谨化 (学习的层变换校准)。

## 进阶 (表示几何)
- **Linear Representation Hypothesis** — 概念是激活空间的线性方向。
- **Probing 的因果局限** — 探针读出≠模型用 (引出 patching, L4 + M12.3)。

## 批判 (接 L4 + M9.3)
- 探针准确率高可能是探针强而非模型编码好 (控制任务对照)。
- attention 权重 ≠ 因果重要性 (Attention is not Explanation)。

## 怎么读 (接 L4)
1. 探针是线性的吗 (否则探针自己算概念)?
2. 有控制任务对照吗 (排除探针作弊)?
3. 有没有因果验证 (probing→patching)?

> 对照本专题: 玩具读「当前值」1.00; 真 gpt2 读「是否数字」1.00 + logit lens 看 Paris 逐层浮现。
