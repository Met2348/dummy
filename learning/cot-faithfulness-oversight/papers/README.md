# 12.6 cot-faithfulness-oversight — 论文清单

> CoT 忠实性 + scalable oversight + 欺骗检测。读法接 M9.3 + 你的 reasoning-r1。

## 必读 (CoT 忠实性)
- **Language Models Don't Always Say What They Think** (Turpin et al., 2023) — 偏置注入证明 CoT 不忠实 (你 N1)。
- **Measuring Faithfulness in CoT** (Anthropic, 2023) — 扰动 CoT 测忠实性 (早答/加错/改写)。
- **CoT Monitorability** (2024-2025) — CoT 监控的前景与脆弱 (优化压力反作用)。

## scalable oversight / 欺骗
- **Weak-to-Strong Generalization** (OpenAI, 2023) — 弱监督引出强能力 (你 N2)。
- **AI Safety via Debate** (Irving et al., 2018) — 辩论式监督。
- **Sleeper Agents** (Anthropic, 2024) — 训练出的欺骗后门难被移除 (欺骗风险)。
- **Sandbagging / deceptive alignment** — 装弱/欺骗性对齐的风险与检测。

## interp × 安全 (接 M12.2-12.5)
- 用探针/SAE/patching 检测「知识-陈述不一致」「欺骗特征」「CoT 真实性」(L4)。

## 怎么读 (接 M9.3 + reasoning-r1)
1. 忠实性怎么测 (干预: 改CoT/加偏置)? 2. CoT 监控的脆弱点? 3. w2s 在系统性错下还成立吗?

> 对照本专题: 真 TinyLlama 偏置敏感性高 (不忠实); w2s 玩具 强学生超弱监督 (归纳偏置平滑噪声)。
