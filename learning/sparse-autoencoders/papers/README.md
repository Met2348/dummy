# 12.4 sparse-autoencoders — 论文清单

> SAE / 字典学习论文。读法接 L4: 特征真实吗? 因果验证了吗? 评估指标合理吗?

## 必读 (核心)
- **Toy Models of Superposition** (Anthropic, 2022) — superposition 奠基 (为什么需要 SAE)。
- **Towards Monosemanticity** (Anthropic, 2023) — SAE 在小 transformer 上提取单义特征。
- **Scaling Monosemanticity (金门大桥)** (Anthropic, 2024) — 大模型 SAE + 可干预单义特征 (里程碑)。
- **Sparse Dictionary Learning** — 经典稀疏编码 (SAE 的数学根, 接你 EE)。

## 进阶 (方法/评估)
- **Gated SAE / TopK SAE / JumpReLU SAE** — SAE 架构改进 (减死特征/更好稀疏-重建权衡)。
- **Evaluating SAEs** — SAE 评估方法论 (重建/稀疏/可解释/因果)。

## 批判 (接 L4 + M9.3)
- SAE 可能强加结构 (控制实验); 特征单义≠模型在用 (因果验证); 评估无公认标准。

## 怎么读 (接 L4)
1. 特征真实吗 (控制实验)? 2. 因果验证 (干预)? 3. 评估指标? 4. 完备? 5. 可扩展?

> 对照本专题: 玩具 SAE 纯度 远超原始神经元 (解叠加硬证据); gpt2 SAE 特征比多义神经元成主题。
