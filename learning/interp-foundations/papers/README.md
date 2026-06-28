# 12.1 interp-foundations — 论文清单

> mech interp 入门论文。读法接 L4 的 5 问: 因果证据? 可证伪? 规模? 完备? 「理解」标准?

## 必读 (纲领)
- **A Mathematical Framework for Transformer Circuits** (Anthropic, 2021) — residual stream / QK-OV 电路框架 (L3)。
- **Toy Models of Superposition** (Anthropic, 2022) — superposition 的奠基 (L2 核心难题)。
- **Zoom In: An Introduction to Circuits** (Olah et al., 2020) — 逆向工程纲领 (视觉版, 思想通用)。

## 进阶 (现象)
- **In-context Learning and Induction Heads** (Anthropic, 2022) — induction circuit (M12.5 预告)。
- **Interpretability in the Wild (IOI)** (Wang et al., 2022) — 一个完整 circuit 的逆向 (从窄行为入手)。
- **The Building Blocks of Interpretability** — 特征可视化 + 归因。

## 批判 (接 L4 + M9.3)
- **Attention is not Explanation** / **Attention is not not Explanation** — attention≠因果重要性 (L4 陷阱一)。
- **Interpretability Illusions** — 警惕电路占星术 (讲故事 vs 可证伪)。

## 怎么读 (接 L4)
1. 有因果证据吗 (patching/ablation 还是只有相关)?
2. 可证伪吗 (定量预测+验证)? 3. 玩具还是真模型? 4. 完备吗? 5. 「理解」标准?

> 对照本专题: 真 gpt2 多义神经元 (N1) = superposition 的现象; 玩具 transformer (N2) = 教方法的受控对象。
