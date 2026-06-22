# papers/ — experiment-design 参考源

本专题教**方法论**, papers/ 收实验设计/统计严谨性的经典 + ML 复现危机的代表作。

## 实验严谨性 / 统计 (方法论)
- **Deep Reinforcement Learning that Matters** — Henderson et al. 2018. RL 实验对种子/超参极度敏感、单种子结论不可靠的著名警钟; L5 的现实依据。
- **Show Your Work / Showing Your Work Doesn't Always Work** — Dodge et al. 关于报告超参搜索预算与公平对照 (L3) 的讨论。
- **A Metric Learning Reality Check** — Musgrave et al. 2020. 「公平对照后, 多年'提升'消失」的经典案例 (L3 同等努力原则)。
- **Statistical significance testing for NLP** — Dror et al. 显著性检验在 NLP 的实践指南 (L5)。
- Cohen, *Statistical Power Analysis* — 效应量 (Cohen's d) 与检验力的来源 (L5)。

## 证伪主义 (L1 的哲学根)
- Karl Popper, *The Logic of Scientific Discovery* — 可证伪性作为科学的分界标准。
- HARKing: Kerr 1998, "HARKing: Hypothesizing After the Results are Known" (L1 反模式)。

## 与本专题运行例子对应的论文 (拿自己复现练)
- DPO — Rafailov et al. 2023 (baseline)
- Robust/noise-robust 偏好优化系列 2024 (本专题假设的现实原型)
- 你自己的 `learning/dpo-family` 复现 —— **最好的练习材料**: 在它上面真排一个噪声鲁棒性消融。

## 为什么 papers/ 这么轻
本专题知识在**可跑的模拟器 + 统计工具** (`experiment.py`/`stats.py`) 和课件方法里。
模拟器让你零算力把「设计→跑→读」循环走几十遍; 真功夫是把这套搬到你自己的真实实验上。
