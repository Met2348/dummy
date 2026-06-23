# Module 12 机制可解释性 Implementation Plan

**Goal**: 新建 Module 12「机制可解释性/对齐前沿」(7 专题)。tiny transformer + 玩具任务, CPU 上真做 probing/patching/SAE/电路分析。

**Design**: `docs/superpowers/specs/2026-06-24-module12-interpretability-design.md`

**Tech Stack**: Python 3.13 / torch (tiny, 可 hook 激活) / numpy / matplotlib / nbformat. 确定性 (固定 seed), 玩具任务 (模运算/括号/induction) 离线可跑。

**依赖**: **独立**, 不依赖 M10/M11/M13。可任意顺序建 (推荐放最后或穿插, 因它最偏纯研究)。强接用户已有 reasoning-r1/process-reward (12.6)。

**构建顺序 (模块内)**: 12.1 foundations → 12.2 probing → 12.3 causal interventions → 12.4 SAE → 12.5 circuits → 12.6 CoT/oversight → 12.7 capstone。

---

## Phases
- **P1 `interp-foundations`**: tiny_transformer.py (可 hook); N1 多义神经元可视化 / N2 玩具任务训 tiny model
- **P2 `probing-and-activations`**: probing.py (线性探针+logit lens); N1 探针读概念 / N2 logit lens 逐层
- **P3 `causal-interventions`**: patching.py (核心: activation patching+ablation); N1 patching 因果定位 / N2 ablation 最小子集
- **P4 `sparse-autoencoders`**: sae.py; N1 mini-SAE 提特征 / N2 单义性检视
- **P5 `circuits-attention`**: circuits.py; N1 找 induction head / N2 归因热图 (复用 9.6)
- **P6 `cot-faithfulness-oversight`**: cot_probe.py; N1 CoT 忠实性扰动 (接 reasoning-r1) / N2 weak-to-strong 玩具
- **P7 `interp-graduation`**: Capstone; N1 对自己模型端到端解剖 / N2 interp×reasoning gap→idea 卡

## 成功标准
- [ ] 7 专题完整, verify_env 全过, 14 notebook 0 报错 (tiny model CPU 可跑)。
- [ ] 至少一个 notebook 做出 activation patching 因果定位 + SAE 特征。
- [ ] 课件公式逐项 (SAE 损失/patching/QK-OV)。
- [ ] Capstone 产出 interp×reasoning 研究 idea 卡 (PhD 候选方向)。
