# papers/ — flow-matching-sota 参考源

## flow matching / rectified flow
- **Flow Matching for Generative Modeling** — Lipman et al. 2023 (L1)。
- **Rectified Flow** — Liu et al. 2023 (reflow 拉直, L2)。
- **Stable Diffusion 3** / **Flux** — 工业级 flow matching 训练 (L4)。

## consistency / 一步生成
- **Consistency Models** — Song et al. 2023 (L3)。
- **Latent Consistency Models** / **SD-Turbo** — 蒸馏少步 (L3)。

## 统一视角 (你 EE 优势)
- "Score-Based Generative Modeling through SDEs" — Song et al. (SDE/ODE 统一, L1/L4)。
- DPM-Solver — 高阶 ODE 积分器 (L4)。

> 本专题知识在可跑的 `flow_matching.py` (速度场 + reflow) 里。
> 真练习: 在 MNIST 上比 DDPM 和 flow matching 的少步生成质量。
