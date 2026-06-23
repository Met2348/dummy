# papers/ — diffusion-foundations 参考源

## 扩散基础
- **DDPM** — "Denoising Diffusion Probabilistic Models" (Ho et al. 2020). 本专题核心 (L2/L3/L4)。
- **Score-based generative modeling** — Song & Ermon (2019/2021). score matching 视角 (L3)。
- **DDIM** — "Denoising Diffusion Implicit Models" (Song et al. 2021). 少步确定性采样 (L4)。

## 连续时间视角 (通往 M13.2)
- "Score-Based Generative Modeling through SDEs" (Song et al. 2021). SDE/ODE 统一视角。
- 你的 EE 随机过程/SDE 背景在这里直接是理解力。

## 综述
- "Understanding Diffusion Models: A Unified Perspective" — 各种推导的统一讲解 (适合补数学)。

> 本专题知识在可跑的 `diffusion.py` (2D DDPM, 看见去噪轨迹) 里。
> 真练习: 在 MNIST 上训一个真 DDPM, 看生成的数字从噪声浮现。
