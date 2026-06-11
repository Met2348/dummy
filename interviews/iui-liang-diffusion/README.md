# IUI Liang Diffusion Interview Prep

This folder was created after the IUI Liang interview feedback on diffusion
models. It contains the three candidate papers, arXiv abstract snapshots, and
Chinese PPT-style deep reading guides.

## Papers

| # | Paper | arXiv | Local PDF | Guide |
|---|---|---|---|---|
| 1 | Derivative-Free Guidance in Continuous and Discrete Diffusion Models with Soft Value-Based Decoding | https://arxiv.org/abs/2408.08252 | `papers/01_derivative_free_guidance.pdf` | `guides/guide_01_derivative_free_guidance_pptstyle.md` |
| 2 | Unlocking Guidance for Discrete State-Space Diffusion and Flow Models | https://arxiv.org/abs/2406.01572 | `papers/02_unlocking_guidance_discrete_diffusion_flow.pdf` | `guides/guide_02_unlocking_guidance_discrete_diffusion_flow_pptstyle.md` |
| 3 | Simple and Effective Masked Diffusion Language Models | https://arxiv.org/abs/2406.07524 | `papers/03_simple_effective_masked_diffusion_lm.pdf` | `guides/guide_03_simple_effective_masked_diffusion_lm_pptstyle.md` |

## Interview Focus

Liang老师明确会考察 diffusion models 的理解，尤其是 methodology 和 experiments。因此每份导读都按下面结构准备：

- 背景和问题设定：为什么这篇论文要存在。
- 数学和算法：用尽量新手友好的方式解释核心公式、变量、采样过程。
- 实验和证据链：数据集、baseline、指标、核心结果、消融实验、局限。
- 面试问答：可能被追问的问题和回答框架。
- 复现路线：如果有 GPU，如何从最小实验开始玩起来。

建议优先顺序：

1. 先读 `guide_03_simple_effective_masked_diffusion_lm_pptstyle.md`，建立 masked diffusion language model 的基础。
2. 再读 `guide_02_unlocking_guidance_discrete_diffusion_flow_pptstyle.md`，理解 discrete guidance 的 rate-matrix 视角。
3. 最后读 `guide_01_derivative_free_guidance_pptstyle.md`，把 reward optimization、value function、derivative-free decoding 串起来。
