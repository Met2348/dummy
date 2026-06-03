# Papers — RL Foundations

> 12 篇核心论文（按 lecture 顺序）。PDF 自取放在本目录，文件名按下表 `code.pdf` 命名。

| # | Lecture | 论文 | Code | 链接 |
|---|---------|------|------|------|
| 1 | L01 mdp-policy | Williams 1992, **REINFORCE** | `reinforce-1992.pdf` | https://link.springer.com/article/10.1007/BF00992696 |
| 2 | L02 actor-critic | Mnih et al. 2016, **A3C** | `a3c-2016.pdf` | https://arxiv.org/abs/1602.01783 |
| 3 | L03 trpo | Schulman et al. 2015, **TRPO** | `trpo-2015.pdf` | https://arxiv.org/abs/1502.05477 |
| 4 | L04 ppo-core | Schulman et al. 2017, **PPO** | `ppo-2017.pdf` | https://arxiv.org/abs/1707.06347 |
| 5 | L05 gae | Schulman et al. 2016, **GAE** | `gae-2016.pdf` | https://arxiv.org/abs/1506.02438 |
| 6 | L06 ppo-tricks | Engstrom et al. 2020, **Implementation Matters** | `ppo-impl-2020.pdf` | https://arxiv.org/abs/2005.12729 |
| 7 | L07 cartpole-lab | Sutton & Barto, **RL: An Introduction** Ch 13 | `sutton-barto-ch13.pdf` | http://incompleteideas.net/book/RLbook2020.pdf |
| 8 | L08 ppo-for-llm | Ziegler et al. 2019, **Fine-Tuning Language Models from Human Preferences** | `ziegler-2019.pdf` | https://arxiv.org/abs/1909.08593 |
| 9 | L09 toy-rl-llm | Stiennon et al. 2020, **Learning to summarize from human feedback** | `stiennon-2020.pdf` | https://arxiv.org/abs/2009.01325 |
| 10 | L10 rl-pitfalls | Gao et al. 2022, **Scaling Laws for Reward Model Overoptimization** | `gao-2022-rh.pdf` | https://arxiv.org/abs/2210.10760 |
| 11 | L11 capstone | TRL 官方 IMDb PPO example | (no paper) | https://huggingface.co/docs/trl |
| 12 | L12 summary | Ouyang et al. 2022, **InstructGPT** (引出专题 2) | `instructgpt-2022.pdf` | https://arxiv.org/abs/2203.02155 |

## 推荐阅读顺序

```
入门：L01 REINFORCE → L02 A3C → L04 PPO（先建直觉）
进阶：L03 TRPO → L05 GAE → L06 PPO 实现细节
延展：L08-L10 LLM-RL 范式 → 专题 2 InstructGPT
```

## 阅读策略

- L04 PPO + L05 GAE 是**必啃**，PPO clip 的几何意义和 GAE 的 λ-return 推导直接决定能否理解后续所有 RL 算法。
- L06 (Engstrom 2020) 是"PPO 工程化清单"，明白 PPO 为何如此挑超参的关键。
- L08 (Ziegler 2019) 是 LLM-RL 鼻祖；L09 (Stiennon 2020) 是 InstructGPT 直接前身。
- L10 (Gao 2022) 是 reward hacking 的奠基论文，专题 2 RLHF 必再啃一次。

## 其他参考

- Lilian Weng 博客《Policy Gradient Algorithms》
- Joschu Schulman《Deep RL Bootcamp》PPT
- HuggingFace《Deep RL Course》 https://huggingface.co/learn/deep-rl-course
