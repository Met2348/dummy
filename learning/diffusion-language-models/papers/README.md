# 13.6 diffusion-language-models — 论文清单

> dLLM 核心论文。读法接 M9.3: 用「真/有条件/打折」三分法审视每条声称。

## 必读 (核心)
- **LLaDA: Large Language Diffusion Models** (2025) — 证明扩散 LM 能 scale 到 ~8B、接近同规模 AR。dLLM 的里程碑。
- **D3PM: Structured Denoising Diffusion in Discrete State-Spaces** (Austin et al., 2021) — 离散扩散奠基 (吸收态/masked 是其特例)。
- **MaskGIT** (Chang et al., 2022) — 图像的 masked 并行迭代解码 (置信度调度), dLLM 解码的思想来源。
- **DiffuSeq / SEDD** — 文本扩散的早期 / score-based 路线。

## 进阶 (机制/对比)
- **Fill-in-the-Middle (FIM)** (OpenAI, 2022) — AR 怎么打补丁做 infilling (对照 dLLM 的原生双向)。
- **Score Entropy Discrete Diffusion (SEDD)** (Lou et al., 2024) — 离散扩散的 score 视角, 质量提升。
- 半自回归 / 块状扩散 LM — AR 与 dLLM 的混合范式探索。

## 现状 / 批判 (gap, 接 L4)
- 少步高质量解码、dLLM 版增量计算 (KV cache 对应物)、似然评估、对齐迁移均为 open。
- 接 M13.2 (少步采样思想迁文本) 与你的 RL/DPO 模块 (对齐怎么迁)。

## 怎么读 (接 M9.3)
1. 对每条声称用「真 / 有条件 / 打折」三分 (L4 的表)。
2. 找它和 AR 的公平对比 (同规模? 同数据? 同评估?)。
3. 对照本专题 toy: 真 dLLM = 你的 masked 双向 transformer + 置信度并行解码 + 规模。
