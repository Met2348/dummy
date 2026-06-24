# 13.4 video-generation — 论文清单

> 视频生成的核心论文。读法接 M9「科研技能」: 先读 abstract/intro/figure, 抓「它在解决视频的哪个难题」。

## 必读 (核心)
- **Video Diffusion Models** (Ho et al., 2022) — 把扩散扩到视频的奠基, 3D U-Net 时空建模。
- **Sora technical report** (OpenAI, 2024) — 时空 patch (spacetime patches) + latent 视频扩散 + 统一表示。本专题 L2 的来源。
- **Latent Diffusion (Stable Diffusion)** (Rombach et al., 2022) — latent 扩散 (M13.3-L2), 视频版的基础。
- **Make-A-Video / Imagen Video** (Meta / Google, 2022) — 早期文生视频, 级联 + 时间超分。

## 进阶 (时序连贯 / 架构)
- **Stable Video Diffusion** (Blattmann et al., 2023) — 图扩散加时间层做视频, 工程细节扎实。
- **Open-Sora / Open-Sora-Plan** (社区, 2024) — 开源复现, 成本量级 (L4) 的公开参考。
- **VDM / Lumiere** (Google, 2024) — 时空 U-Net, 一次生成全部帧 (长程一致思路)。
- **W.A.L.T / Latte** — DiT 式视频扩散 (transformer 去噪 + 时空 patch)。

## 评估 / 难题 (gap, 接 L3)
- **VBench** (Huang et al., 2023) — 视频生成评估基准, 多维度 (含时序一致)。
- 长程一致性、物理合理性仍是 open 问题 — 接 M13.5 世界模型、M11 具身。

## 怎么读 (接 M9)
1. 先问「它在压哪部分成本 / 解哪个尺度的连贯」(L3/L4 的框架)。
2. 找它的「时空 token 怎么切」「在 latent 还是像素空间扩散」。
3. 对照本专题 toy: 真实模型 = 你的 toy 把「2D 点」换成「时空 latent patch」+ 规模。
