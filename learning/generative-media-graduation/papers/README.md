# 13.7 generative-media-graduation — 论文清单 (综述 + 路线)

> Capstone 不再读单点论文, 而是读**综述/路线图**, 建立全局视野。读法接 M9: 抓「领域地图 + open 问题」。

## 综述 / 全局视野
- **Diffusion Models: A Comprehensive Survey** — 扩散全景 (覆盖 M13.1-13.3 的脉络)。
- **Sora technical report** (OpenAI, 2024) — 视频生成 + 世界模拟器愿景 (M13.4-13.5)。
- **World Models 综述 / Dreamer 系列** — 世界模型与 model-based 决策 (M13.5)。
- **LLaDA + 离散扩散综述** — dLLM 的全景与现状 (M13.6)。

## 装配视角 (拆解真实系统)
- 回看各专题 papers/: SD/DiT/Sora/Genie/LLaDA 论文, 用 L2 的 7 问拆解法读。
- 重点不是细节, 而是「它由哪些部件组成 + 创新点在哪 + gap 在哪」。

## 研究 gap (接 L3 + M9)
- 本专题 `generative_capstone.py` 的 `GAPS` 整理了 7 个跨专题 gap。
- 每个 gap 找一两篇最相关论文, 用 M9.3 批判式读, 验证「这个 gap 是否真的还 open」。

## 怎么用这份清单 (capstone 读法)
1. 先读综述建全局地图 (你在地图的哪): M13 全链 + 三个前沿模块三角 (M11/M12)。
2. 挑一个 gap (建议 dLLM 相关), 深读它的 1-2 篇核心论文 (M9.3 批判)。
3. 把读到的 open 问题, 用 N2 的 idea 卡模板固化成你的研究种子。
