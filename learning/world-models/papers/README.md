# 13.5 world-models — 论文清单

> 世界模型核心论文。读法接 M9: 先问「它的状态/动作是什么, 转移怎么学, 误差累积怎么对抗」。

## 必读 (核心)
- **World Models** (Ha & Schmidhuber, 2018) — 奠基: VAE 压观测 + RNN 学动态 + 在「梦」里训练策略。
- **Dreamer / DreamerV3** (Hafner et al., 2020-2023) — 在世界模型的想象里做 RL, 样本效率标杆。
- **Genie** (DeepMind, 2024) — 从无标注视频自学潜在动作, 生成可玩 2D 世界 (L3)。
- **Video generation models as world simulators** (OpenAI Sora report, 2024) — L2 的来源: 视频模型是隐式世界模拟器。

## 进阶 (交互/具身)
- **Vid2World** — 把预训练视频扩散改造成动作条件世界模型 (复用 M13.4 底座)。
- **AVID / 动作条件扩散** — 扩散续帧加动作通道做交互预测。
- **GameNGen** (Google, 2024) — 神经网络实时模拟 DOOM (扩散当游戏引擎)。
- **DayDreamer** — 世界模型直接在真机器人上学 (接 M11)。

## 评测 / 难题 (gap, 接 L4)
- 多步误差累积、长程一致、物理正确性、可评测性 (「懂物理 vs 拟合」) 均为 open。
- 接 M13.4-L3 (长程视频一致) 与 M11 (具身 sim2real)。

## 怎么读 (接 M9)
1. 先问「状态是什么 (低维向量 / 图像 / latent)」「动作从哪来 (标注 / 自学)」。
2. 找它怎么对抗误差累积 (短视野 / 重规划 / 真观测纠偏)。
3. 对照本专题 toy: 真实世界模型 = 你的 (state,action)→Δstate 把状态换成图像 + 规模。
