# 11.5 world-action-models — 论文清单

> 具身世界模型论文。读法接 M11.5: 先问「世界模型怎么学 / 怎么用 (规划 or 想象RL) / 误差累积怎么治」。

## 必读 (核心)
- **World Models** (Ha & Schmidhuber, 2018) — 在"梦"里训策略的奠基。
- **Dreamer / DreamerV3** (Hafner et al., 2020-2023) — model-based RL 标杆 (想象里学策略, L3)。
- **DayDreamer** (2022) — 世界模型直接在真机器人上学。
- **GR00T / NVIDIA world-action** (2024+) — 「先学想象, 再学行动」(L2) + VLA 融合。

## 进阶 (图像世界模型, 接 M13)
- **GameNGen / Genie** (2024) — 视频生成当可玩世界模拟器 (L4, 接 M13.5)。
- **MBPO / PETS** — model-based RL 的经典 (短视野 rollout 对抗 model bias)。
- **TD-MPC2** — model-based + MPC 的现代实现。

## 怎么读 (接 M11.5)
1. 世界模型学什么 (低维状态 / 图像 / latent)? 数据来源 (随机 / 视频 / 真机)?
2. 怎么用 (MPC 规划 / 想象里 RL / 少量 BC 微调)?
3. 误差累积怎么治 (短视野 / 重规划 / 集成)?
4. 和 model-free 比样本效率如何?

> 对照本专题 toy: 真世界模型 = 你的 (state,action)→Δ 把状态换图像 (动作条件视频, M13.4) + 规模。
