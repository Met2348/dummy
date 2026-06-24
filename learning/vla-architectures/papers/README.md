# 11.2 vla-architectures — 论文清单

> VLA 架构论文。读法接 M11.2-L2 的「5 决策」: backbone? 动作头? 世界模型? 冻结? 数据?

## 必读 (架构)
- **OpenVLA** (Kim et al., 2024) — 开源 VLA, 离散 token 动作头, 社区基线。
- **π0** (Physical Intelligence, 2024) — flow-matching 动作头, 高频灵巧操作。
- **RT-2** (Google, 2023) — VLM backbone + 动作 token (两阶段的源头)。
- **GR00T** (NVIDIA, 2024+) — 人形 VLA, 扩散动作头 + 世界模型。

## 进阶 (动作表示)
- **Octo** — 开源通用机器人策略, 模块化动作头。
- **RoboFlamingo / RoboVLM** — 不同 VLM backbone 接动作头的对比研究。
- 离散 vs 连续动作的系统对比 (接 L3)。

## 怎么读 (接 L2 的 5 决策)
1. backbone 选什么 VLM, 多大?
2. 动作头: 离散 token / 连续回归 / 扩散 / flow?
3. 要不要世界模型 (M11.5)?
4. 冻结策略 (M10.3)?
5. 数据配比 (真机 / 仿真 / 人类视频, M11.4/11.6)?

> 对照本专题 toy: 真 VLA = 你的「backbone + 动作头」把 backbone 换 M10 VLM、观测换图像 + 规模。
