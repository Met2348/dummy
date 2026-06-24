# 11.1 embodied-foundations — 论文清单

> 具身基础模型谱系。读法接 M9: 先问「它的感知核是什么 / 动作怎么表示 / 数据从哪来」。

## 必读 (谱系)
- **RT-1** (Google, 2022) — transformer + 动作离散 token, 多任务机器人控制奠基。
- **RT-2** (Google, 2023) — 直接用 VLM + 动作当 token, 涌现泛化 (tokens-as-actions 飞跃)。
- **OpenVLA** (2024) — 开源 VLA, 社区基线 (像 Llama 之于 LLM)。
- **Open X-Embodiment** (2023) — 大规模跨机器人数据集, 「每条 demo 改进所有机器人」的数据基础 (L4)。

## 进阶 (前沿)
- **π0 / π series** (Physical Intelligence, 2024) — flow-matching 动作头 (接 M13.2/M11.3)。
- **GR00T** (NVIDIA, 2024+) — 人形机器人基础模型 + 世界模型 (接 M11.5)。
- **PaLM-E** (Google, 2023) — 具身多模态语言模型, RT-2 的 backbone 思路。

## 怎么读 (接 M9)
1. 先问「感知核 (VLM?) / 动作表示 (离散 token? 连续?) / 数据规模与来源」。
2. 找它的「泛化证据」(零样本新物体/新指令)。
3. 对照本专题 toy: 真 VLA = 你的「状态→动作 token 预测」把状态换成图像 + VLM + 规模。
