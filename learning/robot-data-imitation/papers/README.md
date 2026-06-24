# 11.4 robot-data-imitation — 论文清单

> 模仿学习 + 机器人数据论文。读法接 M11.4: 先问「数据从哪来 / 多大规模 / 怎么防分布漂移」。

## 必读 (核心)
- **ALVINN / behavior cloning 经典** — BC 奠基 (监督学专家)。
- **DAgger** (Ross et al., 2011) — 缓解分布漂移 (专家标注策略走到的状态)。
- **Open X-Embodiment / RT-X** (2023) — 跨机器人大数据 + co-train, 数据 scaling 的代表。
- **RT-2** (2023) — 与 VLM 数据 co-train 防遗忘 (L3)。

## 进阶 (数据来源)
- **Ego4D / Ego-Exo4D** — 大规模 egocentric 人类视频数据集 (L4 的桥)。
- **R3M / VIP / MVP** — 用人类视频预训练机器人视觉表示 (视频→表示→真机微调)。
- **DROID / BridgeData** — 大规模遥操作机器人数据集。

## 怎么读 (接 M11.4)
1. 数据从哪来 (遥操作 / 仿真 / 人类视频), 多大规模?
2. 怎么防分布漂移 (更多数据 / DAgger / 加噪)?
3. 怎么 co-train 防遗忘 (机器人数据 : VLM 数据 配比)?
4. scaling 曲线干净吗 (单变量? 接 9.4)?

> 对照本专题 toy: 真 BC = 你的 state→action 回归把状态换成图像+本体、专家换人类遥操作 + 规模。
