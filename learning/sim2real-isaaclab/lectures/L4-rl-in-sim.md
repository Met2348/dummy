# L4 · 在仿真里用 RL 训练机器人策略

> 20-min lecture · 目标: 把仿真 + DR 用起来 —— 在仿真里用 RL 训策略 (你 RL 模块 + IsaacLab 并行)。收口 11.6。

---

## 0. 仿真的终极用途: 大规模 RL

L1-L3 讲了仿真 (便宜安全并行) + DR (弥合 gap)。它们最终服务于: **在仿真里用 RL 大规模训练机器人策略**。这是你 RL 模块 (PPO 等) + IsaacLab 并行 + DR 的合体。

---

## 1. sim 内 RL 的流程

```
   ① IsaacLab 并行几千个环境 (GPU)
   ② 策略 (你的 RL: PPO/rsl_rl) 在这些环境里试错, 收奖励
   ③ DR (L3): 每个环境随机化参数, 让学到的策略鲁棒
   ④ 训练到收敛 → 在仿真里高成功率
   ⑤ 部署到真机 (DR 让 gap 小) 或真机微调
```

> 你跑的 AntBot (`Isaac-Ant-v0 --headless` + rsl_rl) 就是这个流程的实例: rsl_rl 是 RL 算法库, Isaac-Ant 是任务, 并行环境让 PPO 的样本饥渴被满足。**你已经实际跑过 sim 内 RL。**

---

## 2. 为什么 sim 让机器人 RL 可行

你 RL 模块知道 RL **样本饥渴** (要海量交互)。真机交互贵到 RL 几乎不可行。仿真破局:
- **并行**: 几千环境同时跑 → 样本量暴增几个数量级。
- **快于实时**: 仿真能比真实时间快 → 训练墙钟时间大降。
- **安全试错**: RL 要探索 (会做蠢事), 仿真里随便摔。

> 一句话: **仿真把 RL 的"样本饥渴"喂饱了**。没有 GPU 并行仿真, 机器人 RL 基本不可行; 有了它, AntBot 这种能在几分钟~几小时训出来。这是 IsaacLab 存在的根本理由 (L1 的发动机)。

---

## 3. sim 内 RL vs 你学过的 RL/具身路线

把几条路线连起来:
| 路线 | 在哪练 | 数据 | 你的模块 |
|---|---|---|---|
| BC (M11.4) | 离线 | 专家 demo | 模仿 |
| model-based (M11.5) | 想象里 | 世界模型 | 世界模型+规划 |
| **sim 内 RL (本讲)** | **仿真里** | **仿真交互 + DR** | **你的 RL + IsaacLab** |
| 真机 RL | 真机 | 真实交互 (贵) | 少用 |

> sim 内 RL 是「用仿真交互喂 RL」的路线: 不需专家 (vs BC)、不需学世界模型 (vs model-based), 直接在仿真里试错。代价是要建仿真 + 弥合 gap (DR)。**三条路各有适用**: 有专家用 BC, 动态好建用 model-based, 有仿真用 sim RL, 常组合。

---

## 4. 收口: sim2real 全景

```
   M11.6 sim2real-isaaclab:
   L1 为什么 sim (安全/便宜/并行) + sim2real gap (分布偏移)
   L2 IsaacLab 工作流 (你跑过 AntBot; WSL2 坑 → Windows 原生)
   L3 domain randomization (覆盖→泛化, 弥合 gap)
   L4 sim 内 RL (你的 RL + 并行仿真 + DR, 喂饱样本饥渴)
   ─────────────────────────────────────
   = 在仿真里安全便宜地大规模训练 + DR 弥合 gap → 部署真机
```

> 大图景: **仿真是机器人 RL 的发动机, DR 是它通向真实的桥**。你已经实际趟过 IsaacLab 全流程, 加上本模块的认知 (gap/DR/sim RL), 你对 sim2real 有了「会跑 + 懂原理」的完整掌握。

---

## 5. 本讲小结 (11.6 收口) + 通往 11.7

- **sim 内 RL**: IsaacLab 并行环境 + 你的 RL (PPO/rsl_rl) + DR → 喂饱 RL 样本饥渴 (AntBot 实例)。
- 仿真让机器人 RL **从不可行变可行** (并行/快/安全)。
- 三条训练路线 (BC / model-based / sim RL) 各有适用, 常组合。
- sim2real 全景: 仿真 (发动机) + DR (通向真实的桥) → 部署/微调真机。

> **M11.6 收口**: 仿真便宜安全并行 (IsaacLab); sim2real gap=分布偏移; DR 用覆盖弥合; sim 内 RL 喂饱样本。你已跑过 AntBot。
> **下一专题 M11.7「embodied-graduation」**: Module 11 capstone —— 端到端 mini-VLA 装配 + 评测 (LIBERO/CALVIN 思路) + 研究 gap。下一专题 `embodied-graduation`。

**动手**: 完成 N1 (DR 弥合 gap) + N2 (IsaacLab 真跑指引, 需 NV GPU) 后, 用你 AntBot 的经验, 写一句: sim 内 RL 训出的策略, 要部署到真机还差哪一步? (答案: DR 弥合 gap + 可能真机微调)。
