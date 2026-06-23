# Module 11「具身智能 / VLA / 机器人基础模型」设计 spec

> Date: 2026-06-24 · 用户 (博0, EE 本硕, NLP/LLM; **已有 IsaacLab 文件 = 在碰机器人**)
> 终态: 新增 Module 11 —— 从 VLM (M10) 跨到**具身智能**: Vision-Language-Action (VLA) 机器人基础模型。这是 2026 最爆发、且对 NLP 出身最自然的转型方向。

## 1. 背景与动机

- **2026 行情**: VLA 是「2026 具身 AI 的基石」, ICLR 2026 一口气 164 篇 VLA 投稿; frontier 实验室 (Physical Intelligence π 系列 / NVIDIA GR00T / DeepMind Gemini Robotics) 在抢人 (来源见 portfolio_v4 招聘核对)。
- **对 NLP 出身最自然**: VLA 把 robot 的观测和动作当 **token**, 用 next-token / VLM 那套机器训练。用户的 transformer + RL (reasoning-r1/rl-sota) + VLM (M10) 全套**直接迁移**。
- **用户已有信号**: 仓库里有 `WSL2 + NV591 + UB2204 配置isaacLab` / `初次在Windows原生环境运行isaacLab训练例程AntBot` 文档 —— 他**已经在碰 IsaacLab**, 这个模块把零散尝试系统化。
- **依赖**: 前置 M10 (VLA = VLM backbone + action decoder); 动作头部分用到 M13 (diffusion/flow-matching policy)。

### 关键洞察
1. VLA 的核心是「**动作如何 token 化 / 解码**」—— 离散动作 token vs 连续 (diffusion/flow) 动作头。这是本模块灵魂。
2. **可跑纪律**: 真机器人跑不了, 但用极简 2D/格子/玩具控制环境 + 合成 demo, 在 CPU 上把「VLA 训练→推理→评测」循环跑通 (复用 Module 9 确定性纪律)。IsaacLab 专题给真实配置指引 + 可选真跑。
3. 评测/可复现直接接 9.4/9.5 (RL/机器人方差极大, 9.4-L5 的多种子在这里是刚需)。

## 2. 专题蓝图 (7 专题)

| # | slug | 覆盖技能 | 核心产出 |
|---|---|---|---|
| 11.1 | `embodied-foundations` | 从 LLM/VLM 到具身; robot learning 问题; RT-1/RT-2 谱系; tokens-as-actions 范式 | 具身 AI 地图 + 为什么是基础模型 |
| 11.2 | `vla-architectures` | VLA 设计: VLM backbone + action decoder; OpenVLA/π/GR00T; 离散 vs 连续动作 | 一个 mini-VLA 架构 (接 M10) |
| 11.3 | `action-heads-diffusion-policy` | 动作头: diffusion/flow-matching policy, action chunking, 控制频率(30-100Hz), proprioception | 手搭 diffusion 动作头 (接 M13) |
| 11.4 | `robot-data-imitation` | 模仿学习, 遥操作数据, 数据 scaling 教训, 与 VLM 数据 co-train, egocentric video | 在玩具环境上 behavior cloning |
| 11.5 | `world-action-models` | 具身世界模型 (pretrained to imagine, fine-tuned to act), 基于模型的机器人 RL | 世界模型 + 规划最小实现 |
| 11.6 | `sim2real-isaaclab` | IsaacLab/Isaac Sim (用户已碰!), domain randomization, sim2real gap, sim 内 RL | IsaacLab 配置指引 + 可选真跑 |
| 11.7 | `embodied-graduation` | Capstone: mini-VLA 端到端 + 评测 (LIBERO/CALVIN 思路) + 研究 gap | 端到端 mini-VLA + idea 卡 |

## 3. 逐专题详细设计

### 11.1 embodied-foundations
- **lectures (4)**: L1 什么是具身 AI + 为什么基础模型 (vs 传统逐任务硬编码) · L2 RT-1→RT-2→OpenVLA 谱系 (LLM 化的机器人控制) · L3 tokens-as-actions 范式 (观测/动作都当 token, 复用 next-token) · L4 数据 scaling 的「每条 demo 改进所有机器人」逻辑
- **notebooks (2)**: N1 一个最小玩具控制环境 (格子/2D reacher), 把状态-动作序列化成 token · N2 用一个 tiny transformer 在 token 序列上做下一动作预测 (具身版 next-token)
- **src**: `toy_env.py` (确定性玩具控制环境) · `action_serialize.py` (观测/动作 ↔ token)

### 11.2 vla-architectures
- **lectures (4)**: L1 VLA 两阶段结构 (VLM 感知推理核 + 动作解码器) · L2 OpenVLA / π 系列 / GR00T 架构对比 · L3 离散动作 token vs 连续动作 (各自代价) · L4 把 M10 的 VLM backbone 接动作头
- **notebooks (2)**: N1 用 M10 的 mini-VLM + 一个离散动作头, 组装 mini-VLA · N2 离散 vs 连续动作头在玩具任务上的对比
- **src**: `mini_vla.py` (VLM backbone + 可换动作头), 复用 M10 `mini_vlm.py`

### 11.3 action-heads-diffusion-policy
- **lectures (4)**: L1 为什么动作要 diffusion/flow (多峰分布/平滑) · L2 diffusion policy 原理 (接 M13 扩散) · L3 flow-matching 动作头 + action chunking · L4 控制频率/proprioception/实时约束
- **notebooks (2)**: N1 手搭一个 diffusion 动作头, 在玩具任务上生成平滑动作轨迹 · N2 action chunking 消融 (chunk 大小 vs 平滑度/反应性, 接 9.4)
- **src**: `diffusion_policy.py` (最小 diffusion/flow 动作头), 与 M13 `diffusion.py` 同源

### 11.4 robot-data-imitation
- **lectures (4)**: L1 模仿学习/behavior cloning 基础 · L2 遥操作数据采集 + 质量 · L3 数据 scaling 教训 (LLM 那套搬到机器人) + 与 VLM 数据 co-train · L4 egocentric video 当桥 (人类视频→物理智能)
- **notebooks (2)**: N1 在玩具环境上 behavior cloning (从合成专家 demo 学策略) · N2 数据量 vs 成功率曲线 (scaling, 接 9.4 实验设计)
- **src**: `bc_train.py` (behavior cloning + 合成专家)

### 11.5 world-action-models
- **lectures (4)**: L1 世界模型是什么 (预测下一观测) · L2 「先学想象, 再学行动」(world-action models, NVIDIA 路线) · L3 基于模型的机器人 RL (在想象里 rollout) · L4 视频模型当模拟器 (接 M13 世界模型)
- **notebooks (2)**: N1 在玩具环境学一个状态转移世界模型, 用它做短程规划 · N2 model-based vs model-free 在玩具任务的样本效率对比
- **src**: `world_model.py` (转移模型 + 想象 rollout)

### 11.6 sim2real-isaaclab
- **lectures (4)**: L1 为什么 sim (安全/便宜/可并行) + sim2real gap · L2 IsaacLab/Isaac Sim 架构与工作流 (对接用户已有配置文档) · L3 domain randomization 弥合 gap · L4 sim 内 RL 训练机器人策略
- **notebooks (2)**: N1 一个**不依赖 GPU/Isaac** 的 sim2real 概念演示 (玩具环境加 domain randomization, 看泛化) · N2 IsaacLab 真跑指引 (markdown + 可选脚本, 复用用户 AntBot 经验, 标注「需 NV GPU」)
- **src**: `domain_rand.py` (域随机化包装器) · `isaaclab_notes.md` (踩坑指引, 复用用户文档)
- **特别**: 直接吸收用户仓库里的 `配置isaacLab失败` / `AntBot 例程` 文档经验, 把踩过的坑变成教材。

### 11.7 embodied-graduation (Capstone)
- **lectures (2)**: L1 装配完整 mini-VLA 流水线 (M10 VLM + M11 动作头 + 世界模型) · L2 VLA 研究前沿 (ICLR2026 164 篇趋势: 离散 diffusion VLA / reasoning VLA) + 用 9.3 gap 雷达扫题 (reasoning VLA 对 NLP 人最友好)
- **notebooks (2)**: N1 端到端 mini-VLA: 指令 + 观测 → 动作, 在玩具任务跑通 + 评测 (LIBERO/CALVIN 思路的玩具版, 多种子报方差) · N2 用 9.3/9.4 起一张 VLA 研究 idea 卡
- 接回: 用户从「碰 IsaacLab 跑别人例程」升级到「懂 VLA 怎么造 + 能找 VLA 研究 gap」。

## 4. 与现有资产整合
- **前置 M10**: VLA = M10 VLM backbone + 动作头。
- **接 M13**: diffusion 动作头 (11.3) 和世界模型 (11.5) 用 M13 的扩散/世界模型理论。
- **复用 M4**: reasoning-r1/rl-sota/rl-foundations 的 RL 知识直接用于 sim 内 RL 和 model-based RL。
- **复用 Module 9**: 9.4 多种子 (机器人方差大!) / 9.5 可复现 / 9.6 出图 / 9.3 找 gap 全程用。
- **吸收用户 IsaacLab 文档**: 把他踩过的配置坑变成 11.6 教材。

## 5. 成功标准
- [ ] 7 专题完整落地, 结构同 Module 9。
- [ ] notebook 全 nbconvert 跑通 0 报错, 玩具环境 CPU 可跑 (不强制 GPU/Isaac)。
- [ ] 课件研究生级, 公式逐项 (diffusion policy / behavior cloning / world model)。
- [ ] 至少一个 notebook 端到端跑通 mini-VLA (指令+观测→动作)。
- [ ] 11.6 给出可操作的 IsaacLab 真跑指引 (复用用户经验)。
- [ ] Capstone 产出 VLA 研究 idea 卡 (reasoning VLA 方向)。
