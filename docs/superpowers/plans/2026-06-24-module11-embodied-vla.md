# Module 11 具身/VLA Implementation Plan

**Goal**: 新建 Module 11「具身智能/VLA/机器人基础模型」(7 专题), 从 VLM 跨到机器人基础模型。玩具控制环境 + 合成 demo, CPU 可跑 (不强制 GPU/Isaac); IsaacLab 专题给真跑指引。

**Design**: `docs/superpowers/specs/2026-06-24-module11-embodied-vla-design.md`

**Tech Stack**: Python 3.13 / torch / numpy / matplotlib / nbformat. 确定性玩具环境 (无需 gym/Isaac 也能跑通核心)。11.6 给 IsaacLab 真跑指引 (可选, 需 NV GPU)。

**依赖**: 前置 **M10** (VLA = VLM backbone + 动作头); 动作头(11.3)/世界模型(11.5) 用 **M13** 的扩散/世界模型。**建议 M11 在 M10、M13 之后建。**

**构建顺序 (模块内)**: 11.1 foundations → 11.2 architectures → 11.3 action heads → 11.4 imitation → 11.5 world-action → 11.6 sim2real/IsaacLab → 11.7 capstone。

---

## Phases (每专题 6 步, 同 Module 9/10 套路)
- **P1 `embodied-foundations`**: toy_env.py / action_serialize.py; N1 玩具环境序列化 / N2 具身 next-token
- **P2 `vla-architectures`**: mini_vla.py (复用 M10 mini_vlm); N1 组装 mini-VLA / N2 离散vs连续动作
- **P3 `action-heads-diffusion-policy`**: diffusion_policy.py (同源 M13 diffusion); N1 diffusion 动作头 / N2 action chunking 消融
- **P4 `robot-data-imitation`**: bc_train.py; N1 behavior cloning / N2 数据 scaling 曲线 (复用 9.4)
- **P5 `world-action-models`**: world_model.py (同源 M13); N1 学世界模型+规划 / N2 model-based vs free
- **P6 `sim2real-isaaclab`**: domain_rand.py + isaaclab_notes.md (吸收用户 AntBot/配置文档经验); N1 域随机化概念演示(无需Isaac) / N2 IsaacLab 真跑指引
- **P7 `embodied-graduation`**: Capstone; N1 端到端 mini-VLA + 评测(多种子) / N2 VLA gap→idea 卡

## 成功标准
- [ ] 7 专题完整, verify_env 全过, 14 notebook 0 报错 (玩具环境 CPU 可跑)。
- [ ] 至少一个 notebook 端到端 mini-VLA (指令+观测→动作)。
- [ ] 11.6 给可操作 IsaacLab 真跑指引 (复用用户经验)。
- [ ] Capstone 产出 reasoning-VLA 方向 idea 卡。
