# 11.6 sim2real-isaaclab — 论文/资源清单

> sim2real + 仿真论文。读法接 M11.6: 先问「随机化什么 / 范围怎么定 / 怎么校准到真实」。

## 必读 (核心)
- **Domain Randomization** (Tobin et al., 2017) — DR 弥合 sim2real gap 的奠基。
- **Sim-to-Real via Sim-to-Sim / Dynamics Randomization** (OpenAI, 2018) — 随机化动力学, 真机灵巧手。
- **Learning Dexterity / Rubik's Cube** (OpenAI, 2019) — 大规模 DR + 自适应 (ADR)。
- **IsaacGym / IsaacLab** (NVIDIA) — GPU 并行仿真, sim 内 RL 的发动机。

## 进阶 (方法)
- **Automatic Domain Randomization (ADR)** — 动态调随机化范围 (从易到难)。
- **System Identification + DR** — 用真机数据校准随机化范围。
- **RMA (Rapid Motor Adaptation)** — 在线适应真实动力学 (DR + 自适应)。

## 资源 (你已用过)
- **IsaacLab 官方文档** — 环境/任务/RL 接口 (你跑过 AntBot `Isaac-Ant-v0`)。
- 本仓 `src/isaaclab_notes.md` — 你的 WSL2/Windows 配置踩坑 checklist。

## 怎么读 (接 M11.6)
1. 随机化什么参数 (物理/视觉/传感器), 范围多大?
2. 范围怎么定 (手调 / 系统辨识 / 自适应)?
3. 怎么验证弥合了 gap (真机成功率)?
4. 是否配合真机微调?

> 对照本专题 toy: 真 DR = 你的「目标配置随机化」换成摩擦/光照/噪声随机化 + IsaacLab 并行 + 规模。
