# 11.6 sim2real-isaaclab — 仿真训练 + sim2real gap + DR

> **Module 11「具身智能 / VLA / 机器人」· 第 6 专题 (仿真到真实)**
> 真机训练贵 → 在**仿真**里训 (安全/便宜/并行)。但 sim≠real (sim2real gap)。**domain randomization** 用覆盖弥合 gap。本专题**吸收你的 IsaacLab 实战经验** (AntBot + WSL2 踩坑)。

---

## 这个专题要解决的真问题

- **为什么用仿真?** → 安全/便宜/**GPU 并行几千环境** (IsaacLab), 喂饱 RL 样本饥渴。
- **sim2real gap?** → sim 里好、real 掉点; 本质 = **分布偏移** (同 M11.4 分布漂移)。
- **怎么弥合?** → **domain randomization**: 随机化仿真参数 → 覆盖真实不确定性 → 泛化。
- **怎么在仿真训?** → sim 内 RL (你的 RL + IsaacLab 并行, AntBot 实例)。
- **配置怎么搞?** → 你的踩坑: WSL2 Vulkan 坑 → **Windows 原生** (`isaaclab_notes.md`)。

> **你已经实际跑过 AntBot + 趟过配置坑** —— 本专题把它系统化成「会跑 + 懂原理」。

## 学习路径 (4 讲)
| 讲 | 文件 | 一句话 |
|---|---|---|
| L1 | `lectures/L1-why-sim-and-the-gap.md` | 仿真便宜安全并行; sim2real gap=分布偏移 |
| L2 | `lectures/L2-isaaclab-workflow.md` | IsaacLab 工作流 (对接你 AntBot + WSL2 踩坑) |
| L3 | `lectures/L3-domain-randomization.md` | DR: 随机化覆盖真实 → 泛化 (覆盖=泛化) |
| L4 | `lectures/L4-rl-in-sim.md` | sim 内 RL (你的 RL + 并行仿真 + DR) |

## 动手 (2 个 notebook)
| notebook | 你会真的做什么 |
|---|---|
| `notebooks/N1-domain-randomization.ipynb` | GPU-free: 窄sim训 vs DR全区域训, 看 DR 弥合 real gap |
| `notebooks/N2-isaaclab-run-guide.ipynb` | IsaacLab 真跑指引 (需 NV GPU, 复用你 AntBot 经验) |

## 工具 (`src/`)
- `domain_rand.py` — 域随机化 GPU-free 演示 (目标配置覆盖)。**复用 M11.1 toy_env。**
- `isaaclab_notes.md` — 你的 IsaacLab 踩坑 checklist (WSL2 失败→Windows 原生, AntBot 命令)。

## 环境
```bash
pip install -r environment/requirements.txt
python environment/verify_env.py     # 全部通过 ✅ (N1 GPU-free; N2 指引不需 GPU)
```
Python 3.13 / torch (tiny CPU) / numpy / matplotlib。N1 玩具离线确定性; N2 真跑需 NVIDIA GPU + Isaac Sim。

## 完成本专题后你应该能
- [ ] 解释仿真训练的优势 (安全/便宜/并行) 和 sim2real gap (分布偏移)
- [ ] 描述 IsaacLab 工作流 (验证启动→训练→回放), 说清你的 WSL2 踩坑教训
- [ ] 解释 DR 为何弥合 gap (覆盖→泛化, 同 M11.4)
- [ ] 说清 sim 内 RL 怎么喂饱 RL 样本饥渴 (AntBot)
- [ ] 在 GPU-free 玩具上实测 DR 弥合 gap

---
## 在 Module 11 中的位置
```
  11.1 → 11.2 → 11.3 → 11.4 → 11.5 → 11.6 sim2real ◄你在这 → 11.7 capstone
```
> 设计文档: `docs/superpowers/specs/2026-06-24-module11-embodied-vla-design.md`
> `isaaclab_notes.md` 吸收了你仓库里的 WSL2/AntBot 配置文档经验。
