# L2 · IsaacLab / Isaac Sim 工作流 (对接你的踩坑经验)

> 22-min lecture · 目标: 理解 IsaacLab 仿真器的架构与工作流, 对接你已有的 IsaacLab 配置经验 (AntBot + WSL2 踩坑)。

---

## 0. 你已经碰过它了

这一讲不从零讲 —— **你已经实际配过 IsaacLab、跑过 AntBot 例程、踩过 WSL2 的坑**。本讲把你的实战经验系统化成「仿真工作流」的认知, 并把踩坑变成教材 (详见 `src/isaaclab_notes.md`)。

---

## 1. IsaacLab / Isaac Sim 是什么

```
   Isaac Sim:  NVIDIA 的 GPU 物理仿真器 (基于 Omniverse/PhysX/USD), 渲染 + 物理
   IsaacLab:   建在 Isaac Sim 上的机器人学习框架 (环境/任务/RL 接口)
   关系:       IsaacLab = 机器人 RL 的"健身房", Isaac Sim = 底层物理引擎
```

- **GPU 并行**: 核心卖点 —— 一块 GPU 并行几千个环境 (L1 的发动机)。
- **任务库**: 内置任务 (如 `Isaac-Ant-v0` = 你跑的 AntBot), 接 RL 库 (rsl_rl/rl_games)。
- **USD 场景**: 用 Omniverse 的 USD 格式描述机器人/场景。

> 你跑的 `isaaclab.bat -p scripts/tutorials/00_sim/create_empty.py` 就是最小 Isaac Sim 启动; AntBot 训练用的是 `Isaac-Ant-v0` 任务 + rsl_rl。**你已经走通了从配置到训练的全流程。**

---

## 2. 典型工作流 (你走过的)

```
   ① 装环境:   conda env (env_isaaclab) + Isaac Sim + IsaacLab
   ② 验证:     isaaclab.bat -p scripts/tutorials/00_sim/create_empty.py  (启动空场景)
                → [INFO][AppLauncher]: Using device: cuda:0  (确认用上 GPU)
   ③ 训练:     isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py
                       --task=Isaac-Ant-v0 --headless        (无头并行训 RL)
   ④ 回放/评估: play 脚本看训出来的策略
```

> `--headless` = 无渲染窗口 (纯算, 快, 适合训练); 去掉则开窗口可视化 (慢, 适合 debug)。`cuda:0` 确认 GPU 在用。这套流程你已经在 Windows 原生环境跑通了 (AntBot)。

---

## 3. 你踩的坑 = 最好的教材 (WSL2 vs Windows 原生)

你的真实经验 (详见 `isaaclab_notes.md`):

> **WSL2 + NV591 + UB2204 配置 IsaacLab 5.1.0 → 失败**:
> - 现象: 不支持的 Vulkan 库 → 无法渲染窗口, 连 `--headless` 也死锁; PhysX 降级、Vulkan API 严重错误, 日志停止增长、无训练进程。
> - 根因: WSL2 的图形/Vulkan 栈对 Isaac Sim 支持不全。
> - **决策: 改用 Windows 原生环境 → 成功跑通 AntBot。**

> 这是极有价值的一手教训: **IsaacLab 对图形栈 (Vulkan/驱动) 很挑剔**。WSL2 看似方便但图形支持是雷区; Windows 原生 (或原生 Linux + 正确驱动) 更稳。**环境配置是 sim 的第一道坎, 不是算法问题, 是系统/驱动问题。**

---

## 4. 仿真工作流的通用认知 (带走)

从你的经验提炼通用教训:
- **驱动/图形栈是命门**: Isaac Sim 依赖特定 NVIDIA 驱动 + Vulkan; 版本不匹配直接挂 (你的 NV591+WSL2)。
- **headless ≠ 免图形**: 即使无头, 底层渲染栈仍要可用 (你发现 headless 也死锁)。
- **先验证最小启动**: 跑通 `create_empty.py` 确认 `cuda:0` 再上训练 (你的 ②), 别一上来就训。
- **平台选择**: Windows 原生 / 原生 Ubuntu + 官方推荐驱动 > WSL2 (图形坑多)。

> 这些是「让仿真跑起来」的工程智慧, 和算法无关但极重要 —— 很多人卡在这一步。你已经趟过, 这是你的优势。`isaaclab_notes.md` 把它整理成可复用的 checklist。

---

## 5. 本讲小结 + 通往 L3

- **Isaac Sim** = GPU 物理仿真器; **IsaacLab** = 机器人 RL 框架 (并行几千环境)。
- 工作流: 装环境 → 验证最小启动 (cuda:0) → `--task=Isaac-Ant-v0 --headless` 训 RL → 回放。
- 你的踩坑教训: **WSL2+Vulkan 支持差 → Windows 原生**; 驱动/图形栈是命门, headless 也需渲染栈。
- 通用认知: 先验证最小启动、平台选对、驱动匹配 (整理在 `isaaclab_notes.md`)。

> **下一讲 L3「domain randomization」**: sim2real gap 的主力解法 —— 随机化仿真参数让策略鲁棒。你 N1 会在 GPU-free 玩具上实测它弥合 gap。

**动手**: 翻 `src/isaaclab_notes.md` (你的踩坑整理成 checklist); N2 是 IsaacLab 真跑指引 (需 NV GPU, 复用你 AntBot 经验)。
