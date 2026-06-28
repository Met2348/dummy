# L2 · 端到端 mini-VLA 装配

> 22-min lecture · 目标: 把 OpenVLA / π 拆成 M11/M10/M13 部件, 看一个完整 VLA 怎么从你学的零件拼出来。

---

## 0. 装配 = capstone 核心技能

和 M13.7-L2 一样, 真本事是「看到一个 VLA 能拆回已知部件」。本讲拆两个真实 VLA, 标注每块来自你哪个专题。拆完你会确信: **你懂 VLA 的每一块。**

---

## 1. 装配 OpenVLA (离散动作)

```
   观测 (相机图 + 指令)
     → VLM backbone (视觉编码器 + LLM)         ← M10.1 + M10.2 + M10.3
     → 离散动作 token 头 (tokens-as-actions)    ← M11.1-L3 + M11.2-L3
     → 自回归吐动作 token → 解码成动作           ← 你的 LLM next-token
   训练: 大规模 BC (Open X-Embodiment)          ← M11.4
        + 与 VLM 数据 co-train (防遗忘)          ← M11.4-L3 + M10.3
```
> OpenVLA = VLM (M10) + 离散动作头 (M11.1/11.2) + 大规模 BC (M11.4)。每块你都学过。

---

## 2. 装配 π (flow-matching 动作)

```
   观测 (多相机 + 本体感觉)
     → VLM backbone                            ← M10
     → flow-matching 动作头 (多峰/平滑)         ← M11.3 + M13.2
     → action chunking (一次预测一段)           ← M11.3-L3
     → 高频执行 (~50Hz)                         ← M11.3-L4
   训练: BC (遥操作 demo)                        ← M11.4
```
> π = VLM (M10) + flow-matching 动作头 (M11.3/M13.2) + chunking + 高频。比 OpenVLA 强在动作头 (扩散/flow 解决多峰精细)。

---

## 3. 加上世界模型 (GR00T 式)

```
   π/OpenVLA 的反应式 VLA
     + 世界模型 (预测下一观测)                  ← M11.5 + M13.5
     → 能想象/规划 (不只反应)                    ← M11.5 MPC
     → sim 训练 + DR 弥合 gap                    ← M11.6
   = GR00T 式: VLA + 世界模型 + 仿真大规模训练
```
> 最强形态: VLA (反应) + 世界模型 (规划) + 仿真 (大规模训练) + DR (sim2real)。**M11 全部专题在这里汇成一个系统。**

---

## 4. 装配 VLA 的 7 问 (元技能, 接 M13.7)

拆任何 VLA 的通用流程:
1. **backbone 是什么 VLM, 多大?** (M10)
2. **动作头?** 离散 / 连续 / 扩散 / flow (M11.2/11.3)
3. **动作表示?** tokens-as-actions / 连续 (M11.1)
4. **怎么训?** BC / 仿真 RL / world-action (M11.4/11.5/11.6)
5. **数据?** 真机 / 仿真 / 人类视频, co-train 配比 (M11.4)
6. **要世界模型吗?** 反应 vs 规划 (M11.5)
7. **怎么 sim2real?** DR / 真机微调 (M11.6)

> 拿这 7 问去拆任何 2026 的新 VLA (你没见过的), 你都能十分钟还原成已知部件 + 找创新点。**会拆 = 会造 = 会推进 (找 gap, L3)。**

---

## 5. 本讲小结 + 通往 L3

- OpenVLA = VLM(M10) + 离散动作头(M11.1/2) + 大规模BC(M11.4)。
- π = VLM + flow-matching动作头(M11.3/M13.2) + chunking + 高频。
- GR00T 式 = VLA + 世界模型(M11.5) + 仿真+DR(M11.6); M11 全栈汇成一个系统。
- 装配 7 问 = 拆任何 VLA 的元技能; 每问对应你一个专题。

> **下一讲 L3「具身评测 + 找 gap」**: VLA 怎么评 (LIBERO/CALVIN 思路)? 评测的坑在哪? 在具身全栈上系统找研究 gap。

**动手**: 选一个真实 VLA (你感兴趣的), 用 7 问拆它一遍, 标出每块来自你哪个专题。确认你懂它的每一块。
