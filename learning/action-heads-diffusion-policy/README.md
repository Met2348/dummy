# 11.3 action-heads-diffusion-policy — 扩散动作头解决多峰

> **Module 11「具身智能 / VLA / 机器人」· 第 3 专题 (动作头巅峰)**
> M11.2-L3 的矛盾: 离散精度粗、连续怕多峰。**扩散动作头** (你的 M13!) 两全: 既连续平滑、又能表达多峰。这是 π 系列的核心, 把 M13 直接接到机器人。

---

## 这个专题要解决的真问题

- **动作为什么要扩散?** → 机器人动作**天生多峰** (绕障可上可下); 回归取均值会"既不上也不下"(直冲障碍)。
- **扩散动作头怎么工作?** → 条件 DDPM (= M13.1 + state 条件): 建模分布 + 采样产出动作。
- **怎么够快 (高频)?** → **flow-matching** 动作头 (M13.2 少步采样), π 的选择。
- **怎么够稳?** → **action chunking** (一次预测 H 步), 但权衡反应性 (闭环纠错)。
- **硬约束?** → 30-100Hz 控制频率 + 本体感觉, 决定动作头设计。

> **核心信念**: 扩散动作头 = M13.1 DDPM + state 条件 (同源)。π 动作头 = M10 VLM + M13 flow/扩散 + chunking。

## 学习路径 (4 讲)
| 讲 | 文件 | 一句话 |
|---|---|---|
| L1 | `lectures/L1-why-diffusion-for-actions.md` | 动作多峰 → 回归取均值死穴 → 扩散两全 |
| L2 | `lectures/L2-diffusion-policy-mechanism.md` | diffusion policy = 条件版 M13.1 (一一对应) |
| L3 | `lectures/L3-flow-matching-action-chunking.md` | flow-matching 解决快 + chunking 解决稳 |
| L4 | `lectures/L4-control-freq-proprioception.md` | 控制频率/本体感觉/实时约束决定设计 |

## 动手 (2 个 notebook)
| notebook | 你会真的做什么 |
|---|---|
| `notebooks/N1-diffusion-action-head.ipynb` | 障碍前动作分布: 扩散双峰(上/下) vs 回归塌均值(直冲障碍) |
| `notebooks/N2-action-chunking.ipynb` | chunk 大小消融: 平滑/连贯 vs 反应性(闭环纠错) 权衡 |

## 工具 (`src/`)
- `diffusion_policy.py` — 条件扩散动作头 (= M13.1 DDPM + state 条件) + 回归对照 + action chunking + 绕障环境。**与 M13 diffusion.py 同源。**

## 环境
```bash
pip install -r environment/requirements.txt
python environment/verify_env.py     # 全部通过 ✅
```
Python 3.13 / torch (tiny CPU) / numpy / matplotlib。玩具绕障导航离线确定性。

## 完成本专题后你应该能
- [ ] 解释机器人动作的多峰本质, 及回归为何失败 (取均值)
- [ ] 说清扩散动作头 = 条件版 M13.1 (机制对应)
- [ ] 解释 flow-matching 动作头为何解决高频控制 (接 M13.2)
- [ ] 解释 action chunking 的平滑 vs 反应性权衡
- [ ] 把控制频率/本体感觉/实时约束连到动作头设计

---
## 在 Module 11 中的位置
```
  11.1 → 11.2 → 11.3 扩散动作头 ◄你在这 → 11.4 数据/模仿 → 11.5 世界动作模型 → 11.6 sim2real → 11.7 capstone
```
> 设计文档: `docs/superpowers/specs/2026-06-24-module11-embodied-vla-design.md`
