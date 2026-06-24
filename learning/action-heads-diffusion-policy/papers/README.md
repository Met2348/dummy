# 11.3 action-heads-diffusion-policy — 论文清单

> 扩散动作头论文。读法接 M11.3: 先问「动作头是扩散还是 flow / 怎么处理高频 / 用不用 chunking」。

## 必读 (核心)
- **Diffusion Policy** (Chi et al., 2023) — 扩散当动作头的奠基, 解决多峰 + action chunking。
- **π0** (Physical Intelligence, 2024) — flow-matching 动作头, 50Hz 灵巧操作。
- **ACT (Action Chunking with Transformers)** (Zhao et al., 2023) — action chunking 的代表, 时序集成。

## 进阶 (机制/对比)
- **3D Diffusion Policy / DP3** — 点云观测 + 扩散动作头。
- **Flow Matching for robotics** — flow-matching 少步动作生成 (接 M13.2)。
- 扩散 policy vs 离散 token vs 回归的系统对比 (接 M11.2-L3)。

## 怎么读 (接 M11.3)
1. 动作头: 扩散 / flow / 离散? 为什么 (多峰? 高频?)
2. 怎么满足控制频率 (少步采样? chunking 摊薄?)
3. chunk 大小怎么定 (平滑 vs 反应性权衡, L3)?
4. 本体感觉怎么进 (L4)?

> 对照本专题 toy: 真扩散动作头 = 你的条件 DDPM 动作头把状态换成图像+本体 + flow少步 + 规模。
