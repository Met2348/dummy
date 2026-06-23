# Module 13「扩散 / 生成式媒体 / 世界模型」设计 spec

> Date: 2026-06-24 · 用户 (博0, EE 本硕, NLP/LLM)
> 终态: 新增 Module 13 —— 扩散模型 / 生成式媒体 (图/视频) / 世界模型 / **扩散语言模型 (dLLM)**。补齐用户体系的**范式级空白**: 整个 portfolio 都是自回归文本, 完全没有扩散这条生成范式。

## 1. 背景与动机

- **2026 行情**: 视频扩散→交互式世界模型是 ICLR 2026 主线之一; flow matching 是当红采样范式; **diffusion language models (dLLM)** 是直连 NLP 的新前沿 (AR→diffusion 后训练机制研究)。Open-Sora 2.0 ~$200k 训出商用级视频模型 (来源见 portfolio_v4 招聘核对)。
- **对用户的价值**: ① 扩散是和自回归并列的**另一根生成支柱**, 不懂它 = 生成式 AI 的认知有大洞; ② **dLLM 直接桥回 NLP** (用户本行); ③ 世界模型是 M11 (具身) 的理论底座; ④ 视频/图像生成是高薪生成式媒体方向。
- **依赖**: 独立可建; 是 M11 (diffusion policy / world model) 的理论前置, 与 M11 互为支撑。

### 关键洞察
1. 扩散的核心是「**逐步去噪 = 学习数据分布的梯度 (score)**」。讲清前向加噪/反向去噪/score matching 的数学是本模块灵魂 (用户 EE 数学功底是优势)。
2. **可跑纪律**: 用 2D 玩具分布 (双月/螺旋) + tiny 图 (MNIST 级/合成), 在 CPU 上把「DDPM→flow matching→DiT→条件生成」全跑通, 看见去噪轨迹。复用 Module 9 确定性纪律。
3. dLLM 专题直接接用户的 pretraining/采样知识, 让他在熟悉的 NLP 地盘理解扩散。

## 2. 专题蓝图 (7 专题)

| # | slug | 覆盖技能 | 核心产出 |
|---|---|---|---|
| 13.1 | `diffusion-foundations` | DDPM, 前向/反向过程, score matching, 为什么去噪能生成 | 2D 玩具分布上手搭 DDPM |
| 13.2 | `flow-matching-sota` | flow matching, rectified flow, consistency models, 快速采样 | flow matching vs DDPM 对比 |
| 13.3 | `dit-latent-diffusion` | Diffusion Transformer (DiT), latent diffusion, classifier-free guidance, 条件生成 | tiny DiT 条件生成 |
| 13.4 | `video-generation` | 视频扩散, 时空建模, Sora 式, Open-Sora 成本教训 | 时空扩散最小实现 |
| 13.5 | `world-models` | 世界模型, 视频模型当模拟器, 交互式/可玩 (Vid2World/AVID) | 玩具世界模型 (接 M11) |
| 13.6 | `diffusion-language-models` | dLLM, masked diffusion LM, AR→diffusion 范式迁移, 为什么对 NLP 重要 | mini-dLLM (接回本行) |
| 13.7 | `generative-media-graduation` | Capstone: 装配 + 研究 gap | 端到端生成 + idea 卡 |

## 3. 逐专题详细设计

### 13.1 diffusion-foundations
- **lectures (4)**: L1 生成的两条路: 自回归 vs 扩散 (用户已会前者, 对比建直觉) · L2 前向加噪过程 (马尔可夫链, 逐项交代 βt/αt) · L3 反向去噪 + score matching (去噪 = 估计 ∇log p(x), 每项交代) · L4 DDPM 采样 + 训练目标 (噪声预测)
- **notebooks (2)**: N1 在 2D 玩具分布 (双月/螺旋) 手搭 DDPM, 可视化前向加噪 + 反向去噪轨迹 · N2 在 tiny 图 (合成/MNIST 级) 训 DDPM 生成
- **src**: `diffusion.py` (DDPM 前向/反向/训练, 与 M11 diffusion_policy 同源)

### 13.2 flow-matching-sota
- **lectures (4)**: L1 从 DDPM 到 flow matching (连续视角, ODE vs SDE) · L2 rectified flow (拉直路径 = 更快采样, 逐项交代) · L3 consistency models (一步生成) · L4 采样器全景与权衡 (步数 vs 质量)
- **notebooks (2)**: N1 在同一 2D 分布上 flow matching vs DDPM, 比采样步数/质量 · N2 rectified flow 拉直前后采样轨迹对比
- **src**: `flow_matching.py` (flow matching + rectified flow)

### 13.3 dit-latent-diffusion
- **lectures (4)**: L1 为什么 transformer 做扩散 (DiT, 替代 U-Net) · L2 latent diffusion (在 VAE 隐空间扩散, 省算力; 接 M10 VQ/VAE) · L3 classifier-free guidance (条件强度旋钮, 逐项交代) · L4 条件生成 (文本→图的接口)
- **notebooks (2)**: N1 tiny DiT 在玩具数据上条件生成 (按类别) · N2 classifier-free guidance 强度消融 (接 9.4: guidance scale 的 trade-off)
- **src**: `dit.py` (Diffusion Transformer + CFG), 复用 M10 latent

### 13.4 video-generation
- **lectures (4)**: L1 视频 = 空间 + 时间扩散 (时空 attention) · L2 Sora 式时空 patch / latent 视频扩散 · L3 一致性/时序连贯挑战 · L4 成本工程 (Open-Sora ~$200k 教训, 接 M8 infra)
- **notebooks (2)**: N1 在玩具「运动序列」上时空扩散, 生成连贯短序列 · N2 时序连贯性度量 + 出版级图 (接 9.6)
- **src**: `video_diffusion.py` (时空扩散最小实现)

### 13.5 world-models
- **lectures (4)**: L1 世界模型: 学环境动态 (预测未来观测) · L2 视频生成模型当世界模拟器 (OpenAI 2024 洞察) · L3 交互式/可玩世界模型 (Vid2World/AVID, 加动作条件) · L4 世界模型 × 具身 (接 M11 world-action)
- **notebooks (2)**: N1 在玩具环境学动作条件的世界模型 (给状态+动作→下一状态), 用它想象 rollout · N2 world model 预测质量评测
- **src**: `world_model.py` (动作条件转移模型), 与 M11 同源/共享

### 13.6 diffusion-language-models
- **lectures (4)**: L1 dLLM: 文本也能扩散 (masked diffusion, 非自回归并行生成) · L2 AR vs diffusion LM 的机制差异 (后训练范式迁移研究) · L3 为什么 dLLM 值得 NLP 人关注 (并行解码/可控生成/双向) · L4 dLLM 的现状与开放问题 (用 9.3 批判读)
- **notebooks (2)**: N1 在玩具序列上手搭 masked diffusion LM, 看并行去噪生成文本 · N2 dLLM vs AR 在同任务上的生成对比 (接 9.4)
- **src**: `diffusion_lm.py` (masked diffusion 语言模型最小实现)

### 13.7 generative-media-graduation (Capstone)
- **lectures (2)**: L1 把 13.1-13.6 串成扩散范式全景 (图/视频/世界/语言统一在「去噪」下) · L2 扩散研究前沿 + 用 9.3 gap 雷达扫题 (dLLM 对 NLP 人最友好, 世界模型接 M11)
- **notebooks (2)**: N1 端到端: 训一个条件扩散生成器 + 采样展示 · N2 用 9.3/9.4 起一张扩散/dLLM/世界模型研究 idea 卡
- 接回: 用户补上整条扩散范式, 且 dLLM 直接是他 NLP 本行的新研究入口。

## 4. 与现有资产整合
- **独立可建**, 同时是 M11 (diffusion policy 11.3 / world model 11.5) 的理论前置 —— 建议 M13 在 M11 之前或并行, 让 M11 的动作头/世界模型有理论支撑。
- **接 M10**: latent diffusion 用 M10 的 VAE/VQ; 条件生成接 M10 的视觉 token。
- **接 M3/M8**: dLLM 复用 pretraining 知识; 视频扩散成本工程接 M8 infra。
- **复用 Module 9**: 9.4 (guidance/采样步数消融) / 9.6 (生成质量出图) / 9.3 (批判读扩散论文) / Capstone 找 gap。
- **EE 数学优势**: score matching / ODE-SDE / 概率流的数学在这里直接是生产力。

## 5. 成功标准
- [ ] 7 专题完整落地, 结构同 Module 9。
- [ ] notebook 全 nbconvert 跑通 0 报错, 玩具分布/tiny 图 CPU 可跑。
- [ ] 课件研究生级, 公式逐项 (前向加噪/score matching/flow matching/CFG)。
- [ ] 至少一个 notebook 可视化「去噪轨迹」让人看见扩散在干什么。
- [ ] dLLM 专题让用户在 NLP 本行理解扩散。
- [ ] Capstone 产出扩散/dLLM/世界模型研究 idea 卡。
