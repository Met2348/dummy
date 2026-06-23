# Module 12「机制可解释性 / 对齐前沿」设计 spec

> Date: 2026-06-24 · 用户 (博0, EE 本硕 → 数学功底是优势, NLP/LLM, PhD 找方向)
> 终态: 新增 Module 12 —— 机制可解释性 (mechanistic interpretability) + 对齐前沿。这是 MIT Tech Review **2026 十大突破技术**之一, frontier 安全实验室 (Anthropic/OpenAI/DeepMind) 的硬通货研究方向。

## 1. 背景与动机

- **2026 行情**: 机制可解释性入选 MIT Tech Review 2026 十大突破; Anthropic 的「显微镜」、CoT 监控是当红方向; 这是 frontier 安全实验室最稀缺的研究者画像 (来源见 portfolio_v4 招聘核对)。
- **对用户的独特价值**: ① 这是**研究密集型**方向 (不是工程), 直接服务 PhD 产出新知识; ② 用户 EE 的**数学/信号分析功底**在这里是优势 (线性代数/因果干预); ③ 和用户已有的 reasoning-r1/process-reward/safety 强相关 (CoT 忠实性 = 可解释性 × 推理)。
- **完整空白**: 用户体系无任何可解释性专题。
- **独立性**: M12 横切所有模型, **不依赖 M10/M11**, 可独立建。

### 关键洞察
1. 可解释性的核心方法是「**因果干预**」—— activation patching / SAE / ablation。讲清「相关 vs 因果」是本模块灵魂 (也呼应 9.4 实验设计的因果思想)。
2. **可跑纪律**: 用 tiny transformer (甚至用户已复现的小模型) + 玩具任务 (如括号匹配/induction), 在 CPU 上真的做 probing/patching/SAE, 看见「电路」。复用 Module 9 确定性纪律。
3. 接回用户的 reasoning-r1: 把可解释性工具用到他自己复现的推理模型上, 是最有冲击力的练习 (也是潜在 PhD 题)。

## 2. 专题蓝图 (7 专题)

| # | slug | 覆盖技能 | 核心产出 |
|---|---|---|---|
| 12.1 | `interp-foundations` | 什么/为什么机制可解释性; features/circuits/superposition; 逆向工程 NN 的纲领 | 可解释性研究纲领地图 |
| 12.2 | `probing-and-activations` | 线性探针, residual stream 视角, logit lens, 表示几何 | 在 tiny model 上探针读出概念 |
| 12.3 | `causal-interventions` | activation patching, 因果干预, ablation, 相关 vs 因果 | 用 patching 定位一个行为的因果路径 |
| 12.4 | `sparse-autoencoders` | SAE / 字典学习 / 单义特征 (Anthropic 显微镜), 特征提取 | 训一个 mini-SAE 提取可解释特征 |
| 12.5 | `circuits-attention` | 电路分析, attention head 解释, induction heads, 归因 patching | 找出一个 induction 电路 |
| 12.6 | `cot-faithfulness-oversight` | 推理模型可解释性, CoT 忠实性/监控, scalable oversight, 欺骗/sandbagging 检测 | CoT 忠实性探测 (接 reasoning-r1) |
| 12.7 | `interp-graduation` | Capstone: 把工具用到自己复现的模型 + 研究 gap | 对自己模型做一次 interp 研究 + idea 卡 |

## 3. 逐专题详细设计

### 12.1 interp-foundations
- **lectures (4)**: L1 为什么要打开黑箱 (安全/科学/调试) + 机制可解释性 vs 事后归因 · L2 核心概念: feature / circuit / superposition (为什么神经元多义) · L3 逆向工程纲领 (把 NN 当程序读) · L4 这门学科的方法论与局限 (用 9.3 批判式读它的论文)
- **notebooks (2)**: N1 在 tiny transformer 上可视化激活, 看「多义神经元」现象 · N2 一个玩具任务 (模运算/括号匹配) 训 tiny model, 为后续解剖准备
- **src**: `tiny_transformer.py` (可 hook 中间激活的最小 transformer)

### 12.2 probing-and-activations
- **lectures (4)**: L1 residual stream 视角 (信息在残差流里读写) · L2 线性探针 (probe 出某概念是否被线性编码) · L3 logit lens (每层在「想」什么) · L4 探针的陷阱 (探针能读出 ≠ 模型在用, 相关非因果, 引出 12.3)
- **notebooks (2)**: N1 在 tiny model 上训线性探针, 读出中间层编码的概念 · N2 logit lens: 逐层 decode 看预测如何成形
- **src**: `probing.py` (线性探针 + logit lens), 复用 12.1 hook

### 12.3 causal-interventions
- **lectures (4)**: L1 相关 vs 因果 (探针的根本局限) → 必须干预 · L2 activation patching (把干净激活贴到污染运行, 看行为变化) · L3 ablation / 因果路径 · L4 干预的严谨性 (接 9.4 实验设计: 干预就是消融的极致)
- **notebooks (2)**: N1 用 activation patching 定位「玩具任务里哪个位置/层负责某行为」· N2 ablation 扫描出最小因果子集
- **src**: `patching.py` (activation patching + ablation), 是本模块最核心工具

### 12.4 sparse-autoencoders
- **lectures (4)**: L1 superposition 问题 → 为什么需要 SAE 解叠加 · L2 SAE 原理 (稀疏字典学习, 重建+稀疏损失逐项交代) · L3 单义特征 (Anthropic 的「金门大桥特征」) · L4 SAE 的评估与争议 (特征真实吗, 用 9.3 攻击)
- **notebooks (2)**: N1 在 tiny model 的激活上训一个 mini-SAE, 提取稀疏特征 · N2 检视提取的特征是否单义 (找最大激活样本)
- **src**: `sae.py` (稀疏自编码器 + 特征分析)

### 12.5 circuits-attention
- **lectures (4)**: L1 attention head 在算什么 (QK/OV 电路) · L2 induction heads (in-context learning 的机制) · L3 电路分析方法 (把多个组件连成电路) · L4 归因 patching (规模化的因果归因)
- **notebooks (2)**: N1 在 tiny model 找 induction head (看它复制前文模式) · N2 attribution patching 给出每个组件的贡献热图 (出版级图, 接 9.6)
- **src**: `circuits.py` (head 分析 + induction 检测 + 归因)

### 12.6 cot-faithfulness-oversight
- **lectures (4)**: L1 推理模型可解释性 (CoT 是不是真的「内心独白」) · L2 CoT 忠实性/监控 (模型说的和做的一致吗) · L3 scalable oversight (弱监督强 / debate / weak-to-strong) · L4 欺骗/sandbagging 检测 (对齐安全前沿)
- **notebooks (2)**: N1 CoT 忠实性探测: 扰动 CoT 看答案是否真依赖它 (接用户 reasoning-r1) · N2 一个 weak-to-strong 玩具实验
- **src**: `cot_probe.py` (CoT 忠实性扰动测试)

### 12.7 interp-graduation (Capstone)
- **lectures (2)**: L1 把 12.1-12.6 串成一次完整 interp 研究流程 · L2 可解释性研究前沿 + 用 9.3 gap 雷达扫题 (CoT 忠实性 × 推理 对用户最友好, 直接连 reasoning-r1)
- **notebooks (2)**: N1 对用户自己复现的某个小模型 (或 tiny R1) 做一次端到端解剖 (探针→patching→SAE→电路) · N2 用 9.3/9.4 把一个可解释性 gap 起成 idea 卡 (可证伪假设 + 最小实验)
- 接回: 这是用户**最可能直接转成 PhD 题**的模块 —— interp × reasoning 是 frontier 实验室热点且门槛对 NLP 人友好。

## 4. 与现有资产整合
- **独立于 M10/M11**: 可单独建/学。
- **强接 M4**: reasoning-r1/process-reward 是 12.6 的直接材料 (CoT 忠实性)。
- **接 M6**: safety-defense/red-team 与对齐前沿 (12.6) 互补。
- **复用 Module 9**: 9.4 (干预=消融的极致) / 9.6 (归因热图出版级) / 9.3 (批判读 interp 论文) / Capstone 找 gap 全程用。
- **EE 优势变现**: 线性代数/因果干预/信号分析的数学功底在这里直接是研究生产力。

## 5. 成功标准
- [ ] 7 专题完整落地, 结构同 Module 9。
- [ ] notebook 全 nbconvert 跑通 0 报错, tiny model CPU 可跑。
- [ ] 课件研究生级, 公式逐项 (SAE 损失/patching/QK-OV 电路)。
- [ ] 至少一个 notebook 真的在 tiny model 上做出 activation patching 因果定位 + SAE 特征。
- [ ] Capstone 对「自己的模型」做一次 interp, 产出 interp×reasoning 研究 idea 卡。
- [ ] portfolio 更新, 体现「会造模型 → 会解剖模型」的研究者画像。
