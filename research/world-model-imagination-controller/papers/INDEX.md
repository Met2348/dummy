# 文献库索引 — world-model-imagination-controller(68 篇标注清单:61 篇 arXiv + 7 篇经典/非arXiv引用)

> 这是 [`download_papers.py`](download_papers.py) 下载的全部 arXiv 论文(PDF 不进 git,跑脚本即可重下)加上
> 7 篇无 arXiv 版本的经典/技术报告引用。按 5 路调研分组 A–E,对应
> [`../00-brainstorm-10-ideas.md`](../00-brainstorm-10-ideas.md) 里每条 idea 引用的文献编号。
> 每条标注了核心机制 + 对我们问题("测试时想象预算的自适应分配")的具体相关性,不是泛泛贴标题。
> 建成:2026-07,5 路子代理独立调研 + 主线程对高风险(2025-2026年)条目做 WebFetch 真实核验,零臆造 arXiv ID。
> 部分论文横跨多个 track(如 Genie 同时是 world model 和视频生成),归入最贴切的一个 track,交叉相关性在该论文条目里说明。

---

## A. World Models / 基于模型的规划与决策(16 篇)

想象预算目前是怎么被固定死的——这一组的证据链核心结论:**从 Dreamer 到 TD-MPC2 再到 MuZero,SOTA world model 的"想象多久/想几个候选"普遍是训练前定死的全局超参数,与具体任务难度或模型置信度无关**。

| 论文 | 年份/会议 | 核心机制 | 与我们问题的相关性 |
|---|---|---|---|
| [World Models](https://arxiv.org/abs/1803.10122) (1803.10122) | 2018 NeurIPS | VAE 编码观测 + MDN-RNN 预测下一潜状态,agent 完全在"梦境"环境里训练 controller | 想象/决策解耦范式的起点,训练和测试都用固定长度梦境 rollout,是本问题最原始的反面案例 |
| [PlaNet](https://arxiv.org/abs/1811.04551) (1811.04551) | 2019 ICML | 学隐空间动力学模型,CEM 在隐空间采样固定数量候选动作序列、rollout 固定 horizon | "候选数+horizon 都是训练前定死超参数"的 planning 基线,无按候选质量提前终止的机制 |
| [Dreamer](https://arxiv.org/abs/1912.01603) (1912.01603) | 2020 ICLR (oral) | 隐空间展开固定长度(15步)想象轨迹,价值函数梯度反传学 actor-critic | Dreamer 系列起点,想象 horizon 是全局固定超参数,不随任务/状态/不确定性调整 |
| [DreamerV2](https://arxiv.org/abs/2010.02193) (2010.02193) | 2021 ICLR | 离散(categorical)隐变量替代高斯隐变量,同样固定 horizon 想象,Atari 达人类水平 | 证明固定想象 horizon 方案能 scale,但 horizon 仍是人工常数 |
| [DreamerV3](https://arxiv.org/abs/2301.04104) (2301.04104) | 2023(Nature 2025) | 归一化/KL 平衡等鲁棒化技巧让同一套超参数跨 150+ 任务通用,零人工数据采到 Minecraft 钻石 | 目前最强通用 world model 仍用固定想象步数应付所有任务规模和难度,SOTA 系统里"预算分配粗糙"依然成立的关键证据 |
| [MuZero](https://arxiv.org/abs/1911.08265) (1911.08265) | 2019/2020(Nature) | 学隐动力学模型后用 MCTS 树搜索,每步固定模拟次数(如50/800次) | MCTS 按 PUCT 分数把固定总模拟预算不均匀分给不同分支,是"总预算固定、候选间自适应分配"的早期部分先例,但从不决定要不要减少总预算本身 |
| [EfficientZero](https://arxiv.org/abs/2111.00210) (2111.00210) | 2021 NeurIPS | 自监督一致性损失+值预测修正,固定模拟次数 MCTS 大幅提升样本效率 | 样本效率提升来自模型质量而非搜索预算分配方式改变,是"改模型不改预算分配"的对照组 |
| [TD-MPC](https://arxiv.org/abs/2203.04955) (2203.04955) | 2022 ICML | 隐空间任务导向动力学模型+MPPI/CEM 短 horizon 轨迹优化,可学习终值函数弥补截断 | 用可学习终值函数补偿"horizon 不够长",但候选轨迹数和 horizon 仍是训练前固定超参数 |
| [TD-MPC2](https://arxiv.org/abs/2310.16828) (2310.16828) | 2024 ICLR | scale 到 317M 参数、80 个任务(50 Meta-World+30 DMControl)单一配置,MPPI 用固定数量(512条)候选轨迹迭代重加权 | 当前最强隐空间 MPC 基线之一,候选轨迹数训练前定死,不随任务/状态不确定性调整 |
| [Genie](https://arxiv.org/abs/2402.15391) (2402.15391) | 2024 ICML(最佳论文) | 无标注视频学出 11B 参数时空视频 tokenizer+自回归动力学模型+隐动作模型,逐帧交互生成 | 生成式想象的算力开销随模型规模显著上升,预算分配问题只会更迫切;逐帧自回归结构天然对应"每步是否继续"的停止决策点(交叉关联 Track B/D) |
| [GameNGen](https://arxiv.org/abs/2408.14837) (2408.14837) | 2024(ICLR 2025) | 扩散模型逐帧生成 DOOM 画面,推理时把去噪步数从常规几十步压到 4 步以维持实时帧率 | "为满足固定实时预算而砍生成质量"的典型例子,去噪步数是人工定死常数,不是按当前帧对最终结果的重要性动态决定的(交叉关联 Track B) |
| [IRIS](https://arxiv.org/abs/2209.00588) (2209.00588) | 2022(ICLR 2023) | 离散自编码器把图像转 token,自回归 Transformer 在 token 空间建模动力学 | 又一个"想象 horizon 写死在训练配置里"的高样本效率范例,验证缺口在 RNN 和 Transformer 两类架构上同样存在 |
| [MBPO](https://arxiv.org/abs/1906.08253) (1906.08253) | 2019 NeurIPS | 短分支 rollout(1-15步),理论论证模型误差累积决定该用多长 rollout,按训练进度线性增长 | "rollout 长度怎么定"最早被系统研究的论文之一,但只按全局训练进度做调度,不区分任务/状态/候选,是"预算分配=人工调度表"传统的源头 |
| [MACURA](https://arxiv.org/abs/2405.19014) (2405.19014) | 2024 ICML | 用集成模型不确定性估计,把 rollout 长度按"每个具体状态"自适应截断 | 较强的"按不确定性自适应 rollout 长度"先例,证明自适应分配优于 MBPO 式固定/线性调度;但只调"该想多长"一个维度,未处理"该不该想/该采纳哪个候选" |
| [AVIC: When and How Much to Imagine](https://arxiv.org/abs/2602.08236) (2602.08236) | 2026 | 用 RL(GRPO)训练门控策略,先判断证据是否够用、不够再决定生成多少想象视角,奖励=正确率-想象开销 | **结构上离我们的 controller 最近的现有工作**:显式把"要不要想象"和"想多少"都变成可学习决策,主战场是视觉空间推理 QA;R2R 导航实验里也验证了序贯场景下逐步应用门控,但每步决策彼此独立、没有跨步骤预算调度,也没有多候选比较机制(交叉关联 Track D,已 WebFetch 二轮核验) |
| [AHEAD](https://arxiv.org/abs/2606.02486) (2606.02486) | 2026 | 冻结 VLA 策略接轻量世界模型,前向预测未来视觉特征直到预测不确定性越过阈值才停 | 提供"不确定性阈值触发提前停止想象"的具体机制,但这里是单条轨迹的延迟补偿,不是多候选未来的比较择优(已 WebFetch 核验) |
| 无 arXiv,见 DeepMind 博客(2024.12) | Genie 2 | 技术博客 | 把 Genie 扩展到 3D 一致可玩环境,官方明确区分"未蒸馏基座"与"可实时运行蒸馏版",直接佐证想象存在可调的质量-速度前沿 |

---

## B. 视频生成用于决策,及其计算成本(11 篇)

这一组建立"为什么想象是贵的、为什么预算分配才重要"的动机证据链:**GameNGen 为 20fps 实时把每帧去噪步数从数十步压到 4 步;UniSim 训练耗 512 张 TPU-v3 共 20 天;业界估算 Sora 单条 10 秒视频约需 40 GPU 分钟**。

| 论文 | 年份/会议 | 核心机制 | 与我们问题的相关性 |
|---|---|---|---|
| [UniPi](https://arxiv.org/abs/2302.00111) (2302.00111) | 2023 NeurIPS | 文本条件视频扩散模型生成未来帧作为"计划",逆动力学模型从生成视频提取可执行动作 | video-as-policy 范式奠基之作,是我们要判断"值不值得生成"的想象对象最典型的原型,本身不做任何预算取舍 |
| [Video Language Planning](https://arxiv.org/abs/2310.10625) (2310.10625) | 2023(ICLR 2024) | VLM 作策略与价值函数、text-to-video 模型作动力学模型,树搜索式长程视频规划 | 明确报告"计划质量随计算预算增加而提升"的 scaling 现象,是把想象预算当作显式可调量并观测到收益曲线的最直接先例 |
| [AVDC](https://arxiv.org/abs/2310.08576) (2310.08576) | 2023 | 生成任务完成视频后用帧间稠密光流对应关系反推机器人动作 | 说明生成的想象可以只作中间表征,想象内容是否有效可被独立度量,呼应"生成后该不该采纳"这一设计点 |
| [UniSim](https://arxiv.org/abs/2310.06114) (2310.06114) | 2023(ICLR 2024) | 多源视频/动作/文本数据训练 5.6B 参数交互式视频扩散模拟器 | 训练成本(512张TPU-v3、20天)是"生成式想象引擎本身极贵"的直接量化证据 |
| [DIAMOND](https://arxiv.org/abs/2405.12399) (2405.12399) | 2024 NeurIPS Spotlight | RL agent 完全在逐帧生成下一观测的扩散世界模型内部训练 | 完整展示扩散世界模型想象 rollout 用于策略训练的闭环,是要优化的想象-决策循环最贴近的具体实现对象 |
| [Vista](https://arxiv.org/abs/2405.17398) (2405.17398) | 2024 NeurIPS | 高保真长程(15秒)驾驶视频世界模型,首次用模型自身预测的不确定性作为动作可靠性的奖励信号 | 直接示范"用生成视频的不确定性判断该不该采纳某个想象结果",与"task-conditioned+uncertainty-aware imagination"方向几乎一一对应 |
| [Cosmos](https://arxiv.org/abs/2501.03575) (2501.03575) | 2025 NVIDIA | 覆盖自回归与扩散两类世界基座模型的预训练+后训练平台 | 代表 2025 年最新大规模"世界基座模型"基础设施,对 tokenizer 压缩率与推理吞吐的强调说明生成效率已是核心工程指标 |
| [Diffusion Forcing](https://arxiv.org/abs/2407.01392) (2407.01392) | 2024 NeurIPS | 每 token/帧可独立设定噪声水平,统一因果 next-token 预测与全序列扩散 | 提供让想象按需变长、按需插入 guidance 的架构基础,天然适合把"何时停止生成"的控制器直接接到采样过程里 |
| [iVideoGPT](https://arxiv.org/abs/2405.15223) (2405.15223) | 2024 NeurIPS | 可扩展自回归 transformer 统一压缩视觉观测/动作/奖励为 token 序列 | 明确以"让交互式视频世界模型可扩展、计算更省"为卖点,是生成效率工程改进在架构层面的代表 |
| [Video-T1](https://arxiv.org/abs/2503.18942) (2503.18942) | 2025 | 把视频生成 test-time scaling 重述为搜索问题,Tree-of-Frames 自适应扩展/剪枝候选视频分支 | **与我们要做的 controller 机制几乎同构**,是最直接的同期相关工作,写作时需正面对比 |
| [EvoSearch](https://arxiv.org/abs/2505.17618) (2505.17618) | 2025 | test-time scaling 重述为进化搜索(选择+变异),作用于扩散/流式生成模型去噪过程 | 相比 best-of-N 有约23%-33%的具体提升数字(因生成模型不同而异),另有一组约100倍算力量级的独立scaling曲线观察——二次核验发现两者是原文两个不同实验维度,不是同一组"花100倍换23-33%"的因果配对,为"生成多少候选才够用"提供量化基线 |
| 无 arXiv,见 OpenAI 技术报告(2024) | Sora: Video generation models as world simulators | 技术报告 | "视频生成即世界模拟器"论述的公众认知源头,为本 track 确立核心叙事前提,但未公开算力细节(第三方估算单条10秒视频约需40GPU分钟) |

---

## C. 自适应测试时计算分配的技术工具箱(15 篇)

别的领域怎么解决同构问题——**最可借鉴的模式是 draft-then-verify**:LayerSkip 用同模型浅层出草稿、深层验证;EAGLE-2/Infoprop 用置信度或不确定性动态定深度与停止点。这和用户自己在 `for_real_dummy/inference-serving-deep-dive` 已经吃透的投机解码是同一问题形状。

| 论文 | 年份/会议 | 核心机制 | 与我们问题的相关性 |
|---|---|---|---|
| [EAGLE-2](https://arxiv.org/abs/2406.16858) (2406.16858) | 2024 | 草稿模型自身校准置信度动态决定投机树形状与深度 | 可借鉴"用草稿阶段自身置信度动态决定展开深度/宽度"做候选未来的动态预算分配 |
| [LayerSkip](https://arxiv.org/abs/2404.16710) (2404.16710) | 2024 ACL | 同一模型浅层输出作草稿、深层验证,融合早退与投机解码 | 与 draft-then-verify 结构最贴近的先例,可类比用同一 world model 的浅层/低成本 rollout 做草稿、深层做验证 |
| [CALM](https://arxiv.org/abs/2207.07061) (2207.07061) | 2022 NeurIPS | 每层加早退出口,校准过的置信度度量逐 token 决定是否提前退出 | 可借鉴"置信度校准+序列级早退判定"技术栈,判定某条想象轨迹已"看够"可提前终止 |
| [ACT](https://arxiv.org/abs/1603.08983) (1603.08983) | 2016 | 给 RNN 加可微分 halting unit,按输入难度自主学习该迭代计算几步 | "内部可微分停止概率单元控制计算步数"的最基础范式,可作想象 controller 停止决策的可微分实现方案 |
| [PonderNet](https://arxiv.org/abs/2107.05407) (2107.05407) | 2021 ICML Workshop | ACT 的确定性 halting 改为概率式建模,正则化目标在准确率与计算成本间端到端权衡 | "halting 概率分布+显式计算成本正则项"目标函数设计可直接迁移为想象 controller 的训练目标 |
| [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168) (2110.14168) | 2021 | 生成多候选解后训练独立 verifier 打分取最高者(best-of-N) | best-of-N+verifier 最基础范式,可作"该不该采纳这个想象未来"决策模块的最简基线 |
| [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) (2305.20050) | 2023(ICLR 2024) | 过程级(而非只看最终结果)人工标注训练 process reward model | 可把"逐步过程打分"用于给想象过程中的中间帧打分,不必等整条 rollout 结束才评估是否有用 |
| [Scaling LLM Test-Time Compute Optimally](https://arxiv.org/abs/2408.03314) (2408.03314) | 2024(ICLR 2025) | 最优测试时计算分配应按 prompt 预测难度自适应选择策略,而非统一分配 | "compute-optimal"按难度自适应分配框架直接对应"按想象难度决定生成预算"这一目标(交叉关联 Track E) |
| [s1: Simple Test-Time Scaling](https://arxiv.org/abs/2501.19393) (2501.19393) | 2025 | 用极简"budget forcing"手段强制截断或延长思考 | budget forcing 是几乎不改模型的外部旋钮式控制器,可类比想象 rollout 上"强制停/强制多想一步"的廉价基线 |
| [Large Language Monkeys](https://arxiv.org/abs/2407.21787) (2407.21787) | 2024 | 重复采样 N 个候选解,覆盖率随采样数近似幂律增长,但选择机制数百样本后即饱和 | 提醒预算分配之外必须同时设计好候选筛选器,否则想象预算增加未必转化为决策收益 |
| [Tree of Thoughts](https://arxiv.org/abs/2305.10601) (2305.10601) | 2023 NeurIPS | 生成过程组织成候选思考节点树,每步 self-evaluation 打分后决定扩展/剪枝/回溯 | "生成后先自评分再决定是否继续展开该分支"可直接类比:每个候选未来先打分,分低分支直接剪掉不再想象 |
| [RAP](https://arxiv.org/abs/2305.14992) (2305.14992) | 2023 EMNLP | LLM 同时充当 world model 和推理 agent,MCTS 在候选未来推理路径间搜索 | "world model+MCTS 搜索控制器"结构的直接先例,explore/exploit 回传机制可用于决定该展开哪个候选未来、给多少预算 |
| [Adaptive Rollout Length for Model-Based RL Using Model-Free Deep RL](https://arxiv.org/abs/2206.02380) (2206.02380) | 2022 | 把"该把 rollout 做多长"重新定义为元级决策问题,用 model-free RL agent 周期性调整一个跨批次共享的 rollout 长度超参数(不是逐条 rollout 单独决定何时停,二次核验订正了这个粒度描述) | 与"想象预算控制器"训练范式部分同构:元策略学习粒度是"批量共享超参数",比我们要的"每个候选未来独立决策"更粗,这个粒度差异需要注意 |
| [Infoprop](https://arxiv.org/abs/2501.16918) (2501.16918) | 2025 | 显式拆分 rollout 预测的 aleatoric 与 epistemic 不确定性,追踪累积 epistemic 误差作为终止准则 | 与导师所提"uncertainty aware imagination"直接对应,可将"累积不确定性触发停止"机制搬到视频 world model 想象过程 |
| [REFRAIN](https://arxiv.org/abs/2510.10103) (2510.10103) | 2025 | 训练free的两阶段停止判别器+滑动窗口 UCB 多臂老虎机,依"继续想的边际收益"动态调整停止阈值 | UCB 式"停止阈值随难度自适应调节"的控制器结构,可迁移为按候选未来不确定性调节"是否继续生成下一帧"的阈值,且无需重训 world model |

---

## D. 不确定性量化(12 篇)

这一组是这个 idea 的技术内核——"决策收益"本质要靠某种不确定性/分歧度量来估计。**判断:视频/序列生成中全量集成与 MC-dropout 逐候选重跑代价过高;更适合让模型自身输出校准置信度(如 C3 的 subpatch 校准)配合序列级熵/互信息分解定位不确定性来源,但须警惕潜空间 UQ 可能被系统性低估(Biased Dreams 的警告)**。

| 论文 | 年份/会议 | 核心机制 | 与我们问题的相关性 |
|---|---|---|---|
| [What Uncertainties Do We Need in Bayesian Deep Learning for CV?](https://arxiv.org/abs/1703.04977) (1703.04977) | 2017 NeurIPS | 异方差回归头学逐样本 aleatoric 方差,MC-dropout/集成估计 epistemic 方差,总方差=两者之和 | 把候选未来的"价值不确定性"拆成可靠多生成几个样本降低的部分(继续想象)和环境本身随机的部分(见好就收) |
| [MC-Dropout](https://arxiv.org/abs/1506.02142) (1506.02142) | 2016 ICML | 测试时保持 dropout 开启多次前向采样,输出方差近似模型后验 epistemic 不确定性 | 对同一候选未来用已训练好的世界模型/价值头做 T 次 dropout 前向即得低成本"值不值得生成/采纳"信号 |
| [Deep Ensembles](https://arxiv.org/abs/1612.01474) (1612.01474) | 2017 NeurIPS | 独立训练多网络,集成均值作预测、集成方差/分歧作不确定性 | 维护小型价值/奖励预测集成,对每个候选未来算集成分歧,分歧越大代表信息量越大,越值得生成或保留 |
| [Uncertainty Estimation in Autoregressive Structured Prediction](https://arxiv.org/abs/2002.07650) (2002.07650) | 2020(ICLR 2021) | 把熵/互信息分解推广到自回归结构化预测,定义 token 级与序列级不确定性 | 对世界模型集成做多次 rollout,同时得到"每一帧"和"整条想象轨迹"两级不确定性,定位该在哪截断或延长候选未来 |
| [A Gentle Intro to Conformal Prediction](https://arxiv.org/abs/2107.07511) (2107.07511) | 2021 | 用校准集构造无分布假设、有限样本覆盖保证的预测集/区间 | 给 controller 一个有统计保证的"候选未来预测价值可信到能采纳吗"阈值,替代拍脑袋设定的不确定性 cutoff |
| [Conformal Language Modeling](https://arxiv.org/abs/2306.10193) (2306.10193) | 2023(ICLR 2024) | 对生成模型采样出的一组候选做校准+拒绝规则,保证高概率至少保留一个合格样本且集合尽量小 | 把候选想象未来当作采样集合做同样的校准-拒绝流程,决定生成到第几个候选、留哪些扔哪些,并带统计保证 |
| [HyperDM](https://arxiv.org/abs/2402.03478) (2402.03478) | 2024 NeurIPS | 贝叶斯超网络从单个训练好的扩散模型动态生成等效集成权重,单模型近似真实集成质量 | 若世界模型本身是扩散/流模型,可用此法以远低于真实多集成的算力拿到候选未来的分歧信号 |
| [PETS](https://arxiv.org/abs/1805.12114) (1805.12114) | 2018 NeurIPS | 概率动力学模型集成+轨迹采样,把逐步模型不确定性传播到多步 rollout | 提供"多步想象轨迹"不确定性传播的现成方法论:不只对单步预测打分,而是把不确定性沿整条候选未来传播成轨迹级打分 |
| [Plan2Explore](https://arxiv.org/abs/2005.05960) (2005.05960) | 2020 ICML | Dreamer 式潜空间世界模型里用一步集成分歧做内在奖励,想象规划主动寻找预期分歧大的动作 | 与目标架构几乎同源:展示了如何把潜空间想象 rollout 中的集成分歧直接转成"这个未来值不值得想象"的标量信号 |
| [C3: World Models That Know When They Don't Know](https://arxiv.org/abs/2512.05927) (2512.05927) | 2025-2026 | 训练视频世界模型在潜空间直接输出校准的 subpatch 级置信度,上采样为像素级热力图 | **目前找到最直接对口的工作**:世界模型自己吐出校准置信度,可直接作为给候选想象未来(及局部区域)打分、判断是否可信采纳的现成信号源(已 WebFetch 核验) |
| [FFDC](https://arxiv.org/abs/2605.06222) (2605.06222) | 2026 | 轻量验证器联合看预测动作、预测视觉动态与真实观测,持续判断剩余想象是否还可信 | 对应 controller"该何时停/何时重来"这一半:把不确定性重新定义为预测与真实观测的一致性,而非仅靠生成时内部模型统计量(已 WebFetch 核验) |
| [Biased Dreams](https://arxiv.org/abs/2604.25416) (2604.25416) | 2026 | 实证发现 Dreamer 式 RSSM 潜空间 rollout 会被拉向"表征良好"(通常高奖励)的吸引子区域 | **重要预警而非工具**:controller 无论选哪种不确定性度量,都必须先验证它在潜空间想象里真的追踪真实误差,否则可能在最需要报警时反而给出虚假的低不确定性(已 WebFetch 核验) |

无 arXiv,见 NeurIPS 2023 proceedings / msl.stanford.edu/projects/plancp — **PlanCP**:用 conformal prediction 校准扩散动力学模型,得到轨迹空间中保证覆盖真实 rollout 的置信区域。与本项目场景最接近的现成范例:扩散式(类世界模型)轨迹预测器,覆盖区域"大小"直接就是每个候选想象未来的不确定性打分。

---

## E. 元推理 / 信息价值 / 最优停止理论(7 篇 + 5 篇经典引用)

问题的数学锚点。**主锚点是 Russell & Wefald(1991)的 Value-of-Computation 框架:VOC(c) = 想象 c 后最优决策的期望效用 − 不想象时的期望效用 − 计算成本。"该不该生成"对应 VOC(c)>0,"该不该采纳"对应观测想象结果后 best action 是否切换,配合 Chow-Robbins-Siegmund 最优停止理论给出"何时停"的可操作判据(max_c VOC(c)≤0 即停)**。

| 论文/引用 | 年份/期刊 | 核心机制 | 与我们问题的相关性 |
|---|---|---|---|
| 无 arXiv,经典引用:Russell & Wefald (1991), *Artificial Intelligence* 49(1-3):361-395 | 1991 | 提出 Value of Computation(VOC)框架:理性 agent 应执行"期望效用增益减去计算成本"最大的计算,不为正时停止 | **全项目最核心的数学锚点**,可直接把三个决策点写成 VOC(c) 的具体形式 |
| [Selecting Computations](https://arxiv.org/abs/1207.5879) (1207.5879) | 2012 UAI | "该模拟哪些候选动作序列"形式化为 Bayesian selection problem,给出有限样本下最优/近似最优选择策略及理论界 | 把候选想象未来集合当作待评估的 candidate simulations,直接套用其 selection-problem 框架决定每步该展开哪一个、何时停止 |
| 无 arXiv,经典引用:Howard (1966), *IEEE Trans. Systems Science and Cybernetics* 2:22-26 | 1966 | 首次形式化"信息的价值"=拥有该信息后最优决策期望值减去没有该信息时的期望值 | VOI 理论源头,给出 VOI(c) 最原始的定义模板,可直接映射为"生成候选未来 c"这条信息带来的边际提升量 |
| [VOIMCP](https://arxiv.org/abs/2604.01434) (2604.01434) | 2026 | POMDP 的 MCTS 规划中显式建模"是否值得针对某观测/分支继续推理",VOI 低的分支直接跳过,给出近似值函数与真实最优值的误差界(原文称"bounded regret",但约束对象是这个误差,不是经典bandit意义上随决策轮次累积的regret,二次核验后订正术语)+ MCTS 收敛速率证明 | 把"是否要对每个候选未来展开想象 rollout"当作树搜索中"是否要展开某观测分支"的 meta 决策,可直接复用其剪枝算法与理论保证(已 WebFetch 二轮核验) |
| 无 arXiv,经典引用:Chow, Robbins & Siegmund (1971), *Great Expectations: The Theory of Optimal Stopping* | 1971 | 一般化序贯最优停止理论:用鞅论证明存在最优停止时刻 τ* 使期望回报最大 | 把"该在哪一步停止对某条想象轨迹继续推演"直接写成停止时刻问题,是"何时停"子问题最直接的数学骨架 |
| [Deep Optimal Stopping](https://arxiv.org/abs/1804.05394) (1804.05394) | 2018(JMLR 2019) | 用神经网络直接参数化"停/继续"这一停止决策本身,在高维无解析解问题上学到接近最优停止规则 | 给"该不该继续想象/该在哪一步停"提供可直接套用的训练配方:网络参数化 stop/continue 决策,用蒙特卡洛采样的想象 rollout 序列训练 |
| [BALD](https://arxiv.org/abs/1112.5745) (1112.5745) | 2011 | 预测熵减去期望后验熵得到互信息,挑选"模型整体不确定但一旦观测就会变确定"的点 | 把 BALD 互信息打分改造成"生成这个候选未来是否会改变最终决策"的信息增益打分,即想象预算分配的核心 acquisition function(交叉关联 Track D) |
| [Adaptive Submodularity](https://arxiv.org/abs/1003.3967) (1003.3967) | 2011 JAIR | 定义 adaptive submodularity 性质,满足时贪心自适应策略有 (1−1/e) 近似保证 | 把"预算 k 以内该依次生成哪些候选未来"写成 adaptive stochastic optimization,贪心选边际收益最大的候选生成即有理论保证 |
| 无 arXiv,经典引用:Zilberstein (1996), *AI Magazine* 17(3):73-83 | 1996 | 用 performance profile 刻画每个 anytime 模块的"质量-时间"权衡,给出多模块组合时的 meta-level control 方法 | 把每个候选未来的想象 rollout 看作 anytime computation,performance profile 为"想象步数→决策改进量",controller 任务就是在总预算下调度 |
| [Rational Metareasoning for Large Language Models](https://arxiv.org/abs/2410.05563) (2410.05563) | 2024-2025 | 直接把 Russell & Wefald 的 VOC 搬进 LLM 训练:Expert Iteration + VOC 惩罚项,使模型学会少生成无用思考步骤 | reward=决策收益−λ×计算量的训练目标可原样搬到想象 controller,是 VOC 理论到本项目最直接可操作的模板 |
| [Scaling LLM Test-Time Compute Optimally](https://arxiv.org/abs/2408.03314) (2408.03314) | 2024(ICLR 2025) | 测试时计算量该怎么分配高度依赖 prompt 难度,先估计难度再自适应选策略 | budget(s)=总预算约束下使期望决策收益最大化的分配函数,依赖对 state 当前难度/不确定性的估计(交叉关联 Track C) |
| [Certaindex](https://arxiv.org/abs/2412.20993) (2412.20993) | 2024-2025(NeurIPS 2025) | training-free 统计代理指标衡量推理过程"候选答案是否已收敛稳定",serving 系统据此决定算多久 | 把"该不该继续/终止对某候选未来的想象"表述为监测随想象步数演化的"未来预测收敛度"代理指标,是"何时停"的 training-free 可落地信号设计范式 |

---

## F. 2026-07 二次深挖:直接竞争格局 + VOC/最优停止理论补强(20 篇)

> 老师要求"找对研究问题是最关键的一步,要头脑风暴,千万不能偷懒"后新增的专项调研。全部经 WebFetch
> 逐篇核验(标题/作者/机制均对照原文,非二手转述),核验方法与结果详见
> [`../03-novelty-competitive-landscape.md`](../03-novelty-competitive-landscape.md)。分两组:
> F1 是 2026 年出现的直接竞争/邻近工作(world model 想象门控赛道),F2 是这次挖出的理论补强文献
> (揭示我们 pilot 的"发现一/二"其实是经典 VOC 理论推论,不是新发现)。

### F1. 2026 年直接竞争 / 邻近工作(10 篇)

| 论文 | 年份 | 核心机制 | 与我们问题的相关性 |
|---|---|---|---|
| [Astra: Thinking with Imagination](https://arxiv.org/abs/2606.06476) (2606.06476) | 2026.6 | Astra-VL(RL训练VLM策略)+Astra-WM(基于Bagel的世界模拟器),两阶段RL课程学习训练"何时调用模拟器" | **AVIC最直接的同期竞争者**,框架高度相似(视觉推理+门控何时想象),但资源背景更强;仍是纯RL经验式训练,未涉及理论保证或跨步预算调度——这两点仍是空当 |
| [Imagine-then-Plan (ITP)](https://arxiv.org/abs/2601.08955) (2601.08955) | 2026.3 | POIMDP形式化,RL训练K-head predictor每步自适应选想象深度K_t,代价函数log p(a\*\|s,τ̂)−λ_K·k,ALFWorld/WebShop/ScienceWorld/StableToolBench四个benchmark验证 | **idea 7"该不该想、想多深"这条叙事的框架级实现**(已用WebFetch逐条核对全部技术细节属实)。世界模型是微调LLM(文本生成式,非隐空间动力学),horizon一次性预先选定、无中途止损机制,且没有跨episode步骤的预算结转——这两点是我们仍可差异化的具体缺口 |
| [ELASTIC](https://arxiv.org/abs/2606.31132) (2606.31132) | 2026.6 | CMU(Andrew Zou Li/Gokul Swamy/Yonatan Bisk/Andrea Bajcsy),元RL学习扩散/流式策略的测试时算力调度(去噪步数+并行采样),meta-MDP形式化 | 同属"自适应测试时算力"家族,但**不涉及世界模型/想象rollout**,是对冻结策略的纯推理资源调度,和我们的问题结构不同,风险较低 |
| [Finding the Time to Think](https://arxiv.org/abs/2606.26463) (2606.26463) | 2026.6 | SMDP框架下用轻量gating policy逐决策点选择规划预算(real-time RL,agent自己选延迟),PacMan/Tetris/Snake/SpeedHex/SpeedGo环境验证 | idea 7的最近邻:同样是"每个决策点自适应选投入多少"，但用完美模拟器/MCTS而非学到的生成式world model,且预算基本不跨步结转(除个别子环境例外) |
| [ROI-Reasoning](https://arxiv.org/abs/2601.03822) (2601.03822) | 2026.1 | 把预算受限的序贯推理表述为Ordered Stochastic Multiple-Choice Knapsack Problem(OS-MCKP),两阶段(元认知微调+RL)学长程token分配策略 | idea 7"跨步预算调度"这个机制目前**最接近的匹配**,但应用于LLM多任务批量数学推理,不是具身智能体单episode序贯决策——领域不同,机制同构,必须在differentiation里正面回应 |
| [World-in-World](https://arxiv.org/abs/2510.18135) (2510.18135) | 2025.10 | 多候选动作生成+world model逐一rollout+revision policy打分选优的closed-loop在线规划;经查证提交于cs.CV分类,ICLR接收/Oral状态**未能在摘要页确认**,引用前需单独核实 | idea 1"多候选比较"表层机制目前**最强的既有工作**,但没有"要不要想象"的门控(每步必然想象)、候选是"选哪个动作"而非"该不该信这个想象"、无VOC式差值化打分——差异化必须落在VOC(c)这个反事实统一货币上 |
| [AdaNav](https://arxiv.org/abs/2509.24387) (2509.24387) | 2025.9 | Action Entropy作策略先验触发额外推理,heuristics-to-RL训练难度感知推理策略,视觉语言导航(VLN)场景 | 仅表面相似(不确定性触发推理这个大类里的一个具体实例),无跨步预算结转,风险低 |
| [RARRL: When Should a Robot Think?](https://arxiv.org/abs/2603.16673) (2603.16673) | 2026.3 | 分层orchestration policy(非底层控制)依据观测/执行历史/剩余资源决定是否/用哪种推理角色/分配多少算力,ALFRED基准真实延迟数据验证 | 资源感知调度框架和我们高度同构,但**不涉及世界模型/想象机制**,是通用推理调度而非想象门控,可作为"该不该多算"这一大类问题的方法论参照 |
| [Current Agents Fail to Leverage World Model as Tool for Foresight](https://arxiv.org/abs/2601.03905) (2601.03905) | 2026.1 | 诊断性研究:agent模拟触发率<1%、误用预测结果约15%、强制使用时性能反降最多5%,归因于"何时模拟/如何解读/如何整合"三方面能力缺失 | **极佳的motivation引用**,和AVIC自己的54/14/9/23%诊断数字互相印证,不构成方法竞争,应作为引言开篇证据 |
| [Active Inference as the Test-Time Scaling Law for Physical AI Agents](https://arxiv.org/abs/2606.22813) (2606.22813) | 2026.6 | 主动推断/自由能最小化框架,策略更新建模为软贝叶斯推断,scaling law的自变量是"连续现实世界经验"而非token/模型规模 | 提供的是"想多深"的连续调节机制(软性,非二元门控),和idea 3的VOC路线是**不同的理论传统**(active inference vs 决策论VOC),但同样证明"理论驱动而非RL/GRPO"这条路已经有人在邻近问题上摸索 |

### F2. VOC / 最优停止 / model-based 规划理论补强(10 篇)

> **关键发现**:我们pilot"发现一"(想象与基线同源时,多算期望上不改变决策,只注入方差,已用鞅的全方差公式加强证明)
> **不是新发现**——它是 Russell & Wefald(1991)Value of Computation 停止法则的直接推论,已被下面
> Hay et al.(UAI 2012,已在我们自己 E 组文献库里)形式化到 Bayesian selection 问题层级。"发现二"
> (无关随机性不能翻正)同样是 VOC 的直接推论。这两条应该被重新定位成"用严谨证明把 30 年前的理论
> 在'测试时想象+世界模型'这个具体现代场景钉实",而不是"我们发现了什么新东西"。

| 论文 | 年份 | 核心机制 | 与我们问题的相关性 |
|---|---|---|---|
| [Cognitive Friction](https://arxiv.org/abs/2603.30031) (2603.30031) | 2026.3 | 三元认知架构(TCA),把审议过程建模为belief-congestion联合状态上的随机控制,rollout近似计算信息价值,HJB启发的停止边界决定何时停止查询、采取行动 | **理论驱动(非RL/GRPO)门控**目前唯一命中的2026年工作,但应用于通用工具调用agent,不是world model想象——这正是竞争格局扫描独立指出的"所有8+篇竞品共享的方法论盲区"里,唯一已经有人开始碰但还没碰到我们具体场景的例子 |
| [Sezener & Dayan: Static and Dynamic Values of Computation in MCTS](https://arxiv.org/abs/2002.04335) (2002.04335) | 2020 UAI | 直接量化每次MCTS模拟对最终决策质量的预期影响(含非即时/未来模拟的影响),特定假设下贪婪优化计算价值即最优 | **和我们的Monte Carlo想象rollout设定几乎是同一数学对象**,是VOC理论应用到"该模拟哪个候选、模拟多少次"这个具体问题的现成先例,idea 1/8如果要做理论保证可直接站在这篇肩膀上 |
| [Value Equivalence Principle](https://arxiv.org/abs/2011.03506) (2011.03506) | 2020 NeurIPS | 模型不需要精确复现真实转移,只要在给定策略/价值函数集合下"value-equivalent"即可支撑规划,可用更简单的模型不损失性能 | 从另一个角度支撑我们pilot"想象和基线同源模型下无益"这条结论——如果想象用的模型和基线在value-equivalence意义上就是同一个模型,自然不会带来新信息,这是发现一的一个更一般的理论视角 |
| [Beyond the One-Step Greedy Approach](https://arxiv.org/abs/1802.03654) (1802.03654) | 2018 ICML | 首次系统分析h-step贪婪策略改进(此前无人仔细分析过),推导新算法并证明收敛性,指出多篇著名RL算法是其框架特例 | 我们pilot里"H越深、噪声越多"这个观测和h-step贪婪的理论分析同源,可用于把 Bellman telescoping 论证放进更大的经典理论脉络里 |
| [Multiple-Step Greedy Policies in Online and Approximate RL](https://arxiv.org/abs/1805.07956) (1805.07956) | 2018 NeurIPS | 同一批作者对h-step贪婪的在线/近似RL版本扩展分析(未逐字核验,建立在1802.03654同一脉络) | 与上一条同一脉络的后续工作,补充多步贪婪理论谱系 |
| [Hamrick et al.: On the role of planning in model-based deep RL](https://arxiv.org/abs/2011.04021) (2011.04021) | 2020 | 实证发现planning主要收益在训练/学习阶段而非部署阶段,"planning本身不保证泛化"(未逐字核验,agent二手转述) | 从实证角度呼应"想象要有用,必须真的带来训练时没有的新信息",而不是部署时白算 |
| [When Can Model-Free RL be Enough for Thinking?](https://arxiv.org/abs/2506.17124) (2506.17124) | 2025 | 把"thinking"形式化为"执行一步policy improvement",证明其涌现依赖policy初始化是否已编码未被利用的知识(未逐字核验,agent二手转述) | 提供"何时额外计算有用"的一个形式化视角,和我们"信息优势"框架呼应,可作为idea 3理论部分的候选支撑 |
| [Metacontrol for Adaptive Imagination-Based Optimization](https://arxiv.org/abs/1705.02670) (1705.02670) | 2017 DeepMind | "每个决策点学习投入多少想象"这条脉络的早期鼻祖之一(未逐字核验,agent二手转述,但PDF已下载确认真实存在) | Finding the Time to Think论文自己引用的最近邻之一,写related work时补上这条历史脉络显得调研扎实 |
| [Thinker: Learning to Plan and Act](https://arxiv.org/abs/2307.14993) (2307.14993) | 2023 NeurIPS | 同一条"元控制学习何时/怎么想象"脉络的另一个节点(未逐字核验,agent二手转述,PDF已下载确认真实存在) | 同上,历史脉络补全 |
| [LLMs Cannot Self-Correct Reasoning Yet](https://arxiv.org/abs/2310.01798) (2310.01798) | 2023(ICLR 2024) | 实证证明无外部反馈时intrinsic self-correction不可靠、甚至降低表现(未逐字核验,agent二手转述,PDF已下载确认真实存在) | 跨领域(LLM推理而非MDP决策)呼应发现一"没有新信息,多算不会变好",可作related work里的跨学科佐证 |

**F 组核验状态说明**:F1 全部10篇 + F2 前4篇(Cognitive Friction/Sezener-Dayan/Value Equivalence/
Beyond-One-Step-Greedy)已用 WebFetch 逐篇核对标题、作者、核心机制原文;F2 后6篇(Multiple-Step-Greedy/
Hamrick/Hanna-Corrado/Metacontrol/Thinker/LLMs-Cannot-Self-Correct)来自子agent调研报告、未逐篇二次
核验,但全部已实际下载 PDF 并通过 %PDF magic bytes + 大小校验(见 `download_papers.py` 输出,20/20 成功,
排除了 arXiv ID 完全不存在这种最严重的错误)——**正式写进论文前仍需对这6篇做一次逐字核验**,已在
[`03-novelty-competitive-landscape.md`](../03-novelty-competitive-landscape.md)里明确标注这个待办。

---

## 跨 track 交叉关联小结

- **Genie / GameNGen**(A↔B):既是 world model 又是视频生成架构,是"想象"这一生成动作本身成本和能力上限的共同参照。
- **AVIC**(A↔D):唯一同时出现在两组的 2026 年工作,是目前结构上离本项目 controller 最近的现有系统,写作时是最重要的正面对比对象。
- **BALD**(D↔E):互信息 acquisition function 既是不确定性量化工具也是元推理框架下"信息价值"的具体计算方式,是连接 D 组工具箱和 E 组理论锚点的桥梁。
- **Scaling LLM Test-Time Compute Optimally**(C↔E):C 组里最贴近"按难度自适应分配预算"目标的工程范式,同时也是 E 组 compute-optimal 分配理论最直接的实证案例。
- **F 组与 A/D/E 组**:F1(2026年直接竞品)是 A 组"AVIC 是结构上最近的现有工作"这条判断的最新延伸——
  AVIC 已经不是孤例,而是一个正在快速拥挤的赛道(Astra/ITP/ELASTIC/Finding-the-Time-to-Think/
  RARRL 等);F2(理论补强)直接指向 E 组的核心论断需要收紧——我们 pilot 的"发现一/二"是 E 组
  Russell & Wefald VOC 理论(尤其 [Selecting Computations](https://arxiv.org/abs/1207.5879),已在
  E 组)的推论而非新发现,但"所有 F1 竞品都用 RL/GRPO 经验式门控、没有一篇用有理论保证的 VOC/最优停止
  框架"这一条,是竞争格局扫描独立确认的方法论空当,详见 [`../02-deep-gap-analysis.md`](../02-deep-gap-analysis.md)。
