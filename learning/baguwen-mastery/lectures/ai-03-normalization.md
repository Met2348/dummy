# AI-03 · 归一化

## 这一类在面试里的分量

归一化看起来是"小知识点",但恰恰是面试官用来快速甄别"你是真读过 Transformer 内部结构,还是只会调 `nn.LayerNorm()`"的高效考点——因为它体量小、追问空间却很大:从"BN/LN 区别"这种入门题,几层追问就能钻到"为什么 Post-LN 需要 warmup 而 Pre-LN 不需要"这种要看过论文才答得上来的深度。国内大厂 NLP/推荐算法岗喜欢从"BN 和 LN 区别"起步考基础;海外 Lab 的研究向面试更喜欢直接问"为什么 LLaMA 用 RMSNorm、用 Pre-Norm"——因为这类问题能一次性检验你对现代 LLM 架构选型的理解深度,而不只是背书。这类题目的通用规律是:**"是什么"都很好答,"为什么这么设计"才是采分点**。

## 追问链深挖一:Transformer 为什么放弃 BatchNorm

> **Q1(基础问题):** 为什么 Transformer 用 LayerNorm 而不是 BatchNorm?

**A1:** 主要是三个工程问题。第一,变长序列要靠 padding 对齐成定长 batch,BN 是按特征维在整个 batch(含所有位置)上算统计量,padding token 的激活值会被一起算进均值/方差里,污染真实 token 的统计量,而且不同 batch 里 padding 比例还不一样,统计量本身不稳定。第二,自回归推理(逐 token 生成)经常是单样本甚至逐步只算一个新 token 的激活,batch 维退化到 1,BN 在这种小 batch 下统计量完全不可靠。LayerNorm 对每个样本、每个位置独立按特征维归一化,完全不依赖 batch 里其他样本或位置,天然免疫这几个问题。

> **面试官追问 Q2:** 那如果我在 CV 里用 Transformer(比如 ViT)做检测这种 batch size 天然很小的任务,是不是也应该照搬 LayerNorm,还是有别的选择?

**A2:** ViT 类模型确实沿用了 LayerNorm,这是对的。但如果讨论的是 CV 里的卷积网络(不是 Transformer)在小 batch 场景下要不要用 BN 的替代品,更常见的选择是 GroupNorm 而不是直接搬 LayerNorm——GroupNorm 把通道分成若干组,只在同一组通道内部算统计量,同样不依赖 batch,但保留了"按通道分组"这个更贴近卷积特征结构的先验,在检测/分割这类显存吃紧、batch size 被迫很小的任务上,GroupNorm 通常比直接把 LayerNorm 套到卷积特征图上表现更好、也比小 batch 下的 BN 稳定得多。

> **面试官再追问 Q3:** GroupNorm 的分组数一般怎么定,和 InstanceNorm 是什么关系?

**A3:** GroupNorm 的组数是一个需要根据通道数和任务调的超参,常见取值是 32(原论文的默认设置),组数越多越接近 LayerNorm(每组通道少、归一化粒度细),组数为 1 时就是对所有通道一起算,组数等于通道数时就退化成 InstanceNorm——对每个样本的每个通道单独算统计量。InstanceNorm 因为会抹掉每张图片自己的整体风格/对比度信息,这个特性在风格迁移、图像生成任务里反而有用,因为这些任务恰好希望把"内容"和"风格统计量"解耦开来;但在分类/检测这类需要保留通道间统计关联的任务上,InstanceNorm 通常不如折中的 GroupNorm。

*(完整答案见 `src/ai_qa/qbank_normalization.py` 的 `ai-norm-02` / `ai-norm-06`。)*

## 追问链深挖二:Pre-Norm、Post-Norm、DeepNorm、RMSNorm 怎么组合

> **Q1(基础问题):** Pre-Norm 和 Post-Norm 哪个训练更稳定,为什么?

**A1:** Xiong 等人用均值场理论分析发现,Post-LN(归一化放在残差相加之后)在初始化时,最后一层参数的梯度期望范数和网络深度无关——顶层从一开始就承受和浅层一样"猛"的梯度,如果学习率一开始就设得大,极容易训练不稳定,这正是 Post-LN 必须要 warmup 的理论根源。Pre-LN(归一化放在残差分支内部)则相反,各层梯度范数会被大约 1/√L 压小,深度越深梯度自然越小、更新天然温和,因此可以安全去掉 warmup。

> **面试官追问 Q2:** 那既然 DeepNorm 号称能兼顾"Post-LN 的效果"和"Pre-LN 的稳定性",为什么现在主流大模型(GPT/LLaMA 系列)还是用 Pre-LN,而不是都换成 DeepNorm?

**A2:** DeepNorm 主要解决的是"把 Transformer 堆到上百甚至上千层"这个极端场景下的稳定性问题——论文里验证到 1000 层。但目前主流 LLM 的层数大多在几十到一二百层这个区间,Pre-LN 在这个深度范围内已经足够稳定、不需要额外的残差缩放系数和专门推导的初始化,工程实现也更简单。DeepNorm 引入的额外超参(层数相关的放大系数 α、初始化缩放系数 β)增加了实现和调试成本,在还没触及 Pre-LN 稳定性上限的深度下,收益不足以覆盖这个复杂度增量,所以大多数团队选择"够用就好"的 Pre-LN,只有真正要冲极端深度时才会考虑 DeepNorm 这类方案。

> **面试官再追问 Q3:** RMSNorm 和 Pre-Norm/Post-Norm 是同一个维度的选择吗,LLaMA 里两者是怎么组合的?

**A3:** 不是同一个维度,是两个正交的独立决策,可以自由组合。Pre-Norm/Post-Norm 决定的是"归一化层相对残差连接放在哪个位置";RMSNorm/LayerNorm 决定的是"归一化本身要不要做 re-centering(减均值)"——RMSNorm 只做 re-scaling(按均方根缩放),舍弃了 re-centering,因为 Transformer 里大量的残差连接和线性投影本身就有能力吸收均值偏移。LLaMA 系列采用的是"Pre-Norm 结构 + RMSNorm 统计量"的组合(即 Pre-RMSNorm),两个决策各自独立、互不冲突,你完全可以见到 Post-LayerNorm、Pre-RMSNorm 之类的任意搭配。

*(完整答案见 `src/ai_qa/qbank_normalization.py` 的 `ai-norm-04` / `ai-norm-08` / `ai-norm-03`。)*

## 其余题目速览

- **BN 训练/推理行为差异 + batch=1 问题** —— 推理用移动平均而非当前 batch 统计量,batch=1 时方差恒为 0、统计量失效。完整答案见 `ai-norm-01`。
- **RMSNorm 省了什么** —— 省掉 re-centering(去均值),Transformer 里残差/投影足以吸收均值偏移。完整答案见 `ai-norm-03`。
- **归一化为什么能加速训练** —— 从"减少内部协变量偏移"(Ioffe & Szegedy)到"损失面平滑化"(Santurkar 等,更站得住脚)。完整答案见 `ai-norm-05`。
- **gamma/beta 仿射参数的作用** —— 让网络能"学回"归一化前的尺度/偏移,去掉会限制表达能力。完整答案见 `ai-norm-07`。
- **LayerNorm 具体归一化哪个维度** —— 对每个 (batch, seq_len) 位置的 hidden_dim 向量内部归一化,和 BN 跨样本对齐同一通道正好相反。完整答案见 `ai-norm-09`。
- **Weight Normalization vs 激活归一化** —— 归一化的是权重(w = g/‖v‖·v)而不是激活值,不依赖 batch 但加速效果通常不如 BN/LN。完整答案见 `ai-norm-10`。

## 易错点 / 常见误区清单

1. **只会背"BN 沿 batch 维、LN 沿特征维"却说不出为什么**——一定要能说出 padding/变长/自回归推理这几个具体工程原因,这是面试官判断你是否真懂的关键。
2. **以为 RMSNorm 只是"图省事"**——它是有实验依据的(re-centering 贡献有限),不是随便简化。
3. **把 Pre-Norm/Post-Norm 和 RMSNorm/LayerNorm 混为一谈**——两者是正交的独立选择,LLaMA 是 Pre-Norm + RMSNorm 的组合,不要以为"用了 RMSNorm 就等于 Pre-Norm"。
4. **认为"内部协变量偏移"仍是 BN 有效的标准解释**——Santurkar 等人已经用实验反驳了这个解释,更站得住脚的是损失面平滑化(Lipschitz/β-平滑性),面试官如果读过论文会追问这个点。
5. **忽视 gamma/beta 的作用**——只答归一化的均值方差部分,漏掉可学习仿射参数,等于漏掉了归一化层"保留表达能力"的关键设计。
6. **认为归一化方案之间是互斥的单选题**——GroupNorm/InstanceNorm/RMSNorm/Weight Norm 都是在"沿什么维度、归一化什么对象"这个设计空间里的不同取舍,适合什么场景要具体分析,不是非此即彼。
