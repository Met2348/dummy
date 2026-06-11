# Derivative-Free Guidance in Continuous and Discrete Diffusion Models with Soft Value-Based Decoding

中文深度面试导读。术语第一次出现时保留英文括号。目标：读完这份 deck，能回答 methodology 和 experiments 的追问。

原论文：`../papers/01_derivative_free_guidance.pdf`  
arXiv：https://arxiv.org/abs/2408.08252  
官方代码：https://github.com/masa-ue/SVDD

---

## Slide 01 - 一句话 thesis

- 这篇论文提出 Soft Value-Based Decoding in Diffusion Models，简称 SVDD。
- 它解决的问题是：已有预训练扩散模型（pre-trained diffusion model）会生成“自然”的样本，但我们还希望样本在某个下游奖励函数（reward function）上更高。
- 传统 classifier guidance 需要可微代理模型（differentiable proxy model），而很多科学奖励不可微，比如分子 docking、RDKit 指标、物理模拟。
- 传统 fine-tuning 或 RL fine-tuning 需要重新训练模型，成本高且可能破坏原模型的自然性（naturalness）。
- SVDD 的核心想法：在采样时，每一步从预训练模型多采几个候选 noisy state，用 soft value function 预测这些中间状态未来能带来多高奖励，然后选更好的候选。
- 面试时可以说：它把“末端 reward optimization”变成“每个 denoising step 的 look-ahead selection”。

---

## Slide 02 - 这篇论文在三篇中的位置

- 第三篇 MDLM 解释 masked diffusion language model 为什么可行。
- 第二篇 Discrete Guidance 解释如何在离散状态空间（discrete state-space）里做 principled guidance。
- 这一篇 SVDD 关心的是更广泛的 reward optimization：连续图像、离散序列、分子图都可以。
- 它和第二篇的共同点：都想绕开连续空间 gradient guidance 的限制。
- 它和第二篇的差异：Discrete Guidance 修改 CTMC rate；SVDD 不直接推 guided rate，而是用 value-based candidate selection。
- 这一篇特别适合面试，因为它把 diffusion、KL regularization、soft value、importance sampling、实验设计全串起来。

---

## Slide 03 - 最基础问题：扩散模型在干什么

- 扩散模型（diffusion model）有两个方向。
- 正向过程（forward process）把干净样本 `x_0` 逐步加噪到 `x_T`。
- 反向过程（reverse process / denoising process）学习从 `x_t` 逐步还原到 `x_0`。
- 训练好的模型给出每一步条件分布 `p_pre_t(x_{t-1} | x_t)`。
- 采样时从噪声 `x_T` 出发，反复调用 denoising transition，最后得到样本 `x_0`。
- 论文把预训练扩散模型诱导出的最终分布记为 `p_pre(x_0)`。
- SVDD 的目标不是重新学 `p_pre`，而是在采样时偏向高 reward 的轨迹。

---

## Slide 04 - 为什么不能只用 Best-of-N

- 最直接的方法是 Best-of-N：生成很多完整样本，最后选 reward 最高的。
- 问题是它太浪费，因为 reward 只在末端用一次。
- 如果一个早期 noisy state 已经很可能通向低 reward 的结果，Best-of-N 仍然要把整条 denoising 轨迹跑完。
- SVDD 的想法是“提前看未来”：在每一步 `t`，对候选 `x_{t-1}` 估计它未来通向高 reward 的可能性。
- 因此 SVDD 使用中间状态 value function，而不是只在最终样本上排序。
- 面试回答可以类比搜索：Best-of-N 是随机试完整路径，SVDD 是每一步用启发式评估（heuristic evaluation）剪枝。

---

## Slide 05 - 优化目标：高 reward 但别离开自然分布

- 论文把目标分布写成一个 KL-regularized objective。
- 想要的分布 `p_alpha(x)` 最大化：
  - 期望奖励 `E_p[r(x)]`
  - 减去 `alpha * KL(p || p_pre)`
- 直觉：reward 越高越好，但 `p` 不能离预训练模型太远。
- 解的形式是：
  - `p_alpha(x) proportional exp(r(x) / alpha) * p_pre(x)`
- `alpha` 是温度或正则强度（temperature / regularization strength）。
- `alpha` 小：更贪心，更偏向高 reward。
- `alpha` 大：更接近原始预训练分布。
- 论文实验中常用 `alpha = 0` 的贪心选择近似。

---

## Slide 06 - KL 正则的面试解释

- KL 散度（KL divergence）衡量两个分布之间的差异。
- 这里用 `KL(p || p_pre)` 惩罚新分布离预训练分布太远。
- 为什么要惩罚？
  - 图像里防止 reward hacking，生成不自然但分数高的图。
  - 分子里防止生成化学无效结构。
  - DNA/RNA 里防止序列跑到生物学上不合理区域。
- 这也是为什么论文强调 preserving naturalness。
- 面试时可以说：SVDD 不是无约束优化器，而是 reward optimization under a generative prior。

---

## Slide 07 - Soft value function 是什么

- 软价值函数（soft value function）定义在中间 noisy state 上。
- 记 `v_t(x_t)` 表示：从当前 `x_t` 继续按预训练模型采样，最终 `x_0` 的 exponentiated reward 的 log-expectation。
- 直观写法：
  - `v_t(x_t) = alpha * log E_{x_0 ~ p_pre(.|x_t)}[exp(r(x_0)/alpha)]`
- 它回答的问题是：如果我现在处在这个 noisy state，未来平均能到多好的 reward？
- 这不是普通 value function，而是 entropy-regularized / soft value。
- 当 `alpha` 很小时，它更接近“未来最高 reward”。
- 当 `alpha` 较大时，它更接近“未来平均 reward”。

---

## Slide 08 - soft optimal policy 的核心公式

- 论文定义每一步的 soft optimal policy：
  - `p*_t(x_{t-1}|x_t) proportional p_pre_t(x_{t-1}|x_t) * exp(v_{t-1}(x_{t-1}) / alpha)`
- 这句话很重要。
- 预训练模型提供候选分布 `p_pre_t`。
- `exp(value / alpha)` 重新加权候选。
- value 高的候选更可能被选中。
- 如果每一步都能从这个 soft optimal policy 采样，那么最终样本就来自目标分布 `p_alpha(x)`。
- 这个结论来自 Uehara et al. 的定理，论文把它转化成 inference-time algorithm。

---

## Slide 09 - 为什么 soft optimal policy 难直接采样

- 公式看起来简单，但有两个实际困难。
- 第一，`v_t` 不知道，需要估计。
- 第二，分母归一化常数（normalizing constant）很难算。
- 连续空间里积分难；离散序列空间里枚举所有 token 组合也难。
- SVDD 的处理方式：
  - value function 用 Monte Carlo regression 或 posterior mean approximation 得到。
  - 归一化采样用 importance sampling 近似。
- 这就是 SVDD 的两个核心工程点。

---

## Slide 10 - SVDD 算法直觉

- 在每一步 `t`：
  1. 从预训练模型 `p_pre_t(.|x_t)` 采样 `M` 个候选 `x_{t-1}^{(m)}`。
  2. 对每个候选计算 value `v_{t-1}(x_{t-1}^{(m)})`。
  3. 根据 `exp(value / alpha)` 形成权重。
  4. 从这些候选中按权重选一个，作为下一步状态。
- 当 `alpha = 0` 时，近似变成直接选 value 最大的候选。
- `M` 是 duplication size，也就是每步候选数量。
- 推荐范围通常是 `5` 到 `20`。
- 面试里要强调：这是 per-step look-ahead，不是 full-sample reranking。

---

## Slide 11 - 这为什么叫 derivative-free

- 算法只需要 reward value 或 value estimate。
- 不需要对输入 `x_t` 求梯度。
- 所以 reward model 可以不可微（non-differentiable）。
- 分子任务可以直接用 RDKit 指标、QuickVina docking score。
- 生物序列任务可以用预测模型的输出，不要求对离散 token 做连续梯度。
- 图像 compressibility 是 JPEG 后文件大小，天然不可微，SVDD 也能用。
- 这与 classifier guidance / DPS 的核心区别是：后者通常需要 `grad_x log p(y|x_t)`。

---

## Slide 12 - SVDD-MC：用 Monte Carlo regression 学 value

- SVDD-MC 需要训练一个 value function model。
- 做法：
  1. 用预训练扩散模型生成很多完整轨迹 `x_T, ..., x_0`。
  2. 对每条轨迹的最终样本计算 reward `r(x_0)`。
  3. 用监督学习训练 `f(x_t, t)` 去预测 `r(x_0)`。
- 损失是均方误差（MSE）：
  - minimize sum over trajectories and time of `(r(x_0) - f(x_t, t))^2`
- 它学习的是从中间状态看未来 reward 的回归模型。
- 好处：value 可以比直接 posterior mean 更准确。
- 风险：如果 value model 学不好，SVDD-MC 会受影响。

---

## Slide 13 - SVDD-PM：posterior mean approximation

- SVDD-PM 不额外训练 value model。
- 扩散模型通常已经有一个网络预测干净样本 `x_0`，可记为 `x_hat_0(x_t)`。
- 论文用近似：
  - `v_t(x_t) approx r(x_hat_0(x_t))`
- 也就是：把当前 noisy state 先映射到预测的干净样本，再直接计算 reward。
- 好处：
  - 不需要训练 value function。
  - 只要能调用 reward，就能用。
  - 对不可微 reward 友好。
- 坏处：
  - 如果 `x_hat_0` 在早期很粗糙，value 可能噪声较大。
- 实验中 SVDD-PM 很强，尤其因为它直接利用真实 reward feedback。

---

## Slide 14 - SVDD-MC 和 SVDD-PM 怎么比较

- SVDD-MC：
  - 需要额外 value training。
  - 能学习 expected future reward。
  - 如果 reward 计算很贵，训练一个 value proxy 可能节省推理时成本。
- SVDD-PM：
  - 不额外训练。
  - 直接用 reward。
  - 如果 reward 计算不太贵，通常更简单稳健。
- 论文结论是二者表现有 domain-dependence。
- 面试回答：SVDD-MC 的瓶颈是 value learning quality，SVDD-PM 的瓶颈是 posterior mean approximation quality 和 reward evaluation cost。

---

## Slide 15 - 和 classifier guidance 的区别

- Classifier guidance 在连续扩散里通常修改 reverse dynamics 的均值或 score。
- 它需要 `grad_x log p(y|x_t)`。
- 这要求 predictor 可微，并且输入空间连续或可被连续松弛。
- SVDD 不修改 gradient，不需要 score correction。
- SVDD 从原模型候选中选择，因此天然更贴近 `p_pre`。
- 对离散扩散模型（discrete diffusion model），SVDD 不需要把 token 空间强行变成连续空间。
- 面试里可以说：classifier guidance is gradient-based dynamics steering; SVDD is value-weighted proposal selection。

---

## Slide 16 - 和 DPS 的区别

- DPS（Diffusion Posterior Sampling）是常见 training-free guidance 方法。
- DPS 也常用 `x_hat_0(x_t)` 近似干净样本，再对 measurement/reward 求梯度。
- SVDD-PM 和 DPS 都使用 posterior mean 的直觉。
- 但最终操作不同：
  - DPS 计算 gradient，修改 denoising update。
  - SVDD-PM 计算 reward value，选择候选。
- 所以 SVDD-PM 可以处理不可微 reward。
- 这也是论文 Remark 2 的核心。

---

## Slide 17 - 和 SMC 的区别

- SMC（Sequential Monte Carlo）也在 diffusion sampling 中使用权重和重采样。
- 论文认为标准 SMC 对 reward maximization 不够合适。
- 标准 SMC 往往在整个 batch 上重采样。
- 小 batch 时理论优势不足；大 batch 时可能造成样本多样性下降。
- SVDD 是 per-sample basis：每个样本内部复制 `M` 个候选并选择，不在整个 batch 之间竞争。
- 这使 SVDD 更容易 parallelize，并且 batch size 为 1 也能工作。
- 论文附录指出 SVDD 与 nested importance sampling / nested-IS SMC 有联系，但实践行为不同于 standard SMC。

---

## Slide 18 - computational cost 怎么回答

- 每一步采样 `M` 个候选，因此如果串行计算，时间大约是标准采样的 `M` 倍。
- 如果并行计算，时间不一定线性增加，但显存可能增加。
- 论文主文中说候选计算可 parallelize；图 3 检查了 GPU time 和 memory。
- 文中关于图 3 的文字有一个需要谨慎表述的地方：正文写到 performance plateau，计算和显存复杂度随 `M` 的经验趋势；你面试时不要过度承诺“免费”。
- 稳妥回答：
  - SVDD 用推理时额外候选换取 reward improvement。
  - `M` 增大收益会 plateau。
  - 实际 cost 取决于是否 batch parallel、模型大小、reward 计算成本。

---

## Slide 19 - 实验总览

- 论文覆盖四类 domain：
  1. 图像（images）：Stable Diffusion v1.5。
  2. 分子（molecules）：GDSS on ZINC-250k。
  3. DNA enhancers：masked discrete diffusion model。
  4. RNA 5' UTR：masked discrete diffusion model。
- 关键 reward：
  - 图像 compressibility 和 aesthetic score。
  - 分子 QED、SA、docking score。
  - enhancer 活性预测。
  - 5' UTR 的 mean ribosomal load。
- Baselines：
  - Pre-trained model。
  - Best-of-N。
  - DPS。
  - SMC-based methods。
  - SVDD-MC 与 SVDD-PM。

---

## Slide 20 - 图像实验怎么理解

- 预训练模型：Stable Diffusion v1.5。
- reward 1：compressibility，定义为 JPEG 压缩后的负文件大小，越高表示越可压缩。
- reward 2：aesthetic score，使用 LAION Aesthetic Predictor。
- Compressibility 不可微，所以它很好地展示了 derivative-free 的价值。
- SVDD 在 compressibility 上相对 Best-of-N、DPS、SMC 都更强。
- Aesthetic 上 SVDD-PM 很强，SVDD-MC 不是所有 domain 都最强，说明 value model 学习质量重要。
- 面试可能问：为什么压缩分数高不一定等于好图？答：它是用于证明不可微 reward 可优化的任务，不能把它当作通用视觉质量指标。

---

## Slide 21 - 分子实验怎么理解

- 预训练模型：GDSS，训练在 ZINC-250k。
- reward：
  - QED（Quantitative Estimate of Drug-likeness）：药物相似性。
  - SA（Synthetic Accessibility）：合成可及性，论文归一化为 `(10 - SA) / 9`，越高越好。
  - Docking score：QuickVina 2 计算，针对 parp1、5ht1b、jak2、braf 四个蛋白靶点。
- 这些 reward 多数不可微，尤其 docking。
- 论文报告 top quantile，关注高 reward 样本的质量。
- 结果显示 SVDD 在 QED、SA、docking 上基本都明显强于 baseline。
- 还报告了 validity、uniqueness、novelty 等指标，说明不是单纯 reward hacking。

---

## Slide 22 - DNA/RNA 实验怎么理解

- DNA enhancer 任务：
  - 序列长度 200。
  - reward 是 HepG2 cell line 的 enhancer activity，由 Enformer 预测。
- RNA 5' UTR 任务：
  - 序列长度 50。
  - reward 是 mean ribosomal load，由 ConvGRU 预测。
- 这两个任务展示 SVDD 可以直接应用于离散扩散模型。
- 关键点：离散序列上 classifier guidance 的梯度不自然，而 SVDD 只需要候选和 value。
- 结果中 SVDD-PM 在 enhancer 和 5'UTR 上非常强。

---

## Slide 23 - Table 2 证据链

- Table 2 报告 top 50% 和 top 10% quantiles，越高越好。
- 这不是平均值，而是高 reward 样本的分位表现，适合评估 optimization。
- 代表性结果：
  - Image compressibility 50%：SVDD-PM `-51.1`，强于 Best-N `-71.2`、DPS `-60.1`、SMC `-59.7`。
  - Molecule QED 10%：SVDD-PM `0.928`，SVDD-MC `0.925`，Best-N `0.902`。
  - Molecule SA 10%：SVDD 达到约 `1.000`。
  - Docking parp1 10%：SVDD-MC `13.25`，高于 Best-N `10.67`。
  - Enhancers 10%：SVDD-PM `6.980`，高于 SMC `5.95` 和 DPS `4.879`。
  - 5'UTR 10%：SVDD-PM `1.383`，高于 Best-N `1.064`。
- 面试回答：Table 2 支持“reward optimization 更强”，后续 validity 指标支持“没有完全破坏 naturalness”。

---

## Slide 24 - naturalness / validity 的证据

- 论文不只看 reward，还看样本是否有效。
- 图像：展示生成样本，说明没有明显崩坏。
- 分子：附录 Table 3 比较 pre-trained GDSS 和 SVDD。
- 分子 validity：
  - Pre-trained `1.0`
  - SVDD `1.0`
- Novelty：
  - Pre-trained `1.0`
  - SVDD `1.0`
- Uniqueness：
  - Pre-trained `1.0`
  - SVDD `0.9375`
- 这些指标说明 SVDD 优化 reward 的同时保持较高化学有效性和新颖性。
- 但 uniqueness 有下降，说明强优化仍可能影响 diversity，这是可以主动指出的 limitation。

---

## Slide 25 - ablation on M

- `M` 是每步候选数量。
- 论文发现随着 `M` 增大，reward performance 会提升但逐渐 plateau。
- 这符合 intuition：
  - `M` 越大，越可能在候选里出现高 value state。
  - 但候选质量仍受 `p_pre` 支配，继续增大 `M` 的边际收益下降。
- 论文推荐通常 `M=5` 到 `20`。
- 具体实验：
  - 图像通常 `M=20`。
  - 其他 domain 通常 `M=10`。
- 面试回答：`M` 是 quality-cost trade-off，不能无限增大。

---

## Slide 26 - 为什么 SVDD 能同时支持 continuous 和 discrete

- 对 continuous diffusion，预训练模型能给出候选 `x_{t-1}`。
- 对 discrete diffusion，预训练模型也能给出候选 token/state。
- SVDD 只依赖“从预训练 transition 采样候选”以及“对候选打分”。
- 它不需要在状态空间里定义 gradient。
- 因此 continuous image、discrete DNA/RNA、molecule graph 都可以统一处理。
- 这是它比许多 gradient guidance 更 general 的关键。

---

## Slide 27 - 复现建议：先不要复现全部

- 全量复现实验非常重：
  - Stable Diffusion 图像任务。
  - RDKit + QuickVina docking。
  - Enformer / ConvGRU 序列任务。
- 面试准备建议先做最小复现。
- 最小目标 1：
  - 用官方 SVDD 代码跑一个 toy discrete diffusion 或小分子 QED/SA。
- 最小目标 2：
  - 只实现 Algorithm 1 的 candidate selection 伪代码。
- 最小目标 3：
  - 用一个简单 reward，比如 sequence 中某个 token 出现次数，观察 SVDD 比 Best-of-N 更早偏向高 reward。
- 这样能真正理解方法，不必一开始追求论文级结果。

---

## Slide 28 - 伪代码记忆版

```text
Given pretrained denoiser p_pre, reward r, value estimator v, candidates M
Initialize x_T from noise
For t = T, ..., 1:
    candidates = [sample p_pre(x_{t-1} | x_t) for m in 1..M]
    scores = [v_{t-1}(candidate) for candidate in candidates]
    if alpha == 0:
        x_{t-1} = candidate with max score
    else:
        weights = softmax(scores / alpha)
        x_{t-1} = sample candidate according to weights
Return x_0
```

- 面试时能把这个讲清楚，方法基本就过关一半。
- 重点是说明 `v` 是 look-ahead，不是当前 reward。

---

## Slide 29 - 数学基础补丁：Jensen 近似

- SVDD-MC 使用近似：
  - `v_t(x_t) = alpha log E[exp(r(x_0)/alpha) | x_t]`
  - 近似为 `E[r(x_0) | x_t]`
- 严格来说，`log E exp` 和 `E` 不是一样的。
- 直觉：
  - `log E exp` 是 soft max。
  - `E[r]` 是普通均值。
- 论文选择回归 `r(x_0)`，因为直接回归 `exp(r/alpha)` 在 `alpha` 小时数值爆炸。
- 面试可说：这是为了稳定训练 value function 的 practical approximation。

---

## Slide 30 - 可能问题：为什么不是直接优化 reward

- 因为直接优化 reward 会离开数据流形（data manifold）。
- 生成模型提供 prior，保证样本自然。
- KL 正则就是把 reward optimization 限制在 pre-trained distribution 附近。
- SVDD 通过从 `p_pre` 采样候选来隐式保持这个 prior。
- 如果候选都来自预训练模型，就不太可能一步跳到完全 OOD 的状态。
- 但多步 selection 仍可能积累偏差，所以需要 validity/naturalness 指标验证。

---

## Slide 31 - 可能问题：SVDD 会不会 reward hacking

- 会有风险，尤其 reward model 是 learned proxy 时。
- 论文用 KL regularization、pre-trained candidates、validity metrics 来缓解。
- 分子实验中 validity 和 novelty 保持高，但 uniqueness 有一定下降。
- 如果 reward 是有漏洞的 proxy，SVDD 仍可能找到 proxy 的漏洞。
- 面试回答应诚实：
  - SVDD 解决了不可微和无需 fine-tuning 的问题。
  - 它不自动解决 reward misspecification。
  - 对高风险科学设计，还需要 oracle validation 或 wet-lab / physics validation。

---

## Slide 32 - 可能问题：为什么 top quantile 而不是 average

- 任务是 optimization，不只是拟合分布。
- 如果目标是找高 reward candidate，top 10% 或 top 50% quantile 更相关。
- 平均值可能被大量普通样本稀释。
- 但只看 top quantile 也有风险：
  - 不反映 diversity。
  - 不反映 validity。
  - 不反映 distribution shift。
- 所以论文补充分子 validity、histogram、生成样本可视化。

---

## Slide 33 - 和第二篇 Discrete Guidance 的关系

- Discrete Guidance 修改的是 CTMC 的 rate matrix。
- 它要求有 predictor `p(y|x_t,t)` 或 conditional/unconditional rate。
- SVDD 不直接推导 guided rate，而是在原 transition 上做 value-weighted selection。
- Discrete Guidance 在离散 CTMC 框架下更“解析”。
- SVDD 更黑盒：只要能从 pre-trained model sample candidates、能 evaluate value/reward，就可用。
- 如果面试问三篇怎么串：
  - MDLM 给离散 diffusion LM 基础。
  - Discrete Guidance 给离散 guidance 的 rate view。
  - SVDD 给更通用的 derivative-free reward optimization。

---

## Slide 34 - 这篇论文的强点

- 方法简单，可解释，容易实现。
- 支持不可微 reward。
- 支持 continuous 和 discrete diffusion。
- 不需要 fine-tune generative model。
- 实验覆盖 domain 多：image、molecule、DNA/RNA。
- 和 Best-of-N、DPS、SMC 都有比较。
- 有对 `M` 的消融。
- 给出了 soft value 和 KL-regularized target distribution 的理论连接。

---

## Slide 35 - 这篇论文的弱点和可批判点

- 推理时成本增加，尤其大模型和昂贵 reward 下明显。
- SVDD-MC 依赖 value model 的学习质量。
- SVDD-PM 依赖 posterior mean approximation，早期 noisy state 可能不准。
- 如果 reward 是 proxy，仍有 reward hacking 风险。
- 图像 compressibility 这类 reward 证明了不可微，但不一定代表真实应用价值。
- 多 domain 实验很广，但每个 domain 的深度复现成本高。
- 与 SMC 的理论关系复杂，面试时不要说 SVDD 完全不是 SMC，而要说它不同于 standard batch-level SMC。

---

## Slide 36 - 面试 2 分钟讲法

- 这篇论文想解决 diffusion models 中的 reward-guided generation。
- 现有 classifier guidance 需要可微 predictor，fine-tuning 又贵，Best-of-N 又低效。
- 作者从 KL-regularized reward objective 出发，目标分布是 `exp(r/alpha) p_pre`。
- 根据 soft value theorem，如果每一步按 value-weighted denoising policy 采样，最终就会得到这个目标分布。
- 由于直接采样这个 policy 难，SVDD 每一步从预训练模型采 `M` 个候选，用 soft value function 打分，再按权重或贪心选一个。
- Value 可以通过 Monte Carlo regression 学，也可以用 posterior mean approximation 直接计算 reward。
- 因为只需要 reward value，不需要 gradient，所以可以处理不可微 reward 和离散 diffusion。
- 实验显示它在图像、分子、DNA/RNA reward optimization 上超过 Best-of-N、DPS、SMC，并且保持较好的 validity。

---

## Slide 37 - 高频问答清单

- Q: `alpha` 是什么？
  - A: 控制 reward optimization 和保持 pre-trained distribution 的 trade-off。
- Q: `M` 是什么？
  - A: 每个 denoising step 的候选数量，越大越强但成本更高，收益会 plateau。
- Q: SVDD-PM 为什么不训练？
  - A: 它用 denoiser 的 `x_hat_0(x_t)` 近似未来 clean sample，再直接算 reward。
- Q: 为什么 derivative-free？
  - A: 因为选择候选只需要 value/reward，不需要 `grad_x`。
- Q: 为什么比 Best-of-N 高效？
  - A: 它每一步用 look-ahead value 引导路径，而不是末端才筛选。
- Q: 最大 limitation？
  - A: 推理成本、value 估计误差、reward misspecification。

---

## Slide 38 - 术语表

- 预训练扩散模型（pre-trained diffusion model）：已经学会生成自然样本的 diffusion model。
- 奖励函数（reward function）：衡量样本目标属性的函数。
- 自然性（naturalness）：样本仍像训练数据分布中的真实样本。
- KL 正则（KL regularization）：限制新分布不要离原分布太远。
- 软价值函数（soft value function）：中间状态未来 reward 的 soft look-ahead estimate。
- 重要性采样（importance sampling）：用 proposal sample 加权近似目标分布。
- 后验均值近似（posterior mean approximation）：用 `x_hat_0(x_t)` 近似未来 clean sample。
- 不可微奖励（non-differentiable reward）：不能对输入求梯度的 reward，如 docking 或压缩大小。
