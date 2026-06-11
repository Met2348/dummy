# Unlocking Guidance for Discrete State-Space Diffusion and Flow Models

中文深度面试导读。术语第一次出现时保留英文括号)。目标：能从连续 guidance 讲到离散 CTMC rate，再讲清实验。

原论文：`../papers/02_unlocking_guidance_discrete_diffusion_flow.pdf`  
arXiv：https://arxiv.org/abs/2406.01572

---

## Slide 01 - 一句话 thesis

- 这篇论文提出 Discrete Guidance。
- 它解决的问题是：连续扩散模型（continuous diffusion model）里 guidance 很成熟，但离散状态空间（discrete state-space）里没有自然的 score gradient。
- 作者发现：如果离散扩散/离散 flow 模型用连续时间马尔可夫链（continuous-time Markov chain, CTMC）表示，那么每个瞬间最多只改一个维度。
- 这个性质把原本指数级的归一化问题变成 `D * (S - 1) + 1` 个候选。
- 于是可以用 Bayes rule 在 rate matrix 上做精确的 predictor guidance。
- 核心公式：guided rate = unconditional rate 乘以 predictor likelihood ratio。

---

## Slide 02 - 为什么这篇和面试非常相关

- 老师明确会问 diffusion methodology 和 experiments。
- 这篇论文的方法部分很数学：CTMC、rate matrix、Bayes rule、normalization。
- 实验部分很广：小分子 SMILES、DNA enhancer、protein inverse folding，还有 appendix 中 CIFAR-10 discrete image。
- 它很可能被用来考你是否真正理解“离散 diffusion 为什么不能照搬连续 guidance”。
- 你需要能回答：
  - 为什么 discrete score 不自然？
  - 为什么 CTMC 让 guidance tractable？
  - Exact guidance 和 Taylor-approximated guidance (TAG) 各是什么？
  - 实验指标 MAE、FBD、success rate 分别说明什么？

---

## Slide 03 - 连续空间 classifier guidance 复习

- 连续 diffusion 常用 score function：
  - `score = grad_x log p_t(x)`
- 如果想生成满足条件 `y` 的样本，可以写：
  - `grad_x log p_t(x|y) = grad_x log p_t(x) + grad_x log p_t(y|x)`
- 第一项来自 unconditional diffusion model。
- 第二项来自 classifier / predictor。
- 这就是 classifier guidance 的基本逻辑。
- 问题：离散 token、SMILES 字符、DNA base、protein amino acid 没有自然连续坐标。
- 对离散 one-hot 求梯度往往只是近似，不是原生的概率过程。

---

## Slide 04 - 离散 guidance 的核心难点

- 想用 Bayes rule：
  - `p(x_{t+dt} | x_t, y) proportional p(y | x_{t+dt}, x_t) p(x_{t+dt} | x_t)`
- 如果一个离散状态有 `D` 个位置，每个位置 `S` 种取值。
- 从一个状态到任意另一个状态理论上有 `S^D` 种可能。
- 归一化常数要对所有 next states 求和，指数级不可行。
- 连续空间用 gradient 绕过 normalizing constant。
- 离散空间没有原生 gradient，所以必须找到另一种可行结构。

---

## Slide 05 - CTMC 是什么

- 连续时间马尔可夫链（continuous-time Markov chain, CTMC）是在离散状态空间上连续时间演化的随机过程。
- 状态会保持一段时间（holding time），然后跳到另一个状态（jump）。
- 跳转由速率矩阵（rate matrix）`R_t(x, x')` 控制。
- 对很小时间 `dt`：
  - `p(x_{t+dt}=x' | x_t=x) = delta_{x,x'} + R_t(x,x') dt`
- 非对角元素 `R_t(x,x') >= 0` 表示跳到别的状态的速率。
- 对角元素由概率守恒决定：
  - `R_t(x,x) = - sum_{x' != x} R_t(x,x')`

---

## Slide 06 - CTMC 为什么让离散 diffusion/flow 可行

- CTMC 的关键性质：在无穷小时间 `dt` 内，同时两个维度跳变的概率是 0。
- 因此一个状态 `x` 的可达 next states 不是所有 `S^D` 个状态。
- 可达状态只有：
  - 保持原样的 identity transition。
  - 在 `D` 个维度中选一个维度，并改成其它 `S-1` 个值。
- 总数是 `D * (S - 1) + 1`。
- 这个数量是线性的，不是指数的。
- 这就是论文说“continuous time unlocks tractability”的核心。

---

## Slide 07 - CTDD 和 DFM 背景

- 连续时间离散扩散（continuous-time discrete diffusion, CTDD）用 CTMC 做离散数据 diffusion。
- 离散 flow model（discrete flow model, DFM）也可以用 CTMC 表示生成过程。
- 两者都训练 denoising model `p_theta(x_1 | x_t)`。
- 这个 denoising model 可以诱导一个 rate matrix。
- 采样时从噪声分布出发，沿着时间积分 rate matrix，逐步生成目标样本。
- 论文的 Discrete Guidance 可以用于 CTDD、DFM，以及其它 CTMC-based discrete generative models。

---

## Slide 08 - predictor guidance 的核心公式

- 对目标条件 `y`，作者推导 guided conditional rate：
  - `R_t(x, x' | y) = [p(y | x', t) / p(y | x, t)] * R_t(x, x')`
  - 其中 `x' != x`
- 直觉：
  - 如果跳到 `x'` 以后，predictor 认为 `y` 更可能，则 rate 增大。
  - 如果跳到 `x'` 后 `y` 更不可能，则 rate 减小。
- 对角项仍由概率守恒决定：
  - `R_t(x,x|y) = - sum_{x' != x} R_t(x,x'|y)`
- 这是离散版 classifier guidance 的核心。

---

## Slide 09 - guidance strength gamma

- 和连续 guidance 类似，可以引入 guidance strength `gamma`。
- 公式变成：
  - `R_t^gamma(x,x'|y) = [p(y|x',t) / p(y|x,t)]^gamma * R_t(x,x')`
- `gamma = 0`：回到 unconditional model。
- `gamma = 1`：标准 conditional rate。
- `gamma > 1`：更强地推向条件 `y`。
- 过大的 `gamma` 可能带来 diversity 下降、validity 下降或 predictor 误差放大。
- 实验中小分子、DNA、蛋白都会调 guidance strength。

---

## Slide 10 - predictor-free guidance

- Predictor-free guidance (PFG) 不使用单独 predictor。
- 它需要一个 unconditional rate `R_theta(x,x')` 和一个 conditional rate `R_phi(x,x'|y)`。
- 公式类似 classifier-free guidance：
  - `R_t^gamma(x,x'|y) = R_t(x,x'|y)^gamma * R_t(x,x')^(1-gamma)`
- 直觉：
  - 在 rate 空间中混合 conditional 和 unconditional model。
- 当 `gamma = 1` 是普通 conditional model。
- 当 `gamma = 0` 是 unconditional model。
- 论文同时给出 predictor guidance (PG) 和 predictor-free guidance (PFG)，但实验主要强调 PG。

---

## Slide 11 - Exact guidance 的计算成本

- Exact guidance 需要对当前状态所有可能 one-jump next states 计算 predictor likelihood。
- 数量是 `D * (S - 1) + 1`。
- 对中等 `D` 和 `S` 可行。
- 例如 DNA alphabet 小，蛋白 `S=20`，某些任务还可以。
- 但长序列或大词表时，`D*S` 次 predictor forward 仍然很贵。
- 因此论文提出 Taylor-approximated guidance。

---

## Slide 12 - TAG 是什么

- TAG = Taylor-approximated guidance。
- 它使用 predictor 的一阶泰勒近似（first-order Taylor approximation）。
- 目标是近似：
  - `log p(y|x',t) - log p(y|x,t)`
- 用梯度近似：
  - `(x' - x)^T grad_x log p(y|x,t)`
- 这样只需要一次 forward 和一次 backward，而不是对所有 next states 做 forward。
- 复杂度从 `O(D*S)` predictor calls 变成近似 `O(1)` predictor pass。
- 注意：这是近似，虽然在离散 one-hot 上使用连续梯度，但只用于估计 likelihood ratio，不是直接把状态变成连续采样。

---

## Slide 13 - TAG 和 DiGress 的差异

- DiGress 也是离散图生成里的 guidance 方法，也使用类似 Taylor idea。
- 但 DiGress 基于离散时间离散扩散（discrete-time discrete diffusion）。
- 离散时间下从 `x_t` 到 `x_{t+dt}` 可能同时改变多个维度，所以 Bayes denominator 仍然难。
- 因此 DiGress 需要更多近似。
- Discrete Guidance 基于 CTMC，先用“单维跳变”让 exact guidance 可行，再用 TAG 做效率近似。
- 面试时要说：TAG 和 DiGress 表面相似，但理论基底和可精确版本不同。

---

## Slide 14 - Algorithm 2 记忆版

```text
Given current state x_t, unconditional rates R_t(x_t, .), predictor p(y|x,t), target y

If exact:
    compute log p(y|x_t,t)
    enumerate all one-jump next states x'
    compute log p(y|x',t)
    log_ratio(x') = log p(y|x',t) - log p(y|x_t,t)

If TAG:
    one-hot x_t
    compute gradient g = grad_x log p(y|x_t,t)
    approximate log_ratio for each next token by g difference

guided_rates = R_t * exp(gamma * log_ratio)
fix diagonal by probability conservation
```

- 这个伪代码要能默写。
- 关键是 guided rates 仍然是合法 rate matrix。

---

## Slide 15 - 为什么这不是简单 reranking

- Reranking 是生成完整样本后选择。
- Discrete Guidance 是在 CTMC 每个瞬间修改跳转速率。
- 它改变整个生成过程的 dynamics。
- 它可以在早期就提高通向条件目标的概率。
- 与 SVDD 类似，它比 Best-of-N 更过程化；但机制不同：
  - SVDD 选择候选。
  - Discrete Guidance 修改 rate。

---

## Slide 16 - 实验总览

- 主实验三类：
  1. 小分子 SMILES generation。
  2. DNA enhancer sequence generation。
  3. Protein inverse folding stability guidance。
- 附录还有 CIFAR-10 discrete image guidance。
- 作者的目标不是宣称某个 domain 的 SOTA，而是证明 Discrete Guidance 能跨 domain 工作。
- Baselines 包括：
  - DiGress。
  - Dirichlet Flow Matching (DirFM) for DNA。
  - ProteinMPNN 和 unguided FMIF for protein。
- 主要比较 Exact guidance、TAG、以及相关 baseline。

---

## Slide 17 - 小分子实验设置

- 表示：SMILES strings，离散 token 序列。
- 数据集：QMugs 派生，约 610,575 个 unique molecules，限制分子量、ring 数、LogP 范围。
- 训练 unconditional masking flow model。
- 指导目标：
  - number of rings `N_r`。
  - lipophilicity `LogP`。
- Predictor 预测目标属性。
- 生成时采样到 1,000 个 valid SMILES 再评估。
- guidance strength 使用 `gamma=2`，stochasticity `eta=30`。

---

## Slide 18 - 小分子实验指标和结论

- 论文画 property histogram，看生成分布是否向目标属性移动。
- 还用 mean absolute error (MAE) 比较生成分子的属性和目标值差距。
- DG-Exact 和 DG-TAG 都与 DiGress 比较。
- 结论：
  - Discrete Guidance 能把 molecule property histogram 明显推向指定目标。
  - 在多个目标 `N_r` 和 `LogP` 上，DG 比 DiGress 的 MAE 更低。
  - Exact guidance 通常与 TAG comparable 或更好。
- 面试要强调：这是证明 rate-based guidance 能做 real-valued property conditioning。

---

## Slide 19 - DNA enhancer 实验设置

- 任务：按 cell type 生成 enhancer DNA 序列。
- 数据：104k 条长度 500 的 DNA sequence，81 个 cell type class。
- Baselines：
  - DirFM-CFG。
  - DirFM-CG。
  - DiGress。
  - DG-Exact。
  - DG-TAG。
  - DG-PFG 也训练，但正文主要聚焦 predictor guidance。
- 评估 8 个 cell type。
- 每个条件生成 1,000 条序列。
- guidance strength 多档扫描。

---

## Slide 20 - DNA enhancer 指标

- 指标 1：Class-Conditional FBD。
  - FBD = Frechet Biological Distance。
  - 越低表示生成样本分布更接近该 class 的训练数据。
- 指标 2：Average sample Class Probability。
  - 用 oracle classifier 判断生成序列属于目标 class 的概率。
  - 越高表示条件控制越成功。
- 图 3 的解读：
  - 越靠右下角越好。
  - 右表示目标 class probability 高。
  - 下表示 FBD 低。
- 结果：DG-Exact、DG-TAG、DiGress 大体 comparable；DirFM-CG 较弱。

---

## Slide 21 - 蛋白 inverse folding 实验设置

- Inverse folding：给定蛋白 backbone structure `c`，生成可能折叠成该结构的 protein sequence `x`。
- 作者训练 Flow-Matching Inverse Folding (FMIF) 模型。
- 目标：生成既能折叠回给定结构，又比 wild-type 更稳定的序列。
- Stability predictor 预测 `Delta Delta G`。
- 稳定定义：`Delta Delta G >= 0`。
- Guided model：
  - `p(x | c, Delta Delta G > 0)`
- Baselines：
  - ProteinMPNN。
  - FMIF。
  - DiGress。
  - FMIF-DG-TAG。
  - FMIF-DG-Exact。

---

## Slide 22 - 蛋白实验指标

- 每个方法每个蛋白采样 100 条序列。
- 结构是否正确：
  - 用 AlphaFold2/ColabFold 预测生成序列结构。
  - RMSD <= 2 Angstrom 视为正确 fold。
- 稳定性：
  - `Delta Delta G >= 0`。
- Success：
  - 同时满足正确 fold 和稳定。
- Diversity：
  - 平均 pairwise Hamming distance。
- 这是很强的实验，因为同时要求 function-like stability 和 structure faithfulness。

---

## Slide 23 - 蛋白实验结论

- Figure 4 汇总 8 个 protein。
- FMIF-DG-Exact 和 FMIF-DG-TAG 的 success rate 通常最高。
- FMIF-DG-Exact 在 8 个中 7 个上 outperform 或 match 其它方法。
- 有一个蛋白 6ACV 上 TAG 更好。
- 作者解释可能是 time-dependent predictor miscalibration。
- 这点很重要：exact guidance 并不总是最好，如果 predictor 不准，exact 也会精确地放大错误信号。
- 面试可主动提：guidance quality is only as good as the predictor calibration。

---

## Slide 24 - CIFAR-10 appendix 说明了什么

- 作者还用 pre-trained CTDD 在 CIFAR-10 上做 class-conditional image generation。
- 这里把像素当离散状态。
- 使用 Taylor-approximated predictor guidance。
- 指标：
  - Inception Score (IS)：越高越好。
  - FID：越低越好。
- Table 2：
  - CTDD IS `8.74`, FID `8.10`。
  - CTDD-DG gamma=2 IS `8.91`, FID `7.86`。
  - CTDD-DG gamma=3 IS `9.09`, FID `9.04`。
- 解读：更强 guidance 提升 class confidence，但可能伤害 diversity/FID。

---

## Slide 25 - methodology 的证据链

- Claim 1：离散 guidance 可以 principled。
  - 证据：从 Bayes rule 推导 rate ratio。
- Claim 2：CTMC 让 exact guidance tractable。
  - 证据：单时间点只允许 one-dimension jump，候选数线性。
- Claim 3：TAG 是高效近似。
  - 证据：多个实验中 TAG 接近 exact。
- Claim 4：方法可跨 domain。
  - 证据：molecule、DNA、protein、image。
- Claim 5：相比已有 discrete-time guidance 有优势。
  - 证据：与 DiGress、DirFM 比较。

---

## Slide 26 - 实验的局限

- 小分子使用 SMILES，validity 依赖采样直到有效样本，可能影响效率解释。
- DNA enhancer 的 DG 与 DiGress comparable，不是全面碾压。
- Protein experiment 依赖 stability predictor 和 AlphaFold/ColabFold evaluation，都是 proxy。
- Exact guidance 在 predictor miscalibration 下可能反而更差。
- 全文目标是展示通用性，不是每个领域的最终 SOTA。
- 面试时主动承认这些局限会显得更成熟。

---

## Slide 27 - 复现建议

- 全量复现很重，尤其蛋白 inverse folding 和 AlphaFold evaluation。
- 最小复现优先级：
  1. 实现 toy CTMC rate matrix。
  2. 对一个小序列 alphabet 做 exact guided rates。
  3. 写 TAG 版本，比较 exact log-ratio 和 TAG log-ratio。
  4. 如果 GPU 够，跑作者的 small molecule 或 CIFAR guidance。
- 你真正需要掌握的是 rate adjustment 的计算，而不是一开始复现完整实验。

---

## Slide 28 - 数学基础补丁：rate matrix 合法性

- Rate matrix 必须满足：
  - 非对角元素非负。
  - 每一行求和为 0。
- Guided rate 用 likelihood ratio 乘以原 rate。
- 因为 ratio 非负，所以非对角元素仍非负。
- 对角元素重新设为负的非对角和，所以行和仍为 0。
- 因此 guided rates 仍定义合法 CTMC。
- 这是一个很好的面试点。

---

## Slide 29 - 数学基础补丁：为什么 denominator 变简单

- 原始 Bayes denominator：
  - sum over all possible `x_{t+dt}`。
- 离散空间一般有 `S^D` 个状态。
- CTMC 的无穷小转移只考虑 `x` 自己和 one-jump neighbors。
- 所以 denominator 只对 `D*(S-1)+1` 个状态求和。
- 这不是近似，而是 CTMC 连续时间极限下的结构性质。
- TAG 是近似；tractable candidate set 不是近似。

---

## Slide 30 - 可能问题：为什么连续时间很关键

- 离散时间一步可能改变多个维度。
- 多维同时改变会让 next states 组合爆炸。
- 连续时间 CTMC 在任意无穷小间隔内同时两个 jump 的概率为 0。
- 所以只需考虑 one-coordinate jump。
- 这就是 exact guidance 能成立的原因。
- 回答时一定要说：continuous time is not a cosmetic choice; it changes the transition support。

---

## Slide 31 - 可能问题：Exact 和 TAG 怎么选

- Exact：
  - 更 principled。
  - 需要枚举所有 one-jump next states 并跑 predictor。
  - 对小 alphabet/短序列可行。
- TAG：
  - 用一阶泰勒近似。
  - 更快。
  - 对长序列、大状态空间更现实。
  - 可能受 predictor gradient quality 影响。
- 实验中 TAG 多数接近 exact，但不是保证。
- 如果面试问 trade-off：Exact is accuracy-first, TAG is efficiency-first。

---

## Slide 32 - 可能问题：这和 SVDD 的区别

- 两者都能处理离散空间 guidance。
- Discrete Guidance：
  - 直接修改 CTMC rate。
  - 需要 predictor likelihood 或 conditional rate。
  - 更接近 classifier guidance 的离散版。
- SVDD：
  - 从候选中 value-based selection。
  - 不要求 predictor differentiable。
  - 更像 derivative-free search under diffusion prior。
- 如果 reward 是不可微 docking，SVDD 更自然。
- 如果 predictor likelihood 可用且 CTMC model 明确，Discrete Guidance 更 principled。

---

## Slide 33 - 可能问题：为什么 predictor 要 time-dependent

- Guidance 发生在 noisy state `x_t` 上。
- Predictor 需要估计 `p(y | x_t, t)`，而不是只估计 clean `p(y | x_1)`。
- 不同时间的噪声程度不同。
- 如果 predictor 不知道 `t`，它可能误判 masked/noisy input。
- Protein experiment 中 predictor miscalibration 可能导致 exact guidance 变差。
- 所以 noisy predictor 的 calibration 是方法成败关键。

---

## Slide 34 - 可能问题：实验中 DiGress 为什么弱

- 论文观点：DiGress 基于 discrete-time diffusion，需要额外近似归一化。
- 它不能做 exact guidance。
- 它不自然泛化到 flow matching。
- 在小分子 property MAE 上，DG 显著优于 DiGress。
- 在 DNA enhancer 上 DiGress 与 DG comparable，说明优势不是所有任务都绝对。
- 稳妥回答：DG 提供更 principled and general framework，实验显示多任务优势，但不是每个指标都碾压。

---

## Slide 35 - 2 分钟讲法

- 这篇论文把 continuous diffusion guidance 推广到 discrete state-space。
- 离散空间没有自然 score gradient，而且 Bayes conditioning 需要指数级归一化。
- 作者用 CTMC 表示离散 diffusion/flow，因为连续时间下每个瞬间最多只跳一个维度。
- 这样 next states 从 `S^D` 降到 `D*(S-1)+1`。
- 基于 Bayes rule，guided rate 等于 unconditional rate 乘以 predictor likelihood ratio。
- 为了效率，作者还提出 TAG，用一阶 Taylor approximation 近似 likelihood ratio。
- 实验在小分子、DNA enhancer、蛋白 inverse folding 和 CIFAR image 上验证，说明方法能跨 discrete scientific domains 工作。

---

## Slide 36 - 术语表

- 离散状态空间（discrete state-space）：变量取有限类别，如 token、碱基、氨基酸。
- 连续时间马尔可夫链（continuous-time Markov chain, CTMC）：离散状态在连续时间中跳转的随机过程。
- 速率矩阵（rate matrix）：控制 CTMC 跳转频率和方向的矩阵。
- Predictor guidance：用 predictor `p(y|x,t)` 调整生成过程。
- Predictor-free guidance：混合 conditional 和 unconditional 生成模型。
- Taylor-approximated guidance (TAG)：用一阶泰勒近似降低 guidance 计算成本。
- DiGress：基于 discrete-time discrete diffusion 的离散 guidance baseline。
- FBD：Frechet Biological Distance，用于比较生成生物序列和真实序列分布。

---

## Slide 37 - 最该背下来的公式

- CTMC transition：
  - `p(x_{t+dt}=x'|x_t=x) = delta_{x,x'} + R_t(x,x') dt`
- Predictor-guided rate：
  - `R_t(x,x'|y) = [p(y|x',t) / p(y|x,t)] R_t(x,x')`
- Guidance strength：
  - `R_t^gamma(x,x'|y) = [p(y|x',t) / p(y|x,t)]^gamma R_t(x,x')`
- Predictor-free rate：
  - `R_t^gamma = R_cond^gamma * R_uncond^(1-gamma)`
- TAG log ratio：
  - `log p(y|x',t) - log p(y|x,t) approx (x'-x)^T grad_x log p(y|x,t)`

---

## Slide 38 - 面试答题提醒

- 不要只说“离散也可以 guidance”，要说为什么 CTMC 让 normalization tractable。
- 不要把 TAG 当成理论核心；理论核心是 rate-based Bayes guidance。
- 不要说 Exact 一定最好；predictor miscalibration 会影响它。
- 不要把 DNA enhancer 结果说成全面胜过 DiGress；论文说 comparable。
- 要主动解释每个实验的 metric：
  - molecule: property histogram 和 MAE。
  - DNA: FBD 和 class probability。
  - protein: RMSD、Delta Delta G、success rate。
- 最后要能把它和 SVDD、MDLM 的关系讲出来。
