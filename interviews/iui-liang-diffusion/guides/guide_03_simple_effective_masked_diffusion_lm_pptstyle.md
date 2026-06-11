# Simple and Effective Masked Diffusion Language Models

中文深度面试导读。术语第一次出现时保留英文括号。目标：从基础数学讲到实验，尤其适合先读来补 diffusion language model 基础。

原论文：`../papers/03_simple_effective_masked_diffusion_lm.pdf`  
arXiv：https://arxiv.org/abs/2406.07524  
官方代码：https://github.com/kuleshov-group/mdlm  
项目页：https://s-sahoo.com/mdlm

---

## Slide 01 - 一句话 thesis

- 这篇论文提出 Masked Diffusion Language Model，简称 MDLM。
- 它的核心结论是：简单的 masked discrete diffusion 在语言建模（language modeling）上比过去认为的强很多。
- 作者不发明复杂新架构，而是把 masked diffusion 的目标函数、参数化、训练工程、采样流程做扎实。
- 它把 diffusion objective 化简成加权 masked language modeling loss。
- 这让 BERT-style encoder-only model 获得 principled generation capability。
- 实验上，MDLM 达到 diffusion language model 的新 SOTA，并接近 autoregressive model 的 perplexity。

---

## Slide 02 - 为什么先读这篇

- 另外两篇都涉及 discrete diffusion 或 discrete guidance。
- 如果不理解 masked diffusion，就很难理解：
  - 离散 token 如何加噪。
  - `[MASK]` absorbing state 为什么有用。
  - denoising model 如何预测 clean token。
  - sampling 时如何从全 mask 逐步 unmask。
- 这篇是三篇里最适合作为基础铺垫的一篇。
- 面试时老师问 diffusion models，你可以用这篇解释 discrete diffusion language modeling 的基本机制。

---

## Slide 03 - AR language model vs diffusion language model

- 自回归语言模型（autoregressive language model, AR LM）按顺序生成：
  - `p(x_1, ..., x_L) = product p(x_i | x_<i)`
- 优点：likelihood 计算直接，训练和生成成熟。
- 缺点：生成必须按顺序，难以并行规划整个序列。
- 扩散语言模型（diffusion language model）从噪声序列开始逐步去噪。
- 它可以并行修改多个 token。
- 潜在优势：
  - 非顺序生成。
  - 长程规划。
  - 可控生成。
  - 某些场景下采样速度或 block generation 更灵活。
- 过去问题：diffusion LM perplexity 明显落后 AR。

---

## Slide 04 - 离散 diffusion 的两大路线

- 路线 1：把离散 token embedding 到连续空间，然后做 Gaussian diffusion。
- 路线 2：直接在离散结构上定义 Markov noising process。
- MDLM 属于第二类。
- 它使用 absorbing state / masking diffusion。
- 每个 token 要么保持原 token，要么变成 `[MASK]`。
- 一旦 token 被 mask，在正向加噪过程中会一直保持 mask。
- 这种过程非常接近 BERT 的 masked language modeling，但它有完整 diffusion likelihood interpretation。

---

## Slide 05 - forward masking process

- 对单个 token `x`，正向过程定义：
  - `q(z_t | x) = Cat(alpha_t x + (1 - alpha_t) m)`
- `m` 是 `[MASK]` token 的 one-hot vector。
- `alpha_t` 随时间下降：
  - `t=0` 时接近 1，几乎是 clean token。
  - `t=1` 时接近 0，几乎全 mask。
- 对序列 `x_1:L`，每个 token 独立加 mask。
- 这就是“从 clean sentence 到 all-mask sentence”的 noising process。

---

## Slide 06 - reverse unmasking process

- 反向过程从全 mask 开始。
- 每一步尝试把一些 `[MASK]` token 还原成真实 token。
- 如果一个 token 已经 unmasked，反向过程中它应该保持不变。
- 模型 `x_theta(z_t, t)` 预测 clean token distribution。
- 反向分布 `p_theta(z_s | z_t)` 使用这个预测，把 masked token 变成某个词。
- 直觉：MDLM 就是一个多步的“逐步填空”模型。

---

## Slide 07 - SUBS 参数化是什么

- SUBS = substitution-based parameterization。
- 它把 masked diffusion 的两个结构性质硬编码到模型输出里。
- 性质 1：zero masking probabilities。
  - clean token 不应该预测成 `[MASK]`。
  - 所以把 `[MASK]` logit 设成负无穷。
- 性质 2：carry-over unmasking。
  - 如果 token 已经 unmasked，反向过程直接复制它。
  - 不需要模型再预测它。
- 这两个 substitution 让 objective 更简单、更稳定。

---

## Slide 08 - 为什么 SUBS 重要

- 如果不硬编码这些性质，模型必须自己学：
  - 不要把 token 预测成 mask。
  - 不要改已经 unmasked 的 token。
- 这会浪费容量，也会增加 loss variance。
- SUBS 把已知的条件独立结构（conditional structure）直接放进模型。
- 因此后面可以推导更简洁的 ELBO。
- 实验 ablation 显示 carry-over 对 perplexity 改善明显。

---

## Slide 09 - ELBO / NELBO 复习

- 扩散模型通常通过变分下界（evidence lower bound, ELBO）训练。
- 论文写负 ELBO（negative ELBO, NELBO），越小越好。
- 离散扩散的 NELBO 包括：
  - reconstruction loss。
  - diffusion transition KL terms。
  - prior KL。
- MDLM 在 masking process 和 SUBS 下把 diffusion loss 大幅化简。
- 面试时不必逐行推导，但要知道：这是 principled likelihood objective，不是随便做 MLM。

---

## Slide 10 - Rao-Blackwellized objective

- 作者称目标函数经过 Rao-Blackwellization。
- 直觉：利用模型结构，把某些本来需要采样估计的项解析化简掉。
- 最终 discrete-time diffusion loss 变成类似：
  - 一个权重乘以 `log p_theta(clean token | masked context)`。
- 这比通用 D3PM 形式更稳定。
- 注意：这里“Rao-Blackwellized”带有一定术语借用，核心是通过 SUBS 和 masking structure 降低方差、收紧 bound。

---

## Slide 11 - continuous-time objective

- 作者进一步把步数 `T` 推到连续时间。
- 得到 continuous-time NELBO：
  - integral over `t in [0,1]`
  - 权重是 `alpha'_t / (1 - alpha_t)`
  - loss 是 clean token 的 cross entropy。
- 对序列，目标函数是对所有 token 求和：
  - `sum_l log p_theta(x_l | z_t, t)`
- 但 unmasked token 因为 carry-over 不贡献 loss。
- 所以实际类似：随机 mask 一部分 token，然后训练模型预测被 mask 的 token。

---

## Slide 12 - 和 BERT MLM 的关系

- BERT 的 masked language modeling (MLM) 随机 mask token 并预测。
- MDLM 的 objective 也是 masked token prediction。
- 但区别是：
  - MDLM 的 masking rate 来自 diffusion time `t`。
  - 不同 `t` 有 principled weighting。
  - MDLM 有完整 generative sampling process。
  - MDLM 可以计算 variational likelihood upper bound / perplexity bound。
- 面试回答：MDLM establishes a principled bridge between MLM and diffusion generation。

---

## Slide 13 - noise schedule invariance

- 论文指出 continuous-time objective 对 `alpha_t` 的函数形式有不变性。
- 通过变量替换，可以看到不同 noise schedule 给出同样的 likelihood objective。
- 但不同 schedule 会影响估计方差。
- Appendix Table 9 显示 OWT 上不同 schedule 的 mean BPD 一样，但 variance 不同。
- log-linear schedule variance 最低。
- 这说明 schedule 不只是理论形式，也影响训练估计稳定性。

---

## Slide 14 - 训练工程为什么关键

- 作者强调“simple masked diffusion was underestimated”。
- 性能提升很大部分来自工程细节。
- 关键工程点：
  - tokenizer 选择很重要，太小 vocab 会拉长依赖。
  - 不 materialize full transition matrix，而只处理 masked token indices。
  - 使用现代 diffusion transformer (DiT)。
  - 使用 rotary positional embeddings (RoPE)。
  - 使用 low-discrepancy sampler 降低 ELBO 估计方差。
- 这也解释了为什么旧 D3PM baseline 看起来弱：实现和训练 recipe 不够现代。

---

## Slide 15 - ancestral sampling

- 生成长度 `L` 的序列：
  1. 初始化全 `[MASK]` 序列。
  2. 离散化 reverse process 为 `T` 步。
  3. 每一步对 masked token 独立采样是否/如何 unmask。
  4. 已 unmasked token 保持不变。
- 如果模型不使用 time conditioning，且某一步没有新 token 被 unmask，就可以缓存上次 denoising network 输出。
- 这种 caching 可以减少 forward calls。
- 这是 MDLM 采样效率的重要点。

---

## Slide 16 - semi-autoregressive generation

- MDLM 支持 semi-autoregressive (SAR) 生成任意长度文本。
- 做法：
  - 先生成一段长度 `L`。
  - 下一轮保留末尾一段作为 prefix。
  - prefix token 通过 carry-over unmasking 固定不变。
  - 对后面的新位置从 mask 开始继续 diffusion generation。
- 这样可以像 AR 模型一样生成长文本，但每个 block 内部是 diffusion parallel generation。
- 实验中 MDLM SAR 比 SSD-LM 更快且 generative perplexity 更好。

---

## Slide 17 - 实验总览

- 语言建模：
  - LM1B。
  - OpenWebText (OWT)。
- Zero-shot likelihood：
  - PTB、Wikitext、LM1B、Lambada、AG News、Pubmed、Arxiv。
- 表征学习：
  - C4 pretraining + GLUE fine-tuning。
- Semi-autoregressive decoding：
  - 与 SSD-LM 比较。
- 生物序列：
  - Caduceus DNA model on HG38。
  - Genomics Benchmarks。
- Ablation：
  - SUBS、continuous time、carry-over、zero masking、caching、time conditioning。

---

## Slide 18 - LM1B 实验设置

- 数据集：One Billion Words (LM1B)。
- tokenizer：bert-base-uncased tokenizer。
- context size：128。
- 架构：来自 Lou et al. 的 diffusion transformer，加 RoPE。
- 训练：
  - 33B tokens。
  - 327B tokens。
- 比较：
  - AR Transformer。
  - D3PM。
  - Diffusion-LM。
  - DiffusionBERT。
  - SEDD。
  - MDLM。

---

## Slide 19 - LM1B 结果解读

- Table 1：
  - SEDD 33B tokens：PPL upper bound `32.79`。
  - MDLM 33B tokens：PPL upper bound `27.04`。
  - MDLM 327B tokens：PPL upper bound `23.00`。
  - Retrained AR 33B tokens：PPL `22.32`。
  - Retrained AR 327B tokens：PPL `20.86`。
- 结论：
  - MDLM 是 diffusion LM 中最好的一档。
  - 与 AR 仍有差距，但差距缩小到约 14% 左右。
- 面试表述：MDLM does not beat AR on standard likelihood, but it closes much of the gap while preserving diffusion advantages。

---

## Slide 20 - OWT 实验结果

- 数据集：OpenWebText。
- tokenizer：GPT-2 tokenizer。
- context size：1024。
- 训练约 262B tokens。
- Table 2：
  - AR retrained：PPL `17.54`。
  - SEDD retrained：PPL upper bound `24.10`。
  - MDLM：PPL upper bound `23.21`。
- 结论：
  - MDLM 超过 SEDD。
  - AR 仍更强。
  - MDLM 在更大 context / real web text 上仍保持优势。

---

## Slide 21 - Zero-shot likelihood

- Table 3 使用 OWT 训练模型，在 unseen datasets 上评估。
- 数据集：PTB、Wikitext、LM1B、Lambada、AG News、Pubmed、Arxiv。
- MDLM consistently outperform SEDD。
- 一些 out-of-domain 数据，如 Lambada、Scientific Papers，MDLM 甚至比 AR 更好。
- 论文猜测 diffusion unmasking objective 对 out-of-domain evaluation 更 robust。
- 面试谨慎说法：
  - 这是一个有趣观察，不意味着 MDLM 全面超过 AR。
  - 说明 diffusion LM 可能有不同泛化行为。

---

## Slide 22 - GLUE representation learning

- 作者测试：BERT-style encoder-only model 经过 MDLM fine-tuning 后，能不能获得生成能力，同时不损害表示学习。
- 设置：
  - MosaicBERT architecture。
  - 先用 MLM 在 C4 上预训练。
  - 再用 MDLM fine-tuning 少量 token。
- C4 validation：
  - AR PPL 约 22。
  - BERT 的 MDLM variational PPL upper bound 约 78。
  - BERT + MDLM-FT 约 35。
- GLUE Table 4：
  - BERT average `81.62`。
  - BERT + MDLM-FT average `82.06`。
- 结论：获得 generation capability，没有牺牲 downstream performance。

---

## Slide 23 - Semi-AR 结果

- Table 5 比较生成 2048 token sequence。
- 指标：
  - Gen. PPL：用 GPT-2 评价生成文本。
  - Sec/Seq：每条序列耗时。
- SSD-LM：
  - Gen PPL `35.43`
  - `2473.9` 秒/序列
- MDLM：
  - Gen PPL `27.18`
  - `89.3` 秒/序列
- 结论：MDLM 既更好又快很多。
- 注意：这是特定设置下与 SSD-LM 比较，不是与所有 AR decoding 比较。

---

## Slide 24 - DNA / genomics 实验

- 作者把 MDLM 应用于 biological sequence modeling。
- 使用 Caduceus DNA language model，backbone 是 Mamba-based state space model (SSM)。
- 数据：HG38 human reference genome。
- 任务：
  - generative performance：PPL。
  - downstream performance：Genomics Benchmarks。
- Table 6：
  - AR Mamba PPL `3.067`。
  - HyenaDNA PPL `3.153`。
  - Plaid PPL upper bound `3.240`。
  - SEDD PPL upper bound `3.216`。
  - MDLM PPL upper bound `3.199`。
- MDLM 是 diffusion 里最好，但 AR 仍略好。

---

## Slide 25 - Genomics downstream 结果

- Table 7 比较多个 regulatory element classification task。
- Caduceus + MDLM 在多数任务上保持或改善 MLM pretraining 表现。
- 例如 Human OCR Ensembl：
  - Caduceus MLM `0.821`
  - Caduceus MDLM `0.823`
- Human NonTATA Promoters：
  - Caduceus MLM `0.935`
  - Caduceus MDLM `0.940`
- 结论：generative fine-tuning 不会明显破坏 biological representation。
- 面试回答：这支持 MDLM 可作为 encoder-only biological foundation model 的 generative extension。

---

## Slide 26 - Table 8 ablation

- Table 8 是这篇最重要的 ablation。
- LM1B PPL：
  - MDLM：`27.04`
  - w/o continuous time：`27.19`
  - w/o carry-over：`28.56`
  - w/o zero masking：`28.51`
- 解读：
  - continuous time 有小幅提升。
  - carry-over unmasking 很重要，去掉后 PPL 明显变差。
  - zero masking 的额外影响较小。
- 作者还强调，重新实现的 D3PM-like baseline 比过去认为的强很多。

---

## Slide 27 - sampling speed and caching

- Appendix Table 10：在 A5000 上生成 64 samples。
- `T=5k`：
  - SEDD `85.3` 分钟。
  - MDLM `70.3` 分钟。
  - MDLM + caching `40.1` 分钟。
- `T=10k`：
  - SEDD `155.2` 分钟。
  - MDLM `127.9` 分钟。
  - MDLM + caching `60.4` 分钟。
- caching 的基础是：
  - unmasked token carry over。
  - model 可以不 time-conditioned。
- 这是 MDLM 的实际工程优势。

---

## Slide 28 - T 和 time-conditioning ablation

- Appendix Table 11：
  - 训练 continuous-time MDLM，评估时用不同 discrete `T`。
  - `T` 越大，PPL 越接近 continuous limit。
  - `T=10` PPL `42.18`，`T=1000` PPL `23.15`，continuous PPL `23.05`。
- Appendix Table 12：
  - OWT 上 time-conditioning 影响很小。
  - with time-conditioning PPL `23.21`。
  - without time-conditioning PPL `23.05`。
- 这支持作者使用 no-time-conditioning 来获得 caching speedup。

---

## Slide 29 - 方法强点

- 简单：基于 masked diffusion，不需要复杂离散 CTMC machinery 才能训练。
- 有 principled likelihood objective。
- 和 BERT MLM 连接自然。
- 能给 encoder-only model 加 generation capability。
- 工程实现强，证明旧 baseline 被低估。
- 采样支持 caching 和 semi-autoregressive generation。
- 实验覆盖 language 和 DNA。

---

## Slide 30 - 方法弱点

- 标准 likelihood 仍落后 AR。
- diffusion sampling 通常需要多步，虽然可以缓存和 SAR，但并不天然比 AR 快。
- PPL 是 upper bound，和 AR exact likelihood 比较时要谨慎。
- 生成质量还依赖外部 evaluator，如 GPT-2 generative PPL。
- 论文主打 masked diffusion，泛化到其它 discrete corruption process 不是重点。
- 工程 recipe 贡献很大，理论创新和工程贡献需要区分。

---

## Slide 31 - 复现建议

- 官方代码：https://github.com/kuleshov-group/mdlm
- 全量训练 LM1B/OWT 成本很高。
- 面试准备最小复现：
  1. 实现 toy vocabulary 的 masked forward process。
  2. 实现 SUBS：mask logit 设为负无穷，unmasked token carry over。
  3. 训练一个小 transformer 或 MLP 在 toy text 上做 MDLM objective。
  4. 从 all-mask 采样，观察逐步 unmask。
  5. 比较 with / without carry-over 的 loss 或 sample behavior。
- 如果 GPU 有余力，再跑官方 repo 的小模型配置。

---

## Slide 32 - 数学基础补丁：为什么 objective 是 weighted MLM

- 在普通 MLM 中，随机 mask token，然后预测原 token。
- 在 MDLM 中，`t` 决定 mask rate。
- 不同 `t` 的训练样本有不同权重 `alpha'_t / (1-alpha_t)`。
- 所以它是“对不同 mask rate 的 MLM loss 做加权积分”。
- 这比普通固定 mask rate 更 principled。
- 也解释了为什么 BERT-like encoder 可以被转成 generative model。

---

## Slide 33 - 可能问题：MDLM 和 BERT 有什么本质区别

- BERT：
  - 训练目标是 representation learning。
  - 固定或 heuristic mask rate。
  - 没有完整 reverse diffusion sampling 解释。
- MDLM：
  - 训练目标来自 ELBO。
  - mask rate 来自 diffusion time。
  - 可以从 all-mask 序列逐步采样生成文本。
  - 可以评估 variational likelihood。
- 简短回答：MDLM turns MLM from a denoising pretraining trick into a generative diffusion model。

---

## Slide 34 - 可能问题：为什么 carry-over 很重要

- 在反向 unmasking 中，已经确定的 token 不应该被改掉。
- 如果模型每步都可能重写 unmasked token，生成过程不稳定。
- Carry-over 把这个结构硬编码。
- 它减少模型需要学习的内容。
- Ablation 显示去掉 carry-over PPL 从 `27.04` 变差到 `28.56`。
- 这是 SUBS 中最重要的部分之一。

---

## Slide 35 - 可能问题：为什么 AR 仍然强

- AR 直接 factorize likelihood，训练目标和 evaluation PPL 完全对齐。
- Diffusion LM 的 likelihood 是 variational bound，可能不够紧。
- Diffusion sampling 多步，训练和推理还有额外复杂度。
- 语言有强顺序结构，AR inductive bias 很合适。
- MDLM 的贡献不是“打败 AR”，而是证明 diffusion LM 可以接近 AR，同时带来非顺序生成和 encoder-only generation 的优势。

---

## Slide 36 - 可能问题：它和另外两篇怎么连

- MDLM 提供 masked discrete diffusion LM 的基础。
- Discrete Guidance 可以在 CTMC-based discrete diffusion/flow 上做 conditional guidance。
- SVDD 可以在 masked diffusion / discrete diffusion 上做 derivative-free reward optimization。
- 如果老师问“为什么离散 diffusion 值得研究”：
  - MDLM 说明离散 diffusion 可以做强 language modeling。
  - Guidance 两篇说明它还可以做 controllable generation 和 reward optimization。

---

## Slide 37 - 2 分钟讲法

- 这篇论文重新审视 masked diffusion language model。
- 它使用 absorbing mask process：正向逐步把 token 变成 `[MASK]`，反向从全 mask 逐步 unmask。
- 作者提出 SUBS 参数化，把 zero masking probability 和 carry-over unmasking 硬编码。
- 这样可以推导出 Rao-Blackwellized continuous-time ELBO，形式上就是加权的 masked language modeling loss。
- 这个目标让 BERT-style encoder-only models 具备 principled generation。
- 实验显示 MDLM 在 LM1B、OWT、DNA 上超过之前 diffusion LM，并接近 AR perplexity；同时 SAR decoding 比 SSD-LM 更快。
- 主要 takeaway：masked diffusion is simple, principled, and much stronger with the right objective and engineering。

---

## Slide 38 - 术语表

- 自回归模型（autoregressive model, AR）：按顺序预测下一个 token 的模型。
- 离散扩散（discrete diffusion）：直接在离散 token/类别空间中加噪和去噪。
- 吸收态（absorbing state）：一旦进入就保持不变的状态，这里是 `[MASK]`。
- SUBS 参数化（substitution-based parameterization）：把 mask logit 和 carry-over 规则硬编码进输出。
- Rao-Blackwellized objective：利用结构解析化简、降低方差的训练目标。
- 负 ELBO（negative ELBO, NELBO）：越小越好的变分训练目标。
- 半自回归生成（semi-autoregressive generation, SAR）：按块生成，块内 diffusion 并行，块间顺序推进。
- 缓存（caching）：当状态没有变化且模型不依赖时间时复用 denoiser 输出。
