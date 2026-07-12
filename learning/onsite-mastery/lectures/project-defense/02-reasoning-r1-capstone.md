# 02 · 项目深挖 — reasoning-r1 Capstone（R1-Zero / GRPO 双轨复现设计）

> 素材来源：`learning/reasoning-r1/lectures/13-capstone-r1-zero-track-A.md`、`14-capstone-r1-zero-track-B.md`、`12-spurious-rewards.md`、`02-grpo-derivation.md`、`README.md`；源码 `src/r1_zero_track_a.py`、`src/r1_zero_track_b.py`、`src/grpo_minimal.py`、`src/rloo_minimal.py`、`src/reinforce_pp.py`、`src/rewards/format_reward.py`、`src/rewards/accuracy_reward.py`；测试 `src/tests/test_capstone_and_advantages.py`、`src/tests/test_format_accuracy_reward.py`。
> 用法见 `00-how-to-defend.md` §3。**这篇文档的核心诚实点会反复出现**：capstone 的两条训练轨（Track A/B）目前是**设计 + mock 演示**，不是已经跑通的真实 GPU 训练结果——这一点在第 5 节和整个追问树里都会被反复挑明，因为这是终面最容易被拆穿的地方。

---

## 1. 背景与目标

这个 capstone 要验证/复现的是 **DeepSeek R1-Zero 的训练方法论**：用 GRPO（Group Relative Policy Optimization）+ 纯规则 reward（无需训练 reward model）+ 格式约束（`<think>...</think><answer>...</answer>`），让模型在 RL 过程中自发涌现更长的思维链和"aha moment"式自我修正。

具体规模（如实写，不是"大模型"）：
- **Track A（教学轨，必跑）**：base 是 **GPT-2-medium（355M 参数）**，任务是 **Countdown-3**（给 3 个数字用四则运算凑出 target，比 GSM8K 简单的纯算术任务），横向对比 4 个算法：REINFORCE+mean baseline → RLOO(k=8) → GRPO(k=8) → GRPO+DAPO Clip-Higher。目标硬件是单卡 RTX 5090 24GB，设计时长 4 算法 × 1.5h ≈ 6h。
- **Track B（挑战轨，选跑）**：base 是 **Qwen2.5-1.5B-Base**，4bit NF4 量化 + LoRA（r=16，作用于 q/k/v/o 投影），任务是 **GSM8K-tiny**（500 训练 + 100 测试的极小子集），单一算法 GRPO，目标是观察"aha moment"词频涌现，设计时长约 4h。

capstone 同时手写了三种 RL-for-LLM 算法的核心组件用于横向理解算法演化路径：`grpo_minimal.py`（GRPO：group z-score advantage + PPO clip + Schulman 无偏 KL 估计器）、`rloo_minimal.py`（RLOO：leave-one-out baseline）、`reinforce_pp.py`（REINFORCE++：KL 惩罚加在 reward 上而非 loss 外），以及两套 reward 函数（`format_reward` 严格格式检查 + `accuracy_reward` 支持 Countdown 和 GSM8K 两种验证器）。

---

## 2. 个人贡献

自学项目，个人贡献 = 全部实现，具体做了：
- 从 PPO 推导到 GRPO（`02-grpo-derivation.md`，28 slides 完整推导：去 critic → group baseline → z-score advantage → loss 公式），并手写 `grpo_minimal.py` 里的 `compute_group_advantage()` 和 `grpo_loss()`（PPO clip surrogate + Schulman 无偏 KL 估计器）。
- 独立实现 RLOO（`rloo_minimal.py`：leave-one-out baseline）和 REINFORCE++（`reinforce_pp.py`：KL 惩罚加在 reward 而非 loss 上）两种 baseline 算法，用于和 GRPO 横向对比 advantage 估计方式的差异。
- 设计并实现两套 reward 函数（`rewards/format_reward.py` 严格正则格式检查、`rewards/accuracy_reward.py` 支持 Countdown 和 GSM8K 两种验证器），并写了完整的 pytest 单元测试（`test_format_accuracy_reward.py`，覆盖缺 think/缺 answer/空 think/think 前有多余文字/千分位逗号/负数/浮点等价等 20 个具体断言）。
- 设计 Track A/B 两条 capstone 轨的完整实验方案：模型选型理由、显存预算表、训练超参数、监控指标、预期对照表、退出条件、失败排查表。
- 写 `aha_word_frequency()` 关键词法检测 aha moment，以及 `combined_reward`/`gsm8k_reward_full` 这类把 format+accuracy 组合成单一 reward 的逻辑。
- 写 capstone 自身的测试 `test_capstone_and_advantages.py`，对 RLOO/REINFORCE++ 的 advantage 做手推数值验证，对 Countdown/GSM8K 的 reward 提取逻辑做断言覆盖。

---

## 3. 关键技术决策与理由

| 决策 | 理由（代码/lecture 可查） |
|---|---|
| Track A 选 GPT-2-medium(355M) 而非更大模型 | `13-capstone-r1-zero-track-A.md` Slide 2：小（单卡放得下 4 个 model：actor+ref+old+可能的 RM）、老（pretrain 时无 contamination）、简单（"不会 aha emergence（教学预设说明），只看 pipeline 跑通"）——刻意把"pipeline 正确性"和"aha 涌现"解耦，分别在 Track A/B 验证。 |
| Track A 任务选 Countdown-3 而非 GSM8K | Slide 1："比 GSM8K 简单"——纯算术，`eval()` 直接可判定对错，不需要语言理解，方便先验证 format+accuracy reward 的 pipeline 本身是对的。 |
| reward 权重 `0.1*format + 0.9*accuracy` | Track A/B 两个源码文件的 docstring 和 Slide 5 都写"与 R1-Zero 完全同款"；小权重先引导格式收敛（否则 accuracy reward 在没格式约束时几乎拿不到有效信号，答案都提取不出来），accuracy 主导优化方向。 |
| GRPO 核心超参：k=8, clip_eps=0.2, beta_kl=0.04, lr=5e-6, max_response_len=256 | `13-capstone...` Slide 8。lr 比 REINFORCE 的 1e-5 更保守，配合更长的 response（256 vs 128 tokens）。 |
| DAPO Clip-Higher：`clip_eps_low=0.2, clip_eps_high=0.28` 非对称 clip | Slide 9：让正 advantage 的更新有更大上界，目的是让模型更愿意探索更长输出（预期"length 涨更猛"）。 |
| Track B: LoRA r=16 (q/k/v/o) + 4bit NF4 量化, k 降到 4 | `14-capstone...` Slide 3 显存配方：`4bit base~3GB + LoRA~0.3GB + ref(frozen)~3GB + rollout KV~10GB + grad+adam~6GB ≈ 22GB`（24GB 卡上"紧"），因此 k 必须从 Track A 的 8 降到 4，`max_response_len` 卡在 256。 |
| KL 估计器用 Schulman 无偏 k3 估计器 `kl = exp(log_r) - log_r - 1` | `grpo_minimal.py` 注释明确写"unbiased KL estimator (Schulman)"；这个形式保证 kl ≥ 0（`exp(x)-x-1` 在 x=0 处取最小值 0），比直接用 `log_r`（可能为负，不是合法 KL 近似）更稳。 |
| aha 检测用关键词频率法而非训练分类器 | `aha_word_frequency()`：数 `wait/let me reconsider/actually/i made a mistake/rethink/double-check/verify/重新/等等` 这些词的命中率——简单、可复现、不需要额外模型，是社区通行的 proxy 指标。 |

---

## 4. 踩过的坑与解决

lecture 里**明确记载**的失败排查表（不是抽象讲坛，是具体调参对照）：

| 场景 | 现象 | 修法 |
|---|---|---|
| Track A | 不学 | lr 调大到 2e-5 |
| Track A | reward 飘 | k 加大到 16 |
| Track A | OOM | max_response_len 减半 |
| Track A | KL 爆 | beta 调大至 0.1 |
| Track A | format 学不到 | format 权重 0.1→0.3 |
| Track B | OOM | k 减半 / max_resp 减半 |
| Track B | 不 aha | 加 step / 加 max_resp_len |
| Track B | reward 不动 | beta 调小到 0.02 |
| Track B | KL 飞 | beta 调大到 0.08 |
| Track B | LoRA 不学 | r 加大到 32 |

另一个**明确记载**、并且真正改变了实验设计的坑是 **Spurious Rewards 警示**（`12-spurious-rewards.md`）：2025.06 论文发现 Qwen-2.5-7B 用**随机奖励**做 GRPO 训练，MATH 测试集也涨了 21pp，几乎和真实 reward 的 +20pp 一样；Stanford/Berkeley/Anthropic/DeepSeek 多个独立团队复现出这个现象，归因于 contamination（Qwen 预训练见过 MATH 数据）+ 探索增益 + 格式学习，而非真正学到推理（复现矩阵里 Llama-3-7B 用随机 reward 只涨 +2pp，因为没有污染）。这直接写进了 Track B 的设计防御：必须 held-out test（100 题不可见）、报告 vs base 而非 vs random、人工 spot check 实际推理质量（`14-capstone...` Slide 12/13）——这是**明确写进 Track B 退出条件的防御性设计**，不是事后猜的。

以下是**推测**（非文档明确记载，基于代码里的设计一致性推断）：
- `format_reward` 用严格正则 `^<think>(.+?)</think>\s*<answer>(.+?)</answer>\s*$`（`re.match` 从头开始，而非宽松的 `re.search`），且 `test_format_accuracy_reward.py` 专门测了"think 前有其它字符应失败"（`test_format_text_before`）——（推测）这大概率是刻意收紧格式匹配，防止"模型学会在标签前加废话绕过检查"这类最简单的 reward hacking，但代码/文档没有留下"曾经用宽松正则被绕过"的具体调试记录。
- `countdown_reward` 对用到的数字集合做排序比较（`sorted(used) == sorted(target_nums)`），`test_countdown_wrong_numbers` 专门测了"用了不在 numbers 列表中的数"要判 0 分——这是明确写在测试里的防御性设计（防止模型蒙对答案但用了题目没给的数字）。

---

## 5. 结果与诚实局限（本篇最重要的一节）

### 最重要的诚实点：Track A/B 的"训练"目前是 mock/伪代码，不是真实 GPU 训练结果

逐条对照代码：
- `r1_zero_track_a.py` 的 `mock_train_step()` 用一个**手写公式**模拟 reward：`base = 0.05 + step * 0.001`，再用 `rng.random() < base` 采样二值 reward——这是硬编码的"reward 随 step 线性上升"模拟器，**不是** GPT-2 真的在 rollout + 反向传播中学出来的。`train_track_a()` 打印的 `early avg reward` / `late avg reward` / `Δ=+X pp`，完全是这个硬编码公式的产物，不代表任何真实学习信号。
- 真正的 GRPO 损失函数 `grpo_loss()`（含 PPO clip + KL）在 `r1_zero_track_a.py` 里被 import 了，但**从未被调用**——`mock_train_step` 只调用了 `compute_group_advantage`，连"假装"的 loss 计算都没有发生。
- Track B 的 `setup_lora_qwen()` 函数明确返回一段**字符串形式的伪代码**（源码注释原话："伪代码（CPU 无法跑）"），从未被实际执行；`aha_word_frequency` 的演示用的是 5 条**手写的 mock response**，不是真实训练出的模型生成的文本。
- 仓库里没有 `runs/` 目录、没有训练日志 csv、没有 checkpoint——`13`/`14` 两讲提到的"train log 写入 `runs/track_a/*.csv`"这类路径实际不存在，说明"实战入口"命令从未真正执行过。

### 真实、可验证、已经跑通的部分

- 全部 reward 函数（`format_reward`, `extract_answer`, `gsm8k_extract_answer`, `gsm8k_reward`, `countdown_reward`）有完整 pytest 覆盖且逻辑正确（`test_format_accuracy_reward.py` 20 个测试，例如 `test_gsm8k_extract_with_commas` 验证 `"…#### 1,234"` → `"1234"`）。
- RLOO advantage 手推验证：`rewards=[1,0,1,0]`, k=4，`A[0] = 1 - 0.333 = 0.6667`（`test_rloo_baseline_excludes_self`，容差 1e-3）——真实断言，不是估计。
- REINFORCE++ advantage 的 zero-mean 性质验证（batch 内中心化后均值 < 1e-6）。
- GRPO 的 group z-score advantage 在 `grpo_minimal.py` 的 smoke test 里验证了"组内均值≈0"的性质，但这**只是 print，不是 pytest assert**，没有被自动化测试收录（见追问链 1）。
- Countdown/GSM8K 任务生成与 reward 组合的正确性有具体断言：`test_combined_reward_format_accuracy`（`3*(4+5)=27` 场景 total reward=1.0）、`test_gsm8k_reward_full_extracts_answer` 等。
- `aha_word_frequency` 在 5 条手写 mock response 上算出 `responses_with_aha=3`、`aha_ratio=0.6`（`test_aha_word_frequency`），函数逻辑本身被断言验证过，但**这不是从真实训练产出的 response 里统计出来的**。

### 诚实局限（不走过场）

1. **这个 capstone 本质上是"R1-Zero 训练 pipeline 的设计 + 局部组件正确性验证"，不是一次真正跑通的 GPU 训练实验**。如果被问"你的模型最终 accuracy 涨到多少"，诚实答案是：文档里写的数字（GPT-2-M format 5%→95%、accuracy 5%→15-25%；Qwen accuracy 5%→25%、aha 词频→7%）是根据 R1/TinyZero 等公开复现工作定的**设计预期值 / 退出条件**，不是自己训练测出来的实测数字。
2. **即使真的按计划跑了 Track B，也不能直接把 accuracy 上涨归因为"学到了真推理"**——`12-spurious-rewards.md` 本身的警示就是：Qwen 系列由于 contamination，随机 reward 都能让 MATH 涨 21pp。这意味着即使 Track B 跑出 accuracy 5%→25%，不做 held-out + 多 base 对照 + spot check，就无法排除这是 contamination 被激活或格式学习带来的假涨。
3. **GRPO 核心 loss 函数没有被 pytest 覆盖**，只有 `__main__` 里的 smoke test（print，不是断言）——它的数值正确性（尤其是 Schulman KL 估计器的具体行为）没有像 reward 函数那样被系统验证过。
4. **两条"真训练"轨道所需的重型依赖**（README 开头写"⚠️ 本专题切换到 WSL2：verl + Ray + vllm + Megatron"）在这个环境里未必配置齐全；capstone 假设有单卡 RTX 5090 24GB，这是个人硬件规划，不是通用可复现的实验环境说明。
5. **规模差距巨大，谈不上"复现"**：R1-Zero 原论文是在 DeepSeek-V3-Base（千亿参数级）上训练的；这里的 Track A 是 355M + 极简算术任务，Track B 是 1.5B + 500 条训练数据的 4bit LoRA。参数量差 2-3 个数量级，数据规模差更多——准确的说法是"复现 R1-Zero 的算法 pipeline 设计"，不是"复现 R1-Zero 的结果"。

---

## 6. 追问树

### 链 1 · "你怎么知道你的 GRPO 复现是对的，而不是恰好训出了一个凑巧的结果？"
- **L1**：你的 GRPO 实现里最容易出 bug 的地方是哪，你怎么验证对了？
  → 两个最容易错的地方——group advantage 的归一化方向（z-score）和 KL 估计器的符号/非负性。`compute_group_advantage` 在 `grpo_minimal.py` smoke test 里验证了每组内 advantage 均值≈0；KL 估计器用 Schulman 无偏 k3 形式 `kl=exp(log_r)-log_r-1`，保证 kl≥0，而非可能为负的简单 `log_r`。
- **L2**：但这些都只是 print/smoke test，没有 assert，你说的"验证过"是不是有点勉强？
  → 对，group advantage z-score 性质的验证确实只是 print，不在 pytest 里；真正有断言覆盖的是 RLOO（`test_rloo_baseline_excludes_self`）和 REINFORCE++（zero-mean assert），GRPO 核心组件本身没有专门的 pytest 文件——这是一个诚实的 gap。
- **L3**：那如果现在要你补一个 GRPO 正确性的单元测试，你会怎么写？
  → 至少两个：(1) 构造全相等的 reward，验证 group z-score advantage 应该全为 0；(2) 构造 ratio 明显超出 `[1-eps, 1+eps]` 的情况，验证 clip 确实生效（`min(surr1,surr2)` 选中被 clip 的一侧）。这两个测试目前都没写，是可以立刻补的具体待办。
- **L4 / pitfall**："所以严格说，你现在还不能 100% 确认你的 GRPO 实现没有 bug？"
  → 对，应该这样说而不是"我很确定"。目前的证据链是公式推导 + smoke test 的合理性质 + 和 RLOO/REINFORCE++ 的横向数值对比，构成"逻辑自洽"的证据，但没有到"在真实模型上训练收敛到已知正确结果"这种更强证据。**pitfall**：不要说"跑过很多次都稳定"，因为根本没跑过真实训练。

### 链 2 · "aha moment 现象是真涌现还是你调参调出来的？"
- **L1**：Track B 设计里怎么检测 aha moment？
  → 用关键词频率法（`aha_word_frequency`），数 response 里含 "wait"/"let me reconsider"/"actually"/"rethink" 等词的比例，目标阈值是"≥5%"（`14-capstone...` Slide 7/13）。
- **L2**：关键词命中就等于"真的在做 self-correction"吗？
  → 不等于，这正是这个方法的已知弱点。Slide 11 明确写"不仅看 accuracy，还要 spot check：推理是否合理？是否中英混？长度是否合理？每 100 step 看 10 个样本"——设计里知道纯关键词计数不够，但这个人工抽查本身也从未实际执行过（因为 Track B 从未真正训练）。
- **L3**：L12 spurious rewards 那节课不是说随机 reward 都能让 Qwen 涨 21pp 吗，这跟 aha moment 有什么关系？
  → 关系很直接——Slide 12 专门把这个警示接到 Track B 设计上：即使词频法测出"aha ratio≥5%"，也可能是 Qwen pretrain 阶段就见过类似数据、reward 只是"激活"了预训练已有的话术模式，而不是 RL 真的教会了新能力（L12 引用"RL teaches the model nothing new; it teaches it which capabilities to exhibit"）。
- **L4 / pitfall**："那你现在能不能给我一个真实的、你亲自跑出来的 aha ratio 数字？"
  → 不能，必须诚实说没有。文档里"step 1000: aha 7%"（Slide 8）是预期曲线/退出条件的目标值，不是实测数字。**pitfall**：千万不要把"设计文档里的预期表格"当成"我跑出来的结果"报出去，一旦被要求"给我看 log 文件/checkpoint"就会当场露馅——仓库里确实没有 `runs/` 目录和任何日志文件。

### 链 3 · "和原始 R1-Zero 论文的数字差多少，为什么会有这个差距？"
- **L1**：R1-Zero 原论文和你的复现在规模上差多少？
  → R1-Zero 是在 DeepSeek-V3-Base（千亿参数级）上训练的，我的 Track A 是 GPT-2-medium(355M)+Countdown-3，Track B 是 Qwen2.5-1.5B+GSM8K-tiny(500训/100测)+4bit LoRA——参数量差 2-3 个数量级，数据规模差更多，谈不上"复现 R1-Zero 的结果"，只能说是"复现 R1-Zero 的算法 pipeline 在极小规模下的教学演示"。
- **L2**：既然规模差这么多，为什么 Track A"预期不会出现 aha emergence"这个判断是怎么来的？
  → Slide 2 直接写了理由："GPT-2-M 简单：不会 aha emergence（教学预设说明），只看 pipeline 跑通"——这是有意为之的设计取舍，把"pipeline 正确性"和"aha 涌现现象"解耦验证，不是想复现却失败了。
- **L3**：那 Track B（Qwen-1.5B）有没有可能也复现不出 aha，差距原因是什么？
  → 有很大可能，可能原因（部分来自 lecture 讨论，部分是推测）：模型规模小 2-3 个数量级，self-correction 可能是涌现属性；GSM8K-tiny 只有 500 条训练数据；contamination 问题难以和真实 signal 区分；4bit 量化+LoRA 压缩了表达能力，进一步限制复杂推理模式的涌现。
- **L4 / pitfall**："所以你觉得你的 Track B 设计有意义吗，既然大概率复现不出来？"
  → 有意义，但意义要讲清楚——Slide 13 明确写了降级方案："任一不达成可写为『教学完成+R1-Zero 现象局部观察』"，这从一开始就不是"必须复现出 aha"的赌注式实验，而是在个人硬件预算下尽量走一遍真实训练流程，观察到什么就报告什么。**pitfall**：不要因为"大概率复现不出来"就回避问题，更不要编一个"我复现出来了"的结果去讨好考官。

### 链 4 · "你的 format reward 用严格正则，会不会被模型钻空子（reward hacking）？"
- **L1**：你的 `format_reward` 具体怎么判定格式对不对，为什么用严格模式？
  → 用 `^<think>(.+?)</think>\s*<answer>(.+?)</answer>\s*$` 的严格正则（`re.match` 从头开始），且要求内容非空；`test_format_text_before` 专门验证"think 前有其它字符应失败"，说明设计上刻意不用宽松的 `search()`。
- **L2**：但严格正则会不会导致模型学会"先输出完美格式的空壳，再在里面塞 reward hacking 的内容"？
  → 这是可能的风险，严格正则只防"结构层面"的 hacking，不能防"标签内容层面"的 hacking。这一层防御落在 accuracy reward 上：`countdown_reward` 检查用到的数字集合是否和题目给的一致，`gsm8k_reward` 精确比对数值。
- **L3**：那这套 reward 设计有没有已知的、documented 的漏洞？
  → 代码本身没列出漏洞清单，但可以类比 `12-spurious-rewards.md` 讨论的更广义 reward hacking——即使 reward 函数逻辑无懈可击，GRPO 也可能通过"探索增益"或"格式学习"而非"真推理"拿分。这正好和 interp-graduation capstone 里"CoT 忠实性"的研究 gap 接上，是两个 capstone 之间真实的关联点。
- **L4 / pitfall**："所以你觉得你的 reward 设计能完全避免 reward hacking 吗？"
  → 不能。诚实说法是"防住了几类已知的简单 hacking 模式，但没有、也不可能仅靠 reward 函数设计排除更隐蔽的 hacking，排除这些需要 held-out test+多 base 对照+人工 spot check，这些防御手段设计里有，但从未在真实训练里执行过"。**pitfall**：不要夸口"这套 reward 是抗 hacking 的"。

### 链 5 · "REINFORCE/RLOO/REINFORCE++/GRPO 这几种算法本质区别是什么，为什么要横向对比？"
- **L1**：这几个算法在 advantage 估计上分别怎么不同？
  → REINFORCE+mean baseline（k=1，batch mean 做 baseline，高方差）；RLOO（k=8，leave-one-out：`baseline=(sum_all-rewards)/(k-1)`）；GRPO（k=8，组内 z-score `(R-mean)/std`，KL 惩罚在 loss 外）；GRPO+DAPO Clip-Higher（同 GRPO 但 clip 区间不对称 `[0.2, 0.28]`）。
- **L2**：RLOO 和 REINFORCE++ 又是什么关系，为什么要单独实现 `reinforce_pp.py`？
  → REINFORCE++（OpenRLHF 2025.01）是"去 critic 的 PPO 简化版"——保留 PPO clip，但用 batch baseline 而非 GAE，KL 惩罚加在 reward 上而非 loss 里（`R_corrected = rewards - beta_kl*cumulative_kl`）。这不是 Track A/B 直接用的算法，是独立实现用来理解 GRPO 在"KL 放哪里"这个设计选择上的取舍。
- **L3**：这几种方法在数学上哪个方差更低，你的实现能体现出这个差异吗？
  → `rloo_minimal.py` 的 `compare_rloo_grpo()` 把同一组 reward 分别用 RLOO 和 GRPO(z-score) 算 advantage 并排打印对比，可以肉眼核对数值分布差异；但这只是一次固定 reward 上的并排打印，不是系统性的方差实验，"哪个方差更低"这个理论结论没有被实验数据验证，只是转述自 `02-grpo-derivation.md` 的理论论述。
- **L4 / pitfall**："既然没有真的训练过，你怎么确定 GRPO 在 Track A/B 里比 REINFORCE/RLOO 更适合？"
  → 这个判断目前主要是理论+社区共识（DeepSeekMath/R1 系列证明 GRPO 在无 critic 场景下工程上更稳定、更省显存），不是自己训练对比出来的结果。`13-capstone...` Slide 11 的"预期对照表"是根据算法演化路径的社区认知设定的教学预期值。**pitfall**：如果被追问"对照表数字哪里来的"，诚实说是"教学设计的预期值，参照 GRPO/DAPO 论文与 R1 复现社区共识设定的目标，不是我自己四个算法真跑出来对比的"，不要暗示这是自己的实验结果。
