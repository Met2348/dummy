# 08 · RLHF 与对齐

## 这类题在面试里的分量

RLHF/对齐是 NLP/LLM 方向 PhD 面试里出现频率最高的板块之一——不管你面的是对齐研究岗、post-training 工程岗，还是泛化的 LLM 应用岗，"你们公司/组是怎么做 RLHF 的""PPO 和 DPO 你会怎么选""GRPO 和 PPO 到底差在哪"几乎是必问题。这类题的难点不在于"听说过"，而在于面试官会顺着一条主线一路追问到具体数值和工程细节（比如 clip ε 到底是多少、reward model 的 loss 长什么样），背不出具体数字/公式会被认为"只是听过论文标题"。`interview-prep/src/mlqa/qbank.py` 里已经有 RL/RLHF 类的地板速查（policy gradient、PPO clip 是什么、RLHF 三段流程、KL 惩罚、DPO 优势、reward hacking），本讲义在那 6 题基础上往更细的方向深挖。

## 深挖追问链：PPO vs DPO vs GRPO 全景对比

**Q1（面试官）**：说说标准 RLHF 里 PPO 具体是怎么训练的，需要几个模型？

**A1**：标准 RLHF-PPO 训练时同时驻留四个模型：policy(待训练)、reference(算 KL 用，冻结)、reward model(打分，冻结)、critic(价值网络，待训练)。Critic 的作用是估计状态价值 V(s)，结合实际奖励算优势 A=Q-V(常用 GAE)来降低优势估计的方差；PPO 用 clip 机制把新旧策略概率比 r_t(θ)=π_θ/π_θ_old 限制在 [1-ε,1+ε] 内(ε 常取 0.2)，同时用 KL 惩罚把策略锚在参考模型附近防止 reward hacking。这是一整套"采样-打分-更新"的在线 RL 循环，超参敏感、训练容易震荡。

**面试官追问 Q2**：Critic 这么贵，DPO 是怎么绕开这套流程的？

**A2**：DPO 直接推导出一个等价于"RL + KL 约束"最优解的闭式(closed-form)损失，把偏好学习问题转成一个类似交叉熵的监督损失，只需要 policy 和 reference 两个模型，直接在离线收集好的偏好数据集上训练，不需要在线采样、不需要单独的 reward model、也不需要 critic。工程复杂度骤降——训练曲线更平滑，超参也少得多，几乎和普通 SFT 一样简单部署。代价是它是纯 offline 的：一旦策略在训练中偏离较远，后续遇到的输出已经不在原始偏好数据的覆盖范围内(分布外)，DPO 没有"重新采样验证"的机制去发现这种漂移。

**面试官再追问 Q3**：那 GRPO 呢？它既不像 PPO 那么重、也不像 DPO 那样纯 offline，它是怎么做到的？

**A3**：GRPO 保留了 PPO 的在线采样和 clip 机制，但把最贵的部分——critic 网络——直接砍掉。做法是对同一个 prompt 用当前策略采样一组(group)输出(比如 G=8~16 条)，用奖励模型打分后，把每条输出的优势标准化为 A_i=(r_i-mean(r))/std(r)，即减去组内均值、除以组内标准差，用组内的相对表现直接当作优势信号，不需要 critic 去预测 V(s)。这样训练时只需要 policy、reference、reward model 三个模型，比标准 PPO 少了一整个和策略同规模的价值网络，显存和训练开销明显下降，同时仍然是在线采样、能跟随当前策略更新数据分布——某种意义上是"PPO 的在线性 + 比 DPO 更省的显存友好度"之间的一个新平衡点，这也是 DeepSeekMath/DeepSeek-R1 选它的原因。

**小结对比表**（三者本质区别一句话记忆）：
- PPO：critic 估计基线，在线采样，四个模型，工程最重。
- DPO：无采样无 reward model，闭式监督损失，两个模型，最轻但纯 offline。
- GRPO：组内相对奖励替代 critic 当基线，在线采样，三个模型，介于两者之间。

## 其余题目一览

完整答案见 `src/ai_qa/qbank_rlhf.py`（`ai-rlhf-02/03/04/07/09/10/11/12/13/14`，除上面深挖用到的 `ai-rlhf-01/06/08`）：

- **组内优势具体怎么算**（`ai-rlhf-02`）：A_i=(r_i-mean(r))/std(r)，均值当动态基线、标准差稳定尺度。
- **critic 到底贵在哪、省了几个模型**（`ai-rlhf-03`）：critic 和 policy 同规模，去掉后四个模型降到三个。
- **reward model 具体怎么训练**（`ai-rlhf-04`）：从 SFT 模型初始化、换标量输出头，Bradley-Terry 模型 + pairwise ranking loss。
- **SFT 阶段数据怎么构造**（`ai-rlhf-07`）：人工示范 + 真实请求编辑 + 强模型生成筛选，强调覆盖面和标注质量。
- **拒绝采样微调是什么**（`ai-rlhf-09`）：采样 K 条候选、reward model 选最优一条做 SFT，相当于把 best-of-K 蒸馏进参数里。
- **拒绝采样 vs PPO 优缺点**（`ai-rlhf-10`）：更简单稳定，但计算开销大、只用了组内最优一条、样本利用效率不如 GRPO。
- **Constitutional AI 两阶段**（`ai-rlhf-11`）：监督阶段自我批评修改 + RL 阶段用宪法原则做 AI 偏好判断。
- **RLAIF vs RLHF**（`ai-rlhf-12`）：核心区别是偏好标注来源换成 AI，省成本但引入 AI 打分偏差新问题。
- **奖励过优化(reward overoptimization)**（`ai-rlhf-13`）：奖励模型分数涨但真实质量降，Goodhart 定律，靠 KL/集成/early stopping 缓解。
- **on-policy vs offline 的分布覆盖差异**（`ai-rlhf-14`）：PPO 实时跟随策略采样，DPO 固定数据集易在训练后期遇到分布外样本。

## 易错点 / 常见误区清单

1. **把 GRPO 说成"完全不需要采样"**——错，GRPO 依然是在线算法，仍要对每个 prompt 采样一组输出，省掉的只是 critic 网络，不是采样过程本身。
2. **混淆 clip 和 KL 惩罚**——clip 是单步更新内部的约束（限制概率比），KL 惩罚是整条训练轨迹宏观尺度上的约束（防止策略整体漂移），两者互补、很多实现同时使用，不是二选一。
3. **说不出 PPO clip ε 的具体数值**——常取 0.1~0.2，InstructGPT 等典型实现用 0.2，只说"limit the ratio"而说不出数值范围会被追问到底。
4. **认为 reward model 的分数是绝对可比的**——Bradley-Terry 模型学到的是相对偏好排序，不同 reward model 之间、甚至同一个 reward model 在不同分布上的绝对分数并不天然可比。
5. **把 RLAIF 和 Constitutional AI 划等号**——CAI 是一套包含"自我批评-修改"监督阶段 + RLAIF 强化阶段的完整方法，RLAIF 只是它 RL 阶段用到的"用 AI 而非人类生成偏好"这一个技术点，二者是包含关系不是同义词。
6. **认为 DPO 天然比 PPO"效果更好"**——DPO 的优势是工程简单/稳定，不是效果必然更高；在允许充分在线采样、reward model 质量好的场景下，PPO/GRPO 往往仍有更高的效果上限，这是"稳定性/工程复杂度"和"效果上限"两个不同维度，不要混为一谈。
