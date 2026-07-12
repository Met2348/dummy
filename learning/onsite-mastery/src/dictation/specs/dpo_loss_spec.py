"""DPO（Direct Preference Optimization, Rafailov et al. 2023）损失函数，闭卷从零手写。

面试高频度 *****。对齐/post-training 岗位仅次于 ppo_clip 的必考点，经常和
"讲一下 DPO 怎么从 RLHF 的 KL 约束奖励最大化问题推出来的"一起被连续追问。

接口约定
--------
    dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta) -> 与输入同形状的逐元素 loss

    policy_chosen_logps   : 当前策略模型对 chosen 回复的 **序列级** log-prob（标量或 array-like）
    policy_rejected_logps : 当前策略模型对 rejected 回复的序列级 log-prob
    ref_chosen_logps      : 参考（冻结）模型对 chosen 回复的序列级 log-prob
    ref_rejected_logps    : 参考模型对 rejected 回复的序列级 log-prob
    beta                  : 温度系数（越大 = 越强约束在 ref 模型附近，典型取 0.1）

    返回逐样本 loss，不做 batch 维度的 mean 归约——调用方自己 `.mean()`。

公式推导（面试常被要求现场推一遍）
----------------------------------
RLHF 的目标是在 KL(pi || pi_ref) 约束下最大化奖励期望，这个约束优化问题的闭式解是
    pi*(y|x) ∝ pi_ref(y|x) * exp(r(x,y)/beta)
反解出隐式奖励：
    r(x,y) = beta * log( pi(y|x) / pi_ref(y|x) ) + const(x)
代入 Bradley-Terry 偏好模型 P(y_w > y_l) = sigmoid(r(x,y_w) - r(x,y_l))，const(x) 相减抵消，得到：
    L_DPO = -log sigmoid( beta * [ (log pi(y_w|x)-log pi_ref(y_w|x)) - (log pi(y_l|x)-log pi_ref(y_l|x)) ] )
          = -log sigmoid( beta * (pi_logratios - ref_logratios) )
其中 pi_logratios = policy_chosen_logp - policy_rejected_logp，
     ref_logratios = ref_chosen_logp  - ref_rejected_logp。

本质上就是一个对"隐式奖励差"做 **logistic 回归**——和 RLHF 里训 reward model 用的
Bradley-Terry loss 是同一个函数形式，只是这里奖励是用 policy/ref 的 log-prob 比值
隐式表示出来的，不需要单独训一个 reward model。

面试常问
--------
- 为什么要减 ref 的 logratios，而不是只用 policy 自己的 chosen-rejected 差？
  —— 减掉 ref 是为了做"相对参考模型的改变量"计价，否则模型只要整体推高所有回复
  的绝对概率（不区分好坏）就能刷低 loss，起不到约束在 ref 附近、防止 reward hacking
  / 灾难性遗忘的作用。这是本题最核心的一条检验：**policy 和 ref 完全相同时，
  隐式奖励差恒为 0，loss 必须精确等于 -log(0.5) = ln(2)**，与两个 log-prob 的
  绝对数值大小无关。
- beta 的作用？—— 越大，对隐式奖励差的区分度越敏感（同样的 log-prob 差异，
  beta 越大 loss 下降越快，等价于越不信任 KL 约束，允许 policy 离 ref 更远）。

常见实现陷阱
------------
1. **完全漏掉 ref 项**：直接算 `beta*(policy_chosen_logp - policy_rejected_logp)`，
   这样即便 policy==ref，只要 chosen/rejected 的绝对 log-prob 有差异，loss 就不等于
   ln(2)——check 会专门用 policy==ref 但 chosen != rejected 的输入把这个 bug 揪出来。
2. **chosen/rejected 顺序或符号写反**：应该是 chosen 减 rejected（chosen 的隐式
   奖励应该越高、loss 应该越低），写反会导致 loss 随 chosen 变好反而上升——
   check 里的单调性测试专门测这个方向。
3. **数值稳定性**：直接算 `-log(1/(1+exp(-x)))` 在 x 很负时 exp(-x) 会溢出；
   更稳的写法是用 softplus 恒等式 `-log_sigmoid(x) = log(1+exp(-x)) = softplus(-x)`，
   或者用 `np.logaddexp(0, -x)` / `torch.nn.functional.softplus(-x)`。
4. **归约方式**：这里返回的是逐样本 loss，不要在函数内部偷偷做了 `.mean()`——
   check 会直接对单个标量/单个样本调用，提前 mean 会导致输出维度不对。
"""
from __future__ import annotations


def dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta):
    """见模块 docstring：返回 -log sigmoid(beta * (pi_logratios - ref_logratios))，逐元素、不做 mean。"""
    raise NotImplementedError("闭卷手写：删除这行 raise，实现 DPO 损失")
