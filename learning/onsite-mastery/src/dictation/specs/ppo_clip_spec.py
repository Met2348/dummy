"""PPO-Clip 代理目标函数（surrogate objective），闭卷从零手写。

面试高频度 ***** 。RLHF/post-training 岗位手撕代码 T0 级考点，几乎必考。

接口约定
--------
    ppo_clip_objective(ratio, advantage, clip_eps) -> 与 ratio 同形状的逐元素目标值

    ratio       : 重要性采样比率 pi_theta(a|s) / pi_theta_old(a|s)，标量或 array-like（可以是
                  python float、numpy 数组，也可以是 torch 张量——用你自己习惯的库都行）
    advantage   : 优势估计 A(s,a)，与 ratio 同形状
    clip_eps    : 裁剪半径 eps（PPO 论文取 0.1~0.2，OpenAI 默认 0.2）

    返回值不做 batch 维度的 mean 归约（逐元素返回），如果要塞进优化器当 loss，
    调用方自己取 `-ppo_clip_objective(...).mean()`。

公式（Schulman et al. 2017, PPO 论文式 7）
------------------------------------------
    L^CLIP(theta) = min( ratio * A,  clip(ratio, 1-eps, 1+eps) * A )

这是一个要**最大化**的目标（不是 loss！本函数名是 objective 不是 loss，注意区分）。

为什么长这样（面试常见追问）
----------------------------
- 为什么要 clip？—— 防止一次梯度更新把 ratio 推得离 1 太远：ratio 偏离 1 越多，
  重要性采样的方差越大、off-policy 修正也越不可信，一步走太远可能直接把策略带崩。
- 为什么外面还要套一层 min，而不是直接把 ratio clip 之后再乘 A 就完事？
  —— min 保证目标函数是一个**悲观下界**（pessimistic lower bound）：
  min 之后，目标函数只会在"clip 会让目标变得对 agent 更有利"的那个方向上生效，
  而在"clip 让目标对 agent 更不利"的方向上完全不生效（自动退化为未裁剪值）。
  这正是 min/max 用反是本题头号送分坑的原因——如果错用 max，就变成了乐观上界，
  等于放任 ratio 无限制地朝对 agent"看起来更好"的方向被放大，clip 完全失去意义。
- 四个象限（务必在草稿纸上过一遍，check 会真的按这四种组合去测）：
    ratio > 1+eps 且 A > 0  -> clip 生效（min 选中 clip 后的更小值，压制虚假增益）
    ratio > 1+eps 且 A < 0  -> clip 不生效（min 选中未裁剪值，本来就该继续往下走）
    ratio < 1-eps 且 A < 0  -> clip 生效（min 选中 clip 后的更小值）
    ratio < 1-eps 且 A > 0  -> clip 不生效（min 选中未裁剪值）
  记忆技巧：clip 只在"ratio 已经跑出信任域、且继续跑出去还能让未裁剪目标变得更大"
  的那个方向上才会真正把目标摁下来；另一侧 min 自动选未裁剪值，什么都不用做。

常见实现陷阱
------------
1. **min/max 用反**：写成 max(...) 会变成乐观上界，允许无限制放大更新——这是本题
   最容易犯、也最容易被面试官一眼看穿的错误。
2. **clip 的上下界写反或写错**：应该是 `clip(ratio, 1-eps, 1+eps)`，不是
   `clip(ratio, eps, 1-eps)`，也不是把 1+eps/1-eps 顺序搞反。
3. **对 A 正负写 if/else 分支**：min 这个公式在 A 正负两种情况下都自动正确，
   完全不需要手写 if advantage > 0 的分支——手写分支反而更容易把符号搞反。
4. **信任域内退化检验**：当 ratio 落在 [1-eps, 1+eps] 内时，clip(ratio,...) 恒等于
   ratio 本身，此时 min(ratio*A, ratio*A) 必须精确退化为未裁剪值 ratio*A，
   这是本题最基础的一条防线，check 会逐点验证。
"""
from __future__ import annotations


def ppo_clip_objective(ratio, advantage, clip_eps):
    """见模块 docstring：返回 min(ratio*A, clip(ratio,1-eps,1+eps)*A)，逐元素、不做 mean。"""
    raise NotImplementedError("闭卷手写：删除这行 raise，实现 PPO-Clip 代理目标函数")
