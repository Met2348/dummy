"""GRPO（Group Relative Policy Optimization, DeepSeekMath 2024）组内相对优势，闭卷从零手写。

面试高频度 ****。GRPO 相比 PPO 最大的卖点就是"不需要单独训一个 value model 估计
baseline，用同一个 prompt 采样出的一组 response 的 reward 互相做基准"——这个函数
就是那个"互相做基准"的核心计算，经常被要求当场写 3-5 行代码。

接口约定
--------
    grpo_group_advantage(rewards: list[float], eps: float = 1e-4) -> list[float]

    rewards : 同一个 prompt 采样出的**一组** response 的标量 reward，长度 G（group size）
    eps     : 防止除零的小常数，默认 1e-4（和大多数开源 GRPO 实现一致的量级）
    返回    : 与 rewards 等长的组内标准化 advantage 列表

    **注意**：这个函数只处理"同一个 prompt 的一组 rollout"。真实训练里一个 batch
    通常包含多个不同 prompt 各自的 group，要对每个 group 分别调用这个函数，
    不能把不同 prompt 的 reward 混在一起算 mean/std（否则组间的难度差异会污染
    组内的相对排序，这是概念上最常见的误用）。

公式
----
    mean_r = mean(rewards)
    std_r  = population std(rewards)     # 除以 G，不是 G-1！
    A_i = (r_i - mean_r) / (std_r + eps)

面试常问
--------
- 为什么用 population std（除以 G）而不是 sample std（除以 G-1）？
  —— 大多数开源 GRPO 实现（含 DeepSeekMath 原论文的写法）用的是有偏估计（除以 G），
  这里是对"这一组样本"做描述性标准化，不是在估计某个更大总体的方差，用 G-1
  没有统计学意义；而且 G-1 在 group size=1 时会除零错误更严重（分母直接是 0
  而不是"std 恰好为 0"，两者要用同一个 eps 兜底但含义不同）。
- 为什么需要 eps？—— 当一组 response 的 reward 全部相同时（常见于：全对/全错的
  数学题，reward model 对这批采样打分完全一致），std=0，不加 eps 直接除零。
  加了 eps 之后，这种情况下 advantage 应该全部是 0（没有相对好坏的信号，
  没有梯度也是合理的——这组样本对这次更新不提供有效信息）。
- 这和 PPO 里 GAE+value model 算 advantage 比，好处/坏处是什么？—— 好处是不用
  再训一个 value model（省一半显存/一次前向），坏处是需要对同一个 prompt
  采样多个 response（group size 通常 8~64），推理开销从"每个 prompt 1 次"
  变成"每个 prompt G 次"。

常见实现陷阱
------------
1. **忘记 eps 防护**：一组 reward 全相同时 std=0，直接除会报 ZeroDivisionError
   或者产出 NaN/Inf——check 会专门测这个情况，并且验证输出必须精确是全 0，不是
   NaN。
2. **用错 std 的分母**：用 G-1（样本标准差 ddof=1）而不是 G（总体标准差 ddof=0），
   group size=1 时 G-1=0 会让"看似加了 eps"的代码在算 std 这一步就先崩了
   （eps 加在最终除法上救不了 std 计算本身除零）。
3. **跨组混算**：把多个不同 prompt 的 reward 拼在一起传进来当一个"组"计算 mean/std，
   这不是函数本身的 bug，但是调用方最容易犯的概念错误，面试官经常追问这一点。
4. **输出应该均值约为 0**：标准化之后整组 advantage 的均值应该几乎精确是 0
   （因为减去的就是这组自己的均值），如果实现里减错了 mean（比如用了全局的
   moving average 而不是这一组自己的 mean），这条检验会失败。
"""
from __future__ import annotations


def grpo_group_advantage(rewards: list[float], eps: float = 1e-4) -> list[float]:
    """见模块 docstring：返回 (r_i - mean(r)) / (population_std(r) + eps)，与 rewards 等长。"""
    raise NotImplementedError("闭卷手写：删除这行 raise，实现 GRPO 组内相对优势")
