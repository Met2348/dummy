"""GAE（Generalized Advantage Estimation, Schulman et al. 2016），闭卷从零手写。

面试高频度 ****。RLHF/PPO 训练循环里承上启下的一环——reward/value 都算好之后，
下一步就是它；ppo_clip 吃的 advantage 通常就是这里算出来的。

接口约定
--------
    gae_advantage(rewards, values, gamma, lam) -> list[float]，长度 T

    rewards : list[float]，长度 T，r_0 ... r_{T-1}
    values  : list[float]，长度 **T+1**，V(s_0) ... V(s_T)——注意比 rewards 多一个！
              最后一个 V(s_T) 是 bootstrap 值（如果 episode 在 T 步后自然终止，传 0；
              如果是截断/还没结束，传 critic 对 s_T 的估值）
    gamma   : 折扣因子
    lam     : GAE 的 lambda 混合系数，取值范围 [0, 1]
    返回    : list[float]，长度 T，advantage 估计 A_0 ... A_{T-1}

公式
----
    单步 TD 残差:  delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)      (t = 0 .. T-1)
    GAE:           A_t = delta_t + gamma * lam * A_{t+1}            (从后往前递推)
                   边界条件 A_T = 0（视野之外没有 advantage）

    展开写就是加权指数衰减求和：
        A_t = sum_{l=0}^{T-1-t} (gamma*lam)^l * delta_{t+l}

面试常问 / 两个退化情况（check 会真的拿手算小例子对拍）
--------------------------------------------------------
- **lam = 0** 时退化为什么？—— A_t = delta_t + gamma*0*A_{t+1} = delta_t，
  也就是最朴素的单步 TD 残差（一步 bootstrap，方差最小但有偏）。
- **lam = 1** 时退化为什么？—— 展开成全 gamma 折扣的蒙特卡洛优势：
  A_t = sum_{l=0}^{T-1-t} gamma^l * delta_{t+l} = G_t - V(s_t)，
  其中 G_t 是从 t 开始的（bootstrap 到 V(s_T) 的）折扣回报。也就是"回合走到底"，
  无偏但方差大。GAE 就是在这两个极端之间，用 lambda 做一个偏差-方差的插值旋钮。
- 为什么要从**后往前**递推，而不是从前往后？—— A_t 依赖 A_{t+1}（未来），前往后
  递推时 A_{t+1} 还没算出来，只能倒着算，这是 O(T) 高效实现的关键（避免每个 t 都
  重新展开成 O(T) 的求和，否则整体退化成 O(T^2)）。

常见实现陷阱
------------
1. **方向反了**：delta_t 应该用 V(s_{t+1})（下一个状态的值）加 reward，再减去
   V(s_t)（当前状态的值）；有人会写反成 `V(s_t) - V(s_{t+1})` 或者用错时间下标。
2. **values 长度对不上**：values 必须是 T+1（比 rewards 多一个 bootstrap 值），
   如果误把 values 当成长度 T（漏掉最后的 bootstrap V(s_T)），最后一步的 delta
   就会越界或者算错。
3. **递推方向搞反**：必须从 t=T-1 递减到 t=0；如果从 t=0 递增，此时 A_{t+1}
   根本还没被计算出来（依赖关系错误，是"未来指向过去"，不是"过去指向未来"）。
4. **边界条件漏加**：A_T（视野外）应视为 0，最后一步 A_{T-1} = delta_{T-1}
   （因为没有 A_T 项可加），别漏掉或错误地引入一个不存在的 A_T 值。
"""
from __future__ import annotations


def gae_advantage(rewards: list[float], values: list[float], gamma: float, lam: float) -> list[float]:
    """见模块 docstring：从后往前递推 A_t = delta_t + gamma*lam*A_{t+1}，返回长度 T 的 advantage 列表。"""
    raise NotImplementedError("闭卷手写：删除这行 raise，实现 GAE")
