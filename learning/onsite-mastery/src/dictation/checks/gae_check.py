"""gae_spec 的纯断言检验：不 import solutions/，全部用手算好的小例子对拍。"""
from __future__ import annotations

import math


def _assert_close_list(got, want, msg):
    assert len(got) == len(want), f"{msg}: 长度不对, got len={len(got)}, want len={len(want)}"
    for i, (g, w) in enumerate(zip(got, want)):
        assert math.isclose(g, w, rel_tol=1e-6, abs_tol=1e-9), f"{msg}: 第{i}项 got={g}, want={w}"


def check(target) -> None:
    # 共用小例子: T=3, gamma=0.9
    # delta_0 = 1.0 + 0.9*0.5 - 0.5 = 0.95
    # delta_1 = 2.0 + 0.9*0.5 - 0.5 = 1.95
    # delta_2 = 3.0 + 0.9*0.0 - 0.5 = 2.5
    rewards = [1.0, 2.0, 3.0]
    values = [0.5, 0.5, 0.5, 0.0]
    gamma = 0.9

    # ---- 1) lam=0 应退化为单步 TD 残差: A_t == delta_t ----
    got = target(rewards, values, gamma, 0.0)
    _assert_close_list(got, [0.95, 1.95, 2.5], "lam=0 应退化为单步 TD 残差 delta_t")

    # ---- 2) lam=1 应退化为蒙特卡洛优势 A_t = G_t - V(s_t) ----
    # G_0 = 1 + 0.9*2 + 0.81*3 + 0.729*0 = 5.23  -> A_0 = 5.23-0.5=4.73
    # G_1 = 2 + 0.9*3 + 0.81*0 = 4.7             -> A_1 = 4.7 -0.5=4.2
    # G_2 = 3 + 0.9*0 = 3                        -> A_2 = 3   -0.5=2.5
    got = target(rewards, values, gamma, 1.0)
    _assert_close_list(got, [4.73, 4.2, 2.5], "lam=1 应退化为蒙特卡洛优势 G_t - V(s_t)")

    # ---- 3) 中间 lam 值的手算对拍（全 0 value，更容易手算，交叉验证递推方向没搞反） ----
    # delta = [1,1,1] (因为 values 全 0，delta_t = r_t)
    # A_2 = 1.0
    # A_1 = 1 + 0.9*0.95*1.0        = 1.855
    # A_0 = 1 + 0.9*0.95*1.855      = 2.586025
    got = target([1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0], 0.9, 0.95)
    _assert_close_list(got, [2.586025, 1.855, 1.0], "lam=0.95 中间值手算对拍")

    # ---- 4) 单步(T=1) 退化情况：不管 lam 是多少，A_0 都应该等于 delta_0 ----
    got = target([2.0], [1.0, 0.0], 0.99, 0.7)
    delta0 = 2.0 + 0.99 * 0.0 - 1.0
    _assert_close_list(got, [delta0], "T=1 时 lam 不影响结果，A_0 必须等于 delta_0")
