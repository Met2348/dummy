"""NeuralEnsembleWorldModel:用一个小型神经网络 ensemble(而不是频次计数表格)学出的不完美转移模型。

动机——RESULTS.md 留下的两个"下一步"里,这个模块一次性接下两个:

1. 发现一(想象越深、命中率反而越低,是"想象和基线共享同一个不完美模型"这个结构决定的,
   和 H 本身无关)是不是表格模型(纯频次计数)特有的巧合?换成更接近真实 DreamerV3/TD-MPC2
   (神经网络函数逼近)的模型类,这个结构性质还成立吗?——用 sample_mode="consensus" 复刻
   "想象和基线同源"的原始设计来回答。

2. RESULTS.md 解读二发现 visit_count 这个不确定性代理方向不对,怀疑是代理本身选得太粗糙。
   神经网络 ensemble 天然给出一个更有理论依据的代理——集成分歧(ensemble disagreement,
   PETS[1805.12114]、Deep Ensembles[1612.01474]的标准做法)——用 ensemble_disagreement()
   重新做一次同样的分组分析。

额外顺手做的第三件事——sample_mode="member"(PETS 式 trajectory sampling:想象 rollout 时
每步随机抽一个集成成员而不是用集成平均):这让想象在决策那一刻能看到"基线看不到的东西"
(集成内部的分歧),是一种最廉价、不需要任务条件化上下文就能实现的"打破同源结构"版本,
直接呼应 RESULTS.md 下一步第 1 条,但注意这不等于真正的任务条件化想象(idea 1/7 的方向),
只是最小可行的对照组。

接口刻意和 world_model.LearnedWorldModel 保持一致(.transition_dist / .sample_next),
这样 imagination_planner.py 的 no_imagination_action / imagine_action 可以原样复用,一行不改。
"""
from __future__ import annotations

import random

import numpy as np

from gridworld_env import ACTIONS, N_STATES

N_ACTIONS = len(ACTIONS)
ACTION_IDX = {a: i for i, a in enumerate(ACTIONS)}
INPUT_DIM = N_STATES + N_ACTIONS
HIDDEN_DIM = 32
N_ENSEMBLE = 5
EPOCHS = 500
LR = 0.1


def _featurize(s: int, a: str) -> np.ndarray:
    x = np.zeros(INPUT_DIM, dtype=np.float64)
    x[s] = 1.0
    x[N_STATES + ACTION_IDX[a]] = 1.0
    return x


class _Member:
    """单个 1 隐层 softmax 分类器,手写前向/反向传播(numpy,不引入 torch 依赖)。

    40 输入(one-hot 状态 + one-hot 动作)-> 32 隐层(ReLU) -> 36 路 softmax(预测下一状态分布)。
    规模小到手写反传比拉一个框架进来更透明,也更方便逐行讲清楚每一步在算什么。
    """

    def __init__(self, rng: np.random.Generator):
        self.W1 = rng.normal(0, 0.5, size=(INPUT_DIM, HIDDEN_DIM))
        self.b1 = np.zeros(HIDDEN_DIM)
        self.W2 = rng.normal(0, 0.5, size=(HIDDEN_DIM, N_STATES))
        self.b2 = np.zeros(N_STATES)

    def forward_batch(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Z1 = X @ self.W1 + self.b1
        H1 = np.maximum(Z1, 0.0)
        Z2 = H1 @ self.W2 + self.b2
        Z2 = Z2 - Z2.max(axis=1, keepdims=True)
        E = np.exp(Z2)
        P = E / E.sum(axis=1, keepdims=True)
        return H1, P

    def predict_dist(self, s: int, a: str) -> np.ndarray:
        x = _featurize(s, a)[None, :]
        _, p = self.forward_batch(x)
        return p[0]

    def train_epoch(self, X: np.ndarray, targets: np.ndarray, lr: float) -> None:
        """全批量梯度下降,一次 epoch = 用全部(bootstrap后的)数据更新一次。

        softmax+交叉熵组合梯度是经典结果:dL/dz2 = p - onehot(target)(对每个样本求均值)。
        """
        n = X.shape[0]
        H1, P = self.forward_batch(X)
        dZ2 = P.copy()
        dZ2[np.arange(n), targets] -= 1.0
        dZ2 /= n

        dW2 = H1.T @ dZ2
        db2 = dZ2.sum(axis=0)
        dH1 = dZ2 @ self.W2.T
        dZ1 = dH1 * (H1 > 0)  # ReLU 导数;用 H1>0 判断和用 Z1>0 判断数值上等价
        dW1 = X.T @ dZ1
        db1 = dZ1.sum(axis=0)

        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2


class NeuralEnsembleWorldModel:
    def __init__(self, data: list[tuple[int, str, int]], seed: int, sample_mode: str = "consensus"):
        """sample_mode:
          - "consensus":.sample_next 从集成平均分布采样——和 Vhat 用的分布完全同源,
            复刻表格 pilot"想象和基线零信息差"的原始设计,用于直接对比发现一是否是模型类特有的。
          - "member":.sample_next 每步随机抽一个集成成员,再从那个成员自己的分布采样
            (PETS 式 trajectory sampling)——想象这时候能看到集成内部的分歧,是基线 Vhat
            (用集成平均烘焙)看不到的信息,一种最廉价的"打破同源结构"。
        """
        assert sample_mode in ("consensus", "member")
        self.sample_mode = sample_mode
        self.members: list[_Member] = []
        for m in range(N_ENSEMBLE):
            member_rng = np.random.default_rng(seed * 1000 + m)
            boot_rng = random.Random(seed * 1000 + m)
            bootstrap = [data[boot_rng.randrange(len(data))] for _ in range(len(data))]
            X = np.stack([_featurize(s, a) for s, a, _ in bootstrap])
            targets = np.array([ns for _, _, ns in bootstrap], dtype=np.int64)
            member = _Member(member_rng)
            for _ in range(EPOCHS):
                member.train_epoch(X, targets, LR)
            self.members.append(member)

    def with_sample_mode(self, sample_mode: str) -> "NeuralEnsembleWorldModel":
        """返回共享同一批已训练权重、但 sample_next 行为不同的视图——训练只做一次,
        两种想象采样源(consensus/member)在完全相同的模型上公平对比,不会因为重新训练引入额外噪声变量。
        """
        assert sample_mode in ("consensus", "member")
        view = NeuralEnsembleWorldModel.__new__(NeuralEnsembleWorldModel)
        view.members = self.members
        view.sample_mode = sample_mode
        return view

    def _member_dists(self, s: int, a: str) -> list[np.ndarray]:
        return [m.predict_dist(s, a) for m in self.members]

    def _consensus(self, s: int, a: str) -> np.ndarray:
        return np.mean(self._member_dists(s, a), axis=0)

    def transition_dist(self, s: int, a: str) -> dict[int, float]:
        """给 value_iteration 烘焙 Vhat 用:永远是集成平均(consensus),不受 sample_mode 影响——
        这样 Vhat 的定义在两种 sample_mode 之间保持一致,唯一变量是"想象采样时看不看得到集成分歧"。
        """
        p = self._consensus(s, a)
        return {i: float(p[i]) for i in range(N_STATES) if p[i] > 1e-6}

    def sample_next(self, s: int, a: str, rng: random.Random) -> int:
        if self.sample_mode == "consensus":
            p = self._consensus(s, a)
        else:
            member = self.members[rng.randrange(len(self.members))]
            p = member.predict_dist(s, a)
        return rng.choices(range(N_STATES), weights=p.tolist(), k=1)[0]

    def ensemble_disagreement(self, s: int, a: str) -> float:
        """不确定性代理:集成分歧,mean KL(member_i || consensus)。取代表格 pilot 用的 visit_count。"""
        dists = self._member_dists(s, a)
        consensus = np.mean(dists, axis=0)
        eps = 1e-9
        kls = [float(np.sum(p * np.log((p + eps) / (consensus + eps)))) for p in dists]
        return float(np.mean(kls))


if __name__ == "__main__":
    from world_model import collect_rollout_data

    py_rng = random.Random(0)
    data = collect_rollout_data(400, py_rng)

    model = NeuralEnsembleWorldModel(data, seed=0, sample_mode="consensus")

    # sanity 1: transition_dist 归一化
    dist = model.transition_dist(0, "RIGHT")
    total = sum(dist.values())
    assert abs(total - 1.0) < 1e-6, f"transition_dist 未归一化: {total}"

    # sanity 2: 训练之后,模型对"充分采样过的"(s,a)预测应该明显比均匀分布集中(熵更低)
    def entropy(d: dict[int, float]) -> float:
        return -sum(p * np.log(p + 1e-12) for p in d.values() if p > 0)

    common_dist = model.transition_dist(0, "RIGHT")
    uniform_entropy = float(np.log(N_STATES))
    print(f"state=0,RIGHT 的预测分布熵 = {entropy(common_dist):.3f}(36态均匀分布熵上界 = {uniform_entropy:.3f})")
    assert entropy(common_dist) < uniform_entropy - 0.3, "训练后模型对常见(s,a)的预测太接近均匀分布,训练可能没生效"

    # sanity 3: ensemble_disagreement 非负,且和表格模型的 visit_count 一样,应该在"稀疏采样区域"更大
    # (state=0 附近是均匀随机游走里比较容易到达的区域,预期分歧偏低;这里只做非负性和量级合理性检查)
    disagreement = model.ensemble_disagreement(0, "RIGHT")
    assert disagreement >= -1e-9, f"KL散度不应为负: {disagreement}"
    print(f"state=0,RIGHT 的集成分歧(mean KL) = {disagreement:.4f}")

    # sanity 4: with_sample_mode 共享权重,不重新训练;两种模式采样结果都应该是合法状态
    member_view = model.with_sample_mode("member")
    assert member_view.members is model.members, "with_sample_mode 不应该重新训练/复制权重"
    s_consensus = model.sample_next(0, "RIGHT", random.Random(1))
    s_member = member_view.sample_next(0, "RIGHT", random.Random(1))
    assert 0 <= s_consensus < N_STATES and 0 <= s_member < N_STATES

    # sanity 5: 集成成员之间确实存在分歧(不是5个完全一样的模型)——否则"集成分歧"这个信号毫无意义
    dists_at_rare_pair = model._member_dists(35, "UP")  # 右下角附近,采样覆盖大概率比左上角稀疏
    spread = float(np.mean([np.abs(d - np.mean(dists_at_rare_pair, axis=0)).sum() for d in dists_at_rare_pair]))
    assert spread > 1e-6, "5个集成成员的预测几乎完全一致,bootstrap+随机初始化没有产生任何多样性,检查训练逻辑"
    print(f"sanity 5: 集成成员在 state=35,UP 上的平均分布差异(L1) = {spread:.4f}(应明显 > 0)")

    print("neural_ensemble_model.py 自检通过")
