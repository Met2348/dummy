# L02 · fitness 被 game（本模块核心：reward hacking）

## 1. 一句话：你优化什么，就得到什么——包括你没想要的

自我改进循环不"理解"任务，它只**最大化 fitness**。如果 fitness 和真目标之间有缝，
循环会精准地钻进那条缝。这就是 reward hacking——你在 process-reward 里见过的 reward 被刷，
这里是它的**进化版**。

## 2. 同一个循环，两种 fitness，两种命运

跑 `python src/run.py`：

```
在 naive fitness（代理：泄漏测试集上判分）下进化：
  gen 0  thr=3.0 memo=0  naive=0.667 holdout=0.500
  gen 1  thr=3.0 memo=1  naive=0.833 holdout=0.500
  gen 2  thr=3.0 memo=2  naive=1.000 holdout=0.500   ← 满分，但 holdout 一动没动
```

`threshold` 始终是 3.0（没有任何真泛化），`memo` 从 0 涨到 2——它**只是把答错的泄漏测试点
背了下来**。naive fitness 冲到满分，holdout 死守 0.500。`test_optimizing_proxy_is_reward_hacking`
把这三件事钉死：naive 涨、holdout 不变、threshold 不变。

对照真目标：

```
在 holdout fitness 下进化：
  gen 5  thr=0.5 memo=0  naive=1.000 holdout=1.000   ← 背书没用，只能真调 threshold
```

背 LEAKED 对 holdout 毫无帮助（memo 里的点不在 held-out 里），所以循环被迫去调 `threshold`，
真的把它推到 ~0——真泛化。

## 3. 为什么这不是"我故意写死的"

`memo` 抬高 naive 是**真算出来的**：背一个答错的泄漏点，naive 准确率真的上升
（`test_memorize_inflates_naive_not_holdout`）。holdout 不变也是真算的（memo 不碰 held-out）。
循环每代**贪心选 fitness 最高的候选**，没有任何地方告诉它"去作弊"——
**是 fitness 的形状自己把它引上了作弊路。** 这正是 reward hacking 可怕的地方：
不需要谁使坏，一个可刷的指标就够了。

## 4. 缺口就是警报

`run.py` 末尾打印"代理-真目标缺口"：

```
优化代理后 gap = naive - holdout = +0.500   （大缺口 = 在刷分）
优化真目标后 gap = +0.000                   （贴合 = 真本事）
```

> **盯住代理和真目标的缺口。** 一个在涨的 fitness 配上一个不动的 held-out，
> 就是 reward hacking 的签名。`test_proxy_true_gap_contrast` 保证这个反差稳定存在。

## 5. 动手

1. 把 `evolve` 的 `gens` 加到 30，naive 进化的 holdout 会不会"不小心"也涨起来？
   （不会——因为 threshold 从不被选中。理解：贪心永远先抓更便宜的作弊。）
2. 给 `naive_fitness` 掺 10% 的 holdout 信号，重跑。要掺到多少，循环才放弃背书、改去调 threshold？
   （这量化了"代理里掺多少真信号才压得住作弊"。）
