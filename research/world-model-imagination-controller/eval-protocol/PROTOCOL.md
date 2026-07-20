# Idea 10 Pilot:评测协议设计

> 按 Weikai 2026-07-14 的指导:"開搞第一步永遠是先確立evaluation protocol(數據集、指標、baseline)"。
> 这是 [`00-brainstorm-10-ideas.md`](../00-brainstorm-10-ideas.md) 里 Idea 10(诊断性研究)的原型实现——
> **不是论文最终实验**,是验证测量方法论本身站得住脚、并拿到第一批真实信号的 pilot。范围声明见文末。

## 研究问题

在提出任何新 controller 之前,先系统测量:想象(多步 Monte Carlo rollout 搜索)到底多大比例时候真的改变了
决策、这些改变里有多少是变好、有多少是变坏——把"现有想象预算分配粗糙、容易浪费"这个论文开篇论断,从直觉
变成实测数据。

## 数据集/任务选择,及为什么

选一个真实转移函数完全已知的合成环境(`SlipperyGridWorld`,6x6 随机滑动格子世界),而不是直接上 Atari/
DMControl/机器人。原因:诊断性研究需要一个"裁判"——要判断"想象选的动作是不是真的更好",必须有 ground-truth
最优 Q\*(s,a) 做参照,而这只有在转移函数已知、状态空间小到能精确 value iteration 的环境里才能干净地拿到。
用真实 Atari/机器人环境的话,"哪个动作真的更好"本身就要靠另一个昂贵的评测才能估计,会把"测量工具本身准不准"
和"想象有没有用"这两个问题混在一起。

- Ground truth oracle:`gridworld_env.py` 的 `true_transition_dist` + `value_iteration`,只用来 (a) 生成训练数据
  (b) 事后当裁判,智能体自己的模型/规划器全程看不到这个函数。
- 智能体自己的"世界模型":`world_model.py` 的 `LearnedWorldModel`,只从 400 条随机游走采样的真实交互数据里
  用频次估计学出来的模型——128 个 (s,a) 对里只覆盖到 95 个左右(见运行输出),真实存在误差,不是假装不完美。

## 指标定义

对每个非终止状态 s(共 32 个),比较两种策略给出的动作:
- **决策改变率**(decision change rate):`P(a_imagination ≠ a_no-imagination)`,想象改变了多大比例的决策
- **命中率**(hit rate,仅统计改变了的决策):`P(Q*(s,a_imagination) > Q*(s,a_no-imagination) | 改变了)`
- **帮倒忙率**(hurt rate,仅统计改变了的决策):`P(Q*(s,a_imagination) < Q*(s,a_no-imagination) | 改变了)`
- **不确定性关联**:按状态的平均 `visit_count`(学模型时这个状态被访问的次数,当"模型置信度"的代理信号)
  分高/低两组,比较两组的命中率/帮倒忙率是否有系统性差异

## Baseline 设计,及为什么这样设计

两个策略共享**同一个**不完美 `LearnedWorldModel`,唯一区别是要不要在决策那一刻做额外的多步 Monte Carlo 搜索:

- **no-imagination(H=0)**:先对 `LearnedWorldModel` 跑一次精确 value iteration,烘焙出价值函数 V̂(见
  `imagination_planner.no_imagination_action`),决策时只做一步精确期望展开,不做任何决策时搜索——类比一个
  已经训练好的 critic/amortized policy。
- **imagination(固定预算 K,H)**:决策时用同一个模型跑 K 条深度 H 的 Monte Carlo 想象 rollout(每步真采样,
  不是精确期望),rollout 中途续跑策略=贪心 V̂,跑满 H 步后用 V̂ 做 bootstrap(参照 TD-MPC [2203.04955]、
  MBPO [1906.08253] 的"短 rollout + 学到的终值函数"设计)。

这样设计是刻意的:**两个策略用的是同一个模型、同一个价值函数,想象唯一能带来的"新信息"只可能来自
Monte Carlo 采样本身**,不是因为想象能看到基线看不到的额外知识。这是诊断性研究应该有的"控制变量"纪律——
先把"同一个不完美模型,搜不搜都一样吗"这个最基础的问题测干净,再谈"给想象额外信息(更好的模型/任务条件/
不确定性信号)"这些更复杂的设计。

Budget 扫描:固定 K=5 扫 H∈{1,2,3,5,8};固定 H=3 扫 K∈{1,3,5,10}。每组配置独立重复 5 个随机种子
(数据采样+模型学习+评测全部重新来一遍),报告均值±标准差。

## 范围声明(重要,不要误读这批数字)

**这是原型规模的 pilot,不是论文最终实验**:
- 环境是 6x6 合成格子世界,不是 Atari/DMControl/真实机器人——目的是验证测量方法论、拿第一批方向性信号,
  不是产出可以直接放进论文的最终结果。
- 只有 32 个状态、5 个随机种子,统计功效有限,尤其是"不确定性关联"那部分分组样本量小,结论要谨慎。
- world model 是纯频次估计的表格模型,不是神经网络/扩散模型,不能直接类比视频生成的想象。
- 下一步如果方向选定,需要把同一套测量协议(决策改变率/命中率/预算扫描/不确定性关联)搬到真实 world model
  checkpoint(如 DreamerV3、TD-MPC2)和真实任务(DMControl/Atari)上重新做一遍,这套代码是可复用的骨架,
  不是终点。

结果见 [`RESULTS.md`](RESULTS.md)。
