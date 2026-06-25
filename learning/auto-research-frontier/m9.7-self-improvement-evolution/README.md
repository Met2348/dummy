# 9.7 · 自我改进 / 自动算法发现（Self-Improvement & Evolution）

> 最激进的一支：能不能让 AI **改进 AI 自己**？本模块用一个最小的进化档案循环，
> 让你亲眼看到这条路上最大的坑——**fitness 一旦可刷，自我改进会自动找到那条作弊捷径**（reward hacking）。

## 一句话结论（本模块跑出来的两条进化轨迹）

```
在 naive fitness（代理：泄漏测试集上判分，可背书刷）下进化：
  gen 0  thr=3.0 memo=0  naive=0.667 holdout=0.500
  gen 2  thr=3.0 memo=2  naive=1.000 holdout=0.500   ← 代理涨满，真本事没动
在 holdout fitness（真目标：没见过的数据）下进化：
  gen 5  thr=0.5 memo=0  naive=1.000 holdout=1.000   ← 调对 threshold，真泛化
```

同一个自我改进循环，只换了优化的 fitness：
- 优化**代理**：它 2 代就把泄漏测试集**背**满（`memo` 0→2），naive 冲到 1.0，
  但 `threshold` 一动没动、**holdout 死守 0.500**。档案在涨，系统没变强。
- 优化**真目标**：背书没用，只能把 `threshold` 调到 ~0，holdout 真的 0.5→1.0。

## 自我改进的三种形态

| 形态 | 进化的是什么 | 代表 |
|------|------------|------|
| 进化"解" | 候选答案/程序 | [AlphaEvolve](https://arxiv.org/abs/2506.13131) |
| 进化"自己" | agent 自身的代码 | [Darwin Gödel Machine](https://arxiv.org/abs/2505.22954) |
| agent 设计 agent | agent 架构 | ADAS |

理论上的 Gödel Machine 要求"可证明地更好"才自改；经验版（DGM/AlphaEvolve）退而用
**fitness + keep-if-better + 档案**——也正因如此，**fitness 的可信度就是一切**。

## 跑起来

```powershell
python src/run.py            # 两种 fitness 下的进化轨迹 + reward hacking 对照
python src/run.py --gens 20

python scripts/eric_3080ti_env_audit.py --runbook --tests `
  --modules auto-research-frontier/m9.7-self-improvement-evolution `
  --json-out $env:TEMP/m9.json --md-out $env:TEMP/m9.md
```

## 目录

```
m9.7-self-improvement-evolution/
├── runbook.yaml
├── lectures/
│   ├── 01-three-flavors-of-self-improvement.md  AlphaEvolve/DGM/ADAS + Gödel Machine
│   ├── 02-fitness-gets-gamed.md                 核心：reward hacking
│   └── 03-archive-and-guards.md                 档案/开放式搜索 + 守卫
└── src/
    ├── run.py
    ├── selfimprove/
    │   ├── genome.py     Genome(threshold 真泛化 + memo 作弊查找表) + naive/holdout fitness
    │   └── evolve.py     确定性进化档案循环(变异枚举 + keep-if-better)
    └── tests/test_evolve.py  6 测试：优化代理=刷分(holdout不动)、优化真目标=真涨
```

## Hands-on（轮到你）

`evolve.py` 的 `_mutations` 现在只有两种杠杆：调 threshold（真改进）和背 LEAKED 点（作弊）。

1. **加一种作弊**：让候选能"加宽 LEAKED 命中范围"（比如对接近背过的点也返回背的答案）。
   naive 涨得更快吗？holdout 呢？体会"作弊面"一旦变大，reward hacking 更难防。
2. **做一个抗刷 fitness**：把 `naive_fitness` 换成"泄漏集 + 一个隐藏 mini held-out"的组合，
   看背书还刷不刷得动。多大比例的 held-out 才能压住作弊？（呼应 9.6 的 held-out。）
3. **加守卫**：进化时记录每个被纳入档案的解"靠什么涨的"（threshold 变了还是 memo 变了），
   自动给"只靠 memo 涨"的解打上 ⚠ reward-hacking 标——这就是 9.8 的诚信守卫雏形。

## 桥接

- **process-reward**（你已知 reward 多易被刷）· **rl-sota-2026** · rl-foundations。
- 呼应 **9.6**（held-out 抗刷）；通往 **9.8**（把"靠背书涨分"当成攻击，加独立验证守卫）。
