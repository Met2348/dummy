# L12 · Capstone — 4 method × 3 target ASR 矩阵

## 目标

构建标准 "red-team report card"：
- 3 个 mock target（strong / weak / no_safety）
- 4 攻击方法（gcg / pair / autodan / crescendo）
- 输出 12-cell ASR 矩阵

## 设计

```
src/red_team_matrix.py
├── make_target_zoo()   # 3 mock targets
├── run_matrix()        # 跑全部组合
└── to_md()             # markdown 输出
```

## 跑

```python
from red_team_matrix import run_matrix, to_md

mat = run_matrix()
print(to_md(mat))
```

预期输出（精简）：
```
# Red-team ASR matrix
Rows = target, Cols = attack method

| target \ method  | autodan | crescendo | gcg | pair |
|---|---|---|---|---|
| strong_safety   | 0%      | 0%       | 0%  | 0%   |
| weak_safety     | 67%     | 67%      | 67% | 67%  |
| no_safety       | 100%    | 0%       | 0%  | 0%   |
```

（数字会因 mock 触发器配置而变化）

## 红队报告卡解读

矩阵看：
- **行**：哪个 target 最安全？
  - strong_safety > weak_safety > no_safety
- **列**：哪个攻击最强？
  - crescendo 在 multi-turn 上特别强
  - GCG 需 white-box
- **对角线 zero**：理想情况

## 真世界对照（HarmBench 2024）

```
target \ method  | GCG  | PAIR | AutoDAN | DAN |
Llama-2-7B      |  46% |  10% |   25%   | 1%  |
GPT-3.5         |  86% |  60% |   25%   | 1%  |
GPT-4           |  47% |  40% |   30%   | 0%  |
Claude 3 Opus   |   2% |   9% |    1%   | 0%  |
```

→ frontier model 都已 < 10%，但仍非零。

## 退出条件

- 3 × 4 矩阵生成
- strong_safety 全 0
- weak_safety / no_safety 有 hits
- self_test PASS

## 接 Topic 6

Topic 6 (safety-defense) 会用同样矩阵，加入：
- Llama Guard 3 (input/output classifier)
- Constitutional Classifier
- 看 ASR 降多少

## 一句话

> Red-team 矩阵 = 安全研究的"标准报告卡"。
