# L03 · CoT vs ToT vs GoT

## 推理结构三代

| 代 | 形状 | 论文 | 时间 |
|----|------|------|------|
| **CoT** | Linear chain | Wei 2022 | NeurIPS |
| **ToT** | Tree of Thoughts | Yao 2023 | NeurIPS |
| **GoT** | Graph of Thoughts | Besta 2023 | AAAI |

## CoT (Chain-of-Thought)

```
Q → step 1 → step 2 → step 3 → A
```

- "Let's think step by step"
- 一条线
- 失败：错一步全错

## ToT (Tree of Thoughts)

```
        Q
       /|\
      s1 s1' s1''   ← N candidate
     /|
    s2 s2'
   /
  s3 → A
```

- BFS/DFS 搜索
- 每步生成 N 候选 → vote / value → 留 best k
- 游戏 24 / 创意写作领先 CoT

## GoT (Graph of Thoughts)

```
   s1 ─→ s2
    \   /
     s3 ←── s4
```

- 可合并、可回边
- 比 ToT 灵活，但实现复杂

## 选型

| 场景 | 推荐 |
|------|------|
| 数学 / 简单推理 | CoT 即可 |
| 游戏 24 / 规划问题 | ToT |
| 复杂组合 | GoT (慎用，成本高) |
| Agent loop | ReAct（CoT + 工具）|

## 与 Agent 的关系

Agent loop 的 Thought 段就是 **mini-CoT**。如果想增强：
- LATS (Language Agent Tree Search, Zhou 2023) → ReAct + MCTS = ReAct ToT
- ReWOO → plan 阶段用 ToT，execute 阶段单线

## 关键文献

- CoT: Chain-of-Thought Prompting Elicits Reasoning (Wei 2022)
- ToT: Tree of Thoughts: Deliberate Problem Solving (Yao 2023)
- GoT: Graph of Thoughts: Solving Elaborate Problems (Besta 2023)
- LATS (Zhou 2023)

## 退出条件

- CoT/ToT/GoT 形状会画
- 知道 ToT 在游戏 24 强于 CoT
- 知道 ReAct = CoT + tool

## 一句话

> CoT 是线，ToT 是树，GoT 是图 —— 复杂度递增，agent loop 默认用 CoT-flavor 即可。
