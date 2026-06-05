# L13 · Capstone-2 — mini-Arena + 红队 + 防御

## 三段式 Capstone-2

### Part A: mini-Arena

```python
from mini_arena import run_capstone_arena, to_md
print(to_md(run_capstone_arena()))
```

输出排行（约定）：
```
| 1 | r1_tiny  | 1700+ |
| 2 | phi_tiny | 1600  |
| 3 | dpo      | 1550  |
| 4 | lora     | 1400  |
| 5 | vanilla  | 1300  |
```

### Part B: 红队

```python
from mini_red_team import run_red_team, to_md
print(to_md(run_red_team()))
```

输出：
```
| ckpt | direct | persona_wrap | multi_turn | mean |
|---|---:|---:|---:|---:|
| vanilla   | 100% | 100% | 100% | 100% |
| lora      |   0% |   0% |   0% |   0% |
| dpo       |   0% |   0% |   0% |   0% |
| r1_tiny   |   0% |   0% |   0% |   0% |
| phi_tiny  |   0% |   0% |   0% |   0% |
```

vanilla 全漏。

### Part C: 防御加固

```python
from mini_defense import compare_defense, to_md
print(to_md(compare_defense()))
```

输出：
```
| ckpt | no_def | with_def | reduction |
|---|---:|---:|---:|
| vanilla | 100% | 0% | -100% |
| lora    |   0% | 0% |  -0%  |
| dpo     |   0% | 0% |  -0%  |
| r1_tiny |   0% | 0% |  -0%  |
| phi_tiny|   0% | 0% |  -0%  |
```

→ 关键 demo：**input classifier 救活弱 vanilla**。

## 完整 capstone-2 输出

```python
print("# Capstone-2: Arena + Red-team + Defense")
print()
print(arena.to_md(arena.run_capstone_arena()))
print()
print(red.to_md(red.run_red_team()))
print()
print(defense.to_md(defense.compare_defense()))
```

## 三者关系

```
mini-Arena 测"会聊"
   ↓
红队 测"会守"
   ↓
防御 测"会救"

合起来 = "聊得好 + 守得住 + 救得回"
```

## 真世界对照（HarmBench 2024-2025）

| ckpt class | 红队 ASR | + Llama Guard 3 | + Constit. Cls |
|------------|----------|------------------|----------------|
| Vicuna-7B (类 vanilla) | 80% | 30% | 10% |
| Llama-3 + RLHF (类 dpo) | 25% | 12% | 4% |
| Claude 3.7 (类 phi_tiny) | 5% | — | 0.4% |

我们 mock 数字与 real 趋势一致：vanilla 漏，强 RLHF ckpt 抗。

## 退出条件

- 3 个 markdown 表全部生成
- vanilla 在红队 mean = 100%
- 防御后 vanilla mean = 0%
- Arena Elo 排序合理（vanilla 最低）

## 一句话

> Capstone-2 = 5 ckpt 的"性能/安全/防御"三页画像。
