# L07 · Lookahead Decoding（UCSD 2024）

## 1 · 不同思路：不要 draft model
完全 target model 自己 — 用 **Jacobi iteration**。

## 2 · Jacobi 直觉
设 序列 `y_1, y_2, ..., y_n`，target 给 `y_i = f(y_{<i})`。
不固定 y_i 顺序生成，而是**全部初始化随机** → 反复用 f 更新 → 收敛到正确序列。

```
init: y_1, y_2, y_3, ..., y_n = random
iter 1: y_1' = f(),  y_2' = f(y_1),  y_3' = f(y_1, y_2), ...
iter 2: 用 (y_1', y_2', y_3', ...) 再算
...
```

## 3 · 实际算法
1. **N-gram pool**: 累积过去看到的 n-gram (length 2-7)
2. **Lookahead branches**: 同时 forward 2 类位置
   - **verify**: 验证之前 lookahead 的 token 是否正确
   - **lookahead**: 生成 future n 个位置的 token 猜测
3. 接受的 n-gram 加入 pool，未来直接复用

## 4 · 收益
| 任务 | 加速 |
|------|------|
| 数学（重复 pattern 多） | 2-3x |
| 代码（boilerplate 多） | 2.5x |
| 通用 chat | 1.2-1.5x |

## 5 · 优点
- **无需 draft model + 无需训练**
- 即插即用

## 6 · 缺点
- 加速比不如 EAGLE
- 内存 ↑（n-gram pool）

## 7 · 适用场景
- 部署时 draft 训不完成
- 需要立刻收益的场景
- 重复 pattern 任务

## 8 · 实现：[lookahead.py](../src/lookahead.py)
- N-gram pool 维护
- 简单 Jacobi-like 循环
