# Q&A — 记不住函数参数怎么办（2026-07-06）

## Q：我害怕自己背不下来常用函数的参数和返回值，也不知道什么时候该用哪个函数

## Claude 的回答（要点）

1. **这个担心基于一个错误的专家模型。** 没有人（包括写这些库的人）背下所有 numpy/torch 函数的精确签名。真正的专家技能分两层：
   - 意图 → 大概函数名（模糊记忆，高频使用后自然形成）
   - 函数名 → 精确签名（5 秒内查到，不需要记）

2. **真正需要肌肉记忆的函数很少**（大概 15 个：shape/reshape/sum/mean/matmul/切片等），高频到自然会记住，不用专门背。

3. **剩下的应该按"意图"查表，而不是死记。** 完整的意图索引表、5 个新手常踩的地雷（`*` vs `@`、`cat` vs `stack`、`torch.max` 返回元组等）、以及查阅技巧，见 [03-how-to-look-up-not-memorize.md](../03-how-to-look-up-not-memorize.md)。

4. **建立了自维护速查表** [my-cheatsheet.md](../my-cheatsheet.md)：规则是同一个函数查了 3 次文档就记一条进去。自己踩坑后写的笔记比读现成答案记得牢（生成效应）。

## 详细内容

见 [03-how-to-look-up-not-memorize.md](../03-how-to-look-up-not-memorize.md)
