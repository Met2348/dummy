# L03 — Fat-tree 与 Dragonfly

## Fat-tree (Charles Leiserson 1985)

3-tier (leaf / spine / core)，每层带宽不收敛 = full bisection。

```
   [Core 16]
   /  |  |  \
  [Spine 32]
   /  |  |  \
  [Leaf 32]    ← each leaf 32 ports (16 down, 16 up)
   |
  GPUs (1024)
```

- Radix R 端口 → 每 leaf 服 R/2 个端点
- 1024 端点 → 32 leaf × 32 spine × 16 core ≈ 80 switch
- 2:1 oversubscription = 上行带宽对半 → bisection BW 减半

## Dragonfly+ (Cray 2008)

- 小组 (group) 内全互连，组间稀疏多跳
- 平均 3 hop (2 intra + 1 inter)
- 大 N 性价比远超 fat-tree
- HPE Slingshot / Aurora / GB200 NVL576+ 用

## 选型

- < 1k GPU：fat-tree 简单 + full bisection
- > 4k GPU：Dragonfly 省线 50%+ 同时 BW 满足
- GB200 NVL72：单 rack 内 NVL switch，rack 间 IB/UE
