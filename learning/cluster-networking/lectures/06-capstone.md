# L06 — Capstone: 4 fabric × 4 cluster 选型

## 任务

70B 模型 BF16 梯度 = 140 GB，在 4 个配置上跑 all-reduce，挑最优算法。

## 输出 (近似)

```
Cluster | Link            | Best algo        | Best (s) | Ring (s)
--------|-----------------|------------------|----------|---------
      8 | NVLink 4        | halving_doubling |     0.23 |    0.54
      8 | PCIe Gen5 x16   | halving_doubling |     1.64 |    3.83
     64 | IB NDR 400G     | sharp            |     0.04 |   27.50
    512 | IB XDR 800G     | sharp            |     0.01 |  175.00
```

## 教学结论

- PCIe-only 配置训练 70B 完全不可行 (单次 AR 几秒，秒级 step 下根本跑不动)
- 512 GPU + SHARP vs ring：175 s → 10 ms = **17500× 加速**
- 工程师选型直觉：
  - 单节点：NVLink + 大 tile
  - 8-64 节点：IB NDR + NCCL_COLLNET_ENABLE
  - 512+ 节点：必须 IB + SHARP + 仔细规划 TP/PP/DP 划分

## 退出

```powershell
python learning/cluster-networking/src/capstone_cluster_sim.py
# expect: [OK] capstone_cluster_sim (NVLink 8-GPU 0.23s, IB 512 SHARP 0.01s)
```
