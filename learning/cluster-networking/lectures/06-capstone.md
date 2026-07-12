# L06 — Capstone: 4 fabric × 4 cluster 选型

## 任务

70B 模型 BF16 梯度 = 140 GB，在 4 个配置上跑 all-reduce，挑最优算法。

## 输出 (实测，逐字复制自 `python learning/cluster-networking/src/capstone_cluster_sim.py`)

```
Cluster | Link            | Best algo       | Best (s) | Ring (s)
--------|-----------------|-----------------|----------|---------
      8 | NVLink 4 (per GPU) | halving_doubling |     0.23 |    0.54
      8 | PCIe Gen5 x16   | halving_doubling |     1.64 |    3.83
     64 | IB NDR 400G     | sharp           |     0.09 |    5.51
    512 | IB XDR 800G     | sharp           |     0.01 |     2.8
```

## 教学结论

- PCIe-only 配置训练 70B 完全不可行 (单次 AR 秒级到几秒，秒级 step 下根本跑不动)
- 512 GPU + SHARP vs ring：2.80 s → 0.01 s，实测 **511× 加速**（精确比值，未四舍五入前
  `ring/sharp=511.00x`；64-GPU 那一行同法算出 63×，两个比值都约等于 `n_gpus-1`——见 `sharp_inline.py` 里
  `T_SHARP` 恒 2 步、ring 是 `2(N-1)` 步，大消息下带宽项主导，时间比收敛到步数比）
- 工程师选型直觉：
  - 单节点：NVLink + 大 tile
  - 8-64 节点：IB NDR + NCCL_COLLNET_ENABLE
  - 512+ 节点：必须 IB + SHARP + 仔细规划 TP/PP/DP 划分

## 退出

```powershell
python learning/cluster-networking/src/capstone_cluster_sim.py
# expect: [OK] capstone_cluster_sim (NVLink 8-GPU 0.23s, IB 512 SHARP 0.01s)
```
