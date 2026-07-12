# L06 — Capstone：Attention HBM 节省

## 任务

测算 5 个序列长度 × naive vs flash 的 HBM 流量，验证 N² → N 的 scaling 差异。

## 结果

`capstone_attn_speedup.py` 实测输出（`naive_mb` 按 `S`/`P` 矩阵各"写入 HBM + 读回"计 2 次往返，对应 paper guide §3 的 9 步 HBM 路径）：

```
Seq len | Naive MB | Flash MB | Speedup
--------|----------|----------|--------
    512 |      2.6 |      0.5 |    5.0x
   2048 |     35.7 |      2.1 |   17.0x
   8192 |    545.3 |      8.4 |   65.0x
  32768 |   8623.5 |     33.6 |  257.0x
 131072 | 137573.2 |    134.2 | 1025.0x
```

## 教学结论

- **128k seq 节省 1025×** → 这是为什么 long-context training/inference 必须 FlashAttn
- Naive 在 128k 光是 `S`+`P` 两个 N×N 矩阵的存储footprint 就约 68 GB（`2×N²×dtype_bytes`，不含读回）；算上表格 `naive_mb` 口径的完整写入+读回往返则约 137.6 GB —— 无论哪种口径，单卡都装不下/搬不动
- 实际跑 H100 上，FA3 比 PyTorch SDPA 默认快 4-8× wallclock

## 退出

```powershell
python learning/kernel-engineering/src/capstone_attn_speedup.py
# expect: [OK] capstone_attn_speedup (128k seq: 1025.0x HBM saved)
```
