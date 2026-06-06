# L06 — Capstone：Attention HBM 节省

## 任务

测算 5 个序列长度 × naive vs flash 的 HBM 流量，验证 N² → N 的 scaling 差异。

## 结果

```
Seq len | Naive MB | Flash MB | Speedup
--------|----------|----------|--------
    512 |     1.6  |     0.5  |   3.2x
   2048 |    17.4  |     1.8  |   9.7x
   8192 |   272.6  |     6.6  |  41.0x
  32768 |  4297.7  |    26.2  | 164.0x
 131072 | 68748.8  |    67.1  |1025.0x
```

## 教学结论

- **128k seq 节省 1025×** → 这是为什么 long-context training/inference 必须 FlashAttn
- Naive 在 128k 需 68 GB **临时** HBM (S + P) → 单卡装不下
- 实际跑 H100 上，FA3 比 PyTorch SDPA 默认快 4-8× wallclock

## 退出

```powershell
python learning/kernel-engineering/src/capstone_attn_speedup.py
# expect: [OK] capstone_attn_speedup (128k seq: 1025.0x HBM saved)
```
