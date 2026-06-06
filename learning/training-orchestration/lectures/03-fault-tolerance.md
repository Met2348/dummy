# L03 — 故障容忍与 ckpt 频率

## MTBF 算式

```
1 / cluster_MTBF = N_GPU / per_GPU_MTBF + N_fabric / fabric_MTBF + ...
```

- per-GPU MTBF ≈ 1 year (8760 h)
- 1024 GPU → cluster MTBF ≈ 8.5 h
- 16384 GPU (Llama-3 训练规模) → ≈ 0.5 h

**4 卡训练几乎不会故障，万卡训练 30 分钟一次。**

## Young's Formula (1974)

```
T_opt = sqrt(2 * C * M)
```

- C = ckpt 成本 (s)
- M = MTBF (s)
- C=1s, M=8.5h → T_opt ≈ 247 s ≈ 4 min

## 浪费率

```
wasted % = C/T + T/(2M)
```

- 在 T_opt 时最小化
- C=1s, T=247s, M=8.5h → 0.81% wasted (理想)
- C=1s, T=1h, M=8.5h → 5.9% wasted (ckpt 太稀)
- C=1s, T=30s, M=8.5h → 3.4% wasted (ckpt 太密)

## 大集群应对

- 多 tier ckpt：fast (Lustre) + cold (S3 备份)
- async DCP，blocking 降至 < 10ms → C ≈ 0.01s → T_opt 小到 ~26s 也合理
- torchrun --max-restarts → 自动重新拉起死掉的 rank
