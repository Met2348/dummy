# L03 — Tensor Core 进化史

## 4 代

| Gen | 架构 | MMA 形状 | 主精度 | 备注 |
|----|-----|--------|-------|------|
| 1 | V100 (2017) | m8n8k4 | FP16 → FP32 | 首次 |
| 2 | A100 (2020) | mma.m16n8k16 | BF16/TF32 | sparsity 2:4 |
| 3 | H100 (2022) | wgmma.m64n256k16 | FP8 (E4M3/E5M2) | warp-group 异步 |
| 4 | B200 (2024) | tcgen05.m128n256k128 | **FP4** (MX-FP4) | tensor memory 独立 |

## H100 wgmma 关键概念

- `wgmma` = warp-group MMA，128 个线程协作一条指令
- A 必须在 RF (registers)，B 可以在 SMEM (省 RF)
- 计算与 SMEM→RF load 异步重叠 (`wgmma.commit_group` / `wgmma.wait_group`)
- accumulator 在 RF，输出仍在 RF 或异步 store 回 SMEM

## B200 tcgen05 + Tensor Memory

- 独立 `tmem` 存 accumulator，不再占 RF
- FP4 (E2M1) 需要 microscaling: 每 32 元素一个 E8M0 共享 scale → 等价于 block-quant
- 4× FP8 算力，但精度更敏感，主要服务**推理** (训练仍以 FP8/BF16 为主)
