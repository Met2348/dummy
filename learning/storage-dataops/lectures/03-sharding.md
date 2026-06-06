# L03 — Sharding 策略

## 三种基础

| 策略 | 公式 | 优 | 劣 |
|------|-----|----|----|
| Hash | `sha1(id) % N` | 抗 skew | 不能局部读 |
| Range | `id // (total/N)` | 局部性好 | skew 严重 |
| Round-robin | `id % N` | 简单均匀 | 同 worker 重复看同模式 |

## 进阶

- **Weighted hash**：按 token 数加权，避免长样本压垮一个 shard
- **Stratified**：按类别先分层再均匀 → 每 shard 类别分布一致
- **Reshuffle every epoch**：epoch 间换 hash seed

## LLM 训练实践

- 预先把数据切成 1-10 GB 的 .tar / .parquet shard，存 Lustre
- shard 文件名 = MD5(shard_id) → cluster 跨 batch 自然均匀分散 OSS server 压力
- WebDataset / Mosaic Streaming / litdata 都是这套范式
