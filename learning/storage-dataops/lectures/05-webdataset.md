# L05 — WebDataset / 流式格式

## 为什么 tar shard

- 10× 加速：sequential read >> random IOPS
- 1M 个 100KB JPEG：random read → 2 s (IOPS-bound)
- 同样 1M 个，10000/shard × 100 个 1GB tar → 0.2 s (BW-bound) → **10×**

## 主流格式

| 格式 | 特点 | 使用方 |
|------|------|--------|
| WebDataset (tar) | 简单 + 通用 | OpenCLIP / Stability AI |
| Mosaic Streaming (.mds) | 随机 + epoch shuffle | MosaicML / Composer |
| litdata (.bin) | Lightning 原生 | Lightning AI |
| Parquet + arrow | 列存 + push-down filter | DuckDB / HF datasets |
| .npy / .bin (token id) | 极简 LLM pretraining | Mistral / TinyLlama |

## LLM 预训练特化

- 直接存 token id (uint16/uint32)，跳过 tokenize
- 每 epoch ~10 TB 流过：BW > 1 GB/s 是底线
- Lustre OSS pool 32 server × 25 GB/s = 800 GB/s 上限 → 训练 800 节点都够

## 反例

- 把 JPEG 直接放 S3 用 PyTorch dataset → S3 50 ms latency × 1M sample = 14 h
- 改 WebDataset tar 存 S3 → 1.0 GB/s × 10 TB = 3 h ✓
