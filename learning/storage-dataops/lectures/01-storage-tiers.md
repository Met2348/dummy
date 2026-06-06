# L01 — Storage 分层

| 层 | 介质 | BW | IOPS | 延迟 | 容量 | 用途 |
|---|------|-----|------|------|------|------|
| Host RAM | DDR5 | 80 GB/s | 10M | 0.1 μs | 2 TB | KV cache 临时区 |
| Local NVMe | Gen5 PCIe | 14 GB/s | 2M | 80 μs | 30 TB | 训练 scratch |
| Local NVMe RAID0 | 8× Gen5 | 100 GB/s | 8M | 80 μs | 250 TB | 大 batch staging |
| Lustre/GPFS | OSS pool | 400-600 GB/s | 0.5-0.8M | 0.5 ms | 50 PB | 训练数据集 + 共享 ckpt |
| Object (S3) | regional | 1 GB/s | 10k | 50 ms | unlimited | 长存 + 备份 |

## 工程直觉

- **Training data → Lustre**：sequential read 主导，OSS 聚合 BW 抗得住
- **Hot KV cache → host RAM**：低延迟 + 程序员管
- **Cold ckpt → S3**：弹性 cost，跨 region 备份
- **Local NVMe**：作为 cache / spill，避免 lustre random IO

## 反模式

- 把训练数据放 S3 直读 → IOPS 杀死 batch size
- ckpt 用 NFS → 单点 → 大集群 ckpt 时间塌方
- 不用 RAID0 直读单 NVMe → 14 GB/s 上限堵 dataloader
