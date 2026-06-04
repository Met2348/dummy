# L10 · Attention Backends

## 1 · backend 全谱
| backend | 来源 | 强项 | 弱项 |
|---------|------|------|------|
| naive PyTorch | torch | 调试 | 慢 |
| FlashAttention v2 | Tri Dao 2023 | prefill 快 | decode 一般 |
| FlashAttention v3 | Tri Dao 2024 | Hopper FP8 | 限 H100+ |
| XFormers | Meta | 通用 | 不如 FA |
| **FlashInfer** | UCB 2024 | inference-aware (paged) | 新 |
| vLLM XFormers | vllm | 早期默认 | 已退役 |

## 2 · FA v2 vs v3
- v2: forward + backward + GQA
- v3: + warp specialization + FP8 + 1.5-2× on H100

## 3 · FlashInfer 核心改进
- 原生支持 paged KV（block table input）
- 支持 GQA / MLA / sliding window
- 自动选 prefill / decode kernel

## 4 · 选 backend 经验
- 5090 (Blackwell) + 7B: FlashInfer or FA v3
- T4 / V100: FA v2 (Hopper kernel 跑不动)
- CPU debug: naive

## 5 · vLLM 的 backend
- `--attention-backend FLASHINFER` (推荐)
- `--attention-backend FLASH_ATTN`
- `--attention-backend XFORMERS` (废弃)

## 6 · 性能数字（A100, 7B, seq=2k）
| backend | prefill TPS | decode TPS |
|---------|------------|-----------|
| naive | 80 | 30 |
| FA v2 | 2400 | 220 |
| FlashInfer | 2300 | 240 |

prefill: FA 略胜；decode: FlashInfer 胜（paged-native）。

## 7 · 实现
教学 src 提供 `attention_naive.py` 朴素 baseline 用于 fork 对照实验，flash-infer 仅做 import 检查（不强制安装）。
