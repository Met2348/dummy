# L12 · Capstone — Disaggregated 部署模拟

## 1 · 目标
- 单机模拟 2 GPU：1 prefill + 1 decode
- 与 colocate 对比 throughput
- 估算长 prompt 场景的提升

## 2 · 实验
3 种配置：
1. **colocate**: 1 GPU 既 prefill 又 decode
2. **disagg (近)**: 同机 NVLink 传 KV
3. **disagg (远)**: 跨节点 IB 传 KV

## 3 · 负载
- 32 并发请求
- prompt 长度 1024-2048（长）
- output 长度 128

## 4 · 预期数字
| 配置 | TTFT | TPOT | total tok/s |
|------|------|------|------------|
| colocate | 800 ms | 20 ms | 1200 |
| disagg (近) | 600 ms | 15 ms | 1800 (+50%) |
| disagg (远) | 700 ms | 16 ms | 1500 (+25%) |

## 5 · 实现：[capstone_disagg.py](../src/capstone_disagg.py)
- mock prefill_compute_ms + KV transfer_ms + decode_per_token_ms
- 跑 3 配置 × 32 req
- 输出表

## 6 · 真跑（可选）
2 张 5090：
```bash
# Prefill server (GPU 0)
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-1.5B \
    --port 8001 \
    --kv-transfer-config '{"kv_role":"kv_producer",...}'

# Decode server (GPU 1)
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-1.5B \
    --port 8002 \
    --kv-transfer-config '{"kv_role":"kv_consumer",...}'
```

## 7 · 退出条件
- 3 配置实测/模拟 → 表完成
- 长 prompt disagg ≥ 1.3x 提升
