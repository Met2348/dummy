# L04 · 长上下文推理服务

## 1 · 100k+ ctx 痛点
- KV cache 巨大（100k × 32 layer × 256 KB = 800 GB ⚡）
- 必须 paged + 量化 + 跨节点
- prefill 时间 = O(L²) 注意力或 O(L) sliding

## 2 · 工程组合
- PagedAttention：必须
- KV int8/fp8：必须
- YaRN/NTK：RoPE scaling
- sliding window：长 ctx 截断
- Disaggregated：长 prefill 独立节点

## 3 · 100k ctx 实测（Llama-3 长 ctx 版）
- H100 fp16: 装不下
- H100 + int8 KV: 100k ctx OK
- H100 + fp8 KV + paged: 200k OK
- 多卡 + EP: 1M ctx (Gemini-Pro 风格)

## 4 · 服务接口
```python
{
    "model": "llama-3-100k",
    "messages": [...20k token history...],
    "max_tokens": 1024
}
```
- 长 prompt → router 走 long-ctx pool
- 短 prompt → 走常规 pool

## 5 · 一句话
> 长 ctx 推理 = **paged + 量化 KV + 长 ctx 训练 + 长 prompt 路由**。
