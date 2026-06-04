# L02 · Tensor Parallel for Inference

## 1 · 朴素 idea
矩阵乘法 `Y = X · W` 可按 W 切：
- column split: W = [W₁, W₂], Y = [X·W₁, X·W₂] → concat
- row split: W = [W₁; W₂], X = [X₁, X₂], Y = X₁·W₁ + X₂·W₂ → allreduce

## 2 · Transformer block 切法
QKV projection (W_in_dim × 3·hidden) → **column split**:
```
GPU 0: W_Q, W_K, W_V 前半（n_heads/2）
GPU 1: W_Q, W_K, W_V 后半
```
- attention 算每张卡 head 子集独立
- output projection → **row split** + allreduce
- MLP 同样 down/up：col+row

## 3 · 一个 layer 通信量
- attention out: allreduce(B·S·d)
- MLP out: allreduce(B·S·d)
- 2 × allreduce per layer
- 7B 32 层 → 64 次 allreduce per token

## 4 · 通信带宽
- B=1, S=1 decode: tiny payload
- NVLink 4 (900 GB/s) 一次 allreduce ~10 µs
- 32 layer × 2 × 10 µs = 640 µs / token → 1500 tok/s ceiling

## 5 · vLLM TP 用法
```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3-70B \
    --tensor-parallel-size 8
```

## 6 · TP 限制
- 必须能整除 head 数（n_heads % TP == 0）
- 通信延迟，不适合 batch 极小（< 4）
- 跨节点 TP 不推荐（带宽不够）

## 7 · 实现：[tp_demo.py](../src/tp_demo.py)
- ColumnSplitLinear / RowSplitLinear 模拟
- TP-mlp 完整 forward
- 与 single-GPU 数值一致
