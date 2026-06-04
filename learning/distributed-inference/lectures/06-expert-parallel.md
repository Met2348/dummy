# L06 · Expert Parallel (EP) for MoE

## 1 · MoE 简短回顾（Module 3 已学）
- top-K routing：每 token 选 K 个 expert（典型 K=1,2）
- expert 是独立 MLP
- DeepSeek-V3: 256 expert (每层) + top-8 routing

## 2 · 为什么 EP
- 256 expert × 7B → 1.8T 参数装不下单卡
- 切到 N GPU 上每 GPU 32 expert
- 每 token 路由到 top-K 个 expert (可能跨多 GPU)

## 3 · 数据流：all-to-all
每 token：
1. router 决定 top-K expert ids
2. all-to-all：把 token 送到对应 GPU
3. 每 GPU 算自己 expert
4. all-to-all：把结果送回原 GPU
5. weighted sum

## 4 · all-to-all 带宽要求
- N=64 GPU, batch=B, seq=S, hidden=d, K=top
- send size per GPU = B·S·d / N × K
- NVLink/NVSwitch 必备 (PCIe 慢 10x)

## 5 · DeepSeek-V3 EP=256
- 每 GPU 1 expert
- 单层 2 次 all-to-all
- cross-node 通信占总时间 30-50% → DualPipe 优化

## 6 · EP + TP 组合
- expert 内不切 (单 expert 单卡)
- shared expert 走 TP

## 7 · 路由不均衡
- 实际 routing：某些 expert 热门 → 该 GPU 算力满，其他空
- 解：aux loss (DeepSeek 1.0 时代) → 后转 aux-loss-free (DeepSeek-V3)
- 已是 expert pop 接近均匀

## 8 · 实现：[ep_demo.py](../src/ep_demo.py)
- mock expert assignment
- all-to-all 模拟 (单进程模拟通信开销)
- load balance 度量
