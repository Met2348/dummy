# L01 — Kernel Engineering 三家概览

## Triton (OpenAI 2021)

- Python DSL，编译到 LLVM → PTX
- block-pointer 抽象 (`tl.make_block_ptr`)，自动 swizzle / vectorize
- `@triton.autotune` 自动选 (BLOCK, num_warps, num_stages)
- 适合：研究员快写 attn / norm / 量化 kernel
- 局限：Hopper wgmma 支持 2024 才完善，仍不如 CUTLASS 极致

## CUTLASS / CuTe (NVIDIA 2017+)

- C++ 模板库，对 Tensor Core 全 ISA 直接控制
- CuTe (CUTLASS 3.x)：统一 layout (shape + stride) 代数，编译期推导 swizzle
- 适合：库作者 (FlashAttn / cuBLASLt / cuDNN / FlashInfer)
- 局限：学习曲线陡

## FlashInfer / FlashAttn 3 (2024)

- 针对 LLM 推理的高度专用 attention kernel
- 支持 paged KV, MQA/GQA, prefill+decode 融合
- 2024.07 FA3 在 H100 上达 75% FP16 peak

## Triton vs CUTLASS 选型

| 场景 | 选 |
|------|----|
| 一次性 prototype | Triton |
| 跨 GPU 通用算子库 | CUTLASS |
| 极致性能 (>90% peak) | CUTLASS / 手写 PTX |
| 想自动 autotune | Triton |
| 量化 + MX-FP4 | CUTLASS (Triton 跟进中) |
