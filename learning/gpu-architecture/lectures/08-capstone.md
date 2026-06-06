# L08 — Capstone: Roofline Zoo

## 任务

10 个 LLM 常见 op × 4 代 GPU = 40 个 roofline 分析。验证：

- 大 GEMM (8k³) 在 H100 上 util ≥ 80%
- LayerNorm 在所有 GPU 上 util ≤ 5%
- 多数 LLM op 是 memory bound

## 跑法

```powershell
python learning/gpu-architecture/src/capstone_roofline_zoo.py
```

输出：

```
[OK] capstone_roofline_zoo: 25/40 mem-bound, H100 big GEMM 100.0% util
```

## 教学结论

- **63% LLM op 是 memory bound** → 工业界主旋律是 fuse / 量化 / 改算法 (FlashAttn)
- 大 GEMM 在所有四代上都 compute bound → 训练 backbone 的"好"工作
- 同一 op 在 A100 和 H100 上的 util 都接近 100% → ridge 上移并不会让大 GEMM 变 memory bound (因为 ai 增长率 ~ 模型规模)

## 退出条件

- [x] 7 lectures + 7 src + 1 capstone + tests
- [x] 全部 self-test PASS
- [x] H100 大 GEMM util ≥ 80%
