# L00 · WSL2 + Megablocks 环境配置

> 16 slides | 40 min | MoE Architecture 前置 ⭐⭐⭐

---

## 学习目标

1. 确认 WSL2 已切换（Module 2 末已配）
2. 安装 megablocks + deepspeed
3. verify_env 三段过

---

## Slide 1 · 为什么 MoE 必须 WSL2

```
megablocks: Linux-only kernel
deepspeed-moe: Linux-only
flash-attn: Linux 完善版
```

Windows native 大概率装不上。

---

## Slide 2 · 复用 Module 2 WSL2

如 Module 2 末已切，本专题直接用同 venv。

```bash
cd ~/dummy
source ~/dummy/learning/reasoning-r1/.venv/bin/activate
```

---

## Slide 3 · 新装包

```bash
pip install megablocks
pip install deepspeed
pip install flash-attn --no-build-isolation
```

megablocks 需 GPU + sm_80+ 编译 ~10 min。

---

## Slide 4 · sanity check

```bash
python -c "import megablocks; print(megablocks.__version__)"
python -c "import deepspeed; print(deepspeed.__version__)"
```

---

## Slide 5 · GPU 要求

- megablocks: A100 / H100 / 4090+ / **5090 sm_120 OK**
- deepspeed: 任意 sm_70+
- 5090: 24GB，本课玩具 MoE 都跑得通

---

## Slide 6 · CUDA 版本

```
torch cu130 + sm_120 (5090)
flash-attn 2.6+ 自动选择
```

如 import 报 sm 不匹配 → 重装 flash-attn 用源码 build。

---

## Slide 7 · pip 装 megablocks 常见错误

```
1. nvcc not found  → sudo apt install nvidia-cuda-toolkit
2. Triton 不兼容   → pip install triton==3.0
3. flash-attn 缺   → 先装 flash-attn
```

---

## Slide 8 · NCCL 版本

deepspeed 需 NCCL ≥ 2.18。WSL2 一般已经满足。

```bash
ldconfig -p | grep nccl
```

---

## Slide 9 · 工作目录约定

WSL2 中：
```
/mnt/c/Workspace/dummy/learning/moe-architecture/
```

可直接读写 Windows 路径，便于 IDE 编辑。

---

## Slide 10 · verify_env.py 三段

```
Part A: import (megablocks 可选)
Part B: GPU sanity
Part C: 自训 top-2 MoE 10 step smoke
```

---

## Slide 11 · 如何 fallback

如 megablocks 装不上：
- 用 `src/moe_layer_naive.py` (纯 PyTorch)
- 性能差但教学 OK

---

## Slide 12 · 已知问题

- megablocks 在 5090 (sm_120) 偶尔需重 build
- deepspeed-moe 在 single GPU 模式较慢（多机才发挥）

---

## Slide 13 · Module 2 已完成的 setup 复用清单

```
[ ] torch cu130 + sm_120
[ ] verl + vllm + ray
[ ] flash-attn 2.6
[ ] transformers 5.0
```

加：
```
[ ] megablocks
[ ] deepspeed
```

---

## Slide 14 · GPU memory hygiene

MoE 训练显存峰值可能高：
```
n_expert = 4
top_k = 2
显存 ≈ dense × top_k ≈ 2× dense
```

5090 24GB 玩具 4-expert OK。

---

## Slide 15 · 测试 minimum run

```bash
python src/moe_layer_naive.py
```

应输出 forward + aux loss。

---

## Slide 16 · 接下来

L01 起进入算法主线：
- L01 Shazeer 2017 sparse MoE
- L02 GShard top-2
- L03 Switch top-1
- L04 Expert Choice
- L05 Mixtral
- L06 DeepSeekMoE
- L07 Aux-Loss-Free ⭐⭐⭐⭐⭐
- L08 Phi-MoE / Qwen3-MoE / MoR
- L11-L12 训稳 + 推理
- L13 Capstone 4-expert mini-MoE

---

## 参考

- megablocks docs
- deepspeed-moe paper
- Module 2 WSL2 setup notes
