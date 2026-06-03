# L00 · WSL2 + verl + vllm 环境配置

> 28 slides | 70 min | Reasoning R1 系列开篇 — **环境切换专题**

⚠️ **本专题第一个 lecture 切换到 WSL2**：verl + Ray + Megatron + vllm 在 Windows native 装不上。

---

## Slide 1 · 为什么切 WSL2

| 组件 | Windows native | WSL2 |
|------|--------------|------|
| torch cu130 | ✓ | ✓ |
| transformers | ✓ | ✓ |
| trl | ✓ | ✓ |
| **vllm** | ✗ | ✓ |
| **verl** | ✗ | ✓ |
| **Ray** | 部分 | ✓ |
| **Megatron** | ✗ | ✓ |
| **DeepSpeed** | 部分 | ✓ |

→ 专题 5-7 的 R1/DAPO/VAPO 都需要 verl+vllm，必须 WSL2。

---

## Slide 2 · WSL2 安装

```powershell
# Windows PowerShell (管理员)
wsl --install -d Ubuntu-22.04
# 重启 → 设置用户名 / 密码
wsl --list --verbose  # 检查 Version 2
```

如果遇到 GPU 不见：
- 更新 nvidia driver ≥ 555.x
- 安装 [CUDA on WSL2 toolkit](https://docs.nvidia.com/cuda/wsl-user-guide/)

---

## Slide 3 · CUDA 13 + cuDNN

WSL2 内：
```bash
sudo apt update && sudo apt install -y build-essential
wget https://developer.download.nvidia.com/compute/cuda/13.0.0/local_installers/cuda_13.0_linux.run
sudo sh cuda_13.0_linux.run --silent --toolkit
```

验证：
```bash
nvidia-smi   # 应看到 GPU
nvcc --version  # CUDA 13
```

---

## Slide 4 · Python 3.11 venv

```bash
sudo apt install python3.11 python3.11-venv python3.11-dev
python3.11 -m venv ~/.venv/r1
source ~/.venv/r1/bin/activate
pip install --upgrade pip setuptools wheel
```

→ verl 0.4 要求 Python 3.10+。

---

## Slide 5 · torch cu130 nightly

```bash
pip install torch torchvision torchaudio --pre \
  --index-url https://download.pytorch.org/whl/nightly/cu130
```

验证：
```python
import torch
print(torch.cuda.is_available())  # True
print(torch.cuda.get_device_capability(0))  # (12, 0) for 5090 Blackwell
```

---

## Slide 6 · vllm 安装

```bash
pip install vllm>=0.7
```

**已知坑**：
- 必须 PyTorch >= 2.5
- Blackwell sm_120 需 vllm 0.7+
- 内存 < 32GB 时单 GPU 加载大 model OOM

验证：
```python
from vllm import LLM
llm = LLM(model="Qwen/Qwen2.5-0.5B")
print(llm.generate("hello", sampling_params=...))
```

---

## Slide 7 · verl 0.4 安装

```bash
pip install verl>=0.4
# 同时安装 ray
pip install ray[default]>=2.30
```

**已知坑**：
- 编译 megatron 失败 → 安装 nvidia-cuda-12-2 旧版可救
- protobuf 版本冲突 → `pip install "protobuf<4"`

---

## Slide 8 · 完整 requirements.txt

参考 `environment/requirements.txt`：
```
torch>=2.5+cu130
transformers>=5.0
trl>=0.13
verl>=0.4
vllm>=0.7
ray[default]>=2.30
peft>=0.13
bitsandbytes>=0.43
math-verify
sympy
datasets>=2.14
accelerate>=0.30
```

---

## Slide 9 · 从 Windows 访问

WSL2 内的 `/mnt/c/Workspace/dummy` 就是 Windows 的 `c:\Workspace\dummy`。

工作流：
1. 用 VSCode 在 Windows 编辑代码（WSL Remote 扩展）
2. 在 WSL terminal 运行 `python ...`
3. 工作目录 = `cd /mnt/c/Workspace/dummy/learning/reasoning-r1`

⚠️ 不要把训练 ckpt 存到 `/mnt/c/`，IO 慢。存到 `~/r1-ckpts/`。

---

## Slide 10 · verify_env.py 三段式

`environment/verify_env.py` 检查：

**Part A**: torch + transformers + trl + verl + vllm import ok

**Part B**: GPU + sm_120 + CUDA 13.0 ok

**Part C**: vllm 单卡 GPT-2 推理 smoke + verl 5-step GRPO smoke + Ray cluster init

→ 全 PASS 才能进 L01。

---

## Slide 11 · 备用：Docker

verl 提供 Dockerfile（如本地装不上）：

```bash
docker pull verlai/verl:latest
docker run --gpus all -it -v $(pwd):/workspace verlai/verl:latest
```

镜像内已装好 verl + vllm + torch cu126。

---

## Slide 12 · Ray cluster 启动

```bash
ray start --head --port=6379
ray status
```

verl 内部用 Ray 做 distributed rollout + RM。

单 GPU 也用 Ray，因 verl 的 worker 模型用 Ray actor 封装。

---

## Slide 13 · 单 GPU vs 多 GPU

R1 真训练用 8-32 GPU。我们的 capstone 双轨：
- Track A: GPT-2-M (355M) → 单 5090 24GB 跑得动
- Track B: Qwen-1.5B + 4bit LoRA → 单 5090 24GB 勉强

如果有 4× 5090，可直接跑 Qwen-7B + LoRA。

---

## Slide 14 · 显存账（Qwen-1.5B + LoRA + GRPO）

| 组件 | size |
|------|------|
| actor (Qwen-1.5B fp16) | 3 GB |
| actor + LoRA opt state | 200 MB |
| ref (frozen Qwen-1.5B fp16) | 3 GB |
| vllm rollout cache | 6 GB |
| activations | 6 GB |
| **总** | **~18 GB** |

→ 5090 24GB OK，但 batch_size 紧。

---

## Slide 15 · 关键 verl 配置文件

verl 用 OmegaConf YAML：
```yaml
data:
  train_files: ./data/countdown.parquet
trainer:
  total_epochs: 1
  n_gpus_per_node: 1
actor_rollout_ref:
  actor:
    optim:
      lr: 1e-6
  rollout:
    n: 8         # group size for GRPO
    temperature: 1.0
  ref:
    fsdp_config:
      param_offload: true
algorithm:
  adv_estimator: grpo
```

每个 lecture 会示范一个完整 config。

---

## Slide 16 · vllm 加速 rollout

R1 训练 90% 时间在 rollout（生成 response）。vllm 把 rollout 加速 10-15×。

verl 自动用 vllm 做 rollout，PPO update 用 standard transformers。

→ 没 vllm 你训不动 R1。

---

## Slide 17 · 常见报错 1 · OOM in rollout

修：
- batch_size ÷ 2
- max_response_length ÷ 2
- n (group size) ÷ 2
- gpu_memory_utilization 0.9 → 0.7

---

## Slide 18 · 常见报错 2 · NaN loss

修：
- bf16 而非 fp16（更稳）
- max_grad_norm = 0.5
- lr 1e-6 起步（不要 1e-5）

---

## Slide 19 · 常见报错 3 · KL 爆炸

修：
- kl_coef 0.01 → 0.05
- adaptive_kl_ctrl=True

---

## Slide 20 · 常见报错 4 · Ray 启动卡

```bash
ray stop --force
ray start --head --port=6379 --num-cpus=8 --num-gpus=1
```

如果 firewall 拦截：`--include-dashboard False`

---

## Slide 21 · 测试 verl 是否可用

```bash
python -c "
from verl import PPOConfig
print('verl OK')
"
```

如果 import 失败 → 检查 cuda 12 vs 13 兼容、protobuf 版本。

---

## Slide 22 · 工作流推荐

```
1. Windows VSCode 编辑代码（WSL Remote）
2. WSL terminal cd /mnt/c/Workspace/dummy/learning/reasoning-r1
3. 启动 ray: ray start --head
4. 跑训练: python src/r1_zero_track_a/train_grpo.py
5. 监控: tensorboard --logdir runs/
6. 训完拷贝 ckpt 出去（避免 WSL fs 滥用）
```

---

## Slide 23 · WSL2 资源调度

`%UserProfile%\.wslconfig`:
```ini
[wsl2]
memory=24GB
processors=12
swap=8GB
```

→ 让 WSL 用 RAM 24GB（Windows 仍有 8GB 余）。

---

## Slide 24 · 跨 WSL/Windows 文件同步

代码：`/mnt/c/` 双向访问。
数据：训练数据放 `~/data/`，避免 `/mnt/c/` 的 IO penalty。
ckpt：训完 `cp ~/r1-ckpts/final.bin /mnt/c/Workspace/.../runs/`。

---

## Slide 25 · 备用：Docker 单 container

如果 verl 装不上：
```bash
docker run --gpus all -it --rm \
  -v $(pwd):/workspace -p 6006:6006 -p 6379:6379 \
  verlai/verl:latest bash
```

镜像内有 ray + verl + vllm + torch。

---

## Slide 26 · 自测题

1. 为什么 verl 在 Windows native 装不上？
2. WSL2 与 Windows 的 GPU 访问关系？
3. vllm 在 rollout 中起什么作用？
4. Ray 在 verl 中扮演什么角色？
5. 5090 24GB 跑 Qwen-1.5B + LoRA + GRPO 显存是否充足？

---

## Slide 27 · 入口

```bash
# WSL2 内
cd /mnt/c/Workspace/dummy/learning/reasoning-r1
python environment/verify_env.py  # 三段式必须 PASS

# 下一讲：L01 OpenAI o1 范式
```

---

## Slide 28 · 检查清单

- [ ] WSL2 Ubuntu 22.04 装好
- [ ] CUDA 13 + cuDNN 装好
- [ ] Python 3.11 venv 创建
- [ ] torch cu130 nightly 装好
- [ ] vllm 0.7+ 装好
- [ ] verl 0.4+ 装好
- [ ] ray 装好
- [ ] verify_env.py 三段式全 PASS
