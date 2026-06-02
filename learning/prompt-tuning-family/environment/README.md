# 环境配置

## 依赖列表

见 [`requirements.txt`](requirements.txt)。核心：

- `torch >= 2.5`（CPU 即可，GPU 自动加速）
- `transformers >= 5.0`（注意：本专题使用 5.x 的 `DynamicCache` 接口）
- `peft >= 0.10`
- `jupyterlab + ipykernel`
- `matplotlib + numpy`

## 安装

```powershell
python -m pip install -r learning/prompt-tuning-family/environment/requirements.txt
```

## 自检

```powershell
python learning/prompt-tuning-family/environment/verify_env.py
```

预期最后一行：`Environment ready.`

## 故障排查

### 1. `transformers 4.x` 与本专题代码

本专题代码使用 `transformers 5.x` 的 `DynamicCache` 接口注入 prefix。

若您仍在 4.x 上，需要把：

```python
from transformers import DynamicCache
cache = DynamicCache(ddp_cache_data=kv_list)
```

替换为旧的 tuple 写法：

```python
past_key_values = tuple(kv_list)  # tuple of (K, V) tuples
```

### 2. Huggingface 模型下载慢

设置镜像：

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

或预下载到本地缓存：

```powershell
$env:HF_HOME = "D:/hf_cache"
```

### 3. Windows symlink 警告

Hugging Face 缓存默认使用 symlinks。Windows 普通用户没有创建 symlink 的权限，会有警告但不影响功能。两种方案：

- 忽略警告：`$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"`
- 启用开发者模式：设置 → 隐私和安全性 → 开发者选项 → 开发人员模式

### 4. CUDA 检测

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

本专题所有代码均**支持 CPU 运行**，演示用 `gpt2` 模型（117M 参数）。
