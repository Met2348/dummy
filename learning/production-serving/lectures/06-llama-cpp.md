# L06 · llama.cpp + GGUF

## 1 · llama.cpp
- ggerganov 主导，纯 C++ 推理引擎
- 支持 CPU / Metal / CUDA / ROCm
- Ollama / LM Studio 底层都是它

## 2 · GGUF 格式
- 单文件包含 weights + tokenizer + config
- 量化 in-place (Q2/Q3/Q4/Q5/Q6/Q8)
- 跨平台

## 3 · 量化类型
| 类型 | bits/weight | 精度 |
|------|------------|------|
| Q2_K | ~2.6 | 差 |
| Q3_K_M | ~3.4 | 中 |
| **Q4_K_M** | ~4.6 | **好（推荐）** |
| Q5_K_M | ~5.5 | 高 |
| Q6_K | ~6.6 | 极高 |
| Q8_0 | 8.5 | 几乎无损 |

## 4 · 用
```bash
# 编译
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make

# 推理
./main -m qwen2.5-7b-Q4_K_M.gguf \
    -p "Hello" \
    -n 100
```

## 5 · 量化命令
```bash
# 从 HF 模型转 GGUF
python convert.py /path/to/Qwen2.5-7B
# 量化
./quantize ggml-model-f16.gguf qwen-Q4_K_M.gguf Q4_K_M
```

## 6 · server 模式
```bash
./server -m qwen-Q4_K_M.gguf --port 8080
# OpenAI 兼容 API
```

## 7 · 性能（M3 Max, 7B Q4）
| 模式 | tok/s |
|------|-----|
| Metal | 25 |
| CPU | 8 |

## 8 · 与 vLLM 对比
| | llama.cpp | vLLM |
|---|---------|------|
| 平台 | 全（CPU/Mac/Win/Linux） | Linux + CUDA |
| 单请求 | 快 | 普通 |
| 大 batch | 慢 | **快** |
| KV mgmt | 简单 | PagedAttention |

→ 个人 / 端侧用 llama.cpp，server 用 vLLM
