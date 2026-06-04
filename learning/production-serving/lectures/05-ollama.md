# L05 · Ollama（端侧部署王）

## 1 · 定位
- 一行命令跑 LLM
- 适合个人电脑 / 端侧
- 基于 llama.cpp + GGUF

## 2 · 安装 + 用
```bash
# Mac / Linux
brew install ollama
ollama serve

# 拉模型
ollama pull qwen2.5:7b

# chat
ollama run qwen2.5:7b
```

## 3 · Modelfile
```
FROM qwen2.5:7b
PARAMETER temperature 0.7
PARAMETER top_p 0.9
SYSTEM """You are a helpful assistant."""
```
build:
```bash
ollama create my-qwen -f Modelfile
```

## 4 · OpenAI 兼容 API
```bash
curl http://localhost:11434/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"qwen2.5:7b","messages":[{"role":"user","content":"hi"}]}'
```

## 5 · 后端选择
- Metal (Mac)
- CUDA (Linux/Win)
- ROCm (AMD)
- CPU (慢)

## 6 · 性能（M3 Max）
| 模型 | tok/s |
|------|------|
| Qwen2.5-7B Q4 | 25 |
| Qwen2.5-14B Q4 | 12 |
| Qwen2.5-32B Q4 | 5 |
| Llama-3.3-70B Q4 | 2 |

## 7 · 何时用
- 端侧 (Mac, 个人 PC)
- 隐私场景（数据不离机）
- 快速原型

## 8 · 实现：[ollama_modelfile/](../src/ollama_modelfile/)
- 示例 Modelfile + 启动脚本
