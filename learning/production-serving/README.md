# Topic 6: Production Serving（生产部署）

> Module 5 「用大模型」第 6 专题 · 12 lectures · 12 notebooks · ~12h

## 总览

| Lecture | 主题 | 代码 |
|---------|------|------|
| L01 | 生产部署全图 | — |
| L02 | TensorRT-LLM 概览 | — |
| L03 | TRT-LLM build 实战 | `trtllm_build.py` |
| L04 | Triton Inference Server | `triton_model_repo/` |
| L05 | Ollama | `ollama_modelfile/` |
| L06 | llama.cpp + GGUF | — |
| L07 | LM Studio | — |
| L08 | OpenAI API 规范 | `openai_api_server.py` |
| L09 | FastAPI + SSE | `streaming_sse.py` |
| L10 | 监控 (Prometheus) | `metrics_prometheus.py` |
| L11 | 成本工程 | `cost_calc.py` |
| L12 | **Capstone: 生产栈** ⭐ | `openai_api_server.py` (复用) |

## Tags

- `prod-serving` — 最终（含 Capstone + README）

## 关键组件

```
client → FastAPI (OpenAI compat) → mock/vLLM backend
              ↘ SSE streaming
              ↘ /metrics (Prometheus)
              ↘ /health
```

## 决策树

```
端侧 (Mac/PC)?
 ├── 是 + 极易 → LM Studio
 ├── 是 + 进阶 → Ollama
 └── 是 + 工程师 → llama.cpp

服务器?
 ├── H100/H200 + 极致速度 → TensorRT-LLM + Triton
 ├── 通用 + 易上手 → vLLM (Topic 1)
 ├── agent 场景 → SGLang (Topic 2)
 └── 多模型 ensemble → Triton
```

## 成本经济学

`$/M token = GPU_$_per_h / tok_per_s / 3.6`

| 配置 | $/M token |
|------|----------|
| OpenAI GPT-4 | $2.50 |
| Claude 3.7 | $3.00 |
| H100 + Llama-3 70B | $0.50 |
| 5090 + Qwen-7B AWQ | **$0.05** |

应用 AWQ + EAGLE + cache 全套：**降 70-85%**。

## 环境

```powershell
python environment/verify_env.py
```

## 运行

```powershell
# 测试 (22/22 全绿)
python -c "import sys; sys.path.insert(0,'src'); sys.path.insert(0,'src/tests'); import test_serving"

# 启动 mock OpenAI server
uvicorn src.openai_api_server:app --port 8000

# 用 openai-python 客户端测
python -c "
import openai
c = openai.OpenAI(base_url='http://localhost:8000/v1', api_key='x')
print(c.chat.completions.create(model='mock', messages=[{'role':'user','content':'hi'}]))
"
```

## 退出条件 checklist

- [x] 12 lecture + 12 notebook
- [x] 22 tests pass
- [x] OpenAI compat API (validate/build/error/stream)
- [x] SSE encode/parse helpers
- [x] Prometheus Counter/Histogram + render
- [x] Cost calculator + cache savings + routing
- [x] TRT-LLM build CLI template
- [x] Triton model_repository 模板
- [x] Ollama Modelfile 模板
- [x] git tag `prod-serving` ✓

## 关键文献

- TensorRT-LLM docs (NVIDIA)
- Triton Inference Server docs
- vLLM / SGLang OpenAI server source
- Anthropic / OpenAI API specs
- llama.cpp / gguf 文档

## 一句话

> 生产 = **TRT-LLM 速度 + Triton 编排 + FastAPI 包装 + Prometheus 监控 + 成本工程**。
