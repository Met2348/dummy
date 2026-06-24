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

## 运行验证（Runbook）

> 本模块的"可运行入口"即 [`runbook.yaml`](runbook.yaml) 登记的 6 个直跑 demo，已在 ERIC-3080Ti（RTX 3080 Ti 16GB）V1 验证通过。全部纯 CPU 秒级、无需起服务。
> 一键复验：
> ```powershell
> python scripts/eric_3080ti_env_audit.py --runbook --modules production-serving
> ```

6 个 demo 均无需传参：

```powershell
python learning/production-serving/src/cost_calc.py                 # $/M-token + 缓存节省 + 成本感知路由
python learning/production-serving/src/metrics_prometheus.py        # Counter/Histogram + p50/p95 渲染
python learning/production-serving/src/streaming_sse.py             # SSE 编码/解析往返
python learning/production-serving/src/clipper_original_minimal.py  # 自适应批处理(AIMD) + ensemble + EXP3
python learning/production-serving/src/openai_api_server.py         # OpenAI 协议层 demo（不起服务）
python learning/production-serving/src/trtllm_build.py              # TensorRT-LLM build 配置(mock)
```

> 注：
> - `cost_calc`/`metrics_prometheus`/`streaming_sse`/`clipper_original_minimal`/`openai_api_server` 原先缺 `__main__`（直跑无输出），本轮补了 `demo()` + `__main__` 使其可跑。
> - `openai_api_server.py` 的协议逻辑与 FastAPI 层**解耦**：`app` 仅在 fastapi 可用时构建，纯函数(`build_completion_response`/`validate_chat_request`/`mock_generate`)不依赖起服务即可测/跑。

**起真 mock OpenAI server**（可选，本机已装 fastapi+uvicorn）：

```powershell
# 注意从 src 目录起，模块名才是 openai_api_server
cd learning/production-serving/src; uvicorn openai_api_server:app --port 8000
# 另开一个终端：
#   curl http://localhost:8000/v1/models
#   curl -X POST http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" `
#        -d '{"model":"mock-7b","messages":[{"role":"user","content":"hi"}]}'
```

**测试（V2）**：27 个测试覆盖成本/指标/SSE/Clipper/OpenAI 协议：

```powershell
python -m pytest learning/production-serving/src/tests/ -v
# 或经审计 harness：python scripts/eric_3080ti_env_audit.py --modules production-serving --tests
```

## 退出条件 checklist

- [x] 12 lecture + 12 notebook
- [x] 27 tests pass
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
