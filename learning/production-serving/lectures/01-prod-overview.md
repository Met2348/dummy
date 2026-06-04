# L01 · 生产部署全图

## 1 · 生产 vs 研究
| 维度 | 研究 | 生产 |
|------|------|------|
| latency | "<5s 可接受" | p50 < 500 ms, p99 < 2s |
| availability | 99% | 99.9%+ |
| QPS | "跑通就行" | 100-1000+ |
| 成本 | 不重要 | **极重要** ($/M token) |
| 监控 | 看 stdout | Prometheus + Grafana |
| 滚动升级 | 手动 | canary + blue/green |

## 2 · 部署栈
```
+------------------------------+
| 业务层（agent / chatbot）     |
+------------------------------+
| API 网关（FastAPI + SSE）     |  ← L09
+------------------------------+
| 模型推理（vLLM / SGLang / TRT）|  ← Topic 1-5 + L02-L04
+------------------------------+
| 监控（Prometheus / Grafana）   |  ← L10
+------------------------------+
| 编排（Triton / K8s / Ray）     |  ← L04 / L11
+------------------------------+
```

## 3 · 主要框架对照
| 框架 | 强项 | 弱项 |
|------|-----|-----|
| **vLLM** | 易上手、OpenAI API | 编译慢、kernel 不如 TRT |
| **SGLang** | agent 极佳 | 部署生态弱 |
| **TensorRT-LLM** | 最快、Hopper 优化 | build 复杂 |
| **Triton** | 多模型 / ensemble | 配置文件多 |
| **Ollama** | 端侧极简 | 推理慢 |
| **llama.cpp** | CPU / Metal | 仅推理 |

## 4 · 路线
- L02-L03 TRT-LLM build
- L04 Triton serving
- L05-L07 Ollama / llama.cpp / LM Studio
- L08-L09 OpenAI API + FastAPI 实现
- L10 监控
- L11 成本工程
- L12 Capstone

## 5 · 一句话
> 生产 = **vLLM 跑通 → FastAPI 包装 → Prometheus 监控 → cost engineering**。
