# L10 · 生产监控

## 1 · 关键指标
| 类型 | 指标 |
|------|------|
| latency | TTFT p50/p95/p99, ITL p50/p99 |
| throughput | requests/s, tokens/s |
| GPU | utilisation, memory, temperature |
| KV cache | hit_rate, used_blocks |
| 业务 | error_rate, retry_rate |

## 2 · Prometheus
- pull-based metrics
- exporter 暴露 `/metrics` endpoint
- 历史数据 Prometheus 拉取存储

```python
from prometheus_client import Counter, Histogram, generate_latest

REQS = Counter("llm_requests_total", "Total requests", ["model"])
TTFT = Histogram("llm_ttft_seconds", "Time to first token", ["model"])

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

## 3 · Grafana
- 连 Prometheus 数据源
- 图表 / alert
- 标准 LLM dashboard 有现成模板

## 4 · 报警
- p99 latency > 2s → alert
- error rate > 1% → alert
- GPU util < 30% (浪费) → notify
- GPU memory > 95% → alert

## 5 · 日志
- 结构化 (JSON) 易解析
- request_id 贯穿全链路
- 采样（不全存）

## 6 · 实现：[metrics_prometheus.py](../src/metrics_prometheus.py)
- 几个标准 Counter/Histogram
- /metrics endpoint mock
