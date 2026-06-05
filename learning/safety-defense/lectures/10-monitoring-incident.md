# L10 · 在线监控 + 事件响应

## 4 大监控指标

| 指标 | 测什么 | 工具 |
|------|------|------|
| **Refusal rate** | 模型拒绝率 % | Prometheus |
| **Unsafe output rate** | classifier flag % | Llama Guard |
| **Latency p99** | 推理 + safety 总延迟 | Datadog |
| **PII leak count** | 输出含 PII 次数 | regex monitor |

## 完整 logging schema

```json
{
  "request_id": "abc-123",
  "timestamp": "2026-06-05T10:00:00Z",
  "user_id_hash": "sha256(user_id)",
  "input_safety": "safe / unsafe / S1,S9",
  "input_classifier_score": 0.02,
  "output_safety": "safe",
  "output_classifier_score": 0.01,
  "refusal": false,
  "tokens_in": 234,
  "tokens_out": 156,
  "latency_ms": 1450,
  "model_version": "claude-3.7-sonnet"
}
```

注意：**user_id_hash**, **不存 raw prompt** (privacy)。

## Kill switch

如果发现 model 大量输出 harmful：

```python
# 后台 cron 每分钟检查
if unsafe_rate(last_5min) > 0.1:
    rollback_to_previous_model()
    page_oncall()
```

## 事件响应流程

```
1. Alert (sentry / page)
2. Triage (15 min)
   - 哪个 model? 哪个 endpoint?
   - 实际影响范围?
3. Mitigate (1 hour)
   - rollback / kill-switch / 加规则
4. Communicate (4 hours)
   - status page 更新
5. Root cause (24-72 hours)
   - 训练数据? deploy 错误?
6. Post-mortem (1 week)
   - 不指责 + action items
```

## 真实事件

- **2023.03 ChatGPT 显示他人聊天记录**：cache key bug
- **2023.12 Microsoft Copilot 输出 Tay 名梗**：prompt injection
- **2024.08 Grok 输出政治偏见**：合成数据偏
- **2025.02 DeepSeek 数据泄露**：DB 未加密暴露

## 工具栈

| 类 | 选择 |
|----|------|
| Logs | Datadog / Splunk / ELK |
| Metrics | Prometheus + Grafana |
| Tracing | OpenTelemetry |
| Alerting | PagerDuty / Opsgenie |
| Sentry | error tracking |

## Audit trail（合规）

```
For each LLM call:
  - log request_id + input_hash + output_hash
  - retain 90 days
  - encrypted at rest
```

GDPR Article 22 + EU AI Act 要求。

## 一句话

> 监控不是可选，是上线前提；事件响应平时演练 → 真出事 1 小时内 mitigate。
