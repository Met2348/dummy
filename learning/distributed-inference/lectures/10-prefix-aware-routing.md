# L10 · Prefix-Aware Routing

## 1 · 痛点
多副本部署时（如 4 GPU 各跑同 model）：
- naive round-robin 路由：每 GPU 看到不同 prompt → prefix cache 命中率低
- prefix-aware：让带相同 prefix 的请求路由到同 GPU → 命中率最大化

## 2 · 路由策略
| 策略 | 命中率 | 负载均衡 |
|------|-------|---------|
| Round-Robin | 低 | 高 |
| Random | 低 | 高 |
| Hash-based prefix | 高 | 中 |
| **Consistent hashing** | 高 | 高 |
| Load-aware + prefix | 最佳 | 最佳 |

## 3 · Consistent hashing
- 把 GPU 放圈环 (consistent hash ring)
- prompt 前缀 hash → 对应圈环位置
- 取最近 GPU

加 GPU/删 GPU 时只影响 1/N 流量 → 缓存不全失效。

## 4 · Mooncake 路由
- 全局 prefix index 服务（Redis）
- 维护 (hash → GPU set) 映射
- 实时更新（cache evict 时移除）

## 5 · 负载感知
单独 prefix hit 路由可能导致 hot GPU。
解：在 hit + load 之间 trade-off：
```
target = argmin_{gpu} (load[gpu] - λ * hits[gpu, prompt])
```

## 6 · 实现：[routing_policies.py](../src/routing_policies.py)
- round_robin / prefix_hash / consistent_hash / load_aware
- 命中率 + 负载均衡度量
