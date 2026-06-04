# L09 · 冷启动

## 1 · 痛点
- 70B 模型加载 = 1-2 分钟
- 容器启动 = 30s
- 首请求等很久 → 用户体验差

## 2 · 缓解
| 策略 | 减少 |
|------|-----|
| 预热: container 启动后跑 dummy request | warmup 30s |
| pinned hot containers | 永远不冷启 |
| lazy model load | 后台异步加载 |
| 模型量化 | 加载快 4x |
| Mmap load | OS 缓存 |

## 3 · canary deploy
- 新版本上线，先 5% 流量
- 监控错误率 / latency
- 慢慢 ramp 到 100%

## 4 · blue-green
- 蓝绿两个版本同时跑
- DNS / LB 切换瞬间完成
- 出问题秒回滚

## 5 · 实现
本课概念课，无独立 src。
