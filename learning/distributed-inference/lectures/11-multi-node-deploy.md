# L11 · 多节点部署

## 1 · 部署框架
| 框架 | 用途 |
|------|-----|
| **Ray Serve** | 通用 Python serve，集成 vLLM |
| **KubeRay** | k8s 上跑 Ray |
| **Triton + ensemble** | NVIDIA 官方 |
| **TGI** | HuggingFace 的多节点 LLM serve |
| **vLLM Cluster** | vLLM 0.7+ 内置 multi-node |

## 2 · Ray Serve 关键 API
```python
import ray
from ray import serve
from vllm import LLM

@serve.deployment(num_replicas=4)
class LlmDeployment:
    def __init__(self):
        self.llm = LLM("Qwen/Qwen2.5-7B", tensor_parallel_size=2)
    async def __call__(self, prompt):
        return self.llm.generate(prompt)

serve.run(LlmDeployment.bind())
```

## 3 · 多节点拓扑
- Head node + worker nodes
- Ray cluster 管理 (object store 共享)
- vLLM placement group 跨节点

## 4 · 故障与自愈
- Ray actor 死了自动重启
- replicate state 通过 object store
- KV cache 不跨副本共享（每副本独立）

## 5 · 监控
- Ray dashboard
- Prometheus metric per worker
- p50/p99 latency tracking

## 6 · 经验
- TP 不跨节点（带宽不够）
- PP 跨节点（流式数据少）
- replication 跨节点（独立）
- EP 跨节点（DeepSeek-V3 → IB 必须）

## 7 · 实现
留 placeholder + 文档；多节点真跑要 actual cluster。
