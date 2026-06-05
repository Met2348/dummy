# L04 · Vector DB 选型

## 2025-2026 主流

| DB | 类型 | 主语言 | 强项 |
|----|------|--------|------|
| **Pinecone** | SaaS | Rust/Go | 商业首选 |
| **Weaviate** | OSS / SaaS | Go | hybrid + multi-modal |
| **Chroma** | OSS (embedded) | Python | dev 快 |
| **Qdrant** | OSS / SaaS | Rust | 性能强 |
| **Milvus** | OSS | Go | 大规模 |
| **Postgres pgvector** | SQL + ext | C | SQL 友好 |
| **LanceDB** | OSS (embedded) | Rust | 列存 |
| **Vespa** | OSS | Java | YH 出品 |
| **Turbopuffer** | SaaS | Rust | 2024 新秀 |
| **Redis** + RediSearch | OSS | C | 内存 KV + vector |

## 选型矩阵

| 场景 | 推荐 |
|------|------|
| Quick prototype | Chroma (内嵌) |
| Production SaaS | Pinecone |
| Self-host + 性能 | Qdrant |
| SQL 团队 | pgvector |
| Hybrid + multi-modal | Weaviate |
| 10B+ vectors | Milvus |
| Edge / 小流量 | LanceDB |
| 极低延迟 KV | Redis |

## 索引类型

| Index | 算法 | 用 |
|-------|------|---|
| **HNSW** | Hierarchical NSW | 默认 |
| **IVF** | Inverted File | 大规模 |
| **IVFPQ** | IVF + Product Quant | 内存省 |
| **Flat** | 暴力 | 小规模/baseline |
| **DiskANN** | Disk-friendly | 极大规模 |

## HNSW 关键参数

| 参数 | 含义 | 调 |
|------|------|---|
| `M` | 每节点邻居数 | 16-32 |
| `ef_construction` | 建索引精度 | 100-500 |
| `ef_search` | 查询精度 | 50-200 |

→ 更高 = 更准但更慢/更耗内存。

## 实现 (`vector_store.py` 预告)

简化版 brute-force vector store：

```python
class SimpleVectorStore:
    def __init__(self):
        self.vectors = {}  # id → vec
        self.payloads = {} # id → metadata

    def upsert(self, id, vec, payload): ...
    def search(self, query_vec, k=5, filter_fn=None):
        scored = [(id, cosine(v, query_vec)) for id, v in self.vectors.items()
                  if filter_fn is None or filter_fn(self.payloads.get(id, {}))]
        return sorted(scored, key=lambda x: x[1], reverse=True)[:k]
    def delete(self, id): ...
```

## 元数据 filter（生产关键）

```python
# Memory per-user 隔离
store.search(query_vec, k=5, filter={"user_id": "alice"})
```

→ 不能跨 user 检索。pgvector / Pinecone / Qdrant / Weaviate 都原生支持。

## 嵌入 dim vs 存储

| dim | 1M docs 存储 |
|-----|------------:|
| 256 | 1 GB |
| 768 | 3 GB |
| 1024 | 4 GB |
| 1536 (OpenAI default) | 6 GB |
| 3072 (OpenAI large) | 12 GB |

Matryoshka 嵌入可截断 → 索引 1024 但查询 256 → 速度 4×。

## 退出条件

- 能列 10 vector DB
- 能讲 4 索引类型
- 知道 HNSW 3 参数

## 一句话

> Vector DB 10 家选型：开发 Chroma / 生产 Pinecone / 自托管 Qdrant / SQL pgvector — HNSW 默认索引。
