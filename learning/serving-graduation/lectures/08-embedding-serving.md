# L08 · Embedding 服务

## 1 · 与 LLM 推理区别
- embedding model: BERT-style encoder
- 一次 forward 出向量（不 autoregressive）
- 显存 / latency 远小于 LLM

## 2 · 用途
- RAG: 文档检索
- semantic search
- clustering

## 3 · 模型
- BGE / E5 / mxbai-embed-large
- 顶尖：voyage-3, gte-large

## 4 · 性能
- 一卡 4090: 10k embedding/s
- batch=256 极快

## 5 · 部署
```python
# 用 vLLM (0.7+ 支持 embedding)
from vllm import LLM
llm = LLM("BAAI/bge-large-en-v1.5", task="embed")
emb = llm.encode(["hello", "world"])

# 或专用 TEI (text-embeddings-inference)
docker run -p 8080:80 ghcr.io/huggingface/text-embeddings-inference \
    --model-id BAAI/bge-large-en-v1.5
```

## 6 · 集成 FAISS
```python
import faiss
index = faiss.IndexFlatIP(1024)
index.add(embeddings)
D, I = index.search(query_emb, k=10)
```

## 7 · 实现：[embedding_serve.py](../src/embedding_serve.py)
- mock embedding endpoint
