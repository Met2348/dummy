# L04 · Triton Inference Server

## 1 · 概述
NVIDIA 官方多模型服务器，支持：
- TensorRT / PyTorch / ONNX / Python backend
- ensemble pipeline
- 动态 batching / multi-model
- gRPC / HTTP API

## 2 · 模型仓库结构
```
model_repository/
├── llm/
│   ├── config.pbtxt          # 配置
│   └── 1/                    # version 1
│       └── model.py          # Python backend
├── tokenizer/
│   ├── config.pbtxt
│   └── 1/
│       └── model.py
└── llm_pipeline/             # ensemble
    └── config.pbtxt
```

## 3 · config.pbtxt
```
name: "llm"
backend: "python"
max_batch_size: 32
input [
  { name: "input_ids", data_type: TYPE_INT32, dims: [-1] }
]
output [
  { name: "logits", data_type: TYPE_FP16, dims: [-1, -1] }
]
instance_group [ { kind: KIND_GPU, count: 1 } ]
```

## 4 · ensemble pipeline
```
tokenizer → llm → detokenizer
```
在 `llm_pipeline/config.pbtxt` 串起来。

## 5 · 启动
```bash
docker run --gpus all \
    -v $(pwd)/model_repository:/models \
    -p 8000:8000 \
    nvcr.io/nvidia/tritonserver:24.05-py3 \
    tritonserver --model-repository=/models
```

## 6 · 客户端
```python
import tritonclient.http as client
c = client.InferenceServerClient("localhost:8000")
inp = client.InferInput("input_ids", [1, 10], "INT32")
inp.set_data_from_numpy(np.array([[1,2,3,...]], dtype=np.int32))
result = c.infer("llm", [inp])
```

## 7 · 优点
- 多模型一起服务（embedding + chat + rerank）
- 完善 metrics
- gRPC 高效

## 8 · 缺点
- 配置文件繁
- Python backend 比 TRT-LLM runtime 慢

## 9 · 实现：[triton_model_repo/](../src/triton_model_repo/)
- 完整 llm 模型 + config 示例
