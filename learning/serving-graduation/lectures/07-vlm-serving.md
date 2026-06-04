# L07 · VLM 推理服务

## 1 · 流程
1. Image encoder（ViT）算 visual tokens
2. visual + text token 拼起进 LLM
3. LLM decode 输出

## 2 · 关键工程
- **image encoder offload**: 单独 GPU 跑 ViT
- **async vision**: ViT 跑同时 LLM 准备
- **paged visual cache**: 同一 image 多 query 复用

## 3 · 显存
- Qwen2-VL 7B: 14 GB (LLM) + 4 GB (ViT) = 18 GB
- ViT 不能切 TP（小，不值）
- 多 image 输入时 visual token 累积

## 4 · 部署
```python
# vLLM
llm = LLM("Qwen/Qwen2-VL-7B", limit_mm_per_prompt={"image": 4})
out = llm.generate({"prompt": "Describe", "multi_modal_data": {"image": pil_img}})
```

## 5 · API 兼容
OpenAI vision API:
```json
{
    "messages": [{
        "role": "user",
        "content": [
            {"type": "text", "text": "What is this?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
        ]
    }]
}
```

## 6 · 实现：[vlm_serve.py](../src/vlm_serve.py)
- mock VLM serving 模板
