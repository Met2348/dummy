"""vLLM 离线推理 + OpenAI server 启动模板.

教学版: 仅打印 setup. 真跑需 WSL2 + vllm 安装.
"""
from __future__ import annotations


def offline_inference_template():
    print("=== Offline Inference ===")
    print("""
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.2-3B-Instruct",
    gpu_memory_utilization=0.85,
    max_model_len=4096,
    dtype="bfloat16",
)
sp = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=200)
outs = llm.generate(["Hello, world."], sp)
for o in outs:
    print(o.outputs[0].text)
""")


def server_template():
    print("=== OpenAI-compatible Server ===")
    print("""
# bash:
python -m vllm.entrypoints.openai.api_server \\
  --model meta-llama/Llama-3.2-3B-Instruct \\
  --port 8000 \\
  --gpu-memory-utilization 0.85 \\
  --max-model-len 4096

# client:
from openai import OpenAI
c = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
resp = c.chat.completions.create(
    model="meta-llama/Llama-3.2-3B-Instruct",
    messages=[{"role": "user", "content": "Hi"}],
)
""")


def quant_options():
    print("=== Quantization ===")
    opts = [
        ("awq", "Llama-3-8B-AWQ", "4-bit, NVIDIA"),
        ("fp8", "Llama-3-8B-FP8", "8-bit FP, H100+ best"),
        ("gptq", "Llama-3-8B-GPTQ", "4-bit, group_size 128"),
        ("bitsandbytes", "—", "4/8-bit, 慢"),
    ]
    for q, ex, note in opts:
        print(f"  quantization={q:14s} ex={ex:25s}  {note}")


if __name__ == "__main__":
    offline_inference_template()
    server_template()
    quant_options()
