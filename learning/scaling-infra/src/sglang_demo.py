"""SGLang DSL 示例 - 仅打印模板."""
from __future__ import annotations


def chat_dsl():
    print("=== SGLang chat DSL ===")
    print("""
import sglang as sgl
sgl.set_default_backend(sgl.RuntimeEndpoint("http://localhost:30000"))

@sgl.function
def chat(s, system_prompt, q):
    s += sgl.system(system_prompt)
    s += sgl.user(q)
    s += sgl.assistant(sgl.gen("ans", max_tokens=200))

state = chat.run(system_prompt="You are an expert.", q="Explain RoPE.")
print(state["ans"])
""")


def multi_branch_dsl():
    print("=== Multi-branch (judging) ===")
    print("""
@sgl.function
def judge(s, q, choices):
    s += sgl.user(f"Q: {q}\\nChoose best:")
    forks = s.fork(len(choices))
    for i, c in enumerate(choices):
        forks[i] += sgl.user(c)
        forks[i] += sgl.assistant(sgl.gen("score", max_tokens=20))
    return [f["score"] for f in forks]
""")


def json_output():
    print("=== Schema-guided JSON ===")
    print("""
schema = {
  "type": "object",
  "properties": {
    "name": {"type": "string"},
    "age":  {"type": "integer"},
  },
  "required": ["name", "age"],
}
s += sgl.gen("person", schema=schema)
""")


def launch_server():
    print("=== Launch server ===")
    print("""
python -m sglang.launch_server \\
  --model meta-llama/Llama-3.2-3B-Instruct \\
  --port 30000 \\
  --mem-fraction-static 0.85 \\
  --enable-torch-compile
""")


if __name__ == "__main__":
    chat_dsl()
    multi_branch_dsl()
    json_output()
    launch_server()
