"""
专题环境自检脚本。

运行：python environment/verify_env.py
预期：全部 [OK]，最后打印 "Environment ready."
"""
from __future__ import annotations

import sys


def check(name: str, ok: bool, detail: str = "") -> bool:
    flag = "[OK]  " if ok else "[FAIL]"
    print(f"{flag} {name} {detail}")
    return ok


def main() -> int:
    all_ok = True

    all_ok &= check(
        "Python >= 3.10",
        sys.version_info >= (3, 10),
        f"(got {sys.version.split()[0]})",
    )

    try:
        import torch
        all_ok &= check(
            "torch",
            True,
            f"version={torch.__version__}, cuda={torch.cuda.is_available()}",
        )
    except ImportError as e:
        all_ok &= check("torch", False, str(e))

    try:
        import transformers
        all_ok &= check("transformers", True, f"version={transformers.__version__}")
        if int(transformers.__version__.split(".")[0]) < 5:
            print("       注意：本专题使用 transformers 5.x 接口（DynamicCache）。"
                  "若您是 4.x 用户，需自行替换 past_key_values 构造方式。")
    except ImportError as e:
        all_ok &= check("transformers", False, str(e))

    try:
        import peft
        all_ok &= check("peft", True, f"version={peft.__version__}")
    except ImportError as e:
        all_ok &= check("peft", False, str(e))

    try:
        import jupyterlab
        all_ok &= check("jupyterlab", True, f"version={jupyterlab.__version__}")
    except ImportError as e:
        all_ok &= check("jupyterlab", False, str(e))

    try:
        import ipykernel
        all_ok &= check("ipykernel", True, f"version={ipykernel.__version__}")
    except ImportError as e:
        all_ok &= check("ipykernel", False, str(e))

    try:
        import matplotlib
        all_ok &= check("matplotlib", True, f"version={matplotlib.__version__}")
    except ImportError as e:
        all_ok &= check("matplotlib", False, str(e))

    try:
        import numpy
        all_ok &= check("numpy", True, f"version={numpy.__version__}")
    except ImportError as e:
        all_ok &= check("numpy", False, str(e))

    # 端到端冒烟测试：加载 gpt2 并跑一次推理
    try:
        from transformers import GPT2Tokenizer, GPT2LMHeadModel
        import torch as _torch
        tok = GPT2Tokenizer.from_pretrained("gpt2")
        tok.pad_token = tok.eos_token
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        with _torch.no_grad():
            out = model(**tok("Hello", return_tensors="pt"))
        all_ok &= check(
            "gpt2 推理冒烟测试",
            True,
            f"logits.shape={tuple(out.logits.shape)}",
        )
    except Exception as e:
        all_ok &= check("gpt2 推理冒烟测试", False, str(e))

    print()
    if all_ok:
        print("Environment ready.")
        return 0
    print("Environment NOT ready. 请检查上面 [FAIL] 项。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
