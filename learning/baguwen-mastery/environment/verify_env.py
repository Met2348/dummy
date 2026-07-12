"""环境自检：本 track 纯 stdlib（backend_coding.sync_primitives 用到 threading，仍是 stdlib）。"""
from __future__ import annotations

import os
import sys


def main() -> None:
    assert sys.version_info >= (3, 9), f"需要 Python >= 3.9，当前 {sys.version_info}"
    print(f"Python {sys.version_info[:3]} OK（纯 stdlib，无第三方依赖）")

    src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    sys.path.insert(0, src_dir)

    import ai_qa
    import backend_qa
    import tracker  # noqa: F401

    print(f"ai_qa: {len(ai_qa.ALL_QA)} 题 导入 OK")
    print(f"backend_qa: {len(backend_qa.ALL_QA)} 题 导入 OK")
    print("tracker: 导入 OK")
    print("SUCCESS: 环境就绪，可直接运行 src/tests/test_all.py")


if __name__ == "__main__":
    main()
