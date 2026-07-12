"""环境检查：确认深水区知识库 + 默写套件可以正常导入。dictation 部分复用
interview-prep 同款真 torch（本仓库唯一破例），其余纯 stdlib。
"""
from __future__ import annotations

import os
import sys

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC_DIR)

import torch  # noqa: E402

from ai_deep import ALL_DP as AI_DP  # noqa: E402
from backend_deep import ALL_DP as BE_DP  # noqa: E402
from dictation import harness  # noqa: E402

print(f"torch: {torch.__version__}")
print(f"ai_deep: {len(AI_DP)} 个深水知识点 导入 OK")
print(f"backend_deep: {len(BE_DP)} 个深水知识点 导入 OK")
print(f"dictation: {len(harness.REGISTRY)} 个默写目标已注册")
