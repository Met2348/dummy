"""mini_eval：方法 spec + rubric 真 exec 判分，演示弱 rubric 被刷、强 rubric 抗刷。"""
from .sandbox import FORBIDDEN, SAFE_BUILTINS, SafeExecError, safe_exec
from .task import HELDOUT, SPEC, VISIBLE, true_label
from .candidates import CANDIDATES, MALICIOUS_SRC
from .rubrics import RUBRICS, heldout, trust_print, visible_only
from .evaluate import run_eval

__all__ = [
    "FORBIDDEN", "SAFE_BUILTINS", "SafeExecError", "safe_exec",
    "HELDOUT", "SPEC", "VISIBLE", "true_label",
    "CANDIDATES", "MALICIOUS_SRC",
    "RUBRICS", "heldout", "trust_print", "visible_only",
    "run_eval",
]
