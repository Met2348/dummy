"""integrity：红队一个 mini-scientist 的 4 种造假 + 4 个诚信守卫（毕业 capstone）。"""
from .data import (
    KNOWN_FINGERPRINTS, accuracy, fingerprint, make_dataset, rule_predict,
)
from .naive_scientist import ATTACKS, Report, naive_accept, run_science
from .guards import (
    GUARDS, GuardResult, audit, guard_dataset, guard_independent_review,
    guard_metric, guard_provenance,
)

__all__ = [
    "KNOWN_FINGERPRINTS", "accuracy", "fingerprint", "make_dataset", "rule_predict",
    "ATTACKS", "Report", "naive_accept", "run_science",
    "GUARDS", "GuardResult", "audit", "guard_dataset", "guard_independent_review",
    "guard_metric", "guard_provenance",
]
