"""ideation：创意/假设生成 + 锦标赛排序，演示 novelty≠feasibility 与评委自偏好。"""
from .task import DEFAULT, make_data, train_logreg
from .ideas import Idea, IDEA_BANK, get_idea
from .judge import BUZZWORDS, SELF_BIAS, judge_score, novelty, prefer
from .tournament import (
    feasibility_of, rank_by_feasibility, run_tournament,
)

__all__ = [
    "DEFAULT", "make_data", "train_logreg",
    "Idea", "IDEA_BANK", "get_idea",
    "BUZZWORDS", "SELF_BIAS", "judge_score", "novelty", "prefer",
    "feasibility_of", "rank_by_feasibility", "run_tournament",
]
