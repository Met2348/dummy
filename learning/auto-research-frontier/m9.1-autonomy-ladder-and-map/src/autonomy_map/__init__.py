"""autonomy_map：把"AI 科研系统"按证据归到 Tool/Analyst/Scientist 三级阶梯，
并铺在"自主性阶梯 × 科研生命周期"二维地图上。

核心立场：**按证据分级，不按自称分级**。systems.py 里每个系统的 `claimed_level`
是它对外宣称的级别，`automates/human_sets_problem/independent_verification` 是证据；
classifier.py 只用证据推 `evidenced_level`，两者之差即 hype gap。
"""
from .systems import STAGES, System, SYSTEM_CATALOG
from .classifier import (
    LEVELS, LEVEL_RANK, Classification, classify, evidenced_level,
)
from .mapping import classify_all, ladder_lifecycle_grid, render_map, render_table

__all__ = [
    "STAGES", "System", "SYSTEM_CATALOG",
    "LEVELS", "LEVEL_RANK", "Classification", "classify", "evidenced_level",
    "classify_all", "ladder_lifecycle_grid", "render_map", "render_table",
]
