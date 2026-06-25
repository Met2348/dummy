"""科研生命周期七环 + 一份"真实系统"目录（带诚实的证据标注）。

STAGES 是科研生命周期七环。每个 System 记两类信息：
- **自称**：`claimed_level`——论文/厂商对外宣称的自主级别。
- **证据**：`automates`（真正自动化的环）、`human_sets_problem`（问题是否人给）、
  `independent_verification`（结果有没有经过"非自评"的独立验证，如 held-out 测试/湿实验/排行榜）。

注：这些标注是**面向教学的、可辩护的刻画**，不是权威定级；目的是练"看证据不看话术"。
所有判断都来自 papers/INDEX.md 的 40 篇文献的公开描述与各系统的 Limitations 节。
"""
from __future__ import annotations

from dataclasses import dataclass

# 科研生命周期七环（lifecycle stages）——列坐标
STAGES = ["ideation", "hypothesis", "design", "experiment", "analysis", "writeup", "review"]
# 简码（地图列头，避免中英混排错位）
STAGE_CODE = {
    "ideation": "ide", "hypothesis": "hyp", "design": "des", "experiment": "exp",
    "analysis": "ana", "writeup": "wri", "review": "rev",
}


@dataclass(frozen=True)
class System:
    name: str
    year: str
    claimed_level: str                 # 自称：tool|analyst|scientist
    automates: tuple                    # 证据：真正自动化的 lifecycle 阶段（STAGES 子集）
    human_sets_problem: bool            # 证据：问题/假设是否由人给定
    independent_verification: bool      # 证据：结果是否经独立（非自评）验证
    note: str = ""


# —— 真实系统目录（~15 个，覆盖 tool/analyst/scientist 与各生命周期环）——
SYSTEM_CATALOG = (
    System("AI Scientist v1", "2024", "scientist",
           ("ideation", "experiment", "analysis", "writeup", "review"),
           human_sets_problem=False, independent_verification=False,
           note="Sakana：模板起步，自生成 idea、自跑实验、自写自评。"),
    System("AI Scientist v2", "2025", "scientist",
           ("ideation", "experiment", "analysis", "writeup", "review"),
           human_sets_problem=False, independent_verification=False,
           note="去模板+树搜索+VLM 看图；更自主反而成功率更低。"),
    System("AI-Researcher", "2025", "scientist",
           ("ideation", "experiment", "analysis", "writeup"),
           human_sets_problem=False, independent_verification=False,
           note="数学↔代码双向映射降幻觉，仍自评。"),
    System("NovelSeek", "2025", "scientist",
           ("ideation", "experiment", "analysis", "writeup"),
           human_sets_problem=False, independent_verification=False,
           note="闭环自主研究，结果自报。"),
    System("Google co-scientist", "2025", "analyst",
           ("ideation", "hypothesis", "design"),
           human_sets_problem=True, independent_verification=True,
           note="给定研究目标，生成/排序假设；湿实验验证（独立）。"),
    System("Agent Laboratory", "2025", "analyst",
           ("design", "experiment", "analysis", "writeup"),
           human_sets_problem=True, independent_verification=False,
           note="人给 idea，agent 把它做成论文。"),
    System("AlphaEvolve", "2025", "analyst",
           ("design", "experiment", "analysis"),
           human_sets_problem=True, independent_verification=True,
           note="进化搜索，自动评测器打分（可独立验证）。"),
    System("Darwin Godel Machine", "2025", "scientist",
           ("design", "experiment", "analysis"),
           human_sets_problem=True, independent_verification=True,
           note="自改写 agent；但在给定 benchmark 上改进 → 证据更像 analyst。"),
    System("ADAS", "2024", "analyst",
           ("design", "experiment", "analysis"),
           human_sets_problem=True, independent_verification=True,
           note="meta-agent 搜索 agent 设计，benchmark 评分。"),
    System("MLE-bench agents", "2024", "analyst",
           ("design", "experiment", "analysis"),
           human_sets_problem=True, independent_verification=True,
           note="Kaggle 任务，排行榜独立评分。"),
    System("Coscientist (chem)", "2023", "analyst",
           ("design", "experiment", "analysis"),
           human_sets_problem=True, independent_verification=True,
           note="自主化学实验，真实验室结果。"),
    System("ResearchAgent", "2024", "analyst",
           ("ideation", "hypothesis"),
           human_sets_problem=True, independent_verification=False,
           note="迭代生成 idea+ReviewingAgents；不执行实验。"),
    System("STORM", "2024", "tool",
           ("ideation", "writeup"),
           human_sets_problem=True, independent_verification=False,
           note="多视角提问→写一篇带引用的维基式综述；不做实验。"),
    System("GPT-Researcher", "2023", "tool",
           ("ideation", "writeup"),
           human_sets_problem=True, independent_verification=False,
           note="检索+综合成报告。"),
    System("AgentRxiv", "2025", "analyst",
           ("writeup", "review"),
           human_sets_problem=True, independent_verification=False,
           note="agent 之间共享/评审预印本；自称协作科研。"),
)
