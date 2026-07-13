"""Social Hire Gauntlet:国内大厂资深社招面试全流程通关训练营(8关,合计158点)。

与 staff_gauntlet(对标 Frontier Lab research/staff 面试loop)不同,这里对标的是国内大厂
(阿里/字节/腾讯等)资深社招真实流程阶段:项目拷打(个人贡献量化)→手撕代码(AI作弊检测时代)→
隐藏bug调试深水→系统设计主导权判断→交叉面/委员会面(P7/P8业务视角)→离职原因与谈薪谈判→
背景调查时代的诚实一致性→反问环节与快速融入新业务。GATES 按这个真实顺序排列,练法同样是
按顺序过关,不是随机抽题(见 lectures/00-how-to-defend.md)。

与 staff_gauntlet 的差异化边界(避免重复,详见各 gate 文件内的说明):
- G4(系统设计主导权)vs staff_gauntlet G4(约束临场改判断):这里练"主动驱动设计对话"本身。
- G5(交叉面/委员会面)vs staff_gauntlet G9(国内资深社招视角):这里专练P7/P8业务向委员会
  评审机制,不重复分布式训练等知识点。
- G6(离职与谈薪)vs staff_gauntlet G10(定级谈判):这里是国内语境的离职话术+谈薪战术,
  不是frontier lab的title/equity比较。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deep_common import categories  # noqa: E402

from .gate1_project_deep_dive_impact import BANK as _gate1
from .gate2_coding_ai_cheating_era import BANK as _gate2
from .gate3_debugging_round import BANK as _gate3
from .gate4_system_design_ownership import BANK as _gate4
from .gate5_cross_review_committee import BANK as _gate5
from .gate6_leaving_and_negotiation import BANK as _gate6
from .gate7_background_check_honesty import BANK as _gate7
from .gate8_reverse_questions_and_onboarding import BANK as _gate8

ALL_DP = (
    list(_gate1) + list(_gate2) + list(_gate3) + list(_gate5) + list(_gate8)
)
ALL_SP = (
    list(_gate4) + list(_gate6) + list(_gate7)
)

# GATES:按真实社招面试loop顺序排列的关卡元数据,练习时应按 n 从小到大顺序过关。
GATES: tuple[dict, ...] = (
    {"n": 1, "name": "project_deep_dive_impact", "cat": "社招关卡一:项目拷打与个人贡献量化", "kind": "dp", "bank": _gate1},
    {"n": 2, "name": "coding_ai_cheating_era", "cat": "社招关卡二:手撕代码与AI作弊检测时代", "kind": "dp", "bank": _gate2},
    {"n": 3, "name": "debugging_round", "cat": "社招关卡三:隐藏Bug调试深水", "kind": "dp", "bank": _gate3},
    {"n": 4, "name": "system_design_ownership", "cat": "社招关卡四:系统设计主导权判断", "kind": "sp", "bank": _gate4},
    {"n": 5, "name": "cross_review_committee", "cat": "社招关卡五:交叉面与委员会面(P7/P8业务视角)", "kind": "dp", "bank": _gate5},
    {"n": 6, "name": "leaving_and_negotiation", "cat": "社招关卡六:离职原因与谈薪谈判", "kind": "sp", "bank": _gate6},
    {"n": 7, "name": "background_check_honesty", "cat": "社招关卡七:背景调查时代的诚实一致性", "kind": "sp", "bank": _gate7},
    {"n": 8, "name": "reverse_questions_and_onboarding", "cat": "社招关卡八:反问环节与快速融入新业务", "kind": "dp", "bank": _gate8},
)


def _self_test() -> None:
    assert len(ALL_DP) >= 90, len(ALL_DP)
    assert len(ALL_SP) >= 50, len(ALL_SP)
    assert len(ALL_DP) + len(ALL_SP) >= 150, len(ALL_DP) + len(ALL_SP)
    assert len(categories(ALL_DP)) == 5, len(categories(ALL_DP))
    assert len(categories(ALL_SP)) == 3, len(categories(ALL_SP))

    dp_ids = [dp.id for dp in ALL_DP]
    sp_ids = [sp.id for sp in ALL_SP]
    assert len(dp_ids) == len(set(dp_ids)), "social_hire_gauntlet ALL_DP 内存在重复 id"
    assert len(sp_ids) == len(set(sp_ids)), "social_hire_gauntlet ALL_SP 内存在重复 id"
    assert all(i.startswith("dp-sh-") for i in dp_ids), "存在不以dp-sh-开头的DeepPoint id"
    assert all(i.startswith("sc-sh-") for i in sp_ids), "存在不以sc-sh-开头的ScenarioPoint id"

    dp_triggers = [dp.trigger for dp in ALL_DP]
    sp_triggers = [sp.trigger for sp in ALL_SP]
    assert len(dp_triggers) == len(set(dp_triggers)), "social_hire_gauntlet ALL_DP 内存在重复 trigger"
    assert len(sp_triggers) == len(set(sp_triggers)), "social_hire_gauntlet ALL_SP 内存在重复 trigger"

    assert len(GATES) == 8, len(GATES)
    assert [g["n"] for g in GATES] == list(range(1, 9)), "GATES 顺序不连续"
    gate_bank_total = sum(len(g["bank"]) for g in GATES)
    assert gate_bank_total == len(ALL_DP) + len(ALL_SP), "GATES 里的bank总数与ALL_DP+ALL_SP不一致"
    dp_gate_count = sum(1 for g in GATES if g["kind"] == "dp")
    sp_gate_count = sum(1 for g in GATES if g["kind"] == "sp")
    assert dp_gate_count == 5, dp_gate_count
    assert sp_gate_count == 3, sp_gate_count

    print(
        f"[PASS] social_hire_gauntlet: 8关(5个DeepPoint关+3个ScenarioPoint关) 汇总完整,"
        f"合计{len(ALL_DP)}个DeepPoint + {len(ALL_SP)}个ScenarioPoint = {len(ALL_DP) + len(ALL_SP)}点"
    )


if __name__ == "__main__":
    _self_test()
