"""生成 research-team-operations notebooks (N1 团队运营计划自检). 跑后 nbconvert --execute."""
from __future__ import annotations
import sys
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
def md(s): return new_markdown_cell(s)
def code(s): return new_code_cell(s)

PATHS = """import sys
from pathlib import Path
SRC = Path.cwd().parent / "src"
sys.path.insert(0, str(SRC))"""

# ───────────────────────── N1 audit-team-ops ─────────────────────────
n1 = new_notebook()
n1.cells = [
    md("""# N1 · 自检一份团队运营计划 (Audit Your Team Ops Plan)

> 配套 L1-L5 · **真实科研动作**: 写一版团队运营计划五块骨架(哪怕是雏形), 用 `team_ops_audit.py`
> 自动检查漏项/敷衍项, 覆盖你现在(或未来可能)真实要运营的一个团队/合作关系, 不需要虚构材料。"""),

    code(PATHS + """
import team_ops_audit as toa
print("五块骨架:", [name for _, name, _ in toa.SECTIONS])"""),

    md("""## 1. 写一版完整的运营计划(五块都填)"""),

    code("""good = toa.blank_ops_plan()
good["time_allocation"] = "周一三五上午专属主线课题(深度块), 周二四上午留给合作组的消融项目, 下午统一处理审稿/助教等突发协作。"
good["recruiting_criteria"] = "必须: 能读懂并修改现有PyTorch训练脚本、每周保证15小时以上。可现学: 长上下文推理这个子领域的背景知识。附一个3小时的trial task: 给已有评测脚本加一个输出字段, 评分标准提前写好。"
good["async_protocol"] = "阻塞型消息24小时内响应, 非阻塞型FYI消息一周内在书面周报里统一回应, 真正紧急的情况需明确标注并几小时内响应。每周五写一份200字以内的书面进展更新, 替代原本的例行进度通报会, 每两周才约一次真正需要同步讨论的通话。"
good["onboarding_docs"] = "第一周: 精读2篇奠基性论文+一份200字项目说明, 跑通baseline.py并核对准确率≈0.72(±0.01), 附团队术语表和'谁负责什么'地图(环境问题找B, 数据问题找C)。"
good["cross_discipline_bridge"] = "和合作的工程团队共享一份双向术语对照表(ablation↔A/B测试, eval集↔测试用例集), 并明确模型架构选择归研究者拍板、上线节奏归工程负责人拍板。"

print(toa.render(good))"""),

    md("## 2. 完整性自检"),

    code("""chk = toa.audit(good)
print("✅ 骨架完整" if chk["ready"] else chk["issues"])"""),

    md("""## 3. 反面: 敷衍的运营计划(常见新手版本)"""),

    code("""bad = toa.blank_ops_plan()
bad["time_allocation"] = "有空就做。"
bad["recruiting_criteria"] = "找个厉害的人。"
for i in toa.audit(bad)["issues"]:
    print("⚠", i)
print("\\n→ 缺async协作规范+缺onboarding文档+缺跨专业桥接, 外加两句敷衍的空话,"
      " 这正是'大家自己看着办'式隐性协调的典型样子——没有一件事真正被写清楚。")"""),

    md("""## 4. 从骨架到行动: 每一节对应回哪一讲

如果`audit()`报出某一节缺失或过短, 直接对应回具体的讲义去补, 而不是凭空硬凑一句话:"""),

    code("""followup = {
    "time_allocation": "回 L1, 用时间块分配法画一张真实的每周骨架",
    "recruiting_criteria": "回 L2, 拆出必须项/可现学项并设计一个trial task",
    "async_protocol": "回 L3, 按紧急程度分层写出响应时限约定",
    "onboarding_docs": "回 L4, 明确第一周该看什么+该跑通什么的golden path",
    "cross_discipline_bridge": "回 L5, 建一张双向术语对照表+决策边界",
}
for i in toa.audit(bad)["issues"]:
    for key, name, _ in toa.SECTIONS:
        if name in i:
            print(f"「{name}」 → {followup[key]}")
            break"""),

    md("""## 5. 反思(本专题L1-L5收官)

你刚把"团队该怎么运营"这句抽象的焦虑, 变成了一份可核查、可复盘的五块骨架。带走:

- 五块骨架对应五个完全不同粒度的问题(时间分配/招募/异步协作/onboarding/跨专业桥接), 缺一块就是团队里一处隐性协调的空白, 迟早要靠某个人临场补救。
- 骨架不是写一次就一劳永逸——项目阶段会变(L1)、团队会招新人(L2/L4)、协作对象会变得更跨专业(L5), 值得定期(比如每学期)重新跑一次这份自检。
- 这份骨架和 `team-leadership-for-researchers`(9.12)的带教/反馈/会议/冲突/健康诊断是同一条"科研生涯周期"线上互补的两半: 那个专题回答"团队里的人该怎么被带、被反馈、被诊断", 本专题回答"团队本身的日常运转该怎么被结构化地安排"。

> **本专题(research-team-operations)收官**: L1时间块分配 → L2招募筛选 → L3异步协作 → L4 onboarding文档 → L5跨专业桥接, 完整走完了团队运营实务的五块日常操作基本功。"""),
]
nbformat.write(n1, HERE / "N1-audit-team-ops.ipynb")
print("written N1")
