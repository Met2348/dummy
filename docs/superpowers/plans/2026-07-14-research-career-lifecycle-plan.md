# 科研生涯与共同体扩展(9.10-9.17 + 3专题追加十讲) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为"科研技能"家族新增8个专题(9.10-9.17,覆盖求职生涯/团队管理/经费合规/学术共同体,每个5讲)+ 3个现有专题追加10讲(深化文献综述/实验统计/论文发表策略),合计50讲,全部严格复用`research-direction-proposal`已验证的房屋风格。

**Architecture:** 每个新专题是`learning/<topic>/`下的独立目录(README+lectures+templates+src+notebooks+environment+papers),讲义由subagent按统一文风模板写作;每个专题配1-2个纯stdlib工具(`blank_*()`/`audit()`/`render()`模式)。现有专题追加讲次只新增文件,不改动已有内容。

**Tech Stack:** Python 3.13 (纯stdlib工具) + nbformat/jupyter(notebook生成与执行) + Markdown讲义。

## Global Constraints

- 讲义文风严格复刻`learning/critical-reading-gap/lectures/L4-idea-generation.md`和`learning/research-direction-proposal/lectures/L1-choosing-a-research-direction.md`——不是CS-mastery教科书体。
- 每讲实质内容(不含标题行/代码块)≥4000字符。
- 每讲必须引用至少1处仓库里真实存在的其他专题作为具体例子(执行前用Glob/Grep确认该专题目录真实存在),以及至少1本真实存在的书籍/官方指南(不确定真实性宁可留空)。
- 工具:纯stdlib,`from __future__ import annotations`,`sys.stdout.reconfigure(encoding="utf-8")`兼容Windows控制台,每个工具文件末尾`if __name__ == "__main__":`跑一个好例子+一个反面例子demo。
- 新专题目录结构:`README.md` / `lectures/L1-L5.md` / `templates/*.md` / `src/*.py` / `notebooks/_gen_notebooks.py`+`N1/N2.ipynb` / `papers/README.md` / `environment/requirements.txt`+`verify_env.py`。
- 现有专题追加讲次:只新增`lectures/L{n}.md`文件+更新该专题`README.md`的学习路径表,不修改已有L1-L4/L5内容。
- 全局去重:所有新增讲义标题/`src`文件名与仓库现有专题(含`_shared`)不冲突。
- 每完成一个专题(或一组追加讲次)提交一次commit。

---

## 通用组件:讲义写作 agent 派发模板

写每一篇`lectures/L{n}-{slug}.md`时,派发一个 subagent,使用以下固定 prompt 模板(方括号内按对应任务的表格填空):

```
你要为 [专题目录路径] 写一份讲义 lectures/L[n]-[slug].md。

在写之前,请先完整阅读以下两篇参考文件,精确摸清"科研技能"专题家族的讲义文风(不是CS-mastery
系列的教科书体,不要用"引言/核心概念系统讲解/常见误区澄清/本章小结"这种结构):
- c:\Workspace\dummy\learning\critical-reading-gap\lectures\L4-idea-generation.md
- c:\Workspace\dummy\learning\research-direction-proposal\lectures\L1-choosing-a-research-direction.md

讲义必须满足:
- 标题格式 `# L[n] · [中文标题] ([English Subtitle])`
- 开头 `> XX-min lecture · 目标: ...`(必要时加第二行"关键区分")
- 正文用 `## 0.` 到 `## N.` 编号章节,0号必须是"为什么这是个真问题"式的动机引入
- 大量使用ASCII图表/对比表
- 至少引用1-2处仓库里真实存在的其他专题作为具体例子(用Glob/Grep先确认真实存在再引用,
  禁止编造不存在的专题名)
- 结尾必须有编号章节"本讲小结 + 通往 L[n+1]"(或本专题收官总结)+ "**动手**:"指向
  templates/或notebooks/里的具体文件
- 至少引用1本真实存在的书籍/官方指南(不确定真实性宁可不写,禁止编造书名/作者)
- 正文实质内容(不含标题行/代码块)不少于4000字符

本讲的具体内容范围: [content_scope]

写完后直接返回完整markdown内容,我会保存到 [file_path]。
```

验证每篇讲义:`grep -c "^## " <file>` 确认有编号章节结构;字符数检查 `python -c "print(len(open(r'<file>', encoding='utf-8').read()))"` ≥ 4000。

---

### Task 1: 专题 9.10 `career-pathways`(学术界/工业界/博后路径选择)

**Files:**
- Create: `learning/career-pathways/README.md`
- Create: `learning/career-pathways/lectures/L1-academic-job-market.md` ... `L5-early-faculty-survival.md`(5个文件)
- Create: `learning/career-pathways/src/career_path_scorer.py`
- Create: `learning/career-pathways/templates/career-path-scorecard.md`
- Create: `learning/career-pathways/notebooks/_gen_notebooks.py` + 生成 `N1-score-a-career-path.ipynb`
- Create: `learning/career-pathways/papers/README.md`
- Create: `learning/career-pathways/environment/requirements.txt` + `verify_env.py`

**Interfaces:**
- Produces: `career_path_scorer.DIMENSIONS`(dict,4个key:`skill_fit`/`entry_barrier`/`stability_growth`/`market_timing`) / `blank_path(name) -> dict` / `total(path) -> int` / `audit(path) -> dict` / `compare(paths) -> list` / `render(paths) -> str`——供 notebook N1 消费。

- [ ] **Step 1: 讲义派发表(5讲,用通用模板,替换方括号内容)**

| n | slug | content_scope |
|---|---|---|
| 1 | academic-job-market | 学术界求职全流程:cover letter/research statement/job talk邀请/校园面试各环节的目的和常见失误,和`research-presentation` L1-L3(常规会议报告)的区别在于这是"卖自己"导向 |
| 2 | industry-research-job-market | 工业界研究岗求职全流程:和`interview-prep`已覆盖的算法coding不同,这里讲research portfolio包装、怎么和hiring manager谈、case study展示、reference如何被联系 |
| 3 | academic-industry-transition | 学界⇄业界转换策略:什么信号说明该考虑转换、简历/故事怎么重新包装、常见的双向偏见与如何应对 |
| 4 | choosing-a-postdoc | 博士后阶段选择与规划:要不要读博后的决策框架、怎么选组(区别于`research-direction-proposal` L1的"选研究方向",这里是"选人和组织环境")、博后期间怎么为下一步铺路 |
| 5 | early-faculty-survival | 教职早期生存:前3年怎么起步实验室、招第一批学生、抢第一笔启动经费,与`research-funding-and-resources`(9.14)的关系是"这里讲心态和优先级,9.14讲具体经费操作" |

按上表用讲义派发模板逐篇生成(可2-3篇一批并行派发,每批完成后校验字符数与`## `标题结构),写入对应文件。

- [ ] **Step 2: 编写 `src/career_path_scorer.py`**

```python
"""
career_path_scorer.py — 职业路径候选打分工具: 把"academic还是industry还是postdoc-first"
这种开放性选择, 变成可比较的候选清单。

四个维度 (对应 L1-L5):
  skill_fit        技能匹配度 —— 现有技能栈离这条路径的入门要求差多远
  entry_barrier    入行门槛与准备度 —— 需要补多少东西(证书/作品集/人脉)才够格投递
  stability_growth 长期稳定性与成长空间 —— 5-10年后这条路径的天花板和地板
  market_timing    当前市场窗口期 —— 这条赛道现在是扩张期还是收缩期

纯 stdlib。
"""
from __future__ import annotations
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DIMENSIONS = {
    "skill_fit": ("技能匹配度", "现有技能栈离这条路径的入门要求差多远?"),
    "entry_barrier": ("入行门槛与准备度", "需要补多少东西(证书/作品集/人脉)才够格投递?"),
    "stability_growth": ("长期稳定性与成长空间", "5-10年后这条路径的天花板和地板分别在哪?"),
    "market_timing": ("当前市场窗口期", "这条赛道现在是扩张期还是收缩期?"),
}


def blank_path(name: str) -> dict:
    return {"name": name, "scores": {k: {"score": 0, "note": ""} for k in DIMENSIONS}}


def total(path: dict) -> int:
    return sum(d["score"] for d in path["scores"].values())


def audit(path: dict) -> dict:
    issues = []
    for key, (name, _) in DIMENSIONS.items():
        d = path["scores"].get(key, {})
        score = d.get("score", 0)
        note = d.get("note", "")
        if not (1 <= score <= 5):
            issues.append(f"「{name}」缺分数或分数越界(应为1-5): 当前 {score}")
        if score and not note.strip():
            issues.append(f"「{name}」打了 {score} 分却没写依据")
    return {"issues": issues, "ready": not issues}


def compare(paths: list[dict]) -> list[dict]:
    ranked = sorted(paths, key=total, reverse=True)
    for p in ranked:
        weakest_key = min(p["scores"], key=lambda k: p["scores"][k]["score"])
        p["_weakest"] = DIMENSIONS[weakest_key][0]
    return ranked


def render(paths: list[dict]) -> str:
    ranked = compare(paths)
    lines = ["=== 职业路径候选对比 ==="]
    for i, p in enumerate(ranked, 1):
        lines.append(f"\n{i}. {p['name']}  (总分 {total(p)}/20, 最弱项: {p['_weakest']})")
        for key, (name, _) in DIMENSIONS.items():
            d = p["scores"][key]
            lines.append(f"   {name}: {d['score']}分 —— {d['note'] or '(未填依据)'}")
    return "\n".join(lines)


if __name__ == "__main__":
    a = blank_path("路径A: 工业界research scientist")
    a["scores"]["skill_fit"] = {"score": 4, "note": "已有的复现/工程能力直接对口"}
    a["scores"]["entry_barrier"] = {"score": 3, "note": "还缺一篇一作顶会论文撑门面"}
    a["scores"]["stability_growth"] = {"score": 4, "note": "头部lab research scientist晋升路径清楚"}
    a["scores"]["market_timing"] = {"score": 4, "note": "前沿lab仍在扩招research岗"}

    b = blank_path("路径B: 走tenure-track学术界")
    b["scores"]["skill_fit"] = {"score": 3, "note": "教学/带组经验几乎为零"}
    b["scores"]["entry_barrier"] = {"score": 2, "note": "需要博后+多篇一作+成体系的研究agenda"}
    b["scores"]["stability_growth"] = {"score": 3, "note": "tenure后稳定但晋升周期长"}
    b["scores"]["market_timing"] = {"score": 2, "note": "faculty岗位僧多粥少, 竞争空前激烈"}

    print(render([a, b]))
    print("\n=== 打分完整性自检 ===")
    for p in [a, b]:
        chk = audit(p)
        status = "✅ 完整" if chk["ready"] else "⚠ " + "; ".join(chk["issues"])
        print(f"{p['name']}: {status}")
```

- [ ] **Step 3: 验证工具**

Run: `python learning/career-pathways/src/career_path_scorer.py`
Expected: 打印候选对比表 + "✅ 完整"自检,无异常。

- [ ] **Step 4: 编写 `templates/career-path-scorecard.md`**

一份markdown打分卡,四个小节对应`DIMENSIONS`的四个key(技能匹配度/入行门槛与准备度/长期稳定性与成长空间/当前市场窗口期),每节下面留"候选A:"/"候选B:"两行空白供手填分数+依据。

- [ ] **Step 5: 编写 `notebooks/_gen_notebooks.py` 生成 `N1-score-a-career-path.ipynb`**

严格复用`learning/research-direction-proposal/notebooks/_gen_notebooks.py`的cell结构模式(markdown说明→`import career_path_scorer as cs`→建2个真实候选路径打分→`audit`完整性自检→反面敷衍例子→`compare`排序+反思段落),把`direction_scorer`换成`career_path_scorer`,候选从"研究方向"换成"职业路径"(如"工业界research scientist" vs "学术界tenure-track"),反思段落呼应L1-L5内容。

- [ ] **Step 6: 生成并验证notebook**

Run: `cd learning/career-pathways/notebooks && python _gen_notebooks.py && jupyter nbconvert --to notebook --execute --inplace N1-score-a-career-path.ipynb`
Expected: 无报错,notebook输出包含打分对比表。

- [ ] **Step 7: 编写 `papers/README.md`**

引用真实存在的职业发展类书籍/指南,例如《The Chicago Guide to Your Career in Science》(Herman/Cohen-Fix)、Feibelman《A PhD Is Not Enough》(职业路径相关章节,和`research-life`/`research-direction-proposal`的papers/README.md分别侧重不同章节,不重复摘抄)、以及各大高校研究生院公开的博士生就业统计报告(以官网为准,不编造具体URL)。

- [ ] **Step 8: 编写 `environment/requirements.txt` 与 `verify_env.py`**

`requirements.txt`内容:
```
jupyter
nbformat
```
`verify_env.py`完全参照`learning/research-direction-proposal/environment/verify_env.py`的检查逻辑(检查Python版本、import jupyter/nbformat、检查`src/career_path_scorer.py`可import且`__main__`跑通),终端应输出"全部通过 ✅"。

- [ ] **Step 9: 编写 `README.md`**

结构完全参照`learning/research-direction-proposal/README.md`:一句话标题(`# 9.10 career-pathways — ...`)+ "这个专题要解决的真问题"对比图 + 5讲学习路径表 + notebook表 + 模板列表 + 工具列表 + 环境说明 + checklist + "在Module 9中的位置"图(标注本专题为9.10,插入在9.9`research-life`之后,属于"科研生涯周期"而非"20环节项目周期",与`research-direction-proposal`并列说明两套体系关系)。

- [ ] **Step 10: 验证与提交**

Run: `python learning/career-pathways/environment/verify_env.py`
Expected: 全部通过 ✅

```bash
git add learning/career-pathways/
git commit -m "feat(career-pathways): 新增9.10专题——学术界/工业界/博后职业路径选择(5讲+工具+notebook)"
```

---

### Task 2: 专题 9.11 `research-visibility-negotiation`(科研品牌/推荐信/谈判)

**Files:** 同Task 1结构,目录改为`learning/research-visibility-negotiation/`。

**Interfaces:**
- Produces: `application_kit_audit.SECTIONS`(5元组list)/ `blank_package() -> dict` / `audit(package) -> dict` / `render(package) -> str` / `negotiation_focus(package) -> list[str]`。

- [ ] **Step 1: 讲义派发表**

| n | slug | content_scope |
|---|---|---|
| 1 | building-research-visibility | 个人科研品牌建设:学术社群可见度经营(不是运营网红号,是让同行知道你在做什么),和9.16`academic-community-engagement`的区别是"这里讲个人输出策略,9.16讲共同体角色参与" |
| 2 | recommendation-letter-strategy | 推荐信策略:怎么请、提前给推荐人什么素材、怎么维护长期关系,避免"临时抱佛脚" |
| 3 | designing-a-job-talk | 面试中的research/job talk设计与演练:和`research-presentation` L1-L3(常规会议报告)的区别是这是"说服评委录用你"导向的15-45分钟报告 |
| 4 | negotiation-for-researchers | 谈判技巧:offer谈判、startup package、合作条件谈判的基本框架(锚定/底线/BATNA) |
| 5 | versioning-your-cv | CV/简历/学术履历的版本化管理:不同受众(学术版/工业版/grant申请版)该强调什么、怎么避免版本混乱 |

- [ ] **Step 2: 编写 `src/application_kit_audit.py`**

```python
"""
application_kit_audit.py — 求职材料包自检工具: 检查一套申请材料是否具备说服力, 而不是
简历关键词堆砌 + 千篇一律的cover letter。

五块骨架 (对应 L2-L5):
  cv_tailoring          CV/履历定制
  recommenders          推荐人网络与请求策略
  talk_outline          Job talk/面试陈述大纲
  negotiation_prep      谈判准备
  narrative_consistency 跨材料叙事一致性 (CV/cover letter/job talk讲的是不是同一个故事)

纯 stdlib。
"""
from __future__ import annotations
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SECTIONS = [
    ("cv_tailoring", "CV/履历定制", "针对具体岗位调整了侧重, 而不是海投同一份CV"),
    ("recommenders", "推荐人网络与请求策略", "提前沟通、给推荐人素材包, 而不是临时才开口"),
    ("talk_outline", "Job talk/面试陈述大纲", "有清晰的一条研究主线, 而不是罗列做过的项目"),
    ("negotiation_prep", "谈判准备", "明确底线/优先级(薪资/起始经费/团队规模), 而不是被动接受首次offer"),
    ("narrative_consistency", "跨材料叙事一致性", "CV/cover letter/job talk讲的是同一个故事"),
]

_VAGUE_HINTS = ["综合能力强", "学习能力强", "沟通能力好"]


def blank_package() -> dict:
    return {key: "" for key, _, _ in SECTIONS}


def audit(package: dict) -> dict:
    issues = []
    for key, name, hint in SECTIONS:
        content = package.get(key, "").strip()
        if not content:
            issues.append(f"缺「{name}」这一节 —— {hint}")
            continue
        if any(h in content for h in _VAGUE_HINTS):
            issues.append(f"「{name}」用了空泛套话(如'综合能力强') —— 换成具体可验证的事实")
    return {"issues": issues, "ready": not issues,
            "weak_sections": [i.split("「")[1].split("」")[0] for i in issues if "「" in i]}


def render(package: dict) -> str:
    lines = ["=== 求职材料包骨架 ==="]
    for key, name, hint in SECTIONS:
        content = package.get(key, "")
        lines.append(f"\n## {name}")
        lines.append(content if content else f"  (空 —— {hint})")
    return "\n".join(lines)


def negotiation_focus(package: dict) -> list[str]:
    chk = audit(package)
    prompts = {
        "CV/履历定制": "针对目标岗位重新排列CV的重点顺序了吗?",
        "推荐人网络与请求策略": "给推荐人发过你的CV+这份岗位的具体侧重了吗?",
        "Job talk/面试陈述大纲": "如果只能讲一条研究主线, 是哪一条?",
        "谈判准备": "薪资/起始经费/团队规模, 你的底线分别是多少?",
        "跨材料叙事一致性": "CV和job talk里的故事是否会让人觉得是两个人?",
    }
    return [prompts.get(w, f"「{w}」这一节请补完") for w in chk["weak_sections"]]


if __name__ == "__main__":
    good = blank_package()
    good["cv_tailoring"] = "针对XX Lab的招聘方向, 把interp相关项目排在前面, 弱化早期工程复现经历。"
    good["recommenders"] = "提前2周联系导师+合作者, 附上目标岗位描述和自己想强调的3个点。"
    good["talk_outline"] = "主线: 从机制可解释性方法论到大规模验证, 一条技术演进链。"
    good["negotiation_prep"] = "底线: 起薪不低于X, 优先要独立预算而非团队规模。"
    good["narrative_consistency"] = "三份材料统一用'从机制理解到工程验证'这条主线, 已交叉核对。"
    print(render(good))
    chk = audit(good)
    print("\n完整性自检:", "✅ 骨架完整" if chk["ready"] else chk["issues"])

    print("\n--- 反面: 敷衍的材料包 ---")
    bad = blank_package()
    bad["cv_tailoring"] = "综合能力强, 适合任何岗位。"
    for i in audit(bad)["issues"]:
        print("⚠", i)
    print("\n--- 谈判/面试前准备清单 ---")
    for q in negotiation_focus(bad):
        print("?", q)
```

- [ ] **Step 3: 验证工具**

Run: `python learning/research-visibility-negotiation/src/application_kit_audit.py`
Expected: 打印骨架+完整性自检+反面例子警告+谈判准备清单,无异常。

- [ ] **Step 4-9:** 同Task 1的Step 4-9模式(templates对应5个SECTIONS的填空worksheet;notebook `N1-audit-application-kit.ipynb`复用`research-direction-proposal` N2(`proposal_audit`用法)的cell结构,把`proposal_audit`换成`application_kit_audit`;papers/README.md引用求职/谈判类真实参考,如《Never Split the Difference》(Voss,谈判)、《The Professor Is In》(Kelsky,学术求职);environment同Step 8模板;README同Step 9模板,标注9.11)。

- [ ] **Step 10: 验证与提交**

```bash
python learning/research-visibility-negotiation/environment/verify_env.py
git add learning/research-visibility-negotiation/
git commit -m "feat(research-visibility-negotiation): 新增9.11专题——科研品牌/推荐信/job talk/谈判(5讲+工具+notebook)"
```

---

### Task 3: 专题 9.12 `team-leadership-for-researchers`(带教与向下管理)

**Files:** 目录`learning/team-leadership-for-researchers/`,含**2个工具**:`src/feedback_quality.py` + `src/team_health_check.py`。

**Interfaces:**
- Produces: `feedback_quality.DIMENSIONS` / `blank_feedback(mentee) -> dict` / `audit(feedback) -> dict` / `render(feedback) -> str`
- Produces: `team_health_check.SIGNALS` / `blank_checkup() -> dict` / `diagnose(checkup) -> dict` / `render(checkup) -> str`

- [ ] **Step 1: 讲义派发表**

| n | slug | content_scope |
|---|---|---|
| 1 | mentoring-junior-researchers | 指导低年级学生/实习生:怎么分配任务难度、怎么做code review给新手看、避免自己变成瓶颈(事必躬亲) |
| 2 | giving-downward-feedback | 向下管理与建设性反馈:和`research-life` L2"向上管理"(对导师)相对,这里是对junior的反馈技巧,强调具体/可执行/时效性 |
| 3 | running-effective-meetings | 高效组会/brainstorm会主持:议程设计、避免"一言堂"、怎么让安静的成员发言 |
| 4 | handling-team-conflict | 团队协作冲突处理:authorship纠纷、功劳分配分歧的公开化处理原则 |
| 5 | diagnosing-team-health | 团队健康度诊断:识别功能失调的早期信号(心理安全感/工作量失衡/冲突被回避/成长不可见) |

- [ ] **Step 2: 编写 `src/feedback_quality.py`**

```python
"""
feedback_quality.py — 向下反馈质量自检: 给junior研究者的反馈是否具体、可执行, 而不是
"继续努力"式的空话。

四个维度 (对应 L2):
  specificity   具体行为而非泛泛评价
  actionability 给出可执行的下一步
  balance       认可与改进建议的平衡
  timing        问题发生后多久给出反馈

纯 stdlib。
"""
from __future__ import annotations
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DIMENSIONS = {
    "specificity": ("具体行为而非泛泛评价", "是否点名了具体的代码/实验/会议发言, 而不是'做得不错'"),
    "actionability": ("给出可执行的下一步", "对方听完知道下一步具体做什么"),
    "balance": ("认可与改进建议的平衡", "只有批评或只有表扬都会失真"),
    "timing": ("时效性", "问题发生后多久给出反馈, 拖太久对方记不清上下文"),
}


def blank_feedback(mentee: str) -> dict:
    return {"mentee": mentee, "scores": {k: {"score": 0, "note": ""} for k in DIMENSIONS}}


def audit(feedback: dict) -> dict:
    issues = []
    for key, (name, _) in DIMENSIONS.items():
        d = feedback["scores"].get(key, {})
        score = d.get("score", 0)
        note = d.get("note", "")
        if not (1 <= score <= 5):
            issues.append(f"「{name}」缺分数或越界: 当前 {score}")
        if score and not note.strip():
            issues.append(f"「{name}」打了{score}分却没写依据")
    return {"issues": issues, "ready": not issues}


def render(feedback: dict) -> str:
    lines = [f"=== 给 {feedback['mentee']} 的反馈质量自检 ==="]
    for key, (name, _) in DIMENSIONS.items():
        d = feedback["scores"][key]
        lines.append(f"{name}: {d['score']}分 —— {d['note'] or '(未填依据)'}")
    return "\n".join(lines)


if __name__ == "__main__":
    fb = blank_feedback("师弟A")
    fb["scores"]["specificity"] = {"score": 2, "note": "只说了'实验部分再仔细点', 没点名具体哪个脚本"}
    fb["scores"]["actionability"] = {"score": 2, "note": "没说清'仔细点'具体指做什么动作"}
    fb["scores"]["balance"] = {"score": 3, "note": "全是改进意见, 没提认可的部分"}
    fb["scores"]["timing"] = {"score": 4, "note": "当天代码review时就给了"}
    print(render(fb))
    chk = audit(fb)
    print("\n" + ("✅ 反馈质量合格" if chk["ready"] else "⚠ " + "; ".join(chk["issues"])))
    print("\n→ specificity和actionability都偏低: '仔细点'不是反馈, 是叹气。"
          " 改成'第47行的seed没有固定, 三次跑出的结果对不上, 建议这周把所有实验脚本的"
          "seed显式写进config' 才是可执行的反馈。")
```

- [ ] **Step 3: 编写 `src/team_health_check.py`**

```python
"""
team_health_check.py — 团队健康度诊断: 识别团队功能失调的早期信号, 而不是等到有人
离职才发现问题。

四个信号 (对应 L5):
  psychological_safety 心理安全感 —— 成员敢不敢说"我不知道"/"我错了"
  workload_balance      工作量分配均衡
  conflict_visibility   冲突是否被公开处理还是被回避/憋着
  growth_visibility     成员能否看到自己的成长路径

纯 stdlib。
"""
from __future__ import annotations
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SIGNALS = [
    ("psychological_safety", "心理安全感", "成员敢在组会上说'我不知道'或承认失败吗?"),
    ("workload_balance", "工作量分配均衡", "是否有人长期超负荷而有人明显清闲?"),
    ("conflict_visibility", "冲突是否被公开处理", "上次意见分歧是摊开谈了还是不了了之?"),
    ("growth_visibility", "成长路径可见度", "成员知道自己半年后会成长成什么样子吗?"),
]


def blank_checkup() -> dict:
    return {key: {"score": 0, "note": ""} for key, _, _ in SIGNALS}


def diagnose(checkup: dict) -> dict:
    risks = []
    for key, name, hint in SIGNALS:
        d = checkup.get(key, {})
        score = d.get("score", 0)
        if score and score <= 2:
            risks.append(f"「{name}」偏低({score}分) —— {hint}")
    return {"risks": risks, "healthy": not risks}


def render(checkup: dict) -> str:
    lines = ["=== 团队健康度诊断 ==="]
    for key, name, hint in SIGNALS:
        d = checkup.get(key, {"score": 0, "note": ""})
        lines.append(f"{name}: {d['score']}分 —— {d['note'] or '(未填)'}")
    diag = diagnose(checkup)
    lines.append("\n" + ("✅ 暂无明显风险信号" if diag["healthy"]
                          else "⚠ 风险信号:\n  " + "\n  ".join(diag["risks"])))
    return "\n".join(lines)


if __name__ == "__main__":
    c = blank_checkup()
    c["psychological_safety"] = {"score": 2, "note": "组会上很少有人主动说自己实验失败了"}
    c["workload_balance"] = {"score": 4, "note": "目前3个人任务量分布均衡"}
    c["conflict_visibility"] = {"score": 2, "note": "上次authorship意见不合, 私下嘀咕但没摊开谈"}
    c["growth_visibility"] = {"score": 3, "note": "有semi-annual review但反馈比较笼统"}
    print(render(c))
```

- [ ] **Step 4: 验证两个工具**

Run: `python learning/team-leadership-for-researchers/src/feedback_quality.py && python learning/team-leadership-for-researchers/src/team_health_check.py`
Expected: 两者均无异常输出诊断结果。

- [ ] **Step 5-9:** 同Task 1模式(templates 2份分别对应两个工具;notebook `N1-give-feedback.ipynb`(用`feedback_quality`)+`N2-diagnose-team.ipynb`(用`team_health_check`);papers/README.md引用团队管理类真实参考,如《The Manager's Path》(Fournier,虽面向工程管理但带教/反馈章节可迁移用于研究团队,需在papers/README.md里明确注明"工程管理书籍,概念迁移到科研团队场景")、Google re:Work公开的团队效能研究(Project Aristotle);environment/README同标准模板,标注9.12)。

- [ ] **Step 10: 验证与提交**

```bash
python learning/team-leadership-for-researchers/environment/verify_env.py
git add learning/team-leadership-for-researchers/
git commit -m "feat(team-leadership-for-researchers): 新增9.12专题——带教/向下反馈/团队健康度(5讲+2工具+2notebook)"
```

---

### Task 4: 专题 9.13 `research-team-operations`(团队运营实务)

**Files:** 目录`learning/research-team-operations/`。

**Interfaces:**
- Produces: `team_ops_audit.SECTIONS` / `blank_ops_plan() -> dict` / `audit(plan) -> dict` / `render(plan) -> str`

- [ ] **Step 1: 讲义派发表**

| n | slug | content_scope |
|---|---|---|
| 1 | multi-project-time-management | 多项目并行时间管理与优先级排序:时间块分配法、避免"救火式"被动切换 |
| 2 | recruiting-collaborators | 招募筛选合作者/实习生:标准怎么写(必须项vs可现学项)、面试/试做任务设计 |
| 3 | async-remote-collaboration | 远程/异步跨时区科研团队协作:响应时限约定、文档同步替代部分会议 |
| 4 | onboarding-new-members | 团队知识传承与onboarding文档建设:新人第一周该看什么、该跑通什么 |
| 5 | cross-disciplinary-teamwork | 跨专业背景团队协作:和工程师/PM/设计师混合团队协作的术语对齐与协作边界(工业界研究团队常见场景) |

- [ ] **Step 2: 编写 `src/team_ops_audit.py`**

```python
"""
team_ops_audit.py — 团队运营计划自检: 检查一份多人协作计划是否具备可执行的运营骨架,
而不是"大家自己看着办"式的隐性协调。

五块骨架 (对应 L1-L5):
  time_allocation         多项目并行时间分配计划
  recruiting_criteria     合作者/实习生招募标准
  async_protocol          远程/异步协作规范
  onboarding_docs         新人onboarding文档计划
  cross_discipline_bridge 跨专业背景沟通桥接计划

纯 stdlib。
"""
from __future__ import annotations
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SECTIONS = [
    ("time_allocation", "多项目并行时间分配", "每周固定时间块分给哪个项目, 而不是'有空就做'"),
    ("recruiting_criteria", "合作者/实习生招募标准", "写清楚必须具备的能力和可以现学的能力, 不是'越强越好'"),
    ("async_protocol", "远程/异步协作规范", "响应时限/会议时区/文档同步方式写清楚"),
    ("onboarding_docs", "新人onboarding文档", "新人第一周能不看你就跑起来baseline"),
    ("cross_discipline_bridge", "跨专业沟通桥接", "和非ML背景的合作者(工程师/PM)有没有共享术语表"),
]


def blank_ops_plan() -> dict:
    return {key: "" for key, _, _ in SECTIONS}


def audit(plan: dict) -> dict:
    issues = []
    for key, name, hint in SECTIONS:
        content = plan.get(key, "").strip()
        if not content:
            issues.append(f"缺「{name}」这一节 —— {hint}")
        elif len(content) < 15:
            issues.append(f"「{name}」内容过短, 疑似敷衍")
    return {"issues": issues, "ready": not issues}


def render(plan: dict) -> str:
    lines = ["=== 团队运营计划骨架 ==="]
    for key, name, hint in SECTIONS:
        content = plan.get(key, "")
        lines.append(f"\n## {name}")
        lines.append(content if content else f"  (空 —— {hint})")
    return "\n".join(lines)


if __name__ == "__main__":
    good = blank_ops_plan()
    good["time_allocation"] = "周一三五上午专属项目A, 周二四上午项目B, 下午统一留给突发协作。"
    good["recruiting_criteria"] = "必须: 会读PyTorch代码。可现学: 具体某个子领域背景。"
    good["async_protocol"] = "24小时内响应non-urgent消息, 每周一份async周报替代部分会议。"
    good["onboarding_docs"] = "首周: 环境搭建doc + 跑通一个baseline脚本 + 结对review一次PR。"
    good["cross_discipline_bridge"] = "和工程团队共享一份术语对照表(如'ablation'对应他们的'A/B测试')。"
    print(render(good))
    chk = audit(good)
    print("\n" + ("✅ 骨架完整" if chk["ready"] else "⚠ " + "; ".join(chk["issues"])))
```

- [ ] **Step 3: 验证工具**

Run: `python learning/research-team-operations/src/team_ops_audit.py`
Expected: 打印骨架+"✅ 骨架完整"。

- [ ] **Step 4-8:** 同Task 1模式(templates对应5个SECTIONS;notebook `N1-audit-team-ops.ipynb`;papers/README.md引用团队运营/远程协作类真实参考,如GitLab公开的《The Remote Playbook》、《The Manager's Path》相关章节(需注明"工程团队管理书籍,概念迁移");environment/README标注9.13)。

- [ ] **Step 9: 验证与提交**

```bash
python learning/research-team-operations/environment/verify_env.py
git add learning/research-team-operations/
git commit -m "feat(research-team-operations): 新增9.13专题——团队运营实务(5讲+工具+notebook)"
```

---

### Task 5: 专题 9.14 `research-funding-and-resources`(经费与资源规划)

**Files:** 目录`learning/research-funding-and-resources/`。

**Interfaces:**
- Produces: `funding_plan_audit.SECTIONS` / `blank_funding_plan() -> dict` / `audit(plan) -> dict` / `render(plan) -> str` / `reviewer_focus(plan) -> list[str]`

- [ ] **Step 1: 讲义派发表**

| n | slug | content_scope |
|---|---|---|
| 1 | grant-application-lifecycle | 经费申请全流程:proposal撰写→预算论证→执行→结题报告,和`research-direction-proposal` L3(开题报告写作)的区别是这里聚焦"钱怎么申请和管理",不是研究内容本身 |
| 2 | compute-resource-planning | 算力资源规划与申请:集群quota申请流程、云计算预算估算、GPU资源battle的谈判技巧 |
| 3 | data-management-plans | 数据管理规划(DMP):FAIR原则、存储/共享/留存策略,很多资助机构强制要求 |
| 4 | multi-institution-collaboration | 大型多机构合作项目管理:责任划分、资源分摊、多方署名协议 |
| 5 | vendor-and-api-compliance | 供应商/API/第三方合规评估:数据处理协议、模型使用许可核查(工业界研究常见场景) |

- [ ] **Step 2: 编写 `src/funding_plan_audit.py`**

```python
"""
funding_plan_audit.py — 经费与资源计划自检: 检查一份经费/资源申请是否具备评审会追问的
几块内容, 而不是只有一句"需要更多算力"。

五块骨架 (对应 L1-L5):
  budget_justification 预算论证 (每一项花费为什么必要)
  compute_plan          算力资源规划 (需要多少/什么时候用/备选方案)
  data_management       数据管理规划 (DMP, 存储/共享/留存策略)
  collaboration_mou     多机构合作协议 (谁出资源/谁担责任/怎么分署名)
  vendor_compliance     第三方/供应商合规审查 (数据处理协议/使用许可)

纯 stdlib。
"""
from __future__ import annotations
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SECTIONS = [
    ("budget_justification", "预算论证", "每一项花费都对应具体产出, 而不是笼统的'研究经费'"),
    ("compute_plan", "算力资源规划", "写清楚需要多少GPU-hours/什么时候用/申请不到时的备选方案"),
    ("data_management", "数据管理规划", "数据存哪/谁能访问/论文发表后是否公开, 全部写清楚"),
    ("collaboration_mou", "多机构合作协议", "谁出资源/谁担责任/怎么分署名, 提前写清楚而不是出结果后再吵"),
    ("vendor_compliance", "第三方/供应商合规审查", "用到的API/数据集的使用许可是否核实过"),
]


def blank_funding_plan() -> dict:
    return {key: "" for key, _, _ in SECTIONS}


def audit(plan: dict) -> dict:
    issues = []
    for key, name, hint in SECTIONS:
        content = plan.get(key, "").strip()
        if not content:
            issues.append(f"缺「{name}」这一节 —— {hint}")
    return {"issues": issues, "ready": not issues,
            "weak_sections": [i.split("「")[1].split("」")[0] for i in issues if "「" in i]}


def render(plan: dict) -> str:
    lines = ["=== 经费与资源计划骨架 ==="]
    for key, name, hint in SECTIONS:
        content = plan.get(key, "")
        lines.append(f"\n## {name}")
        lines.append(content if content else f"  (空 —— {hint})")
    return "\n".join(lines)


def reviewer_focus(plan: dict) -> list[str]:
    chk = audit(plan)
    prompts = {
        "预算论证": "这一项花费如果砍掉一半, 研究计划还能执行吗?",
        "算力资源规划": "申请的算力被砍到一半, B计划是什么?",
        "数据管理规划": "论文发表后数据/代码是否公开? 隐私数据怎么处理?",
        "多机构合作协议": "如果合作方中途退出, 责任怎么划分?",
        "第三方/供应商合规审查": "这些数据/API的使用许可覆盖你打算发表的用途吗?",
    }
    return [prompts.get(w, f"「{w}」请准备被追问") for w in chk["weak_sections"]]


if __name__ == "__main__":
    good = blank_funding_plan()
    good["budget_justification"] = "80% 预算用于GPU-hours(对应3个大规模实验), 20%用于会议差旅。"
    good["compute_plan"] = "需要8×A100持续2个月; 若砍半, 优先保留可解释性实验, 砍训练规模。"
    good["data_management"] = "训练数据存内部集群, 论文发表后仅公开代码和评测脚本, 不公开原始训练数据。"
    good["collaboration_mou"] = "对方实验室出算力, 我方出算法设计, 通讯作者由数据主要贡献方担任, 已书面确认。"
    good["vendor_compliance"] = "核实过所用API的商用条款, 确认允许用于论文发表和非商业研究。"
    print(render(good))
    chk = audit(good)
    print("\n" + ("✅ 骨架完整" if chk["ready"] else "⚠ " + "; ".join(chk["issues"])))
```

- [ ] **Step 3: 验证工具**

Run: `python learning/research-funding-and-resources/src/funding_plan_audit.py`
Expected: 打印骨架+"✅ 骨架完整"。

- [ ] **Step 4-8:** 同Task 1模式(templates对应5个SECTIONS;notebook `N1-audit-funding-plan.ipynb`(含`reviewer_focus`调用演示);papers/README.md引用NSF/NIH proposal撰写公开指南、DMPTool(dmptool.org,真实存在的DMP撰写辅助工具网站)相关说明;environment/README标注9.14)。

- [ ] **Step 9: 验证与提交**

```bash
python learning/research-funding-and-resources/environment/verify_env.py
git add learning/research-funding-and-resources/
git commit -m "feat(research-funding-and-resources): 新增9.14专题——经费与资源规划(5讲+工具+notebook)"
```

---

### Task 6: 专题 9.15 `research-integrity-and-compliance`(科研诚信与合规深水)

**Files:** 目录`learning/research-integrity-and-compliance/`。

**Interfaces:**
- Produces: `compliance_checklist.SECTIONS` / `blank_compliance() -> dict` / `audit(project) -> dict` / `render(project) -> str`

- [ ] **Step 1: 讲义派发表**

| n | slug | content_scope |
|---|---|---|
| 1 | research-misconduct-deep-dive | 学术诚信深水:不端调查流程、authorship纠纷仲裁机制、图像/数据造假识别,和`research-life` L3(署名规则+伦理红线)的区别是这里讲"红线被踩之后走什么流程",L3讲"怎么不踩红线" |
| 2 | irb-and-ethics-review | IRB/伦理审查全流程:涉及人类被试/敏感数据的项目怎么申请审查、常见驳回原因 |
| 3 | ip-and-tech-transfer | 知识产权与成果转化:专利disclosure流程、tech transfer办公室的角色、创业spin-off的基本路径 |
| 4 | international-collaboration-compliance | 国际合作合规:出口管制(export control)基础概念、跨境数据传输限制 |
| 5 | responsible-disclosure | 安全与负责任披露:AI安全/red-teaming相关研究发布前的coordinated disclosure norm,呼应`red-team-jailbreak`专题但聚焦"披露流程"而非"攻击技术本身" |

- [ ] **Step 2: 编写 `src/compliance_checklist.py`**

```python
"""
compliance_checklist.py — 科研合规自检: 检查一个项目在诚信/伦理/合规上是否有明显漏洞,
这些漏洞往往不是能力问题, 而是"没人提醒就压根没想到要处理"的盲区。

五块骨架 (对应 L1-L5):
  authorship_agreement 署名协议 (谁一作/贡献声明是否书面化)
  irb_status            IRB/伦理审查状态 (涉及人类被试/敏感数据是否走过审查)
  ip_disclosure         知识产权披露 (涉及专利/商业敏感内容是否提前披露)
  export_control        出口管制/跨境合规 (国际合作涉及的技术/数据跨境限制)
  disclosure_plan       负责任的风险披露计划 (发现安全漏洞/危险能力后的披露流程)

纯 stdlib。
"""
from __future__ import annotations
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SECTIONS = [
    ("authorship_agreement", "署名协议", "谁一作/谁通讯/贡献比例是否书面确认过, 而非默认约定俗成"),
    ("irb_status", "IRB/伦理审查状态", "涉及人类被试/敏感数据的项目是否已过伦理审查"),
    ("ip_disclosure", "知识产权披露", "涉及潜在专利/商业敏感方法, 是否已按机构要求提前披露"),
    ("export_control", "出口管制/跨境合规", "国际合作中涉及的技术/数据是否核实过出口管制限制"),
    ("disclosure_plan", "负责任的风险披露计划", "如果研究涉及安全漏洞/危险能力, 披露流程和时间线是否提前规划"),
]


def blank_compliance() -> dict:
    return {key: "" for key, _, _ in SECTIONS}


def audit(project: dict) -> dict:
    issues = []
    for key, name, hint in SECTIONS:
        content = project.get(key, "").strip()
        if not content:
            issues.append(f"缺「{name}」这一节 —— {hint}")
    return {"issues": issues, "ready": not issues,
            "risk_flags": [i.split("「")[1].split("」")[0] for i in issues if "「" in i]}


def render(project: dict) -> str:
    lines = ["=== 科研合规自检 ==="]
    for key, name, hint in SECTIONS:
        content = project.get(key, "")
        lines.append(f"\n## {name}")
        lines.append(content if content else f"  (空 —— {hint})")
    return "\n".join(lines)


if __name__ == "__main__":
    good = blank_compliance()
    good["authorship_agreement"] = "项目启动会上书面确认: A一作(实现+实验), B通讯(idea+资源), 已发邮件存档。"
    good["irb_status"] = "本项目不涉及人类被试或敏感个人数据, 已在项目文档标注'不适用'并说明理由。"
    good["ip_disclosure"] = "方法涉及的核心算法已按学校要求提交disclosure表, 等待技术转移办公室反馈。"
    good["export_control"] = "国际合作方来自受限清单外地区, 已核实无出口管制限制, 存档确认邮件。"
    good["disclosure_plan"] = "若发现模型存在可被滥用的能力, 先内部上报安全团队, 90天coordinated disclosure后再公开细节。"
    print(render(good))
    chk = audit(good)
    print("\n" + ("✅ 骨架完整" if chk["ready"] else "⚠ 风险: " + "; ".join(chk["issues"])))

    print("\n--- 反面: 完全没考虑合规的项目 ---")
    bad = blank_compliance()
    bad["authorship_agreement"] = "大家心照不宣。"
    for i in audit(bad)["issues"]:
        print("⚠", i)
```

- [ ] **Step 3: 验证工具**

Run: `python learning/research-integrity-and-compliance/src/compliance_checklist.py`
Expected: 打印骨架+"✅ 骨架完整"+反面例子警告。

- [ ] **Step 4-8:** 同Task 1模式(templates对应5个SECTIONS;notebook `N1-audit-compliance.ipynb`;papers/README.md引用COPE(Committee on Publication Ethics)公开指南、美国研究诚信办公室(ORI)公开的misconduct案例分析、各校IRB官网流程说明;environment/README标注9.15)。

- [ ] **Step 9: 验证与提交**

```bash
python learning/research-integrity-and-compliance/environment/verify_env.py
git add learning/research-integrity-and-compliance/
git commit -m "feat(research-integrity-and-compliance): 新增9.15专题——科研诚信与合规深水(5讲+工具+notebook)"
```

---

### Task 7: 专题 9.16 `academic-community-engagement`(学术共同体参与)

**Files:** 目录`learning/academic-community-engagement/`。

**Interfaces:**
- Produces: `engagement_scorer.DIMENSIONS` / `blank_engagement(name) -> dict` / `total(engagement) -> int` / `audit(engagement) -> dict` / `compare(engagements) -> list` / `render(engagements) -> str`

- [ ] **Step 1: 讲义派发表**

| n | slug | content_scope |
|---|---|---|
| 1 | academic-service-beyond-reviewing | 审稿之外的学术服务:program committee/area chair的角色和时间投入,和`research-life` L1(建设性审稿)的区别是这里讲"要不要接、接了之后的角色定位",L1讲"接了之后怎么审好" |
| 2 | organizing-workshops | 组织workshop/研讨会:从提案到执行的基本流程、常见的组织失误 |
| 3 | conference-networking | 学术会议社交与人脉网络建设:怎么主动搭话、poster session社交技巧、避免"社恐式陪跑" |
| 4 | building-long-term-collaborations | 建立长期合作关系:怎么把一次会议偶遇变成持续多年的合作 |
| 5 | entering-review-networks | 跨机构review/评审网络建设:怎么进入某个细分领域的评审圈子、被邀请审稿的正反馈循环 |

- [ ] **Step 2: 编写 `src/engagement_scorer.py`**

```python
"""
engagement_scorer.py — 学术共同体参与邀约打分: 审稿邀请/PC职位/workshop组织邀约,
不是来者不拒, 也不是全部推掉, 而是按四个维度系统比较该不该接。

四个维度 (对应 L1-L5):
  time_cost       时间成本 —— 这项service大概占用多少小时
  visibility_gain 可见度/影响力增益 —— 对你在这个领域的存在感有多大帮助
  network_value   人脉/合作价值 —— 会不会认识潜在合作者/推荐人
  reciprocity     对共同体的回馈价值 —— 你是不是也在"占便宜不还"

纯 stdlib。
"""
from __future__ import annotations
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DIMENSIONS = {
    "time_cost": ("时间成本", "这项service大概占用多少小时? 值不值当前阶段投入?"),
    "visibility_gain": ("可见度/影响力增益", "接了之后, 领域内对你的存在感会有实质提升吗?"),
    "network_value": ("人脉/合作价值", "会不会因此认识潜在合作者/推荐人?"),
    "reciprocity": ("对共同体的回馈价值", "你是不是也一直在'索取审稿意见却从不审别人'?"),
}


def blank_engagement(name: str) -> dict:
    return {"name": name, "scores": {k: {"score": 0, "note": ""} for k in DIMENSIONS}}


def total(engagement: dict) -> int:
    return sum(d["score"] for d in engagement["scores"].values())


def audit(engagement: dict) -> dict:
    issues = []
    for key, (name, _) in DIMENSIONS.items():
        d = engagement["scores"].get(key, {})
        score = d.get("score", 0)
        note = d.get("note", "")
        if not (1 <= score <= 5):
            issues.append(f"「{name}」缺分数或越界: 当前 {score}")
        if score and not note.strip():
            issues.append(f"「{name}」打了{score}分却没写依据")
    return {"issues": issues, "ready": not issues}


def compare(engagements: list[dict]) -> list[dict]:
    ranked = sorted(engagements, key=total, reverse=True)
    for e in ranked:
        weakest_key = min(e["scores"], key=lambda k: e["scores"][k]["score"])
        e["_weakest"] = DIMENSIONS[weakest_key][0]
    return ranked


def render(engagements: list[dict]) -> str:
    ranked = compare(engagements)
    lines = ["=== 学术共同体参与邀约对比 ==="]
    for i, e in enumerate(ranked, 1):
        lines.append(f"\n{i}. {e['name']}  (总分 {total(e)}/20, 最弱项: {e['_weakest']})")
        for key, (name, _) in DIMENSIONS.items():
            d = e["scores"][key]
            lines.append(f"   {name}: {d['score']}分 —— {d['note'] or '(未填依据)'}")
    return "\n".join(lines)


if __name__ == "__main__":
    a = blank_engagement("邀约A: 顶会workshop program committee")
    a["scores"]["time_cost"] = {"score": 3, "note": "预计需审6-8篇, 约15小时"}
    a["scores"]["visibility_gain"] = {"score": 4, "note": "PC名单会公开挂在workshop官网"}
    a["scores"]["network_value"] = {"score": 4, "note": "PC群里都是这个细分领域的活跃研究者"}
    a["scores"]["reciprocity"] = {"score": 5, "note": "自己至今没审过一次稿, 欠了共同体不少"}

    b = blank_engagement("邀约B: 不知名期刊审稿(单篇)")
    b["scores"]["time_cost"] = {"score": 4, "note": "单篇预计3小时"}
    b["scores"]["visibility_gain"] = {"score": 1, "note": "期刊领域内认可度低"}
    b["scores"]["network_value"] = {"score": 1, "note": "匿名审稿, 认识不到任何人"}
    b["scores"]["reciprocity"] = {"score": 5, "note": "同样是在回馈共同体"}

    print(render([a, b]))
```

- [ ] **Step 3: 验证工具**

Run: `python learning/academic-community-engagement/src/engagement_scorer.py`
Expected: 打印邀约对比表。

- [ ] **Step 4-8:** 同Task 1模式(templates对应4个DIMENSIONS;notebook `N1-score-an-engagement.ipynb`(复用`career_path_scorer`的notebook cell结构模式);papers/README.md引用学术共同体建设类真实参考,如各大顶会官网公开的PC/AC招募说明、Katherine Hayles或学术生涯类博客(以官方可查为准,不确定则只引用会议官网流程说明);environment/README标注9.16)。

- [ ] **Step 9: 验证与提交**

```bash
python learning/academic-community-engagement/environment/verify_env.py
git add learning/academic-community-engagement/
git commit -m "feat(academic-community-engagement): 新增9.16专题——学术共同体参与(5讲+工具+notebook)"
```

---

### Task 8: 专题 9.17 `open-science-and-communication`(开放科学与科学传播)

**Files:** 目录`learning/open-science-and-communication/`。

**Interfaces:**
- Produces: `open_science_audit.SECTIONS` / `blank_release_plan() -> dict` / `audit(plan) -> dict` / `render(plan) -> str`

- [ ] **Step 1: 讲义派发表**

| n | slug | content_scope |
|---|---|---|
| 1 | interdisciplinary_collaboration | 跨学科合作方法论:不同学科的默认假设/术语/评审标准差异,怎么建立共享词汇表 |
| 2 | science-communication | 科学传播与公众沟通:科普写作、media interview应对、避免过度简化导致的误读 |
| 3 | open-science-practices | 开放科学实践:预注册(preregistration)/registered report、开源发布规范 |
| 4 | organizing-competitions | 竞赛/challenge的组织与参与策略:Kaggle-style/NeurIPS竞赛track的设计与参赛策略 |
| 5 | academic-social-media-boundaries | 学术社交媒体的边界与风险管理:公开发言哪些代表个人、哪些代表机构,职业风险规避 |

- [ ] **Step 2: 编写 `src/open_science_audit.py`**

```python
"""
open_science_audit.py — 开放科学实践自检: 检查一个项目是否具备可复现/可核验的开放
科学实践, 而不是发表后"代码以后会整理"的空头支票。

五块骨架 (对应 L1-L5):
  interdisciplinary_glossary 跨学科术语对照表 (和其他学科合作者对齐关键概念)
  public_communication        公众沟通材料 (面向非专业读者的准确摘要)
  preregistration             预注册计划 (提前公开假设和分析计划, 防止事后诸葛亮)
  artifact_release_plan       代码/数据发布规范 (发表时同步公开, 而非"以后")
  social_media_boundary       学术社交媒体边界 (公开发言哪些代表个人、哪些代表机构)

纯 stdlib。
"""
from __future__ import annotations
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SECTIONS = [
    ("interdisciplinary_glossary", "跨学科术语对照表", "和其他学科合作者是否对齐了关键概念的定义"),
    ("public_communication", "公众沟通材料", "有没有一份面向非专业读者、准确不夸大的摘要"),
    ("preregistration", "预注册计划", "假设和分析计划是否在跑实验前就已公开记录"),
    ("artifact_release_plan", "代码/数据发布规范", "发表时是否同步公开代码/数据, 而不是'以后整理'"),
    ("social_media_boundary", "学术社交媒体边界", "公开发言哪些代表个人观点、哪些代表机构, 是否写清楚过"),
]


def blank_release_plan() -> dict:
    return {key: "" for key, _, _ in SECTIONS}


def audit(plan: dict) -> dict:
    issues = []
    for key, name, hint in SECTIONS:
        content = plan.get(key, "").strip()
        if not content:
            issues.append(f"缺「{name}」这一节 —— {hint}")
    return {"issues": issues, "ready": not issues}


def render(plan: dict) -> str:
    lines = ["=== 开放科学实践自检 ==="]
    for key, name, hint in SECTIONS:
        content = plan.get(key, "")
        lines.append(f"\n## {name}")
        lines.append(content if content else f"  (空 —— {hint})")
    return "\n".join(lines)


if __name__ == "__main__":
    good = blank_release_plan()
    good["interdisciplinary_glossary"] = "和认知科学合作者共建了一份'注意力'一词在两个学科的定义对照。"
    good["public_communication"] = "写了一段200字的非专业摘要, 请非本领域朋友试读确认能看懂且不夸大结论。"
    good["preregistration"] = "在OSF上预注册了核心假设和统计检验方法, 时间戳早于正式实验开始。"
    good["artifact_release_plan"] = "投稿时代码已整理好放在匿名仓库, accept后24小时内公开正式仓库。"
    good["social_media_boundary"] = "个人twitter简介已声明'观点仅代表个人, 不代表所在机构'。"
    print(render(good))
    chk = audit(good)
    print("\n" + ("✅ 骨架完整" if chk["ready"] else "⚠ " + "; ".join(chk["issues"])))
```

- [ ] **Step 3: 验证工具**

Run: `python learning/open-science-and-communication/src/open_science_audit.py`
Expected: 打印骨架+"✅ 骨架完整"。

- [ ] **Step 4-8:** 同Task 1模式(templates对应5个SECTIONS;notebook `N1-audit-open-science.ipynb`;papers/README.md引用真实存在的开放科学资源,如Center for Open Science(cos.io)的OSF预注册说明、ACM Artifact Review and Badging政策官网;environment/README标注9.17)。

- [ ] **Step 9: 验证与提交**

```bash
python learning/open-science-and-communication/environment/verify_env.py
git add learning/open-science-and-communication/
git commit -m "feat(open-science-and-communication): 新增9.17专题——开放科学与科学传播(5讲+工具+notebook)"
```

---

### Task 9: `literature-mapping` 追加 L5(系统性文献综述/meta-analysis方法论)

**Files:**
- Create: `learning/literature-mapping/lectures/L5-systematic-review-methodology.md`
- Modify: `learning/literature-mapping/README.md`(学习路径表追加一行)

- [ ] **Step 1**: 先Read `learning/literature-mapping/README.md`和`lectures/L4*.md`确认现有讲次编号与收尾方式。

- [ ] **Step 2**: 用通用讲义派发模板生成L5,content_scope: "系统性文献综述/meta-analysis方法论——和`literature-mapping` L1-L4(摸清一个方向的文献版图,产出是给自己用的地图)的区别是,系统性综述是**产出本身就是一篇可发表的成果**,PRISMA流程(检索策略预注册/筛选标准/双人独立筛选/PICO框架)、meta-analysis的效应量合并方法(fixed vs random effects model)基础"。

- [ ] **Step 3**: Read后Edit `README.md`学习路径表,追加一行L5,不改动L1-L4现有行。

- [ ] **Step 4: 验证**

Run: `grep -c "^## " learning/literature-mapping/lectures/L5-systematic-review-methodology.md`
Expected: ≥5(章节编号存在)

```bash
python -c "print(len(open('learning/literature-mapping/lectures/L5-systematic-review-methodology.md', encoding='utf-8').read()))"
```
Expected: ≥4000

- [ ] **Step 5: 提交**

```bash
git add learning/literature-mapping/
git commit -m "feat(literature-mapping): 追加L5系统性文献综述/meta-analysis方法论(深化9.10-9.17之外的E组缺口)"
```

---

### Task 10: `experiment-design` 追加 L6-L8

**Files:**
- Create: `learning/experiment-design/lectures/L6-bayesian-vs-frequentist.md`
- Create: `learning/experiment-design/lectures/L7-compute-budget-planning.md`
- Create: `learning/experiment-design/lectures/L8-systematic-ablation-design.md`
- Modify: `learning/experiment-design/README.md`

- [ ] **Step 1**: 先Read `learning/experiment-design/README.md`和现有L5确认编号/收尾方式,并Glob确认`interp-graduation`/`harness-engineering`等将引用的专题真实存在。

- [ ] **Step 2**: 逐篇用讲义派发模板生成:
  - L6 content_scope: "贝叶斯vs频率派统计方法选择——和`experiment-design` L5(方差与显著性检验,频率派范式下的p-value/置信区间)互补,讲清楚什么场景该换成贝叶斯方法(如小样本、需要持续更新的在线实验),贝叶斯因子vs p-value的实践取舍"
  - L7 content_scope: "大规模实验的算力预算规划与调度——训练实验的compute budget估算方法、集群调度排队现实、和`research-funding-and-resources`(9.14) L2的区别是这里是'实验设计阶段就该纳入的技术判断',9.14 L2是'资源申请的行政流程'"
  - L8 content_scope: "Ablation设计系统化方法论深水——怎么系统枚举该做哪些ablation(不是想到哪做到哪),消融矩阵的完备性检查,常见的'漏做关键ablation导致审稿人一击致命'案例模式"

- [ ] **Step 3**: Edit `README.md`追加L6-L8三行。

- [ ] **Step 4: 验证**

```bash
for f in L6-bayesian-vs-frequentist L7-compute-budget-planning L8-systematic-ablation-design; do
  python -c "print('$f:', len(open('learning/experiment-design/lectures/$f.md', encoding='utf-8').read()))"
done
```
Expected: 三者均 ≥4000

- [ ] **Step 5: 提交**

```bash
git add learning/experiment-design/
git commit -m "feat(experiment-design): 追加L6-L8(贝叶斯vs频率派/算力预算规划/ablation深水)"
```

---

### Task 11: `paper-writing-submission` 追加 L5-L10

**Files:**
- Create: `learning/paper-writing-submission/lectures/L5-negative-results.md`
- Create: `learning/paper-writing-submission/lectures/L6-paper-series-strategy.md`
- Create: `learning/paper-writing-submission/lectures/L7-venue-selection.md`
- Create: `learning/paper-writing-submission/lectures/L8-camera-ready-and-artifacts.md`
- Create: `learning/paper-writing-submission/lectures/L9-building-long-term-impact.md`
- Create: `learning/paper-writing-submission/lectures/L10-writing-as-non-native-speaker.md`
- Modify: `learning/paper-writing-submission/README.md`

- [ ] **Step 1**: 先Read `learning/paper-writing-submission/README.md`和现有L1-L4确认编号/收尾方式。

- [ ] **Step 2**: 分两批(L5-L7一批,L8-L10一批)用讲义派发模板生成:
  - L5 content_scope: "负结果的处理与发表策略——file drawer problem、什么情况下负结果值得写成论文(如颠覆了广泛假设的负结果)、怎么在正面工作里诚实报告失败的分支而不是藏起来"
  - L6 content_scope: "多篇论文组合发表策略——一个大项目怎么切成几篇可独立发表的论文而不是憋一篇'大而全',常见的'切太碎'和'攒太久'两种反模式"
  - L7 content_scope: "会议vs期刊选择策略+dual submission规则——不同venue的时间线/影响力/审稿风格差异,双重投稿的红线和常见误解"
  - L8 content_scope: "Camera-ready与后续维护——accept后到camera-ready deadline之间要做什么、artifact evaluation(如果venue有)怎么准备、代码/数据发布的时间点"
  - L9 content_scope: "论文的长期影响力经营——怎么让一篇论文被更多人看到和引用(不是灌水引用而是让真正相关的人看到)、follow-up工作的规划"
  - L10 content_scope: "非英语母语者的学术写作策略——常见的中式英语学术写作陷阱、怎么高效利用写作辅助工具而不丢失自己的论证逻辑"

- [ ] **Step 3**: Edit `README.md`追加L5-L10六行。

- [ ] **Step 4: 验证**

```bash
for f in L5-negative-results L6-paper-series-strategy L7-venue-selection L8-camera-ready-and-artifacts L9-building-long-term-impact L10-writing-as-non-native-speaker; do
  python -c "print('$f:', len(open('learning/paper-writing-submission/lectures/$f.md', encoding='utf-8').read()))"
done
```
Expected: 六者均 ≥4000

- [ ] **Step 5: 提交**

```bash
git add learning/paper-writing-submission/
git commit -m "feat(paper-writing-submission): 追加L5-L10(负结果/论文组合策略/venue选择/camera-ready/长期影响力/非英语母语写作)"
```

---

### Task 12: 收尾——L0地图更新、全局去重、memory

- [ ] **Step 1**: Read `learning/research-direction-proposal/lectures/L0-research-lifecycle-map.md`,在文末("本讲小结"之前)新增一节"## 6. 科研生涯周期(9.10-9.17)与本地图的关系",说明:20环节地图是"单个项目周期",9.10-9.17是与之并列、跨项目反复出现的"科研生涯周期",两者不是包含关系,并列出9.10-9.17的一句话索引表(专题名+一句话)。

- [ ] **Step 2**: 全局去重检查

```bash
grep -rn "^# L[0-9]" learning/career-pathways/lectures/ learning/research-visibility-negotiation/lectures/ learning/team-leadership-for-researchers/lectures/ learning/research-team-operations/lectures/ learning/research-funding-and-resources/lectures/ learning/research-integrity-and-compliance/lectures/ learning/academic-community-engagement/lectures/ learning/open-science-and-communication/lectures/ | sort
```
逐行确认50个新讲次标题彼此不重复,且与`experiment-design`/`literature-mapping`/`paper-writing-submission`原有讲次标题不冲突。

- [ ] **Step 3**: 全部8个新专题跑一次环境验证汇总

```bash
for d in career-pathways research-visibility-negotiation team-leadership-for-researchers research-team-operations research-funding-and-resources research-integrity-and-compliance academic-community-engagement open-science-and-communication; do
  echo "=== $d ==="
  python "learning/$d/environment/verify_env.py"
done
```
Expected: 8个均输出"全部通过 ✅"。

- [ ] **Step 4**: 提交L0更新

```bash
git add learning/research-direction-proposal/lectures/L0-research-lifecycle-map.md
git commit -m "docs(research-direction-proposal): L0地图追加9.10-9.17科研生涯周期索引说明"
```

- [ ] **Step 5**: 写memory文件 `C:\Users\ericp\.claude\projects\c--Workspace-dummy\memory\research-career-lifecycle-modules.md`(YAML frontmatter: name/description/metadata.node_type=memory/metadata.type=project/metadata.originSessionId),内容涵盖:老师提出50个专题的完整要求、"五组均衡不做取舍"决策、"Module 10编号冲突→改用9.10-9.17"的命名修正、8个新专题+3个现有专题追加讲次的清单、Why(科研生涯周期与项目周期是两个不同尺度,不能塞进20环节地图)/How to apply(未来若还有"科研技能"相关新需求,先检查这18个专题——原10个+新8个——有没有覆盖)。更新`MEMORY.md`追加一行索引。

- [ ] **Step 6**: 最终确认

```bash
git log --oneline -15
git status
```
Expected: working tree clean,最近的commit包含全部12个task的提交记录。

---

## 执行顺序提醒

严格按 Task 1 → 2 → ... → 12 顺序执行(每个新专题内部讲义可批量并行派发subagent,但专题与专题之间顺序推进,降低单一时间窗口的并发峰值,沿用CS六件套已验证的分批策略)。若某个agent因429/503失败,先用`ls`确认磁盘上是否已有完整文件再决定是否重派。
