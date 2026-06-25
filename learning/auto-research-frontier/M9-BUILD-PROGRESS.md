# M9 专题建设进度账本（source of truth，抗 compaction）

> 任务：把 CURRICULUM.md 的 8 专题里**除 9.5 外的 7 个**全部建成像 9.5 那样的
> **可跑可验证模块**（手写缩小版 + 每 demo 配批判 + runbook.yaml/pytest，V0/V1/V2 全绿）。
> 用户指令：「把之前没有做完的专题全部彻底做完」（2026-06-25）。自主推进，预期多次 compaction。

## 验证标准（每个模块都要过）
- **V0** 文档静态：`run.py --help` exit 0（runbook `v0:true`）。
- **V1** 冒烟跑通：harness `--runbook` 跑 smoke 形式 exit 0、打印预期。
- **V2** 测试：`src/tests/test_*.py` 真断言（pytest，含 `__main__` 兜底）。
- 验证命令（**必须**带 `--json-out/--md-out` 指向 temp，**别**用默认覆盖已提交基线）：
  ```bash
  python scripts/eric_3080ti_env_audit.py --runbook \
    --modules auto-research-frontier/<模块目录> \
    --json-out C:/Users/ericp/AppData/Local/Temp/m9build.json \
    --md-out  C:/Users/ericp/AppData/Local/Temp/m9build.md
  ```
- 脚手架照抄 9.5：`src/run.py`(sys.path 引导+argparse)、`src/<pkg>/`、`src/tests/`、`runbook.yaml`(module 传**全相对路径**)、`README.md`+`lectures/`。

## 进度表
| 专题 | 目录 | 可跑交付物 | 批判点(防踩坑) | 状态 |
|------|------|-----------|---------------|------|
| 9.1 自主性阶梯与全景 | `m9.1-autonomy-ladder-and-map/` | `taxonomy_classifier`：按**证据**(非自称)把系统归 Tool/Analyst/Scientist + 生命周期覆盖图 | claim≠evidence 的 hype gap；"自称 Scientist 的都是自评的" | ✅ 4/4 绿 |
| 9.2 研究 Agent 内核 | `m9.2-research-agent-core/` | `mini_research_agent`：ReAct 环 问题→检索→拟idea→自我批判→结构化计划 | 无 critic 会过度自信/幻觉引用；检索 grounding | ✅ 4/4 绿 |
| 9.3 创意与假设生成 | `m9.3-ideation-and-tournament/` | `idea_tournament`：生成K→novelty+judge→Elo锦标赛→top-k | LLM 给自己点子打高分；novelty≠feasibility | ✅ 4/4 绿 |
| 9.4 Deep Research 综述 | `m9.4-deep-research-storm/` | `mini_storm`：多视角提问→检索→带引用合成→**引用忠实度核查** | 引用存在≠引用忠实（植入假引用被抓） | ✅ 3/3 绿 |
| 9.6 评测 Research Agent | `m9.6-evaluating-research-agents/` | `mini_replication_eval`：method spec+rubric 真 exec 判分（扩 safe_exec） | 弱 rubric 被刷(硬编码/print指标)，强 rubric 抓住 | ⬜ |
| 9.7 自我改进/进化 | `m9.7-self-improvement-evolution/` | `mini_self_improve`：变异自己→评估→keep-if-better→档案 | fitness 被 game（泄漏信号过拟合，held-out 戳穿） | ⬜ |
| 9.8 批判·安全·诚信(capstone) | `m9.8-redteam-and-integrity/` | 红队 mini-scientist：幻觉表/换数据/硬编码指标 + 守卫 | 每种攻击对裸 pipeline 成功、被守卫抓住（防御教育） | ⬜ |

## 顺序与依赖
建设序：9.1→9.2→9.3→9.4→9.6→9.7→9.8（9.8 红队 9.5，放最后做 capstone）。
每个模块**自包含**（不跨模块 import；要复用 safe_exec/judge 就内联缩小版）。

## 已完成提交记录
（建好一个填一行：commit hash + 模块 + V0/V1/V2 结果）
- **9.1** autonomy-ladder-and-map → harness 4/4 绿（V0 --help / V1 ×2 / V2 pytest 8 测试）。commit aadc042
- **9.2** research-agent-core → harness 4/4 绿（V0 / V1 ×2 / V2 pytest 7 测试）。commit 152a5b6
- **9.3** ideation-and-tournament → harness 4/4 绿（V0 / V1 ×2 / V2 pytest 6 测试）。commit e931f82
- **9.4** deep-research-storm → harness 3/3 绿（V0 / V1 / V2 pytest 6 测试）。

## 最近进度
- 2026-06-25：9.1/9.2/9.3/9.4 建成并全绿。已完成 4/7。
  9.4 核心：mini-STORM 多视角综述 + 忠实度核查；植入 2 句"引真论文但论文不支持此论断"
  （v2 没做湿实验/STORM 没高引用准确率），naive 存在性检查 5/5 全过、忠实度揪出 2 句。
  下一步建 9.6 评测 Research Agent（safe_exec 真沙箱判分）。
