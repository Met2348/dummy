# TRACE-H 本机证据强度更新：Source Policy v2 与 REPLAN 边际效用

- **日期：** 2026-07-13
- **最终本机决策：** 8B source scaffold 通过 competence gate；当前 REPLAN 机制未通过 advantage gate；不进入 target transport PK。

## 直接结果

| 实验 | 4B | 8B | 结论 |
|---|---:|---:|---|
| 旧 command-trie baseline | 0/3 | 0/3 | 原 scaffold 退化 |
| Source Policy v2 | 0/3 | 2/3 | deliberation、长期记忆与 anti-loop 对8B有效，但不跨到4B |
| v2 parser failure | 0/150 | 0/82 | 结果不是 action legality 造成 |
| 8B failure-prefix replay | - | 2/2 suffix 完全一致 | paired branch 因果起点可信 |
| v2 NONE/REPLAN | - | 6/6 pairs 全零 | REPLAN 没有正 terminal advantage |

8B 在 heat-egg 与 book-to-sofa 两题均16步成功，双 newspaper 任务50步失败。失败任务抽取两个 prefixes，deterministic NONE 后缀分别47/47与38/38动作完全复现。六个 REPLAN plans 均非空，并使每对后续轨迹产生32-47个不同动作，但 NONE 与 REPLAN terminal scores 仍全部为0。因此失败不是“plan 没进入执行器”，而是改变行为后仍未完成多实例任务。

## 证据强度分层

| 证据层 | 强度 | 置信度 | 解释 |
|---|---:|---|---|
| Runner、parser、append-only、replay | 9/10 | 高 | 多轮真实GPU运行、0 parser failure、完整suffix复现 |
| Partial transport/LCB 软件机制 | 8.5/10 | 高 | 5 scenarios x 20 seeds 重跑，4项断言全过且报告hash完全一致 |
| 8B source executor competence | 6/10 | 中 | 同三题由0/3升至2/3，但样本极小且每步两次调用 |
| 跨模型 source competence | 3/10 | 中高 | 4B仍0/3，改善不具备模型普适性 |
| REPLAN action advantage | 1/10 | 高 | 两个失败prefix、三seed均无正效用；这是强局部反证 |
| TRACE-H真实 transport 效果 | 0/10 | 高 | 尚未形成非退化多动作 response bank，target未运行 |
| AAAI主会实证就绪度 | 2.5/10 | 中高 | 有可信工程与一个scaffold正结果，但没有方法主表或baseline PK |

## 科学结论

当前最强正结论是：**显式 deliberation、长期可见历史和确定性 anti-loop 能把 Qwen3-8B 的 ALFWorld 三题成功率从0/3提高到2/3。** 这证明原先全零并非任务或本机量化环境绝对不可解，也证明 source-policy qualification 必须先于 policy transport。

当前最强负结论是：**在已经具备2/3任务能力的8B executor 上，额外一次 REPLAN 虽能系统性改变轨迹，却没有在困难的双对象任务上产生任何终局边际收益。** 因而不能继续把当前自然语言 REPLAN 当作 response bank 的有效 action，也不能运行 CHECK/RETRY 后直接进入 target PK。

## 本机停止边界

本机已完成当前路径仍有信息增益的 GPU 实验：4B/8B competence gate、8B failure-prefix repeats、L4 synthetic 重跑。下一步需要改变机制而非扩量：将 REPLAN 从一次性自然语言计划改成可验证的结构化 subgoal-state update，或引入合法的中间 goal-condition signal。该变化会形成新 action contract，必须重新预注册；在此之前不消耗14B或B200 target预算。

证据文件：

- [8B Source Policy v2 gate](local-dev/reports/L3-source-policy-v2-gate-qwen3-8b.json)
- [4B Source Policy v2 gate](local-dev/reports/L3-source-policy-v2-gate-qwen3-4b.json)
- [v2 prefix replay](local-dev/reports/L3-source-policy-v2-qwen3-8b-prefix-replay-audit.json)
- [v2 NONE integrity](local-dev/reports/L3-source-policy-v2-qwen3-8b-none-integrity.json)
- [v2 NONE/REPLAN branches](local-dev/reports/L3-source-policy-v2-qwen3-8b-none-replan-branch.json)
- [L4 deterministic rerun](local-dev/reports/L4-synthetic-transport-rerun-20260713.json)

