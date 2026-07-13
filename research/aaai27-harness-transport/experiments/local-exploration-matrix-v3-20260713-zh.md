# TRACE-H 本机探索矩阵 v3

- **日期：** 2026-07-13
- **性质：** exploratory development，不作为论文确认性结果
- **冻结原则：** 全部尝试进入账本；不删除失败；不触碰14B/32B/Gemma target outcome

## E1：8B source competence 扩展

- 已有开发任务：offsets 0-2；
- 新增描述性任务：offsets 3-9；
- 固定 full Source Policy v2、NF4、seed 20260712、50 steps；
- 报告 task type、success、steps、calls、parser failure；
- 不根据新增任务结果修改 offsets 0-9。

## A1/A2：三任务机制消融

| ID | Deliberation | Persistent memory | Anti-loop | 目的 |
|---|---|---|---|---|
| V1 | 否 | 4-step raw history | 否 | 历史旧 baseline |
| V2 | 是 | 12 transitions | 是，repeat=2 | 当前 full method |
| A1 | 是 | 12 transitions | 否，repeat=999 | 测 anti-loop 边际作用 |
| A2 | 是 | 0 transitions | 是，repeat=2 | 测长期记忆边际作用 |

三题样本只用于机制定位，不做显著性检验。成本同时报告，不把两调用方法与一调用方法写成同预算胜利。

## M1：结构化 REPLAN

额外一次调用必须输出一个 JSON object：

```json
{
  "phase": "locate|acquire|transform|deliver|recover",
  "subgoal": "short text",
  "known_facts": ["visible fact"],
  "avoid_actions": ["exact action"],
  "next_action": "exact admissible command"
}
```

确定性 verifier 检查字段、phase、字符串长度、列表类型和 `next_action` 合法性。验证失败仍计一次 REPLAN 成本，但 guidance 为空。开发集为8B双 newspaper 失败轨迹的p003/p012，seeds 0/1/2；与已完成 natural-language REPLAN 同预算比较。

### M1 筛选门

- 至少一个 seed-0 candidate 获得正 terminal advantage，才允许进入确认集；
- 若只改变轨迹、不改变utility，不晋级；
- 若 JSON validity <90%，先判 contract 不可执行，不通过放宽parser补救；
- 最终论文机制门仍要求两个candidate为正且重复符号稳定率 >=70%。

## S1：Synthetic stress grid

- private ratio：0%、10%、30%、50%、70%；
- feature noise：0.03、0.10、0.25；
- response conflict：0%、25%、50%、75%、100%；
- 每格20 seeds；
- 比较 balanced OT、partial transport、partial+LCB；
- 检查 private coverage、NONE rate、utility与response-aware相对semantic-only优势。

## Confirmatory holdout

任何本轮选出的机制，只能在未参与选择的 offsets 10-14 或后续B200预注册任务上确认。不得用 offsets 0-9 的重复调参结果冒充独立复现。
