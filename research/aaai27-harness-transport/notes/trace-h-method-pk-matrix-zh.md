# TRACE-H 方法 PK 矩阵与公平比较协议

- **日期：** 2026-07-11
- **主原则：** 第一主表比较 actual execution；所有方法按 target feedback、source data 和 inference cost 分账。
- **配套 Proposal：** [TRACE-H Policy Transport](../proposals/trace-h-policy-transport-proposal-zh.md)

## 1. 三种预算

每个方法必须报告三列，任何一列不同都不能宣称“同预算击败”：

1. `B_source`：来源 executor 的训练/branch episodes；
2. `B_target_action`：适配阶段读取的目标非 NONE action outcomes；
3. `B_infer`：测试时额外 LLM calls、tokens、steps 和 wall-clock。

TRACE-H 的核心 operating point 是：

```text
B_target_action = 0
```

目标 baseline-only calibration episodes 单列为 `B_target_base`，所有零反馈方法共享。

## 2. 第一主表：零目标干预

| ID | 方法 | 学习/决策机制 | Target action outcomes | 实现来源 |
|---|---|---|---:|---|
| B0 | No Harness | NONE | 0 | 统一 runner |
| B1 | Best Fixed | source validation 选 CHECK/RETRY/REPLAN | 0 | 本项目 |
| B2 | Source-AW | pooled source offline advantage weighting | 0 | Agentic-RL-harness code |
| B3 | Nearest-AW | target baseline fingerprint 选 source AW | 0 | 本项目 + AW code |
| B4 | MF-Gated AW | Metric Freedom gate source AW | 0 | 忠实复现 MF + AW |
| B5 | Category Router | task/event category 选择 source policy | 0 | Adaptive Auto-Harness-style controlled baseline |
| B6 | kNN-Branch | 最近 source event response | 0 | 本项目 |
| B7 | Balanced-OT | 全质量强制对齐 | 0 | POT |
| B8 | PAR-style Penalty | representation mismatch 降权 source transitions | 0 或少量 baseline transitions | 忠实机制重实现 |
| M | TRACE-H | counterfactual branch + partial OT + LCB/NONE | 0 | 本项目 |

预声明 primary contrast：`TRACE-H - max(B0...B8)`，按 task-paired utility 计算。不能事后挑一个弱 baseline 做 headline。

## 3. 第二主表：完整公开系统

| 方法 | 共同环境 | Target feedback | 运行状态 | 解释 |
|---|---|---:|---|---|
| MASA Base Skill | ALFWorld/WebShop | 0 | 官方 JSON 可运行 | model-agnostic static skill |
| MASA DS-Adapter | ALFWorld/WebShop | 0 action reward | 官方 JSON 可运行 | model-card one-shot rewrite |
| MASA Evolved | ALFWorld/WebShop/Qwen | 大量 | 官方 JSON 可运行 | target-search upper comparator |
| SkillAdaptor | WebShop/统一任务 wrapper | 多轮 | 官方 code snapshot | target failure + qualification baseline |
| Target-AW | 统一 Harness MDP | rollout buffer | 官方 AW code | target-trained policy upper comparator |
| SkVM-style AOT | 支持 capability transform 的 task slice | capability probes | 需重实现，标非官方 | static compiler comparator |
| TRACE-H | ALFWorld/WebShop | 0 action reward | 待实现 | cross-executor policy transport |

若 action spaces 不同，不做虚假的 per-action accuracy 比较；只比较相同模型、任务、成功判据和完整成本下的 system utility。

## 4. 第三张图：Target Feedback Frontier

横轴：`B_target_action = 0, 5, 10, 20, 40, ...`

纵轴：target end-to-end normalized utility。

- `x=0`：TRACE-H 与零反馈 baselines；
- `x>0`：SkillAdaptor、target-AW、MASA evolution 的 budgeted variants；
- 目标不是声称零反馈永远胜有反馈，而是证明 TRACE-H 在低预算区间占优，并减少达到某一效用所需的 target trials。

## 5. 共同运行规范

- 相同 executor checkpoint/quantization；
- 相同 task manifest 与 order；
- 相同 environment state、max steps、tool set；
- 相同 sampling temperature/seed policy；
- 相同 success verifier；
- action 增加的 tokens/steps/latency全部计入；
- parser failure 计为真实失败，不静默重跑；
- 每个方法的 target information 写入 append-only ledger；
- 作者 artifact、作者 code、本项目修复版、机制重实现四种状态必须分开标注。

## 6. 结果表模板

| Method | `B_target_base` | `B_target_action` | Success ↑ | Utility ↑ | Tokens ↓ | Neg. intervention ↓ | Oracle regret ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| No Harness | 0 | 0 |  |  |  |  |  |
| Source-AW | shared | 0 |  |  |  |  |  |
| MF-Gated AW | shared | 0 |  |  |  |  |  |
| kNN-Branch | shared | 0 |  |  |  |  |  |
| Balanced-OT | shared | 0 |  |  |  |  |  |
| TRACE-H | shared | 0 |  |  |  |  |  |
| SkillAdaptor | shared | >0 |  |  |  |  |  |
| Target-AW | shared | >0 |  |  |  |  |  |

## 7. 必须击败的顺序

1. `No Harness`：只能证明方法不荒谬；
2. `Best Fixed/Source-AW`：证明动态 policy 与跨模型问题存在；
3. `MF-Gated/Nearest/Category`：证明 target baseline 简单适配不够；
4. `kNN/Balanced-OT/PAR-style`：证明新运输机制有增量；
5. MASA/SkillAdaptor/Target-AW 成本前沿：证明相对已有设计方法有实际价值。

论文至少必须完成第 4 层；只完成前两层不具备 AAAI 主会方法论文强度。

## 8. Kill 条件

- TRACE-H 不胜最强零反馈 baseline；
- 优势只存在于 prediction/diagnostic metric；
- MASA DS-Adapter 在相同零 action feedback 下稳定更强且更便宜；
- 少量 target feedback 即让 SkillAdaptor/Target-AW 以极低成本全面超过；
- partial OT 不胜 balanced OT/kNN；
- cross-family target 出现实质负效用。

满足任一条时，不得用更多 related-work 限定词挽救 design claim。
