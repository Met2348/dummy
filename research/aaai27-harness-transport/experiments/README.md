# TRACE-H 实验工作区

## 2026-07-13 ledgered source-first macro 证据

[Ledgered macro evidence](local-ledgered-macro-evidence-20260713-zh.md)：真实 Qwen3-8B Source Policy v2 no-progress candidates 上，`source_first + delivered/transformed/deposit ledger` 达到 5/5 success；`target_first` 对照为 3/5。当前决策改为把单步 `bundle_conservative` 降级为 ablation，把主线推进到状态化 macro mechanism。

## 2026-07-13 源策略分支最终判定

[Qwen3-4B/8B NONE vs REPLAN 最终报告](local-none-replan-source-pilot-final-20260713-zh.md)：9/9 prefix 与 deterministic NONE continuation 完整一致；17 对 paired branches 的 raw terminal scores 全为 0。4B 的 REPLAN 改变 10/10 action suffix，8B 改变 3/7，但均无 utility。当前决策为 `PIVOT_SOURCE_POLICY_OR_TASK_SIGNAL`；不补跑 CHECK/RETRY，不形成 response bank，不进入 target transport PK。

[Source Policy v2 证据更新](local-source-policy-v2-evidence-assessment-20260713-zh.md)：8B由0/3提升至2/3，4B仍为0/3；但8B困难任务的6对 NONE/REPLAN 仍全零，因此 source competence 已修复一部分，当前 REPLAN action mechanism 仍被停止。

## 当前里程碑

1. [2026-07-12 L2/L3 续跑状态](local-run-continuation-20260712-zh.md)：可信 command-trie baseline 已跑 3x50 steps，parser 0/150 failure，提取 6 个 NO_PROGRESS prefixes 并完成 6/6 精确重放；
2. [2026-07-12 本机实验启动状态](local-run-status-20260712-zh.md)：Qwen3-4B NF4、97 tests、8-task ALFWorld replay 与 100 次合成 transport 已实跑；
3. [本机可运行性历史审计](local-readiness-audit-20260711-zh.md)：保留启动前 blockers 与首轮 smoke；
4. [本机具体实验计划](local-development-experiment-plan-zh.md)：约 314 条 development episodes，验证 runner、branch/replay、transport、seal 与 B200 handoff；
5. [72 小时方法杀伤实验](../notes/trace-h-72-hour-method-kill-test-zh.md)：在 B200 上运行 750-900 条统一 BF16 episodes，产生 DR-0004；
6. 完整 ALFWorld/WebShop 主矩阵：只有 DR-0004 Go 后展开。

本机 micro result 不是论文主结果。Qwen3-14B 在本机被明确标为 `DEV_TARGET`；正式 sealed targets 仍是 Qwen3-32B 与 Gemma。

## 目录契约

- `manifests/`：不可变 model/task/block manifests；
- `contracts/`：`NONE/CHECK/RETRY/REPLAN` 的版本化可执行契约；
- `runs/`：append-only episode/branch JSONL 与 raw artifacts；
- `artifacts/`：AW、kNN、OT、TRACE-H router 等冻结方法文件；
- `seals/`：freeze manifest、hash 与 target information ledger；
- `analysis/`：统一 primary table、bootstrap、error audit；
- `schemas/`：contract、episode、branch、router 与 ledger schemas；
- `local-dev/`：本机 development manifests、reports 和 B200 handoff bundle。

大模型权重、ALFWorld 数据和大型 raw traces 位于外部运行目录，只在本目录保存不可变 manifest、hash 与小型报告。

## 当前 schema 状态

当前已建立 TRACE-H action-contract、episode、branch、information-ledger、router-artifact 与 freeze schemas，并纳入 97 项自动测试。三个 `forecast-h` schema 是上一版 effect-prediction 方向的历史文件，只保留溯源，不能用于当前 TRACE-H policy-transport runs。

## 数据隔离

- `DEV_SOURCE_BRANCH`、`DEV_TARGET_BASELINE`、`DEV_TARGET_FINAL` 仅用于本机工程协议；
- 正式实验使用 `SOURCE_BRANCH`、`SOURCE_VALIDATION`、`TARGET_BASELINE_CALIBRATION`、`TARGET_FINAL_TEST`；
- target baseline 与 target final 必须使用不同 output roots；
- target final raw records 在全部方法完成前不能进入调参或 artifact 生成流程；
- 任何 infrastructure failure、parser failure 和超预算都必须保留，不能静默重跑。
