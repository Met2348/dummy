# Runbook 验证 & README 框架 — 设计文档

> Branch: `ERIC-3080Ti/paper-guides`
> Machine: ERIC-3080Ti（RTX 3080 Ti Laptop 16GB, driver 595.97, torch 2.11.0+cu128, Python 3.13.9, repo-local `.venv`）
> Date: 2026-06-20
> 关联计划：[2026-06-20-runbook-verification.md](../plans/2026-06-20-runbook-verification.md)
> 活动账本：[ERIC-3080Ti-runbook-progress.md](../../local-env/ERIC-3080Ti-runbook-progress.md)

---

## 1. 目标（Why）

确保**系统学习时，照着每个模块文档里的运行指示走，不会踩坑**。

具体地：
1. 文档里写的"实用入口 / 怎么运行"命令，**照着敲就能跑通**（不只是单元测试绿）。
2. 命令的 flag、路径、预期输出与代码**一致**；不一致则修。
3. 补齐 README 缺口：根 README + Module 8 的 7 个模块 README。
4. **自主、可续**：中途会多次 compact，全部状态落盘，任何新会话读 3 个文件即可接续。

## 2. 现状基线（已完成，不重复劳动）

[ERIC-3080Ti-final-report.md](../../local-env/ERIC-3080Ti-final-report.md) 显示，单元测试层面已绿：

| Check | Result |
|---|---:|
| `src/tests/` 模块测试 | 46 / 46 PASS |
| `verify_env.py` | 39 / 39 PASS |
| Module 8（无 env 脚本）| 7 / 7 由测试覆盖 |

**已有工具**：`scripts/eric_3080ti_env_audit.py` —— 带 UTF-8、隔离 PYTHONPATH、超时控制、脚本式 `_self_test()` 回退的测试 harness。

**关键缺口（本设计要补的）**：现有审计跑的是 `pytest src/tests/` 和 `verify_env.py`，**没有验证 README/讲义里写的"实用入口"命令本身能否照着跑通**，也没碰 notebook。单元测试通过 ≠ 文档命令可用。

## 3. 验证标准（What "跑通" means）

每模块按阶梯验证。本轮做 **V0 + V1 + V2**；V3/V4 见 §8。

| 层 | 名称 | 判定 |
|---|---|---|
| **V0** | 文档静态 | README/讲义"运行"段每条命令：① 引用的文件路径存在；② 脚本 argparse 定义了命令用到的每个 flag；③ `python <script> --help` exit 0（import 链通）。 |
| **V1** | smoke 跑通 | 每条文档入口命令用**最小预算**（tiny steps/iters/batch）跑到 **exit 0**；stdout 与文档里的"预期"声明（如"reward 上升""shape=…"）大致对得上。 |
| **V2** | 测试复核 | `src/tests/` 经 audit harness 仍绿；记录"测试是否覆盖文档入口命令"的缺口（覆盖缺口 = 潜在踩坑点）。 |

**重型/可选栈**（vllm / verl / ray / playwright / 大模型权重）：**尽量改出在 3080 Ti(16GB)/CPU 上真实可跑的缩小版**（更小模型、更短序列、mock 数据）；确实无法本地真实运行的，才退化为文档标注 + mock smoke 路径。

**修复策略**：
- 代码跑不通 → **修代码 + 修文档**。
- 代码能跑但文档指示错（flag/路径/预期对不上）→ **修文档**。
- 每个修复都要让"照 README 敲"这条路径成立。

## 4. 单一事实源：每模块 `runbook.yaml`

放在 `learning/<module>/runbook.yaml`，机器可读地登记该模块的"文档入口命令"：

```yaml
module: rl-foundations
commands:
  - id: cartpole-ppo
    desc: "CartPole PPO 横向（README 实用入口）"
    cmd: "python src/cartpole_full.py --algo ppo --total-steps {steps}"
    full:  { steps: 30000 }     # 文档里给学生的真实预算
    smoke: { steps: 2000 }      # 验证器实际执行的最小预算
    expect: "exit 0; mean reward 随 step 上升"
    tier: V1
    gpu: false
  - id: capstone-imdb
    desc: "Capstone IMDb PPO"
    cmd: "python src/capstone_imdb_ppo.py --total-iters {iters} --batch-size {bs}"
    full:  { iters: 200, bs: 16 }
    smoke: { iters: 2, bs: 4 }
    expect: "exit 0; 生成样本情感分上升"
    tier: V1
    gpu: true
```

设计意图：
- **README 的"运行验证"段从 / 据 `runbook.yaml` 校验**，所以文档和被测命令**无法漂移**。
- 验证器执行 `smoke` 形态；README 给学生展示 `full` 形态 + 一句"smoke 用 `--total-steps 2000`"。
- `runbook.yaml` 本身就是该模块"有哪些可跑入口"的清单，对学习者也是导航。

## 5. 工具：扩展 `eric_3080ti_env_audit.py`

加一个 `--runbook` 模式，**复用**现有 harness 的进程运行管线（UTF-8 环境、隔离 `PYTHONPATH=<module>/src`、timeout、stdout/stderr 捕获），避免重复造轮子。

行为：
1. 遍历指定模块的 `runbook.yaml`。
2. 对每条命令跑 V0（静态：路径存在 + `--help`）。
3. 对 `tier: V1` 命令跑 V1（smoke 形态，到 exit 0）。
4. 输出账本 md + json（见 §6）。

调用示例：
```powershell
$env:PYTHONUTF8="1"; $env:PYTHONIOENCODING="utf-8"
.\.venv\Scripts\python.exe scripts\eric_3080ti_env_audit.py --runbook --modules rl-foundations `
  --json-out docs\local-env\ERIC-3080Ti-runbook-results.json `
  --md-out  docs\local-env\ERIC-3080Ti-runbook-matrix.md
```

## 6. 持久化 & 续跑（扛 compact）

三份落盘文件 = 全部状态，任何新会话读它们即可接续：

1. **spec（本文）** — 标准与方法，基本不变。
2. **plan** — `docs/superpowers/plans/2026-06-20-runbook-verification.md` — 可执行的每模块循环 + 模块顺序 + RESUME 协议。
3. **活动账本** — `docs/local-env/ERIC-3080Ti-runbook-progress.md`（人读）+ `.json`（机读）：46 模块 × {README, runbook.yaml, V0, V1, V2} 状态 + 发现的问题 + 修复 + commit sha。**账本顶部写死 RESUME PROTOCOL。**

外加 **每模块一个 git commit** 于 `ERIC-3080Ti/paper-guides`，git 历史本身即第二条进度轨迹。

**Checkpoint 纪律**：每 ~3 个模块、以及任何有风险/长耗时操作之前，更新账本并 commit。

## 7. README 计划

| 对象 | 数量 | 内容 |
|---|---|---|
| 根 `README.md`（新建）| 1 | 知识组织框架：8 模块 / 46 专题地图；7 件套模块解剖；`.venv` 安装；"怎么运行/验证任意模块"；导航。 |
| Module 8 模块 README（新建）| 7 | `gpu-architecture` `cuda-essentials` `kernel-engineering` `cluster-networking` `storage-dataops` `training-orchestration` `infra-graduation`；以 rl-foundations README 为模板。 |
| 已有模块 README（修）| 39 | 补/修"运行验证 / Runbook"段，与 `runbook.yaml` 对齐（V0/V1 期间顺手修文档）。 |

## 8. 范围边界（YAGNI）

- **本轮做**：src 代码 + README/讲义文档命令的 V0/V1/V2；根 + 7 个 README；39 个 README 的运行段修订。
- **本轮不做**：V3 notebook 端到端执行（410 个，留作后续单列 pass）；V4 完整训练复现（capstone 真实达标，太慢，仅按需 opt-in）。

## 9. 顺序

- **Phase 0（pilot）**：在 `rl-foundations`（README 已是范本、含重型 + capstone + GPU/非 GPU 混合）上把 `--runbook` 模式、账本、`runbook.yaml` 模板、README 运行段模板全部打磨好。**先去工具风险。**
- **Phase 1（fan-out）**：按课程序 M1→M8 逐模块跑循环。到 M8 时一并写 7 个缺失 README。
- **Phase 2**：根 README（积累足够模块信息后写，描述才准确）。
- **Phase 3（后续）**：notebook pass。

## 10. 每模块循环（可重复单元）

1. 读 README + lectures → 抽取文档入口命令 → 写/更新 `runbook.yaml`。
2. **V0** 静态检查 → 修文档/代码不一致。
3. **V1** 逐命令 smoke → 修代码（或文档）到 exit 0；重型栈做缩小版。
4. **V2** 复核 `src/tests/`。
5. 更新 README"运行验证"段，与 `runbook.yaml` 对齐。
6. 更新账本；`git commit`（`verify(<module>): ...`）。
