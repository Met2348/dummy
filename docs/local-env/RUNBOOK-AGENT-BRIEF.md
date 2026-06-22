# Runbook 验证 — Subagent 作业简报（复用）

你被派来对**一个**学习模块做"文档入口命令"验证。读完本简报照做，返回结构化报告。**不要 git commit**（编排者负责提交）。

## 背景

仓库是 LLM 全栈自学库，46 个模块在 `learning/<module>/`。目标：让"照着模块文档的运行指示走"真能跑通，不踩坑。
- 标准/方法：`docs/superpowers/specs/2026-06-20-runbook-verification-design.md`
- 活动账本（含**系统性坑目录**，必读）：`docs/local-env/ERIC-3080Ti-runbook-progress.md`
- 模板参考：`learning/rl-foundations/runbook.yaml`（带 flag 的 CLI）、`learning/prompt-tuning-family/runbook.yaml`（无参直跑 demo）；README 运行段模板见这两个模块 README 的「运行验证（Runbook）」段。

## 环境

- Windows + repo-local venv。**Python 解释器**：在 Bash 工具里用正斜杠 `.venv/Scripts/python.exe`（反斜杠会被吞）。
- 跑命令前 `export PYTHONUTF8=1 PYTHONIOENCODING=utf-8`。
- GPU：RTX 3080 Ti Laptop 16GB。可用 cuda。

## 任务：对你被指派的模块 `M` 执行以下循环

### 1. 收割文档入口命令
- 读 `learning/M/README.md`（找"实用入口/运行/使用说明/快速开始"段）+ 扫 `learning/M/lectures/` 里的 `python ...` 命令 + 看 `learning/M/src/` 有哪些可跑脚本。
- 判断每个脚本是否有 argparse：`grep -nE "add_argument|ArgumentParser|__main__" learning/M/src/X.py`。

### 2. 写 `learning/M/runbook.yaml`
格式（见模板）：
```yaml
module: M
commands:
  - id: <短横线 id>
    desc: "中文一句话"
    cmd: "python learning/M/src/<script>.py [--flag {param}]"   # README 原文形态，repo-root 相对
    full:  { param: <真实预算> }      # 有可调预算才需要
    smoke: { param: <最小预算> }      # 验证器实际跑的
    tier: V1                          # V1=要 smoke 跑；V0=只静态检查
    v0: false                         # 脚本无 argparse 时必须 false（跳过 --help 探针）
    gpu: true|false
```
- 有 argparse 的训练脚本：给 smoke 最小预算（steps/iters/batch 调到最小）。
- 无 argparse 的直跑 demo：`v0: false`，无 smoke 参数。
- 测试套件归 V2（不进 runbook）；verify_env 归 --env（不进 runbook）。

### 3. 跑 V0+V1
```bash
.venv/Scripts/python.exe scripts/eric_3080ti_env_audit.py --runbook --modules M --timeout 180 \
  --json-out docs/local-env/ERIC-3080Ti-runbook-results.json --md-out docs/local-env/ERIC-3080Ti-runbook-matrix.md
```
读 results.json 里 FAIL/TIMEOUT 的 stdout/stderr 定位问题。

### 4. 修复（关键）
- **先判性质**：代码 bug → 修代码 + 文档；文档/flag/路径错 → 修文档/runbook。
- **套用账本"系统性问题"目录的标准修法**（trl API 漂移→fail-fast+回退手写/minimal；`load_dataset("x")`→命名空间 id+离线回退；decoder-only 批量生成→`padding_side='left'`；"假成功 print+return"→显式失败或真实回退）。
- **可疑的秒级 PASS**：训练类 demo 若 0.x 秒就 PASS，确认它不是 no-op/假成功（看 stdout 是否真有训练/计算输出）。纯数值 self-test 秒级是正常的。
- 重型/可选栈（vllm/verl/ray/playwright/大权重）跑不动 → **尽量改出 3080 Ti 真实可跑的缩小版**；实在不行才 `tier: skip` + README 标注 + 留 mock 路径。
- 重跑 step 3 直到 V0/V1 全 PASS（或有理由的 skip）。
- **不要 hack**：遇到需要大改架构、或你拿不准的 novel 破坏 → **停手，在报告里"ESCALATE"小节描述清楚**，留给编排者 inline 决策。

### 5. V2 复核
- 若你**改过该模块代码** → 跑 `.venv/Scripts/python.exe scripts/eric_3080ti_env_audit.py --modules M --tests --timeout 600`，确认仍绿。
- 若**没改代码** → V2 记"基线绿"（基线 46/46 已 PASS），不必重跑。

### 6. README 运行段
在 `learning/M/README.md` 加/改一个 `## 运行验证（Runbook）`（或 `### `）段，仿照 rl-foundations/prompt-tuning 模板：指向 runbook.yaml + `--runbook` 一键命令 + 列出可跑入口（full 形态 + smoke 提示）+ 关键坑注记 + 测试(V2)入口。

## 返回报告（结构化）

```
MODULE: M
V0: PASS/—(n/a)    V1: x/y PASS    V2: PASS(改码重跑) / 基线绿(未改码)
CODE_CHANGES: <改了哪些文件 + 一句话原因；没有则 none>
FILES_WRITTEN: runbook.yaml, README.md (+改的 src)
SYSTEMIC_HITS: <命中哪些坑目录条目；没有则 none>
ESCALATE: <需编排者决策的 novel 问题；没有则 none>
NOTES: <异常/可疑点，如秒级 PASS 的判断依据>
```
不要提交。把改动留在工作区。
