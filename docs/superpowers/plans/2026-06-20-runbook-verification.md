# Runbook 验证 & README 框架 — 实现计划

> **For agentic workers:** 用 superpowers:executing-plans 逐任务执行。步骤用 `- [ ]` 跟踪。
> **本计划为自主执行**：用户已授权"不用反复批准，自主测试维护直到完成，中途会多次 compact"。**不要停下来问批准**；每 ~3 模块更新账本 + commit。

**Goal:** 让"照着每个模块文档的运行指示走"真的能跑通——验证文档入口命令（非仅单元测试）、修代码/文档不一致、补齐 README 缺口；全程自主可续。

**Architecture:** 给现有 `scripts/eric_3080ti_env_audit.py` 加 `--runbook` 模式（复用其 UTF-8/隔离 PYTHONPATH/超时管线）。每模块一份 `runbook.yaml` 登记文档入口命令 + smoke 预算，作为"文档=被测命令"的单一事实源。状态全部落盘到活动账本，扛 compact。

**Tech Stack:** Python 3.13 / repo-local `.venv` / PyYAML / pytest / torch 2.11+cu128 / RTX 3080 Ti 16GB。

**关联文档：**
- 标准/方法：[spec](../specs/2026-06-20-runbook-verification-design.md)
- 活动账本（续跑入口）：[ERIC-3080Ti-runbook-progress.md](../../local-env/ERIC-3080Ti-runbook-progress.md)

---

## 🧭 RESUME PROTOCOL（compact 后）

读 spec + 本计划 + 活动账本 → 账本"模块矩阵"里第一个 V1 非 ✅/⏭ 的模块就是下一个 → 执行下方"Phase 1 每模块循环" → 每 ~3 模块更新账本 + commit。分支固定 `ERIC-3080Ti/paper-guides`。

---

## File Structure

| 文件 | 责任 | 动作 |
|---|---|---|
| `scripts/eric_3080ti_env_audit.py` | 加 `--runbook` 模式 | Modify |
| `scripts/tests/test_runbook_mode.py` | runbook 加载/格式化单测 | Create |
| `learning/<module>/runbook.yaml` | 该模块文档入口命令清单（46 份）| Create ×46 |
| `learning/<module>/README.md`「运行验证」段 | 与 runbook.yaml 对齐 | Modify ×39 / Create ×7 |
| `docs/local-env/ERIC-3080Ti-runbook-progress.md` | 活动账本 | Modify（持续）|
| `docs/local-env/ERIC-3080Ti-runbook-matrix.md` / `.json` | 验证器输出 | Create（生成物）|
| `README.md`（根）| 知识组织框架 + 怎么跑 | Create |

---

## Phase 0 — 工具 + Pilot（rl-foundations）

目标：把 `--runbook` 模式、`runbook.yaml` 模板、README「运行验证」段模板，在 rl-foundations 上一次打磨好，去工具风险。

### Task 0.1: 确认 PyYAML + 分支

- [ ] **Step 1: 确认在工作分支**

Run: `git rev-parse --abbrev-ref HEAD`
Expected: `ERIC-3080Ti/paper-guides`

- [ ] **Step 2: 确认 PyYAML 可用**

Run: `.\.venv\Scripts\python.exe -c "import yaml; print(yaml.__version__)"`
Expected: 打印版本号。若 ImportError → `.\.venv\Scripts\python.exe -m pip install pyyaml` 后重试。

### Task 0.2: 给 harness 加 `--runbook` 模式（TDD）

**Files:** Modify `scripts/eric_3080ti_env_audit.py`；Create `scripts/tests/test_runbook_mode.py`

- [ ] **Step 1: 写失败测试**

Create `scripts/tests/test_runbook_mode.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import eric_3080ti_env_audit as audit


def test_format_cmd_substitutes_and_swaps_python():
    tokens = audit._format_cmd("python src/x.py --steps {steps}", {"steps": 2000})
    assert tokens[0] == sys.executable
    assert tokens[-2:] == ["--steps", "2000"]


def test_format_cmd_no_params_is_literal():
    tokens = audit._format_cmd("python src/x.py --flag", {})
    assert tokens[1:] == ["src/x.py", "--flag"]


def test_script_of_finds_py():
    assert audit._script_of([sys.executable, "src/x.py", "--a"]) == "src/x.py"
    assert audit._script_of([sys.executable, "-m", "pytest"]) is None


def test_load_runbook_reads_commands(tmp_path, monkeypatch):
    mod = tmp_path / "learning" / "demo"
    mod.mkdir(parents=True)
    (mod / "runbook.yaml").write_text(
        "module: demo\ncommands:\n  - id: a\n    cmd: 'python src/x.py'\n    tier: V0\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(audit, "LEARNING", tmp_path / "learning")
    cmds = audit._load_runbook("demo")
    assert cmds[0]["id"] == "a"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.\.venv\Scripts\python.exe -m pytest scripts/tests/test_runbook_mode.py -q`
Expected: FAIL（`_format_cmd` / `_script_of` / `_load_runbook` 未定义）。

- [ ] **Step 3: 实现 runbook 函数 + 接线**

在 `scripts/eric_3080ti_env_audit.py` 顶部 import 区加 `import shlex`。在 `_run_test_command` 之后插入：

```python
RUNBOOK_NAME = "runbook.yaml"


def _load_runbook(module: str) -> list[dict]:
    path = LEARNING / module / RUNBOOK_NAME
    if not path.exists():
        return []
    import yaml
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data.get("commands", []) or []


def _format_cmd(template: str, params: dict | None) -> list[str]:
    text = template.format(**(params or {}))
    tokens = [t.strip('"').strip("'") for t in shlex.split(text, posix=False)]
    if tokens and tokens[0] in {"python", "python3", "py"}:
        tokens[0] = sys.executable
    return tokens


def _script_of(tokens: list[str]) -> str | None:
    for t in tokens:
        if t.endswith(".py"):
            return t
    return None


def _run_runbook(module: str, timeout: int) -> list[RunResult]:
    results: list[RunResult] = []
    for entry in _load_runbook(module):
        cid = entry.get("id", "?")
        tier = entry.get("tier", "V1")
        tokens = _format_cmd(entry["cmd"], entry.get("smoke", {}))
        script = _script_of(tokens)
        # V0: 脚本存在 + --help
        if script is not None:
            exists = (ROOT / script).exists() or Path(script).exists()
            if not exists:
                results.append(RunResult(module, f"runbook-v0:{cid}", tokens,
                                         "FAIL", None, 0.0, "", f"script not found: {script}"))
            else:
                results.append(_run(module, f"runbook-v0:{cid}",
                                    [sys.executable, script, "--help"], min(timeout, 120)))
        # V1: smoke 跑通
        if tier == "V1":
            results.append(_run(module, f"runbook-v1:{cid}", tokens, timeout))
    return results
```

在 `main()` 的 argparse 加：

```python
    parser.add_argument("--runbook", action="store_true", help="run documented runbook.yaml commands")
```

在 `main()` 模块循环里，最前面加 runbook 分支（在 `_commands_for` 之前）：

```python
        if args.runbook:
            for result in _run_runbook(module, args.timeout):
                print(f"  -> {result.kind} {result.status} ({result.seconds:.2f}s)", flush=True)
                results.append(result)
                _write_json(results, args.json_out)
                _write_md(results, args.md_out)
            if not args.env and not args.tests:
                continue
```

并把 `if not args.env and not args.tests: args.tests = True` 改为只在**未指定 runbook 时**默认开 tests：

```python
    if not args.env and not args.tests and not args.runbook:
        args.tests = True
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.\.venv\Scripts\python.exe -m pytest scripts/tests/test_runbook_mode.py -q`
Expected: PASS（4 passed）。

- [ ] **Step 5: commit**

```bash
git add scripts/eric_3080ti_env_audit.py scripts/tests/test_runbook_mode.py
git commit -m "feat(audit): add --runbook mode to verify documented entry-point commands"
```

### Task 0.3: 写 rl-foundations 的 `runbook.yaml`

**Files:** Create `learning/rl-foundations/runbook.yaml`

- [ ] **Step 1**: 读 `learning/rl-foundations/README.md`「实用入口」段 + `environment/verify_env.py`，把每条"学生会敲的命令"抄成 `runbook.yaml`。命令写成**与 README 一致的 repo-root 相对形态**（如 `python learning/rl-foundations/src/cartpole_full.py ...`）。每条给 `smoke` 最小预算。模板见 spec §4。至少覆盖：cartpole 5 算法、capstone、verify_env。
- [ ] **Step 2: commit**：`git add learning/rl-foundations/runbook.yaml && git commit -m "docs(rl-foundations): add runbook.yaml"`

### Task 0.4: 在 rl-foundations 上跑 V0+V1+V2，修到全绿

- [ ] **Step 1: 跑 runbook 验证**

Run:
```powershell
$env:PYTHONUTF8="1"; $env:PYTHONIOENCODING="utf-8"
.\.venv\Scripts\python.exe scripts\eric_3080ti_env_audit.py --runbook --modules rl-foundations --timeout 600 --json-out docs\local-env\ERIC-3080Ti-runbook-results.json --md-out docs\local-env\ERIC-3080Ti-runbook-matrix.md
```
Expected: 每条命令 `runbook-v0:* PASS` 且 `runbook-v1:* PASS`。

- [ ] **Step 2**: 对每个 FAIL/TIMEOUT：读 stderr（在 results.json）→ 判定"代码 bug"还是"文档/runbook 写错"→ **代码 bug 修代码 + 文档；文档错修文档/runbook**。重型命令若 smoke 仍超时 → 给脚本加更小的 smoke 友好 flag（如 `--total-steps`），并回填 runbook smoke 值。重复 Step 1 直到全绿。
- [ ] **Step 3: V2 复核**：`.\.venv\Scripts\python.exe scripts\eric_3080ti_env_audit.py --modules rl-foundations --tests --timeout 600` → Expected PASS（基线已绿；若被修复影响则修回）。
- [ ] **Step 4: commit**：`git commit -am "verify(rl-foundations): runbook V0+V1 green; fixes …"`（消息里列实际修的东西）

### Task 0.5: rl-foundations README「运行验证」段（模板）

**Files:** Modify `learning/rl-foundations/README.md`

- [ ] **Step 1**: 在 README 加/改一个固定标题 `## 运行验证（Runbook）` 段：列出 `runbook.yaml` 里每条命令的 `full` 形态（学生真实跑）+ 一句 smoke 提示 + 预期信号 + 一行"一键验证：`python scripts/eric_3080ti_env_audit.py --runbook --modules rl-foundations`"。这个段落即其余 45 个模块的模板。
- [ ] **Step 2: commit**：`git commit -am "docs(rl-foundations): add 运行验证 section aligned to runbook.yaml"`

### Task 0.6: 收尾 pilot

- [ ] **Step 1**: 更新活动账本 rl-foundations 行：README/runbook/V0/V1/V2 → ✅（修过的标 🩹 + Notes）；填 commit sha；更新顶部"最近进度"。
- [ ] **Step 2**: 把"系统性问题"记进账本（pilot 暴露的通用坑）。
- [ ] **Step 3: commit**：`git commit -am "docs: checkpoint ledger after rl-foundations pilot"`

---

## Phase 1 — Fan-out（其余 45 模块）

按账本"模块矩阵"课程序处理。**每个模块跑同一个循环**（下方），具体修复内容记进**账本**，不写进本计划（修复在验证时才发现，无法预先写死）。

### 每模块循环（可重复单元）

对模块 `M`：

- [ ] **1. runbook.yaml**：读 `learning/M/README.md` 运行段 + lectures 里的命令 + `environment/verify_env.py`，写 `learning/M/runbook.yaml`（命令 = README 原文形态；给 smoke 预算；`tier` 标 V0/V1）。
- [ ] **2. V0+V1**：跑 `python scripts/eric_3080ti_env_audit.py --runbook --modules M --timeout 600 --json-out docs/local-env/ERIC-3080Ti-runbook-results.json --md-out docs/local-env/ERIC-3080Ti-runbook-matrix.md`。读 json 里 FAIL 的 stderr。
- [ ] **3. 修复**：代码 bug → 修代码 + 文档；文档/flag/路径错 → 修文档/runbook。重型/可选栈（vllm/verl/ray/playwright/大权重）→ **尽量改出 3080 Ti 真实可跑的缩小版**（更小模型/更短序列/mock 数据）；真不行才 `tier: skip` + README 标注 WSL2-only + 留 mock smoke。重跑直到 V0/V1 绿。
- [ ] **4. V2**：`--modules M --tests` 复核仍绿（基线已绿，确认未被修坏）。
- [ ] **5. README 运行段**：按 Phase 0 模板加/修「运行验证」段，与 runbook.yaml 对齐。
- [ ] **6. 账本 + commit**：更新账本该行（✅/🩹/⏭ + Notes + commit sha）；`git commit -am "verify(M): runbook green; <实际修的>"`。

### 模块顺序（账本矩阵同序）

M1: `prompt-tuning-family` `lora-family` `adapter-tuning-family`
M3: `data-curation` `transformer-deep` `moe-architecture` `ssm-hybrid` `long-context` `scaling-infra` `pretraining-recipe` `small-model-graduation`
M4: `rlhf-classic` `dpo-family` `process-reward` `reasoning-r1` `rl-sota-2026` `multimodal-agent`（rl-foundations 已在 Phase 0）
M5: `inference-engine-core` `sglang-radixattention` `speculative-decoding` `quantization-deploy` `distributed-inference` `production-serving` `serving-graduation`
M6: `eval-foundations` `reasoning-eval` `agent-code-eval` `llm-judge-arena` `red-team-jailbreak` `safety-defense` `eval-graduation`
M7: `agent-foundations` `rag-essential` `tool-use-mcp` `multi-agent-orchestration` `agent-memory-context` `agent-framework-stack` `agent-graduation`
M8: `gpu-architecture` `cuda-essentials` `kernel-engineering` `cluster-networking` `storage-dataops` `training-orchestration` `infra-graduation`
（M8 七个**同时缺 README** → 循环 Step 5 改为"新建 README"，用 Phase 2 Task 模板。）

### Checkpoint 纪律

- 每 ~3 模块：更新账本顶部"最近进度" + `git commit`。
- 任何长耗时（lora/adapter tests 数百秒）前：先更新账本，避免半路 compact 丢状态。

---

## Phase 2 — README 缺口

### Task 2.1: Module 8 的 7 个 README（fan-out 到 M8 时做）

**Files:** Create `learning/{gpu-architecture,cuda-essentials,kernel-engineering,cluster-networking,storage-dataops,training-orchestration,infra-graduation}/README.md`

- [ ] 以 `learning/rl-foundations/README.md` 为模板（专题概览表 / 学习路径图 / 目录结构 / 环境 / 横向对比 / cheatsheet / 自测题 / 运行验证段 / git 里程碑 / 跨专题衔接 / 验收清单）。内容据该模块 lectures + src + portfolio_v3.md Module 8 描述填实。每个 commit 一次。

### Task 2.2: 根 README.md（全部模块验证完后）

**Files:** Create `README.md`

- [ ] 知识组织框架：8 模块 / 46 专题地图（链到各模块 README）；7 件套模块解剖说明；`.venv` 安装（抄 final-report 的 Rebuild Commands）；"怎么运行/验证任意模块"（`--runbook` + tests 用法）；导航（docs/ keynotes/ interviews/ 是什么）。commit。

---

## 完成定义（DoD）—— ✅ 2026-07-12 全部达成

- [x] 46 模块每个都有 `runbook.yaml`，且 `--runbook` 全模块 V0/V1 = PASS 或有理由的 SKIP。（454/454 PASS，零 FAIL/SKIP）
- [x] 46 模块 README 都有与 runbook 对齐的「运行验证」段；M8 七个 README 已建。
- [x] 根 README 已建。（`README.md`，commit `cf66259`）
- [x] 账本矩阵全行收口（无 ⬜/🔧）。（`docs/local-env/ERIC-3080Ti-runbook-progress.md` 46 行全 ✅/🩹）
- [x] `--runbook --modules <all>` 一把过的最终 matrix 已生成并 commit。（`docs/local-env/ERIC-3080Ti-runbook-matrix.md`，commit `ee10ac0`）

---

## Self-Review（写完即查）

- **Spec 覆盖**：spec §3 标准→Phase 0/1 循环；§4 runbook.yaml→Task 0.3 + 循环 Step1；§5 工具→Task 0.2；§6 持久化→账本 + checkpoint 纪律；§7 README→Phase 2 + 循环 Step5；§9 顺序→Phase 0/1/2。✅ 全覆盖。
- **占位符**：Phase 0 工具任务含完整代码；Phase 1 是调查型循环（修复内容据设计无法预写，落账本）——已显式说明，非占位。
- **类型一致**：`_load_runbook/_format_cmd/_script_of/_run_runbook` 在 Task 0.2 定义并在循环中使用，命名一致；复用既有 `_run/RunResult/_write_json/_write_md`。
